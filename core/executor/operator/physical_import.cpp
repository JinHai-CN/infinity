//
// Created by JinHai on 2022/7/28.
//

module;

//extern "C" {
//#include "zsv/api.h"
//#include "zsv/common.h"
//}

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <vector>

import stl;
import txn;
import query_context;
import table_def;
import table;
import parser;
import physical_operator_type;
import operator_state;
import expression_state;
import data_block;
import logger;
import third_party;
import local_file_system;
import defer_op;
import txn_store;
import block_column_entry;
import block_entry;
import db_entry;
import table_collection_meta;
import wal_entry;
import file_system_type;
import file_system;
import object_handle;
import infinity_assert;
import infinity_exception;
import table_collection_entry;
import segment_entry;


module physical_import;

namespace infinity {

void PhysicalImport::Init() {}

/**
 * @brief copy statement execute function
 * @param query_context
 * @param input_state
 * @param output_state
 */
void PhysicalImport::Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) {
    auto import_input_state = static_cast<ImportInputState *>(input_state);
    auto import_output_state = static_cast<ImportOutputState *>(output_state);
    switch (file_type_) {
        case CopyFileType::kCSV: {
            ImportCSV(query_context, import_input_state, import_output_state);
            break;
        }
        case CopyFileType::kJSON: {
            ImportJSON(query_context);
            break;
        }
        case CopyFileType::kFVECS: {
            ImportFVECS(query_context, import_input_state, import_output_state);
            break;
        }
    }
    output_state->SetComplete();
}

void PhysicalImport::Execute(QueryContext *query_context) {
    switch (file_type_) {
        case CopyFileType::kCSV: {
            return ImportCSV(query_context);
        }
        case CopyFileType::kJSON: {
            return ImportJSON(query_context);
        }
        case CopyFileType::kFVECS: {
            return ImportFVECS(query_context);
        }
    }
}

SizeT PhysicalImport::ImportFVECSHelper(QueryContext *query_context) {
    if (table_collection_entry_->columns_.size() != 1) {
        Error<ExecutorException>("FVECS file must have only one column.", __FILE_NAME__, __LINE__);
    }
    auto &column_type = table_collection_entry_->columns_[0]->column_type_;
    if (column_type->type() != kEmbedding) {
        Error<ExecutorException>("FVECS file must have only one embedding column.", __FILE_NAME__, __LINE__);
    }
    auto embedding_info = static_cast<EmbeddingInfo *>(column_type->type_info().get());
    if (embedding_info->Type() != kElemFloat) {
        Error<ExecutorException>("FVECS file must have only one embedding column with float element.", __FILE_NAME__, __LINE__);
    }

    LocalFileSystem fs;

    UniquePtr<FileHandler> file_handler = fs.OpenFile(file_path_, FileFlags::READ_FLAG, FileLockType::kReadLock);
    DeferFn defer_fn([&]() { fs.Close(*file_handler); });

    int dimension = 0;
    i64 nbytes = fs.Read(*file_handler, &dimension, sizeof(dimension));
    fs.Seek(*file_handler, 0);
    if (nbytes != sizeof(dimension)) {
        Error<ExecutorException>(Format("Read dimension which length isn't {}.", nbytes), __FILE_NAME__, __LINE__);
    }
    if (embedding_info->Dimension() != dimension) {
        Error<ExecutorException>(Format("Dimension in file ({}) doesn't match with table definition ({}).", dimension, embedding_info->Dimension()),
                                 __FILE_NAME__,
                                 __LINE__);
    }
    SizeT file_size = fs.GetFileSize(*file_handler);
    SizeT row_size = dimension * sizeof(FloatT) + sizeof(dimension);
    if (file_size % row_size != 0) {
        Error<ExecutorException>("Weird file size.", __FILE_NAME__, __LINE__);
    }
    SizeT vector_n = file_size / row_size;

    Txn *txn = query_context->GetTxn();
    const String &db_name = *TableCollectionEntry::GetDBEntry(table_collection_entry_)->db_name_;
    const String &table_name = *table_collection_entry_->table_collection_name_;
    txn->AddTxnTableStore(table_name, MakeUnique<TxnTableStore>(table_name, table_collection_entry_, txn));
    TxnTableStore *txn_store = txn->GetTxnTableStore(table_name);

    u64 segment_id = TableCollectionEntry::GetNextSegmentID(table_collection_entry_);
    SharedPtr<SegmentEntry> segment_entry = SegmentEntry::MakeNewSegmentEntry(table_collection_entry_,
                                                                              query_context->GetTxn()->TxnID(),
                                                                              segment_id,
                                                                              query_context->GetTxn()->GetBufferMgr());
    BlockEntry *last_block_entry = segment_entry->block_entries_.back().get();
    CommonObjectHandle object_handle(last_block_entry->columns_[0]->buffer_handle_);
    SizeT row_idx = 0;

    while (true) {
        int dim;
        nbytes = fs.Read(*file_handler, &dim, sizeof(dimension));
        if (dim != dimension or nbytes != sizeof(dimension)) {
            Error<ExecutorException>(Format("Dimension in file ({}) doesn't match with table definition ({}).", dim, dimension),
                                     __FILE_NAME__,
                                     __LINE__);
        }
        ptr_t dst_ptr = object_handle.GetData() + row_idx * sizeof(FloatT) * dimension;
        fs.Read(*file_handler, dst_ptr, sizeof(FloatT) * dimension);
        ++segment_entry->current_row_;
        ++last_block_entry->row_count_;

        if (BlockEntry::IsFull(last_block_entry)) {
            segment_entry->block_entries_.emplace_back(MakeUnique<BlockEntry>(segment_entry.get(),
                                                                              segment_entry->block_entries_.size(),
                                                                              segment_entry->column_count_,
                                                                              ++segment_entry->current_row_,
                                                                              txn->GetBufferMgr()));
            last_block_entry = segment_entry->block_entries_.back().get();
        }

        row_idx++;
        if (row_idx == vector_n) {
            txn->AddWalCmd(MakeShared<WalCmdImport>(db_name, table_name, *segment_entry->segment_dir_));
            txn_store->Import(segment_entry);
            break;
        }
        if (segment_entry->AvailableCapacity() == 0) {
            txn->AddWalCmd(MakeShared<WalCmdImport>(db_name, table_name, *segment_entry->segment_dir_));
            txn_store->Import(segment_entry);
            segment_id = TableCollectionEntry::GetNextSegmentID(table_collection_entry_);
            segment_entry = SegmentEntry::MakeNewSegmentEntry(table_collection_entry_,
                                                              query_context->GetTxn()->TxnID(),
                                                              segment_id,
                                                              query_context->GetTxn()->GetBufferMgr());

            last_block_entry = segment_entry->block_entries_.back().get();
            object_handle = CommonObjectHandle(last_block_entry->columns_[0]->buffer_handle_);
        }
    }
    return vector_n;
}

void PhysicalImport::ImportCSVHelper(QueryContext *query_context, ParserContext &parser_context) {
#if 0
    FILE *fp = fopen(file_path_.c_str(), "rb");
    if (!fp) {
        Error<ExecutorException>(strerror(errno), __FILE_NAME__, __LINE__);
    }
    struct zsv_opts opts {};

    // TODO: redesign parser_context
    parser_context.table_collection_entry_ = table_collection_entry_;

    parser_context.txn_ = query_context->GetTxn();

    parser_context.txn_->AddTxnTableStore(
        *table_collection_entry_->table_collection_name_,
        MakeUnique<TxnTableStore>(*table_collection_entry_->table_collection_name_, table_collection_entry_, parser_context.txn_));

    u64 segment_id = TableCollectionEntry::GetNextSegmentID(table_collection_entry_);
    parser_context.segment_entry_ =
        SegmentEntry::MakeNewSegmentEntry(table_collection_entry_, parser_context.txn_->TxnID(), segment_id, parser_context.txn_->GetBufferMgr());
    parser_context.delimiter_ = delimiter_;

    const String &db_name = *TableCollectionEntry::GetDBEntry(table_collection_entry_)->db_name_;
    const String &table_name = *table_collection_entry_->table_collection_name_;

    if (header_) {
        opts.row_handler = CSVHeaderHandler;
    } else {
        opts.row_handler = CSVRowHandler;
    }

    opts.delimiter = delimiter_;
    opts.stream = fp;
    opts.ctx = &parser_context;
    opts.buffsize = (1 << 20); // default buffer size 256k, we use 1M

    parser_context.parser_ = zsv_new(&opts);
    enum zsv_status csv_parser_status;
    while ((csv_parser_status = zsv_parse_more(parser_context.parser_)) == zsv_status_ok) {
        ;
    }

    zsv_finish(parser_context.parser_);
    // flush the last segment entry
    if (parser_context.segment_entry_->current_row_ > 0) {
        parser_context.txn_->AddWalCmd(MakeShared<WalCmdImport>(db_name, table_name, *parser_context.segment_entry_->segment_dir_));
        auto txn_store = parser_context.txn_->GetTxnTableStore(table_name);
        txn_store->Import(parser_context.segment_entry_);
    }

    zsv_delete(parser_context.parser_);

    fclose(fp);
    if (csv_parser_status != zsv_status_no_more_input) {
        if (parser_context.err_msg_.get() != nullptr) {
            Error<ExecutorException>(*parser_context.err_msg_, __FILE_NAME__, __LINE__);
        } else {
            String err_msg = (char *)zsv_parse_status_desc(csv_parser_status);
            Error<ExecutorException>(err_msg, __FILE_NAME__, __LINE__);
        }
    }
    table_collection_entry_->row_count_ += parser_context.row_count_;
#endif
}

void PhysicalImport::ImportFVECS(QueryContext *query_context) {
    SizeT row_count = ImportFVECSHelper(query_context);

    Vector<SharedPtr<ColumnDef>> column_defs;
    SharedPtr<TableDef> result_table_def_ptr = MakeShared<TableDef>(MakeShared<String>("default"), MakeShared<String>("Tables"), column_defs);
    output_ = MakeShared<Table>(result_table_def_ptr, TableType::kDataTable);

    UniquePtr<String> result_msg = MakeUnique<String>(Format("IMPORTED {} Rows", row_count));
    output_->SetResultMsg(Move(result_msg));
}

void PhysicalImport::ImportFVECS(QueryContext *query_context, ImportInputState *input_state, ImportOutputState *output_state) {
    SizeT row_count = ImportFVECSHelper(query_context);
    auto result_msg = MakeUnique<String>(Format("IMPORT {} Rows", row_count));
    output_state->result_msg_ = Move(result_msg);
}

void PhysicalImport::ImportCSV(QueryContext *query_context) {
    ParserContext parser_context{};

    ImportCSVHelper(query_context, parser_context);

    // Generate the result
    Vector<SharedPtr<ColumnDef>> column_defs;

    SharedPtr<TableDef> result_table_def_ptr = MakeShared<TableDef>(MakeShared<String>("default"), MakeShared<String>("Tables"), column_defs);
    output_ = MakeShared<Table>(result_table_def_ptr, TableType::kDataTable);

    UniquePtr<String> result_msg = MakeUnique<String>(Format("IMPORTED {} Rows", parser_context.row_count_));
    output_->SetResultMsg(Move(result_msg));
}

void PhysicalImport::ImportJSON(QueryContext *query_context) {
    DBEntry *db_entry = TableCollectionMeta::GetDBEntry(table_collection_entry_->table_collection_meta_);
    String err = Format("IMPORT Table: {}.{} FROM file: {} WITH format JSON",
                        *db_entry->db_name_,
                        *table_collection_entry_->table_collection_name_,
                        file_path_);
    Error<NotImplementException>(err, __FILE_NAME__, __LINE__);
}

/**
 * @brief copy statement import json function
 * @param query_context
 * @param input_state
 * @param output_state
 */
void PhysicalImport::ImportCSV(QueryContext *query_context, ImportInputState *input_state, ImportOutputState *output_state) {

    ParserContext parser_context{};
    ImportCSVHelper(query_context, parser_context);

    //    // Generate the result
    //    Vector<SharedPtr<ColumnDef>> column_defs;
    //    auto result_table_def_ptr
    //                = MakeShared<TableDef>(MakeShared<String>("default"), MakeShared<String>("Tables"), column_defs);
    //    output_state->table_def_ = Move(result_table_def_ptr);

    auto result_msg = MakeUnique<String>(Format("IMPORT {} Rows", parser_context.row_count_));
    output_state->result_msg_ = Move(result_msg);
}

/**
 * @brief copy statement import csv function
 * @param query_context
 * @param input_state
 * @param output_state
 */
void PhysicalImport::ImportJSON(QueryContext *query_context, ImportInputState *input_state, ImportOutputState *output_state) {}

void PhysicalImport::CSVHeaderHandler(void *context) {
#if 0
    ParserContext *parser_context = static_cast<ParserContext *>(context);
    SizeT csv_column_count = zsv_cell_count(parser_context->parser_);

    SizeT table_column_count = parser_context->table_collection_entry_->columns_.size();
    if (csv_column_count != table_column_count) {
        parser_context->err_msg_ = MakeShared<String>(Format("Unmatched column count ({} != {})", csv_column_count, table_column_count));

        zsv_abort(parser_context->parser_); // return zsv_status_cancelled
        return;
    }

    // Not check the header column name
    for (SizeT idx = 0; idx < csv_column_count; ++idx) {
        auto *csv_col_name = reinterpret_cast<const char *>(zsv_get_cell_str(parser_context->parser_, idx));
        auto *table_col_name = parser_context->table_collection_entry_->columns_[idx]->name().c_str();
        if (!strcmp(csv_col_name, table_col_name)) {
            parser_context->err_msg_ = MakeShared<String>(Format("Unmatched column name({} != {})", csv_col_name, table_col_name));

            zsv_abort(parser_context->parser_); // return zsv_status_cancelled
            return;
        }
    }

    // This is header, doesn't count in row number

    zsv_set_row_handler(parser_context->parser_, CSVRowHandler);
#endif
}

namespace {
Vector<StringView> SplitArrayElement(StringView data, char delimiter) {
    SizeT data_size = data.size();
    if (data_size < 2 || data[0] != '[' || data[data_size - 1] != ']') {
        Error<TypeException>("Embedding data must be surrounded by [ and ]", __FILE_NAME__, __LINE__);
    }
    Vector<StringView> ret;
    SizeT i = 1, j = 1;
    while (true) {
        if (data[i] == delimiter || data[i] == ' ' || i == data_size - 1) {
            if (i > j) {
                ret.emplace_back(data.begin() + j, data.begin() + i);
            }
            j = i + 1;
        }
        if (i == data_size - 1) {
            break;
        }
        i++;
    }
    return ret;
}

template <typename T>
void AppendSimpleData(BlockColumnEntry *column_data_entry, const StringView &str_view, SizeT dst_offset) {
    T ele = DataType::StringToValue<T>(str_view);
    BlockColumnEntry::AppendRaw(column_data_entry, dst_offset, reinterpret_cast<ptr_t>(&ele), sizeof(T));
}

template <typename T>
void AppendEmbeddingData(BlockColumnEntry *column_data_entry, const Vector<StringView> &ele_str_views, SizeT dst_offset) {
    SizeT arr_len = ele_str_views.size();
    auto tmp_buffer = MakeUnique<T[]>(arr_len);
    for (SizeT ele_idx = 0; auto &ele_str_view : ele_str_views) {
        T ele = DataType::StringToValue<T>(ele_str_view);
        tmp_buffer[ele_idx++] = ele;
    }
    BlockColumnEntry::AppendRaw(column_data_entry, dst_offset, reinterpret_cast<ptr_t>(tmp_buffer.get()), sizeof(T) * arr_len);
}

void AppendVarcharData(BlockColumnEntry *column_data_entry, const StringView &str_view, SizeT dst_offset) {
    const char_t *tmp_buffer = str_view.data();
    auto varchar_type = MakeUnique<VarcharT>(str_view.data(), str_view.size());
    // TODO shenyushi: unnecessary copy here.
    BlockColumnEntry::AppendRaw(column_data_entry, dst_offset, reinterpret_cast<ptr_t>(varchar_type.get()), sizeof(VarcharT));
}

} // namespace

void PhysicalImport::CSVRowHandler(void *context) {
#if 0
    ParserContext *parser_context = static_cast<ParserContext *>(context);

    auto *table = parser_context->table_collection_entry_;
    SizeT column_count = zsv_cell_count(parser_context->parser_);
    auto *txn = parser_context->txn_;
    auto txn_store = txn->GetTxnTableStore(*table->table_collection_name_);

    auto segment_entry = parser_context->segment_entry_;
    const String &db_name = *TableCollectionEntry::GetDBEntry(table)->db_name_;
    const String &table_name = *table->table_collection_name_;
    // we have already used all space of the segment
    if (segment_entry->AvailableCapacity() == 0) {
        // add to txn_store
        txn->AddWalCmd(MakeShared<WalCmdImport>(db_name, table_name, *segment_entry->segment_dir_));
        txn_store->Import(segment_entry);

        // create new segment entry
        parser_context->segment_entry_ =
            SegmentEntry::MakeNewSegmentEntry(table, txn->TxnID(), TableCollectionEntry::GetNextSegmentID(table), txn->GetBufferMgr());
        segment_entry = parser_context->segment_entry_;
    }

    SizeT write_row = segment_entry->current_row_;

    BlockEntry *last_block_entry = segment_entry->block_entries_.back().get();
    if (BlockEntry::IsFull(last_block_entry)) {
        segment_entry->block_entries_.emplace_back(MakeUnique<BlockEntry>(segment_entry.get(),
                                                                          segment_entry->block_entries_.size(),
                                                                          segment_entry->column_count_,
                                                                          write_row,
                                                                          txn->GetBufferMgr()));
        last_block_entry = segment_entry->block_entries_.back().get();
    }

    // append data to segment entry
    for (SizeT column_idx = 0; column_idx < column_count; ++column_idx) {
        struct zsv_cell cell = zsv_get_cell(parser_context->parser_, column_idx);
        StringView str_view{};
        if (cell.len) {
            str_view = StringView((char *)cell.str, cell.len);
        }
        BlockColumnEntry *block_column_entry = last_block_entry->columns_[column_idx].get();
        auto column_type = block_column_entry->column_type_.get();
        SizeT dst_offset = write_row * column_type->Size();
        if (column_type->type() == kVarchar) {
            auto varchar_info = dynamic_cast<VarcharInfo *>(column_type->type_info().get());
            if (varchar_info->dimension() < str_view.size()) {
                Error<ExecutorException>("Varchar data size exceeds dimension.", __FILE_NAME__, __LINE__);
            }

            AppendVarcharData(block_column_entry, str_view, dst_offset);
        } else if (column_type->type() == kEmbedding) {
            Vector<StringView> res;
            auto ele_str_views = SplitArrayElement(str_view, parser_context->delimiter_);
            auto embedding_info = dynamic_cast<EmbeddingInfo *>(column_type->type_info().get());
            if (embedding_info->Dimension() < ele_str_views.size()) {
                Error<ExecutorException>("Embedding data size exceeds dimension.", __FILE_NAME__, __LINE__);
            }

            switch (embedding_info->Type()) {
                case kElemBit: {
                    Error<ExecutorException>("Embedding bit type is not implemented.", __FILE_NAME__, __LINE__);
                }
                case kElemInt8: {
                    AppendEmbeddingData<TinyIntT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemInt16: {
                    AppendEmbeddingData<SmallIntT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemInt32: {
                    AppendEmbeddingData<IntegerT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemInt64: {
                    AppendEmbeddingData<BigIntT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemFloat: {
                    AppendEmbeddingData<FloatT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemDouble: {
                    AppendEmbeddingData<DoubleT>(block_column_entry, ele_str_views, dst_offset);
                    break;
                }
                case kElemInvalid: {
                    Error<ExecutorException>("Embedding element type is invalid.", __FILE_NAME__, __LINE__);
                }
            }
        } else {
            switch (column_type->type()) {
                case kBoolean: {
                    AppendSimpleData<BooleanT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kTinyInt: {
                    AppendSimpleData<TinyIntT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kSmallInt: {
                    AppendSimpleData<SmallIntT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kInteger: {
                    AppendSimpleData<IntegerT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kBigInt: {
                    AppendSimpleData<BigIntT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kFloat: {
                    AppendSimpleData<FloatT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kDouble: {
                    AppendSimpleData<DoubleT>(block_column_entry, str_view, dst_offset);
                    break;
                }
                case kMissing:
                case kInvalid: {
                    Error<ExecutorException>("Invalid data type", __FILE_NAME__, __LINE__);
                }
                default: {
                    Error<ExecutorException>("Not supported now in append data in column", __FILE_NAME__, __LINE__);
                }
            }
        }
    }
    ++last_block_entry->row_count_;
    ++segment_entry->current_row_;
    ++parser_context->row_count_;
#endif
}

} // namespace infinity
