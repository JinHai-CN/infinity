//
// Created by jinhai on 23-8-18.
//

#include "storage/meta/entry/table_collection_entry.h"
#include "parser/statement/extra/create_table_info.h"
#include "storage/index_def/ivfflat_index_def.h"
#include "storage/meta/table_collection_meta.h"
#include "storage/txn/txn_store.h"
#include "storage/common/block_index.h"

namespace infinity {

TableCollectionEntry::TableCollectionEntry(const SharedPtr<String> &db_entry_dir,
                                           SharedPtr<String> table_collection_name,
                                           const Vector<SharedPtr<ColumnDef>> &columns,
                                           TableCollectionType table_collection_type,
                                           TableCollectionMeta *table_collection_meta,
                                           u64 txn_id,
                                           TxnTimeStamp begin_ts)
    : BaseEntry(EntryType::kTable),
      table_entry_dir_(MakeShared<String>(*db_entry_dir + "/" + *table_collection_name + "/txn_" + std::to_string(txn_id))),
      table_collection_name_(std::move(table_collection_name)), columns_(columns), table_collection_type_(table_collection_type),
      table_collection_meta_(table_collection_meta) {
    SizeT column_count = columns.size();
    for (SizeT idx = 0; idx < column_count; ++idx) {
        name2id_[columns[idx]->name()] = idx;
    }

    begin_ts_ = begin_ts;
    txn_id_ = txn_id;
}

void TableCollectionEntry::Append(TableCollectionEntry *table_entry, Txn *txn_ptr, void *txn_store, BufferManager *buffer_mgr) {
    TxnTableStore *txn_store_ptr = (TxnTableStore *)txn_store;
    Txn *transaction_ptr = (Txn *)txn_ptr;
    AppendState *append_state_ptr = txn_store_ptr->append_state_.get();
    if (append_state_ptr->Finished()) {
        LOG_TRACE("No append is done.");
        return;
    }
    {
        std::unique_lock<RWMutex> rw_locker(table_entry->rw_locker_); // prevent another read conflict with this append operation

        // No segment, or unsealed_segment_ is already closed(flushed to disk).
        if (table_entry->unsealed_segment_ == nullptr || table_entry->unsealed_segment_->status_ != DataSegmentStatus::kSegmentOpen) {
            u64 next_segment_id = TableCollectionEntry::GetNextSegmentID(table_entry);
            SharedPtr<SegmentEntry> new_segment = SegmentEntry::MakeNewSegmentEntry(table_entry, table_entry->txn_id_, next_segment_id, buffer_mgr);
            table_entry->segments_.emplace(new_segment->segment_id_, new_segment);
            table_entry->unsealed_segment_ = new_segment.get();
            //            table_entry->unsealed_segment_->Init(this->definition_ptr_->columns(), dir_, buffer_mgr_);
            LOG_TRACE("Add a new segment");
        }
    }

    while (!append_state_ptr->Finished()) {
        SizeT current_row = append_state_ptr->current_count_;

        if (table_entry->unsealed_segment_->AvailableCapacity() == 0 && table_entry->unsealed_segment_->row_capacity_ > 0) {
            // uncommitted_segment is full
            std::unique_lock<RWMutex> rw_locker(table_entry->rw_locker_); // prevent another read conflict with this append operation
            // Need double-check
            if (table_entry->unsealed_segment_->AvailableCapacity() == 0) {
                SharedPtr<SegmentEntry> new_segment = SegmentEntry::MakeNewSegmentEntry(table_entry,
                                                                                        table_entry->txn_id_,
                                                                                        TableCollectionEntry::GetNextSegmentID(table_entry),
                                                                                        buffer_mgr);
                table_entry->segments_.emplace(new_segment->segment_id_, new_segment);
                table_entry->unsealed_segment_ = new_segment.get();
                LOG_TRACE("Add a new segment");
            }
        }
        // Append data from app_state_ptr to the buffer in segment. If append all data, then set finish.
        SegmentEntry::AppendData(table_entry->unsealed_segment_, txn_ptr, append_state_ptr, buffer_mgr);
        LOG_TRACE("Segment is appended with {} rows", append_state_ptr->current_count_ - current_row);
    }
}

UniquePtr<String>
TableCollectionEntry::Delete(TableCollectionEntry *table_entry, Txn *txn_ptr, DeleteState &delete_state, BufferManager *buffer_mgr) {
    NotImplementError("TableCollectionEntry::Delete");
    return nullptr;
}

UniquePtr<String> TableCollectionEntry::InitScan(TableCollectionEntry *table_entry, Txn *txn_ptr, ScanState &scan_state, BufferManager *buffer_mgr) {
    NotImplementError("TableCollectionEntry::InitScan");
    return nullptr;
}

UniquePtr<String>
TableCollectionEntry::Scan(TableCollectionEntry *table_entry, Txn *txn_ptr, const ScanState &scan_state, BufferManager *buffer_mgr) {
    NotImplementError("TableCollectionEntry::Scan");
    return nullptr;
}

void TableCollectionEntry::CommitAppend(TableCollectionEntry *table_entry,
                                        Txn *txn_ptr,
                                        const AppendState *append_state_ptr,
                                        BufferManager *buffer_mgr) {

    for (const auto &range : append_state_ptr->append_ranges_) {
        LOG_TRACE("Commit, segment: {}, block: {} start: {}, count: {}", range.segment_id_, range.block_id_, range.start_id_, range.row_count_);
        SegmentEntry *segment_ptr = table_entry->segments_[range.segment_id_].get();
        SegmentEntry::CommitAppend(segment_ptr, txn_ptr, range.block_id_, range.start_id_, range.row_count_);
    }
}

void TableCollectionEntry::RollbackAppend(TableCollectionEntry *table_entry, Txn *txn_ptr, void *txn_store) {
    auto *txn_store_ptr = (TxnTableStore *)txn_store;
    AppendState *append_state_ptr = txn_store_ptr->append_state_.get();
    NotImplementError("Not implemented");
}

UniquePtr<String>
TableCollectionEntry::CommitDelete(TableCollectionEntry *table_entry, Txn *txn_ptr, DeleteState &append_state, BufferManager *buffer_mgr) {
    NotImplementError("TableCollectionEntry::CommitDelete");
    return nullptr;
}

UniquePtr<String>
TableCollectionEntry::RollbackDelete(TableCollectionEntry *table_entry, Txn *txn_ptr, DeleteState &append_state, BufferManager *buffer_mgr) {
    NotImplementError("TableCollectionEntry::RollbackDelete");
    return nullptr;
}

UniquePtr<String> TableCollectionEntry::ImportAppendSegment(TableCollectionEntry *table_entry,
                                                            Txn *txn_ptr,
                                                            SharedPtr<SegmentEntry> segment,
                                                            AppendState &append_state,
                                                            BufferManager *buffer_mgr) {
    for (const auto &block_entry : segment->block_entries_) {
        append_state.append_ranges_.emplace_back(segment->segment_id_, block_entry->block_id_, 0, block_entry->row_count_);
    }

    std::unique_lock<RWMutex> rw_locker(table_entry->rw_locker_);
    table_entry->segments_.emplace(segment->segment_id_, std::move(segment));
    return nullptr;
}

EntryResult TableCollectionEntry::CreateIndex(TableCollectionEntry *table_entry, Txn *txn_ptr, SharedPtr<IndexDef> index_def) {
    const String &column_name = index_def->column_names()[0];
    u64 column_id = table_entry->GetColumnIdByName(column_name);
    ColumnDef &column_def = *table_entry->columns_[column_id];
    switch (index_def->method_type()) {
        case IndexMethod::kIVFFlat: {
            auto &column_def = *table_entry->columns_[column_id];
            if (column_def.column_type_->type() != LogicalType::kEmbedding) {
                StorageError("IVFFlat index should created on Embedding type column")
            }
            auto type_info = column_def.column_type_->type_info().get();
            auto embedding_info = (EmbeddingInfo *)type_info;
            if (embedding_info->Type() != EmbeddingDataType::kElemFloat) {
                StorageError("IVFFlat index should created on float embedding type column")
            }
            for (const auto &[_segment_id, segment_entry] : table_entry->segments_) {
                SegmentEntry::CreateIndexEmbedding(segment_entry.get(), txn_ptr, *index_def, column_id, embedding_info->Dimension());
            }
            break;
        }
        default: {
            NotImplementException("Not implemented.");
        }
    }
    // TODO shenyushi: change meta data
    // table_entry->indexes_.emplace(index_def->index_name(), index_def);
    return {.entry_ = nullptr, .err_ = nullptr};
}

SegmentEntry *TableCollectionEntry::GetSegmentByID(const TableCollectionEntry *table_entry, u64 id) {
    auto iter = table_entry->segments_.find(id);
    if (iter != table_entry->segments_.end()) {
        return iter->second.get();
    } else {
        return nullptr;
    }
}

DBEntry *TableCollectionEntry::GetDBEntry(const TableCollectionEntry *table_entry) {
    TableCollectionMeta *table_meta = (TableCollectionMeta *)table_entry->table_collection_meta_;
    return (DBEntry *)table_meta->db_entry_;
}

SharedPtr<BlockIndex> TableCollectionEntry::GetBlockIndex(TableCollectionEntry *table_collection_entry, u64 txn_id, TxnTimeStamp begin_ts) {
    //    SharedPtr<MultiIndex<u64, u64, SegmentEntry*>> result = MakeShared<MultiIndex<u64, u64, SegmentEntry*>>();
    SharedPtr<BlockIndex> result = MakeShared<BlockIndex>();
    std::shared_lock<RWMutex> rw_locker(table_collection_entry->rw_locker_);
    result->Reserve(table_collection_entry->segments_.size());

    for (const auto &segment_pair : table_collection_entry->segments_) {
        result->Insert(segment_pair.second.get(), txn_id);
    }

    return result;
}

nlohmann::json TableCollectionEntry::Serialize(const TableCollectionEntry *table_entry) {
    nlohmann::json json_res;

    json_res["table_entry_dir"] = *table_entry->table_entry_dir_;
    json_res["table_name"] = *table_entry->table_collection_name_;
    json_res["table_entry_type"] = table_entry->table_collection_type_;
    json_res["row_count"] = table_entry->row_count_;
    json_res["begin_ts"] = table_entry->begin_ts_;
    json_res["commit_ts"] = table_entry->commit_ts_.load();
    json_res["txn_id"] = table_entry->txn_id_.load();
    json_res["deleted"] = table_entry->deleted_;

    for (const auto &column_def : table_entry->columns_) {
        nlohmann::json column_def_json;
        column_def_json["column_type"] = column_def->type()->Serialize();
        column_def_json["column_id"] = column_def->id();
        column_def_json["column_name"] = column_def->name();

        for (const auto &column_constraint : column_def->constraints_) {
            column_def_json["constraints"].emplace_back(column_constraint);
        }

        json_res["column_definition"].emplace_back(column_def_json);
    }

    for (const auto &segment_pair : table_entry->segments_) {
        json_res["segments"].emplace_back(SegmentEntry::Serialize(segment_pair.second.get()));
    }
    u64 next_segment_id = table_entry->next_segment_id_;
    json_res["next_segment_id"] = next_segment_id;

    for (const auto &[_column_id, index_def] : table_entry->indexes_) {
        json_res["indexes"].emplace_back(index_def->Serialize());
    }

    return json_res;
}

UniquePtr<TableCollectionEntry>
TableCollectionEntry::Deserialize(const nlohmann::json &table_entry_json, TableCollectionMeta *table_meta, BufferManager *buffer_mgr) {
    nlohmann::json json_res;

    SharedPtr<String> table_entry_dir = MakeShared<String>(table_entry_json["table_entry_dir"]);
    SharedPtr<String> table_name = MakeShared<String>(table_entry_json["table_name"]);
    TableCollectionType table_entry_type = table_entry_json["table_entry_type"];
    u64 row_count = table_entry_json["row_count"];

    bool deleted = table_entry_json["deleted"];

    Vector<SharedPtr<ColumnDef>> columns;
    if (!deleted) {
        for (const auto &column_def_json : table_entry_json["column_definition"]) {
            SharedPtr<DataType> data_type = DataType::Deserialize(column_def_json["column_type"]);
            i64 column_id = column_def_json["column_id"];
            String column_name = column_def_json["column_name"];

            HashSet<ConstraintType> constraints;
            if (column_def_json.contains("constraints")) {
                for (const auto &column_constraint : column_def_json["constraints"]) {
                    ConstraintType constraint = column_constraint;
                    constraints.emplace(constraint);
                }
            }

            SharedPtr<ColumnDef> column_def = MakeShared<ColumnDef>(column_id, data_type, column_name, constraints);
            columns.emplace_back(column_def);
        }
    }

    u64 txn_id = table_entry_json["txn_id"];
    TxnTimeStamp begin_ts = table_entry_json["begin_ts"];

    UniquePtr<TableCollectionEntry> table_entry =
        MakeUnique<TableCollectionEntry>(table_entry_dir, table_name, columns, table_entry_type, table_meta, txn_id, begin_ts);
    table_entry->row_count_ = row_count;
    table_entry->next_segment_id_ = table_entry_json["next_segment_id"];
    if (table_entry_json.contains("segments")) {
        for (const auto &segment_json : table_entry_json["segments"]) {
            SharedPtr<SegmentEntry> segment_entry = SegmentEntry::Deserialize(segment_json, table_entry.get(), buffer_mgr);
            table_entry->segments_.emplace(segment_entry->segment_id_, segment_entry);
        }
    }

    table_entry->commit_ts_ = table_entry_json["commit_ts"];
    table_entry->deleted_ = deleted;

    if (table_entry_json.contains("indexes")) {
        for (const auto &index_json : table_entry_json["indexes"]) {
            IndexMethod type_method = index_json["type_method"];
            switch (type_method) {
                case IndexMethod::kIVFFlat: {
                    SharedPtr<IVFFlatIndexDef> index_def = IVFFlatIndexDef::Deserialize(index_json);
                    table_entry->indexes_.emplace(index_def->index_name(), index_def);
                    break;
                }
                default: {
                    NotImplementException("Not implemented.");
                }
            }
        }
    }

    return table_entry;
}

u64 TableCollectionEntry::GetColumnIdByName(const String &column_name) {
    auto it = name2id_.find(column_name);
    if (it == name2id_.end()) {
        StorageError(fmt::format("No column name: {}", column_name))
    }
    return it->second;
}

} // namespace infinity