//
// Created by JinHai on 2022/7/28.
//

#include "physical_table_scan.h"
#include "storage/storage.h"
#include "storage/table.h"
#include "storage/table_def.h"
#include "storage/data_block.h"
#include "storage/common/block_index.h"
#include "main/query_context.h"

#include "executor/operator_state.h"
#include "function/table/table_scan.h"
#include "planner/bound/base_table_ref.h"

#include "common/column_vector/column_vector.h"

namespace infinity {

void PhysicalTableScan::Init() {}

void PhysicalTableScan::Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) {
    auto *table_scan_input_state = static_cast<TableScanInputState *>(input_state);
    auto *table_scan_output_state = static_cast<TableScanOutputState *>(output_state);

    ExecuteInternal(query_context, table_scan_input_state, table_scan_output_state);

    //    const MultiIndex<u64, u64, SegmentEntry*>* segment_entry_index_array = *table_scan_input_state->table_scan_function_data_->segment_index_;
    //    for(const u64 segment_idx: segment_entry_index_array) {
    //        SegmentEntry* segment_entry = base_table_ref_->segment_entries_->at(segment_idx);
    //        LOG_TRACE("Segment Entry ID: {}", segment_entry->segment_id_);
    //    }

    //    table_scan_output_state->data_block_->Reset();
    //
    //    base_table_ref_->table_func_->main_function_(query_context,
    //                                                 table_scan_input_state->table_scan_function_data_.get(),
    //                                                 *table_scan_output_state->data_block_);
}

void PhysicalTableScan::Execute(QueryContext *query_context) {
    // Generate the result table definition
    Vector<SharedPtr<ColumnDef>> column_defs;
    size_t column_count = base_table_ref_->column_names_->size();
    for (size_t idx = 0; idx < column_count; ++idx) {
        // Use the column id to fetch column name and type
        String &col_name_ref = base_table_ref_->column_names_->at(idx);
        SharedPtr<DataType> col_type_ref = base_table_ref_->column_types_->at(idx);

        SharedPtr<ColumnDef> col_def = MakeShared<ColumnDef>(idx, col_type_ref, col_name_ref, HashSet<ConstraintType>());
        column_defs.emplace_back(col_def);
    }

    SharedPtr<TableDef> table_def_ptr = MakeShared<TableDef>(MakeShared<String>("default"), MakeShared<String>(base_table_ref_->alias_), column_defs);

    output_ = MakeShared<Table>(table_def_ptr, TableType::kResult);

    while (true) {
        SharedPtr<DataBlock> output_block = MakeShared<DataBlock>();
        output_block->Init(*base_table_ref_->column_types_);
        base_table_ref_->table_func_->main_function_(query_context, nullptr, *output_block);
        if (output_block->row_count() > 0) {
            output_->Append(output_block);
        } else {
            break;
        }
    }
}

SharedPtr<Vector<String>> PhysicalTableScan::GetOutputNames() const { return base_table_ref_->column_names_; }

SharedPtr<Vector<SharedPtr<DataType>>> PhysicalTableScan::GetOutputTypes() const { return base_table_ref_->column_types_; }

String PhysicalTableScan::table_alias() const { return base_table_ref_->alias_; }

u64 PhysicalTableScan::TableIndex() const { return base_table_ref_->table_index_; }

TableCollectionEntry *PhysicalTableScan::TableEntry() const { return base_table_ref_->table_entry_ptr_; }

SizeT PhysicalTableScan::BlockEntryCount() const { return base_table_ref_->block_index_->BlockCount(); }

BlockIndex *PhysicalTableScan::GetBlockIndex() const { return base_table_ref_->block_index_.get(); }

Vector<SizeT> &PhysicalTableScan::ColumnIDs() const { return base_table_ref_->column_ids_; }

Vector<SharedPtr<Vector<GlobalBlockID>>> PhysicalTableScan::PlanBlockEntries(i64 parallel_count) const {
    BlockIndex *block_index = base_table_ref_->block_index_.get();

    u64 all_block_count = block_index->BlockCount();
    u64 block_per_task = all_block_count / parallel_count;
    u64 residual = all_block_count % parallel_count;

    Vector<SharedPtr<Vector<GlobalBlockID>>> result(parallel_count, nullptr);
    for (u64 task_id = 0, global_block_id = 0, residual_idx = 0; task_id < parallel_count; ++task_id) {
        result[task_id] = MakeShared<Vector<GlobalBlockID>>();
        for (u64 block_id_in_task = 0; block_id_in_task < block_per_task; ++block_id_in_task) {
            result[task_id]->emplace_back(block_index->global_blocks_[global_block_id++]);
        }
        if (residual_idx < residual) {
            result[task_id]->emplace_back(block_index->global_blocks_[global_block_id++]);
            ++residual_idx;
        }
    }
    return result;
}

void PhysicalTableScan::ExecuteInternal(QueryContext *query_context,
                                        TableScanInputState *table_scan_input_state,
                                        TableScanOutputState *table_scan_output_state) {
    DataBlock *output_ptr = table_scan_output_state->data_block_.get();
    output_ptr->Reset();

    TableScanFunctionData *table_scan_function_data_ptr = table_scan_input_state->table_scan_function_data_.get();
    const BlockIndex *block_index = table_scan_function_data_ptr->block_index_;
    Vector<GlobalBlockID> *block_ids = table_scan_function_data_ptr->global_block_ids_.get();
    const Vector<SizeT> &column_ids = table_scan_function_data_ptr->column_ids_;
    i64 &block_ids_idx = table_scan_function_data_ptr->current_block_ids_idx_;
    if (block_ids_idx >= block_ids->size()) {
        // No data or all data is read
        table_scan_output_state->SetComplete();
        return;
    }

    SizeT &read_offset = table_scan_function_data_ptr->current_read_offset_;

    // Here we assume output is a fresh data block, we have never written anything into it.
    auto write_capacity = output_ptr->capacity();
    while (write_capacity > 0 && block_ids_idx < block_ids->size()) {
        i32 segment_id = block_ids->at(block_ids_idx).segment_id_;
        i16 block_id = block_ids->at(block_ids_idx).block_id_;

        BlockEntry *current_block_entry = block_index->GetBlockEntry(segment_id, block_id);

        auto remaining_rows = current_block_entry->row_count_ - read_offset;
        auto write_size = std::min(write_capacity, remaining_rows);

        SizeT output_column_id{0};
        for (auto column_id : column_ids) {
            ColumnBuffer column_buffer =
                BlockColumnEntry::GetColumnData(current_block_entry->columns_[column_id].get(), query_context->storage()->buffer_manager());
            output_ptr->column_vectors[output_column_id++]->AppendWith(column_buffer, read_offset, write_size);
        }

        // write_size = already read size = already write size
        write_capacity -= write_size;
        remaining_rows -= write_size;
        read_offset += write_size;

        // we have read all data from current segment, move to next block
        if (remaining_rows == 0) {
            ++block_ids_idx;
            read_offset = 0;
        }
    }
    if (block_ids_idx >= block_ids->size()) {
        table_scan_output_state->SetComplete();
    }

    output_ptr->Finalize();
}

} // namespace infinity
