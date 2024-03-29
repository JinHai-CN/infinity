//
// Created by JinHai on 2022/7/28.
//

module;

import stl;
import txn;
import query_context;
import table_def;
import table;
import parser;
import physical_operator_type;
import operator_state;

module physical_create_table;

namespace infinity {

PhysicalCreateTable::PhysicalCreateTable(SharedPtr<String> schema_name,
                                         SharedPtr<TableDef> table_def_ptr,
                                         SharedPtr<Vector<String>> output_names,
                                         SharedPtr<Vector<SharedPtr<DataType>>> output_types,
                                         ConflictType conflict_type,
                                         u64 table_index,
                                         u64 id)
    : PhysicalOperator(PhysicalOperatorType::kCreateTable, nullptr, nullptr, id), schema_name_(Move(schema_name)),
      output_names_(Move(output_names)), output_types_(Move(output_types)), conflict_type_(conflict_type), table_index_(table_index),
      table_def_ptr_(Move(table_def_ptr)) {}

PhysicalCreateTable::PhysicalCreateTable(SharedPtr<String> schema_name,
                                         const SharedPtr<PhysicalOperator> &input,
                                         SharedPtr<Vector<String>> output_names,
                                         SharedPtr<Vector<SharedPtr<DataType>>> output_types,
                                         ConflictType conflict_type,
                                         u64 table_index,
                                         u64 id)
    : PhysicalOperator(PhysicalOperatorType::kCreateTable, input, nullptr, id), schema_name_(Move(schema_name)),
      output_names_(Move(output_names)), output_types_(Move(output_types)), conflict_type_(conflict_type), table_index_(table_index) {}

void PhysicalCreateTable::Init() {}

void PhysicalCreateTable::Execute(QueryContext *query_context, InputState *input_state, OutputState *output_state) {
    auto txn = query_context->GetTxn();
    auto result = txn->CreateTable(*schema_name_, table_def_ptr_, conflict_type_);
    auto create_table_output_state = (CreateTableOutputState *)output_state;
    if (result.err_.get() != nullptr) {
        create_table_output_state->error_message_ = Move(result.err_);
    }
    output_state->SetComplete();
}

void PhysicalCreateTable::Execute(QueryContext *query_context) {
    //    ResponseError("Execute: Create table: " + table_def_ptr_->name());
    Txn *txn = query_context->GetTxn();
    txn->CreateTable(*schema_name_, table_def_ptr_, conflict_type_);

    // Generate the result
    Vector<SharedPtr<ColumnDef>> column_defs = {
        MakeShared<ColumnDef>(0, MakeShared<DataType>(LogicalType::kInteger), "OK", HashSet<ConstraintType>())};

    output_ = Table::MakeResultTable(column_defs);
}

} // namespace infinity
