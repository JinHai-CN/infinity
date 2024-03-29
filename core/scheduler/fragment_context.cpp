//
// Created by jinhai on 23-9-8.
//

module;

#include <memory>

import stl;
import parser;
import fragment_task;
import infinity_assert;
import infinity_exception;
import operator_state;
import physical_operator;
import physical_operator_type;

import table_scan_function_data;
import knn_scan_data;
import physical_table_scan;
import physical_knn_scan;
import physical_aggregate;

import global_block_id;
import knn_expression;
import value_expression;
import column_expression;
import third_party;
import query_context;
import physical_source;
import physical_sink;
import table;
import data_block;

import plan_fragment;

module fragment_context;

namespace infinity {

template <typename InputStateType, typename OutputStateType>
void BuildSerialTaskStateTemplate(Vector<PhysicalOperator *> &fragment_operators,
                                  Vector<UniquePtr<FragmentTask>> &tasks,
                                  i64 operator_id,
                                  i64 operator_count,
                                  FragmentContext *parent_fragment_context) {
    if (tasks.size() != 1) {
        Error<SchedulerException>("Serial fragment type will only have one task(one concurrency)", __FILE_NAME__, __LINE__);
    }

    FragmentTask *task_ptr = tasks.back().get();

    task_ptr->operator_input_state_[operator_id] = MakeUnique<InputStateType>();
    auto current_operator_input_state = (InputStateType *)(task_ptr->operator_input_state_[operator_id].get());

    task_ptr->operator_output_state_[operator_id] = MakeUnique<OutputStateType>();
    auto current_operator_output_state = (OutputStateType *)(task_ptr->operator_output_state_[operator_id].get());
    auto output_types = fragment_operators[operator_id]->GetOutputTypes();
    if (output_types != nullptr) {
        current_operator_output_state->data_block_ = DataBlock::Make();
        current_operator_output_state->data_block_->Init(*output_types);
    }

    if (operator_id == operator_count - 1) {
        // Set first operator input as the output of source operator
        SourceState *source_state = task_ptr->source_state_.get();
        source_state->SetNextState(current_operator_input_state);
    }

    if (operator_id > 0 && operator_id < operator_count) {
        OutputState *prev_operator_output_state = task_ptr->operator_output_state_[operator_id - 1].get();
        current_operator_input_state->ConnectToPrevOutputOpState(prev_operator_output_state);
        //        current_operator_input_state->input_data_block_ = prev_operator_output_state->data_block_.get();
    }

    if (operator_id == 0) {
        // Set last operator output as the input of sink operator
        SinkState *sink_state = task_ptr->sink_state_.get();
        sink_state->SetPrevState(current_operator_output_state);

        if (parent_fragment_context != nullptr) {
            // this fragment has parent fragment, which means the sink node of current fragment
            // will sink data to the parent fragment.
            switch (sink_state->state_type_) {
                case SinkStateType::kQueue: {
                    auto *queue_sink_state = static_cast<QueueSinkState *>(sink_state);

                    for (const auto &next_fragment_task : parent_fragment_context->Tasks()) {
                        auto *next_fragment_source_state = static_cast<QueueSourceState *>(next_fragment_task->source_state_.get());
                        queue_sink_state->fragment_data_queues_.emplace_back(&next_fragment_source_state->source_queue_);
                    }

                    break;
                }
                case SinkStateType::kInvalid: {
                    Error<SchedulerException>("Invalid sink operator state type.", __FILE_NAME__, __LINE__);
                }
                default: {
                    break;
                }
            }
        }
    }
}

template <typename InputStateType, typename OutputStateType>
void BuildParallelTaskStateTemplate(Vector<PhysicalOperator *> &fragment_operators,
                                    Vector<UniquePtr<FragmentTask>> &tasks,
                                    i64 operator_id,
                                    i64 operator_count,
                                    FragmentContext *parent_fragment_context,
                                    i64 real_parallel_size) {
    for (i64 task_id = 0; task_id < real_parallel_size; ++task_id) {
        FragmentTask *task_ptr = tasks[task_id].get();

        task_ptr->operator_input_state_[operator_id] = MakeUnique<InputStateType>();
        auto current_operator_input_state = (InputStateType *)(task_ptr->operator_input_state_[operator_id].get());

        task_ptr->operator_output_state_[operator_id] = MakeUnique<OutputStateType>();
        auto current_operator_output_state = (OutputStateType *)(task_ptr->operator_output_state_[operator_id].get());

        current_operator_output_state->data_block_ = DataBlock::Make();
        current_operator_output_state->data_block_->Init(*fragment_operators[operator_id]->GetOutputTypes());

        if (operator_id == operator_count - 1) {
            // Set first operator input as the output of source operator
            SourceState *source_state = tasks[task_id]->source_state_.get();
            source_state->SetNextState(current_operator_input_state);
        }

        if (operator_id >= 0 && operator_id < operator_count - 1) {
            OutputState *prev_operator_output_state = task_ptr->operator_output_state_[operator_id + 1].get();
            current_operator_input_state->ConnectToPrevOutputOpState(prev_operator_output_state);
            //            current_operator_input_state->input_data_block_ = prev_output_state->data_block_.get();
        }

        if (operator_id == 0) {
            // Set last operator output as the input of sink operator
            SinkState *sink_state = tasks[task_id]->sink_state_.get();
            sink_state->SetPrevState(current_operator_output_state);

            if (parent_fragment_context != nullptr) {
                // this fragment has parent fragment, which means the sink node of current fragment
                // will sink data to the parent fragment.
                switch (sink_state->state_type_) {
                    case SinkStateType::kQueue: {
                        auto *queue_sink_state = static_cast<QueueSinkState *>(sink_state);

                        for (const auto &next_fragment_task : parent_fragment_context->Tasks()) {
                            auto *next_fragment_source_state = static_cast<QueueSourceState *>(next_fragment_task->source_state_.get());
                            queue_sink_state->fragment_data_queues_.emplace_back(&next_fragment_source_state->source_queue_);
                        }

                        break;
                    }
                    case SinkStateType::kInvalid: {
                        Error<SchedulerException>("Invalid sink operator state type.", __FILE_NAME__, __LINE__);
                    }
                    default: {
                        break;
                    }
                }
            }
        }
    }
}

template <>
void BuildParallelTaskStateTemplate<TableScanInputState, TableScanOutputState>(Vector<PhysicalOperator *> &fragment_operators,
                                                                               Vector<UniquePtr<FragmentTask>> &tasks,
                                                                               i64 operator_id,
                                                                               i64 operator_count,
                                                                               FragmentContext *parent_fragment_context,
                                                                               i64 real_parallel_size) {

    // TableScan must be the first operator of the fragment
    if (operator_id != operator_count - 1) {
        Error<SchedulerException>("Table scan operator must be the first operator of the fragment.", __FILE_NAME__, __LINE__);
    }

    if (operator_id == 0) {
        Error<SchedulerException>("Table scan shouldn't be the last operator of the fragment.", __FILE_NAME__, __LINE__);
    }

    PhysicalOperator *physical_op = fragment_operators[operator_id];
    if (physical_op->operator_type() != PhysicalOperatorType::kTableScan) {
        Error<SchedulerException>("Expect table scan physical operator", __FILE_NAME__, __LINE__);
    }
    auto *physical_table_scan = static_cast<PhysicalTableScan *>(physical_op);

    for (i64 task_id = 0; task_id < real_parallel_size; ++task_id) {
        FragmentTask *task_ptr = tasks[task_id].get();
        SourceState *source_state = tasks[task_id]->source_state_.get();
        //        SinkState* sink_ptr = tasks[task_id]->sink_state_.get();

        if (source_state->state_type_ != SourceStateType::kTableScan) {
            Error<SchedulerException>("Expect table scan source state", __FILE_NAME__, __LINE__);
        }

        auto *table_scan_source_state = static_cast<TableScanSourceState *>(source_state);
        task_ptr->operator_input_state_[operator_id] = MakeUnique<TableScanInputState>();
        auto table_scan_input_state = (TableScanInputState *)(task_ptr->operator_input_state_[operator_id].get());

        //        if(table_scan_source_state->segment_entry_ids_->empty()) {
        //            Error<SchedulerException>("Empty segment entry ids")
        //        }

        table_scan_input_state->table_scan_function_data_ = MakeShared<TableScanFunctionData>(physical_table_scan->GetBlockIndex(),
                                                                                              table_scan_source_state->global_ids_,
                                                                                              physical_table_scan->ColumnIDs());

        task_ptr->operator_output_state_[operator_id] = MakeUnique<TableScanOutputState>();
        auto table_scan_output_state = (TableScanOutputState *)(task_ptr->operator_output_state_[operator_id].get());

        table_scan_output_state->data_block_ = DataBlock::Make();
        table_scan_output_state->data_block_->Init(*fragment_operators[operator_id]->GetOutputTypes());

        source_state->SetNextState(table_scan_input_state);
    }
}

template <>
void BuildParallelTaskStateTemplate<KnnScanInputState, KnnScanOutputState>(Vector<PhysicalOperator *> &fragment_operators,
                                                                           Vector<UniquePtr<FragmentTask>> &tasks,
                                                                           i64 operator_id,
                                                                           i64 operator_count,
                                                                           FragmentContext *parent_fragment_context,
                                                                           i64 real_parallel_size) {

    // KnnScan must be the first operator of the fragment
    if (operator_id != operator_count - 1) {
        Error<SchedulerException>("Knn scan operator must be the first operator of the fragment.", __FILE_NAME__, __LINE__);
    }

    PhysicalOperator *physical_op = fragment_operators[operator_id];
    if (physical_op->operator_type() != PhysicalOperatorType::kKnnScan) {
        Error<SchedulerException>("Expect knn scan physical operator", __FILE_NAME__, __LINE__);
    }
    auto *physical_knn_scan = static_cast<PhysicalKnnScan *>(physical_op);

    for (i64 task_id = 0; task_id < real_parallel_size; ++task_id) {
        FragmentTask *task_ptr = tasks[task_id].get();
        SourceState *source_state = tasks[task_id]->source_state_.get();
        //        SinkState* sink_ptr = tasks[task_id]->sink_state_.get();

        if (source_state->state_type_ != SourceStateType::kKnnScan) {
            Error<SchedulerException>("Expect knn scan source state", __FILE_NAME__, __LINE__);
        }

        auto *knn_scan_source_state = static_cast<KnnScanSourceState *>(source_state);
        task_ptr->operator_input_state_[operator_id] = MakeUnique<KnnScanInputState>();
        auto knn_scan_input_state = (KnnScanInputState *)(task_ptr->operator_input_state_[operator_id].get());

        //        if(table_scan_source_state->segment_entry_ids_->empty()) {
        //            Error<SchedulerException>("Empty segment entry ids")
        //        }
        if (physical_knn_scan->knn_expressions_.size() == 1) {
            KnnExpression *knn_expr = static_cast<KnnExpression *>(physical_knn_scan->knn_expressions_[0].get());
            Assert<SchedulerException>(knn_expr->arguments().size() == 1, "Expect one expression", __FILE_NAME__, __LINE__);
            ColumnExpression *column_expr = static_cast<ColumnExpression *>(knn_expr->arguments()[0].get());

            Vector<SizeT> knn_column_ids = {column_expr->binding().column_idx};
            ValueExpression *limit_expr = static_cast<ValueExpression *>(physical_knn_scan->limit_expression_.get());
            i64 topk = limit_expr->GetValue().GetValue<BigIntT>();

            knn_scan_input_state->knn_scan_function_data_ = MakeShared<KnnScanFunctionData>(physical_knn_scan->GetBlockIndex(),
                                                                                            knn_scan_source_state->global_ids_,
                                                                                            physical_knn_scan->ColumnIDs(),
                                                                                            knn_column_ids,
                                                                                            topk,
                                                                                            knn_expr->dimension_,
                                                                                            1,
                                                                                            knn_expr->query_embedding_.ptr,
                                                                                            knn_expr->embedding_data_type_,
                                                                                            knn_expr->distance_type_);
            knn_scan_input_state->knn_scan_function_data_->Init();
        } else {
            Error<SchedulerException>("Currently, we only support one knn column scenario", __FILE_NAME__, __LINE__);
        }

        task_ptr->operator_output_state_[operator_id] = MakeUnique<KnnScanOutputState>();
        auto knn_scan_output_state = (KnnScanOutputState *)(task_ptr->operator_output_state_[operator_id].get());
        knn_scan_output_state->data_block_ = DataBlock::Make();
        knn_scan_output_state->data_block_->Init(*fragment_operators[operator_id]->GetOutputTypes());

        source_state->SetNextState(knn_scan_input_state);

        if (operator_id == 0) {
            // Set last operator output as the input of sink operator
            SinkState *sink_state = tasks[task_id]->sink_state_.get();
            sink_state->SetPrevState(knn_scan_output_state);

            if (parent_fragment_context != nullptr) {
                // this fragment has parent fragment, which means the sink node of current fragment
                // will sink data to the parent fragment.
                switch (sink_state->state_type_) {
                    case SinkStateType::kQueue: {
                        auto *queue_sink_state = static_cast<QueueSinkState *>(sink_state);

                        for (const auto &next_fragment_task : parent_fragment_context->Tasks()) {
                            auto *next_fragment_source_state = static_cast<QueueSourceState *>(next_fragment_task->source_state_.get());
                            queue_sink_state->fragment_data_queues_.emplace_back(&next_fragment_source_state->source_queue_);
                        }

                        break;
                    }
                    case SinkStateType::kInvalid: {
                        Error<SchedulerException>("Invalid sink operator state type.", __FILE_NAME__, __LINE__);
                    }
                    default: {
                        Error<SchedulerException>("Sink type isn't queue after knn scan operator", __FILE_NAME__, __LINE__);
                    }
                }
            }
        }
    }
}

void FragmentContext::MakeFragmentContext(QueryContext *query_context,
                                          FragmentContext *parent_context,
                                          PlanFragment *fragment_ptr,
                                          Vector<FragmentTask *> &task_array) {
    Vector<PhysicalOperator *> &fragment_operators = fragment_ptr->GetOperators();
    i64 operator_count = fragment_operators.size();
    if (operator_count < 1) {
        Error<SchedulerException>("No operators in the fragment.", __FILE_NAME__, __LINE__);
    }

    UniquePtr<FragmentContext> fragment_context = nullptr;
    switch (fragment_ptr->GetFragmentType()) {
        case FragmentType::kInvalid: {
            Error<SchedulerException>("Invalid fragment type", __FILE_NAME__, __LINE__);
        }
        case FragmentType::kSerialMaterialize: {
            fragment_context = MakeUnique<SerialMaterializedFragmentCtx>(fragment_ptr, query_context);
            break;
        }
        case FragmentType::kParallelMaterialize: {
            fragment_context = MakeUnique<ParallelMaterializedFragmentCtx>(fragment_ptr, query_context);
            break;
        }
        case FragmentType::kParallelStream: {
            fragment_context = MakeUnique<ParallelStreamFragmentCtx>(fragment_ptr, query_context);
            break;
        }
    }

    // Set parallel size
    i64 parallel_size = static_cast<i64>(query_context->cpu_number_limit());
    fragment_context->CreateTasks(parallel_size, operator_count);
    Vector<UniquePtr<FragmentTask>> &tasks = fragment_context->Tasks();
    i64 real_parallel_size = tasks.size();

    for (i64 operator_id = operator_count - 1; operator_id >= 0; --operator_id) {

        switch (fragment_operators[operator_id]->operator_type()) {
            case PhysicalOperatorType::kInvalid: {
                Error<PlannerException>("Invalid physical operator type", __FILE_NAME__, __LINE__);
                break;
            }
            case PhysicalOperatorType::kAggregate: {
                BuildParallelTaskStateTemplate<AggregateInputState, AggregateOutputState>(fragment_operators,
                                                                                          tasks,
                                                                                          operator_id,
                                                                                          operator_count,
                                                                                          parent_context,
                                                                                          real_parallel_size);

                break;
            }
            case PhysicalOperatorType::kParallelAggregate: {
                BuildParallelTaskStateTemplate<ParallelAggregateInputState, ParallelAggregateOutputState>(fragment_operators,
                                                                                                          tasks,
                                                                                                          operator_id,
                                                                                                          operator_count,
                                                                                                          parent_context,
                                                                                                          real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeParallelAggregate: {
                BuildSerialTaskStateTemplate<MergeParallelAggregateInputState, MergeParallelAggregateOutputState>(fragment_operators,
                                                                                                                  tasks,
                                                                                                                  operator_id,
                                                                                                                  operator_count,
                                                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kUnionAll:
            case PhysicalOperatorType::kIntersect:
            case PhysicalOperatorType::kExcept:
            case PhysicalOperatorType::kDummyScan:
            case PhysicalOperatorType::kJoinHash:
            case PhysicalOperatorType::kJoinNestedLoop:
            case PhysicalOperatorType::kJoinMerge:
            case PhysicalOperatorType::kJoinIndex:
            case PhysicalOperatorType::kCrossProduct:
            case PhysicalOperatorType::kUpdate:
            case PhysicalOperatorType::kPreparedPlan:
            case PhysicalOperatorType::kAlter:
            case PhysicalOperatorType::kFlush:
            case PhysicalOperatorType::kSink:
            case PhysicalOperatorType::kSource: {
                Error<SchedulerException>(Format("Not support {} now", PhysicalOperatorToString(fragment_operators[operator_id]->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
                break;
            }
            case PhysicalOperatorType::kTableScan: {
                BuildParallelTaskStateTemplate<TableScanInputState, TableScanOutputState>(fragment_operators,
                                                                                          tasks,
                                                                                          operator_id,
                                                                                          operator_count,
                                                                                          parent_context,
                                                                                          real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kKnnScan: {
                BuildParallelTaskStateTemplate<KnnScanInputState, KnnScanOutputState>(fragment_operators,
                                                                                      tasks,
                                                                                      operator_id,
                                                                                      operator_count,
                                                                                      parent_context,
                                                                                      real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeKnn: {
                BuildParallelTaskStateTemplate<MergeKnnInputState, MergeKnnOutputState>(fragment_operators,
                                                                                        tasks,
                                                                                        operator_id,
                                                                                        operator_count,
                                                                                        parent_context,
                                                                                        real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kFilter: {
                BuildParallelTaskStateTemplate<FilterInputState, FilterOutputState>(fragment_operators,
                                                                                    tasks,
                                                                                    operator_id,
                                                                                    operator_count,
                                                                                    parent_context,
                                                                                    real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kIndexScan: {
                BuildParallelTaskStateTemplate<IndexScanInputState, IndexScanOutputState>(fragment_operators,
                                                                                          tasks,
                                                                                          operator_id,
                                                                                          operator_count,
                                                                                          parent_context,
                                                                                          real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kHash: {
                BuildParallelTaskStateTemplate<HashInputState, HashOutputState>(fragment_operators,
                                                                                tasks,
                                                                                operator_id,
                                                                                operator_count,
                                                                                parent_context,
                                                                                real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeHash: {
                BuildSerialTaskStateTemplate<MergeHashInputState, MergeHashOutputState>(fragment_operators,
                                                                                        tasks,
                                                                                        operator_id,
                                                                                        operator_count,
                                                                                        parent_context);
                break;
            }
            case PhysicalOperatorType::kLimit: {
                BuildParallelTaskStateTemplate<LimitInputState, LimitOutputState>(fragment_operators,
                                                                                  tasks,
                                                                                  operator_id,
                                                                                  operator_count,
                                                                                  parent_context,
                                                                                  real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeLimit: {
                BuildSerialTaskStateTemplate<MergeLimitInputState, MergeLimitOutputState>(fragment_operators,
                                                                                          tasks,
                                                                                          operator_id,
                                                                                          operator_count,
                                                                                          parent_context);
                break;
            }
            case PhysicalOperatorType::kTop: {
                BuildParallelTaskStateTemplate<TopInputState, TopOutputState>(fragment_operators,
                                                                              tasks,
                                                                              operator_id,
                                                                              operator_count,
                                                                              parent_context,
                                                                              real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeTop: {
                BuildSerialTaskStateTemplate<MergeTopInputState, MergeTopOutputState>(fragment_operators,
                                                                                      tasks,
                                                                                      operator_id,
                                                                                      operator_count,
                                                                                      parent_context);
                break;
            }
            case PhysicalOperatorType::kProjection: {
                BuildParallelTaskStateTemplate<ProjectionInputState, ProjectionOutputState>(fragment_operators,
                                                                                            tasks,
                                                                                            operator_id,
                                                                                            operator_count,
                                                                                            parent_context,
                                                                                            real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kSort: {
                BuildParallelTaskStateTemplate<SortInputState, SortOutputState>(fragment_operators,
                                                                                tasks,
                                                                                operator_id,
                                                                                operator_count,
                                                                                parent_context,
                                                                                real_parallel_size);
                break;
            }
            case PhysicalOperatorType::kMergeSort: {
                BuildSerialTaskStateTemplate<MergeSortInputState, MergeSortOutputState>(fragment_operators,
                                                                                        tasks,
                                                                                        operator_id,
                                                                                        operator_count,
                                                                                        parent_context);
                break;
            }
            case PhysicalOperatorType::kDelete: {
                BuildSerialTaskStateTemplate<DeleteInputState, DeleteOutputState>(fragment_operators,
                                                                                  tasks,
                                                                                  operator_id,
                                                                                  operator_count,
                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kInsert: {
                BuildSerialTaskStateTemplate<InsertInputState, InsertOutputState>(fragment_operators,
                                                                                  tasks,
                                                                                  operator_id,
                                                                                  operator_count,
                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kImport: {
                BuildSerialTaskStateTemplate<ImportInputState, ImportOutputState>(fragment_operators,
                                                                                  tasks,
                                                                                  operator_id,
                                                                                  operator_count,
                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kExport: {
                BuildSerialTaskStateTemplate<ExportInputState, ExportOutputState>(fragment_operators,
                                                                                  tasks,
                                                                                  operator_id,
                                                                                  operator_count,
                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kCreateTable: {
                BuildSerialTaskStateTemplate<CreateTableInputState, CreateTableOutputState>(fragment_operators,
                                                                                            tasks,
                                                                                            operator_id,
                                                                                            operator_count,
                                                                                            parent_context);
                break;
            }
            case PhysicalOperatorType::kCreateIndex: {
                BuildSerialTaskStateTemplate<CreateIndexInputState, CreateIndexOutputState>(fragment_operators,
                                                                                            tasks,
                                                                                            operator_id,
                                                                                            operator_count,
                                                                                            parent_context);
                break;
            }
            case PhysicalOperatorType::kCreateCollection: {
                BuildSerialTaskStateTemplate<CreateCollectionInputState, CreateCollectionOutputState>(fragment_operators,
                                                                                                      tasks,
                                                                                                      operator_id,
                                                                                                      operator_count,
                                                                                                      parent_context);
                break;
            }
            case PhysicalOperatorType::kCreateDatabase: {
                BuildSerialTaskStateTemplate<CreateDatabaseInputState, CreateDatabaseOutputState>(fragment_operators,
                                                                                                  tasks,
                                                                                                  operator_id,
                                                                                                  operator_count,
                                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kCreateView: {
                BuildSerialTaskStateTemplate<CreateViewInputState, CreateViewOutputState>(fragment_operators,
                                                                                          tasks,
                                                                                          operator_id,
                                                                                          operator_count,
                                                                                          parent_context);
                break;
            }
            case PhysicalOperatorType::kDropTable: {
                BuildSerialTaskStateTemplate<DropTableInputState, DropTableOutputState>(fragment_operators,
                                                                                        tasks,
                                                                                        operator_id,
                                                                                        operator_count,
                                                                                        parent_context);
                break;
            }
            case PhysicalOperatorType::kDropCollection: {
                BuildSerialTaskStateTemplate<DropCollectionInputState, DropCollectionOutputState>(fragment_operators,
                                                                                                  tasks,
                                                                                                  operator_id,
                                                                                                  operator_count,
                                                                                                  parent_context);
                break;
            }
            case PhysicalOperatorType::kDropDatabase: {
                BuildSerialTaskStateTemplate<DropDatabaseInputState, DropDatabaseOutputState>(fragment_operators,
                                                                                              tasks,
                                                                                              operator_id,
                                                                                              operator_count,
                                                                                              parent_context);
                break;
            }
            case PhysicalOperatorType::kDropView: {
                BuildSerialTaskStateTemplate<DropViewInputState, DropViewOutputState>(fragment_operators,
                                                                                      tasks,
                                                                                      operator_id,
                                                                                      operator_count,
                                                                                      parent_context);
                break;
            }
            case PhysicalOperatorType::kExplain: {
                BuildSerialTaskStateTemplate<ExplainInputState, ExplainOutputState>(fragment_operators,
                                                                                    tasks,
                                                                                    operator_id,
                                                                                    operator_count,
                                                                                    parent_context);
                break;
            }
            case PhysicalOperatorType::kShow: {
                BuildSerialTaskStateTemplate<ShowInputState, ShowOutputState>(fragment_operators, tasks, operator_id, operator_count, parent_context);
                break;
            }
        }
    }

    if (fragment_ptr->HasChild()) {
        // current fragment have children
        for (const auto &child_fragment : fragment_ptr->Children()) {
            FragmentContext::MakeFragmentContext(query_context, fragment_context.get(), child_fragment.get(), task_array);
        }
    }

    for (const auto &task : tasks) {
        task_array.emplace_back(task.get());
    }

    fragment_ptr->SetContext(std::move(fragment_context));
}

FragmentContext::FragmentContext(PlanFragment *fragment_ptr, QueryContext *query_context)
    : fragment_ptr_(fragment_ptr), fragment_type_(fragment_ptr->GetFragmentType()), query_context_(query_context){};

Vector<PhysicalOperator *> &FragmentContext::GetOperators() { return fragment_ptr_->GetOperators(); }

PhysicalSink *FragmentContext::GetSinkOperator() const { return fragment_ptr_->GetSinkNode(); }

PhysicalSource *FragmentContext::GetSourceOperator() const { return fragment_ptr_->GetSourceNode(); }

void FragmentContext::CreateTasks(i64 cpu_count, i64 operator_count) {
    i64 parallel_count = cpu_count;
    PhysicalOperator *first_operator = this->GetOperators().back();
    switch (first_operator->operator_type()) {
        case PhysicalOperatorType::kTableScan: {
            auto *table_scan_operator = static_cast<PhysicalTableScan *>(first_operator);
            parallel_count = std::min(parallel_count, (i64)(table_scan_operator->BlockEntryCount()));
            if (parallel_count == 0) {
                parallel_count = 1;
            }
            break;
        }
        case PhysicalOperatorType::kKnnScan: {
            auto *knn_scan_operator = static_cast<PhysicalKnnScan *>(first_operator);
            parallel_count = std::min(parallel_count, (i64)(knn_scan_operator->BlockEntryCount()));
            if (parallel_count == 0) {
                parallel_count = 1;
            }
            break;
        }
        case PhysicalOperatorType::kMergeKnn:
        case PhysicalOperatorType::kProjection: {
            // Serial Materialize
            parallel_count = 1;
            break;
        }
        default: {
            break;
        }
    }

    switch (fragment_type_) {
        case FragmentType::kInvalid: {
            Error<SchedulerException>("Invalid fragment type", __FILE_NAME__, __LINE__);
        }
        case FragmentType::kSerialMaterialize: {
            UniqueLock<std::mutex> locker(locker_);
            tasks_.reserve(parallel_count);
            tasks_.emplace_back(MakeUnique<FragmentTask>(this, 0, operator_count));
            break;
        }
        case FragmentType::kParallelMaterialize:
        case FragmentType::kParallelStream: {
            UniqueLock<std::mutex> locker(locker_);
            tasks_.reserve(parallel_count);
            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_.emplace_back(MakeUnique<FragmentTask>(this, task_id, operator_count));
            }
            break;
        }
    }

    // Determine which type of source state.
    switch (first_operator->operator_type()) {
        case PhysicalOperatorType::kInvalid: {
            Error<SchedulerException>("Unexpected operator type", __FILE_NAME__, __LINE__);
        }
        case PhysicalOperatorType::kAggregate: {
            if (fragment_type_ != FragmentType::kParallelMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized fragment", PhysicalOperatorToString(first_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(first_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            // Partition the hash range to each source state
            auto *aggregate_operator = (PhysicalAggregate *)first_operator;
            Vector<HashRange> hash_range = aggregate_operator->GetHashRanges(parallel_count);
            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->source_state_ = MakeUnique<AggregateSourceState>(hash_range[task_id].start_, hash_range[task_id].end_);
            }
            break;
        }
        case PhysicalOperatorType::kParallelAggregate:
        case PhysicalOperatorType::kFilter:
        case PhysicalOperatorType::kHash:
        case PhysicalOperatorType::kProjection:
        case PhysicalOperatorType::kLimit:
        case PhysicalOperatorType::kTop:
        case PhysicalOperatorType::kSort:
        case PhysicalOperatorType::kDelete: {
            Error<SchedulerException>(
                Format("{} shouldn't be the first operator of the fragment", PhysicalOperatorToString(first_operator->operator_type())),
                __FILE_NAME__,
                __LINE__);
        }
        case PhysicalOperatorType::kMergeParallelAggregate:
        case PhysicalOperatorType::kMergeHash:
        case PhysicalOperatorType::kMergeLimit:
        case PhysicalOperatorType::kMergeTop:
        case PhysicalOperatorType::kMergeSort:
        case PhysicalOperatorType::kMergeKnn: {
            if (fragment_type_ != FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should be serial materialized fragment", PhysicalOperatorToString(first_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != 1) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(first_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            tasks_[0]->source_state_ = MakeUnique<QueueSourceState>();
            break;
        }
        case PhysicalOperatorType::kUnionAll:
        case PhysicalOperatorType::kIntersect:
        case PhysicalOperatorType::kExcept:
        case PhysicalOperatorType::kDummyScan:
        case PhysicalOperatorType::kIndexScan:
        case PhysicalOperatorType::kJoinHash:
        case PhysicalOperatorType::kJoinNestedLoop:
        case PhysicalOperatorType::kJoinMerge:
        case PhysicalOperatorType::kJoinIndex:
        case PhysicalOperatorType::kCrossProduct:
        case PhysicalOperatorType::kUpdate:
        case PhysicalOperatorType::kPreparedPlan:
        case PhysicalOperatorType::kFlush: {
            Error<SchedulerException>(Format("Not support {} now", PhysicalOperatorToString(first_operator->operator_type())),
                                      __FILE_NAME__,
                                      __LINE__);
        }
        case PhysicalOperatorType::kTableScan: {
            if (fragment_type_ != FragmentType::kParallelMaterialize && fragment_type_ != FragmentType::kParallelStream) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized/stream fragment", PhysicalOperatorToString(first_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(first_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            // Partition the hash range to each source state
            auto *table_scan_operator = (PhysicalTableScan *)first_operator;
            Vector<SharedPtr<Vector<GlobalBlockID>>> blocks_group = table_scan_operator->PlanBlockEntries(parallel_count);
            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->source_state_ = MakeUnique<TableScanSourceState>(blocks_group[task_id]);
            }
            break;
        }
        case PhysicalOperatorType::kKnnScan: {
            if (fragment_type_ != FragmentType::kParallelMaterialize && fragment_type_ != FragmentType::kParallelStream) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized/stream fragment", PhysicalOperatorToString(first_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(first_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            // Partition the hash range to each source state
            auto *knn_scan_operator = (PhysicalKnnScan *)first_operator;
            Vector<SharedPtr<Vector<GlobalBlockID>>> blocks_group = knn_scan_operator->PlanBlockEntries(parallel_count);
            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->source_state_ = MakeUnique<KnnScanSourceState>(blocks_group[task_id]);
            }
            break;
        }
        case PhysicalOperatorType::kInsert:
        case PhysicalOperatorType::kImport:
        case PhysicalOperatorType::kExport:
        case PhysicalOperatorType::kAlter:
        case PhysicalOperatorType::kCreateTable:
        case PhysicalOperatorType::kCreateIndex:
        case PhysicalOperatorType::kCreateCollection:
        case PhysicalOperatorType::kCreateDatabase:
        case PhysicalOperatorType::kCreateView:
        case PhysicalOperatorType::kDropTable:
        case PhysicalOperatorType::kDropCollection:
        case PhysicalOperatorType::kDropDatabase:
        case PhysicalOperatorType::kDropView:
        case PhysicalOperatorType::kExplain:
        case PhysicalOperatorType::kShow: {
            if (fragment_type_ != FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in serial materialized fragment", PhysicalOperatorToString(first_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != 1) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(first_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            tasks_[0]->source_state_ = MakeUnique<EmptySourceState>();
            break;
        }

        default: {
            Error<SchedulerException>(Format("Unexpected operator type: {}", PhysicalOperatorToString(first_operator->operator_type())),
                                      __FILE_NAME__,
                                      __LINE__);
        }
    }

    // Determine which type of the sink state.
    PhysicalOperator *last_operator = this->GetOperators().front();
    switch (last_operator->operator_type()) {

        case PhysicalOperatorType::kInvalid: {
            Error<SchedulerException>("Unexpected operator type", __FILE_NAME__, __LINE__);
        }
        case PhysicalOperatorType::kAggregate:
        case PhysicalOperatorType::kParallelAggregate:
        case PhysicalOperatorType::kHash:
        case PhysicalOperatorType::kLimit:
        case PhysicalOperatorType::kTop:
        case PhysicalOperatorType::kSort: {
            if (fragment_type_ != FragmentType::kParallelMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->sink_state_ = MakeUnique<MaterializeSinkState>();
            }
            break;
        }
        case PhysicalOperatorType::kMergeParallelAggregate:
        case PhysicalOperatorType::kMergeHash:
        case PhysicalOperatorType::kMergeLimit:
        case PhysicalOperatorType::kMergeTop:
        case PhysicalOperatorType::kMergeSort:
        case PhysicalOperatorType::kMergeKnn:
        case PhysicalOperatorType::kExplain:
        case PhysicalOperatorType::kShow: {
            if (fragment_type_ != FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in serial materialized fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != 1) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            tasks_[0]->sink_state_ = MakeUnique<MaterializeSinkState>();
            MaterializeSinkState *sink_state_ptr = static_cast<MaterializeSinkState *>(tasks_[0]->sink_state_.get());
            sink_state_ptr->column_types_ = last_operator->GetOutputTypes();
            sink_state_ptr->column_names_ = last_operator->GetOutputNames();
            break;
        }
        case PhysicalOperatorType::kKnnScan: {
            if (fragment_type_ == FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized/stream fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->sink_state_ = MakeUnique<QueueSinkState>();
                //                QueueSinkState* sink_state_ptr = static_cast<QueueSinkState*>(tasks_[task_id]->sink_state_.get());
            }
            break;
        }
        case PhysicalOperatorType::kTableScan:
        case PhysicalOperatorType::kFilter:
        case PhysicalOperatorType::kIndexScan: {
            if (fragment_type_ == FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in parallel materialized/stream fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != parallel_count) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                tasks_[task_id]->sink_state_ = MakeUnique<MaterializeSinkState>();
                MaterializeSinkState *sink_state_ptr = static_cast<MaterializeSinkState *>(tasks_[task_id]->sink_state_.get());
                sink_state_ptr->column_types_ = last_operator->GetOutputTypes();
                sink_state_ptr->column_names_ = last_operator->GetOutputNames();
            }
            break;
        }
        case PhysicalOperatorType::kProjection: {
            if (fragment_type_ == FragmentType::kSerialMaterialize) {
                if (tasks_.size() != 1) {
                    Error<SchedulerException>("SerialMaterialize type fragment should only have 1 task.", __FILE_NAME__, __LINE__);
                }

                tasks_[0]->sink_state_ = MakeUnique<MaterializeSinkState>();
                MaterializeSinkState *sink_state_ptr = static_cast<MaterializeSinkState *>(tasks_[0]->sink_state_.get());
                sink_state_ptr->column_types_ = last_operator->GetOutputTypes();
                sink_state_ptr->column_names_ = last_operator->GetOutputNames();
            } else {
                if (tasks_.size() != parallel_count) {
                    Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                              __FILE_NAME__,
                                              __LINE__);
                }

                for (i64 task_id = 0; task_id < parallel_count; ++task_id) {
                    tasks_[task_id]->sink_state_ = MakeUnique<MaterializeSinkState>();
                    MaterializeSinkState *sink_state_ptr = static_cast<MaterializeSinkState *>(tasks_[task_id]->sink_state_.get());
                    sink_state_ptr->column_types_ = last_operator->GetOutputTypes();
                    sink_state_ptr->column_names_ = last_operator->GetOutputNames();
                }
            }
            break;
        }
        case PhysicalOperatorType::kUnionAll:
        case PhysicalOperatorType::kIntersect:
        case PhysicalOperatorType::kExcept:
        case PhysicalOperatorType::kDummyScan:
        case PhysicalOperatorType::kUpdate:
        case PhysicalOperatorType::kJoinHash:
        case PhysicalOperatorType::kJoinNestedLoop:
        case PhysicalOperatorType::kJoinMerge:
        case PhysicalOperatorType::kJoinIndex:
        case PhysicalOperatorType::kCrossProduct:
        case PhysicalOperatorType::kAlter:
        case PhysicalOperatorType::kPreparedPlan:
        case PhysicalOperatorType::kFlush: {
            Error<SchedulerException>(Format("Not support {} now", PhysicalOperatorToString(last_operator->operator_type())),
                                      __FILE_NAME__,
                                      __LINE__);
        }
        case PhysicalOperatorType::kDelete:
        case PhysicalOperatorType::kInsert:
        case PhysicalOperatorType::kImport:
        case PhysicalOperatorType::kExport: {
            if (fragment_type_ != FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in serial materialized fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != 1) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            tasks_[0]->sink_state_ = MakeUnique<MessageSinkState>();
            break;
        }
        case PhysicalOperatorType::kCreateTable:
        case PhysicalOperatorType::kCreateIndex:
        case PhysicalOperatorType::kCreateCollection:
        case PhysicalOperatorType::kCreateDatabase:
        case PhysicalOperatorType::kCreateView:
        case PhysicalOperatorType::kDropTable:
        case PhysicalOperatorType::kDropCollection:
        case PhysicalOperatorType::kDropDatabase:
        case PhysicalOperatorType::kDropView: {
            if (fragment_type_ != FragmentType::kSerialMaterialize) {
                Error<SchedulerException>(
                    Format("{} should in serial materialized fragment", PhysicalOperatorToString(last_operator->operator_type())),
                    __FILE_NAME__,
                    __LINE__);
            }

            if (tasks_.size() != 1) {
                Error<SchedulerException>(Format("{} task count isn't correct.", PhysicalOperatorToString(last_operator->operator_type())),
                                          __FILE_NAME__,
                                          __LINE__);
            }

            tasks_[0]->sink_state_ = MakeUnique<ResultSinkState>();
            break;
        }
        default: {
            Error<SchedulerException>(Format("Unexpected operator type: {}", PhysicalOperatorToString(last_operator->operator_type())),
                                      __FILE_NAME__,
                                      __LINE__);
        }
    }
}

SharedPtr<Table> SerialMaterializedFragmentCtx::GetResultInternal() {

    // Only one sink state
    if (tasks_.size() != 1) {
        Error<SchedulerException>("There should be one sink state in serial materialized fragment", __FILE_NAME__, __LINE__);
    }

    switch (tasks_[0]->sink_state_->state_type()) {
        case SinkStateType::kInvalid: {
            Error<SchedulerException>("Invalid sink state type", __FILE_NAME__, __LINE__);
            break;
        }
        case SinkStateType::kMaterialize: {
            auto *materialize_sink_state = static_cast<MaterializeSinkState *>(tasks_[0]->sink_state_.get());

            Vector<SharedPtr<ColumnDef>> column_defs;
            SizeT column_count = materialize_sink_state->column_names_->size();
            column_defs.reserve(column_count);
            for (SizeT col_idx = 0; col_idx < column_count; ++col_idx) {
                column_defs.emplace_back(MakeShared<ColumnDef>(col_idx,
                                                               materialize_sink_state->column_types_->at(col_idx),
                                                               materialize_sink_state->column_names_->at(col_idx),
                                                               HashSet<ConstraintType>()));
            }

            SharedPtr<Table> result_table = Table::MakeResultTable(column_defs);
            result_table->data_blocks_ = std::move(materialize_sink_state->data_block_array_);
            return result_table;
        }
        case SinkStateType::kResult: {
            auto *result_sink_state = static_cast<ResultSinkState *>(tasks_[0]->sink_state_.get());
            if (result_sink_state->error_message_ == nullptr) {
                SharedPtr<Table> result_table = Table::MakeResultTable({result_sink_state->result_def_});
                return result_table;
            }
            Error<ExecutorException>(*result_sink_state->error_message_, __FILE_NAME__, __LINE__);
        }
        case SinkStateType::kMessage: {
            auto *message_sink_state = static_cast<MessageSinkState *>(tasks_[0]->sink_state_.get());
            if (message_sink_state->error_message_ != nullptr) {
                throw Exception(*message_sink_state->error_message_);
            }

            if (message_sink_state->message_ == nullptr) {
                Error<SchedulerException>("No response message", __FILE_NAME__, __LINE__);
            }

            SharedPtr<Table> result_table = Table::MakeEmptyResultTable();
            result_table->SetResultMsg(std::move(message_sink_state->message_));
            return result_table;
        }
        case SinkStateType::kQueue: {
            Error<SchedulerException>("Can't get result from Queue sink type.", __FILE_NAME__, __LINE__);
        }
    }
    Error<SchedulerException>("Unreachable", __FILE_NAME__, __LINE__);
}

SharedPtr<Table> ParallelMaterializedFragmentCtx::GetResultInternal() {

    SharedPtr<Table> result_table = nullptr;

    auto *first_materialize_sink_state = static_cast<MaterializeSinkState *>(tasks_[0]->sink_state_.get());
    if (first_materialize_sink_state->error_message_ != nullptr) {
        Error<ExecutorException>(*first_materialize_sink_state->error_message_, __FILE_NAME__, __LINE__);
    }

    Vector<SharedPtr<ColumnDef>> column_defs;
    SizeT column_count = first_materialize_sink_state->column_names_->size();
    column_defs.reserve(column_count);
    for (SizeT col_idx = 0; col_idx < column_count; ++col_idx) {
        column_defs.emplace_back(MakeShared<ColumnDef>(col_idx,
                                                       first_materialize_sink_state->column_types_->at(col_idx),
                                                       first_materialize_sink_state->column_names_->at(col_idx),
                                                       HashSet<ConstraintType>()));
    }

    for (const auto &task : tasks_) {
        if (task->sink_state_->state_type() != SinkStateType::kMaterialize) {
            Error<SchedulerException>("Parallel materialized fragment will only have common sink stte", __FILE_NAME__, __LINE__);
        }

        auto *materialize_sink_state = static_cast<MaterializeSinkState *>(task->sink_state_.get());
        if (materialize_sink_state->error_message_ != nullptr) {
            Error<ExecutorException>(*materialize_sink_state->error_message_, __FILE_NAME__, __LINE__);
        }

        if (result_table == nullptr) {
            result_table = Table::MakeResultTable(column_defs);
        }

        for (const auto &result_data_block : materialize_sink_state->data_block_array_) {
            result_table->Append(result_data_block);
        }
    }

    return result_table;
}

SharedPtr<Table> ParallelStreamFragmentCtx::GetResultInternal() {
    SharedPtr<Table> result_table = nullptr;

    auto *first_materialize_sink_state = static_cast<MaterializeSinkState *>(tasks_[0]->sink_state_.get());

    Vector<SharedPtr<ColumnDef>> column_defs;
    SizeT column_count = first_materialize_sink_state->column_names_->size();
    column_defs.reserve(column_count);
    for (SizeT col_idx = 0; col_idx < column_count; ++col_idx) {
        column_defs.emplace_back(MakeShared<ColumnDef>(col_idx,
                                                       first_materialize_sink_state->column_types_->at(col_idx),
                                                       first_materialize_sink_state->column_names_->at(col_idx),
                                                       HashSet<ConstraintType>()));
    }

    for (const auto &task : tasks_) {
        if (task->sink_state_->state_type() != SinkStateType::kMaterialize) {
            Error<SchedulerException>("Parallel materialized fragment will only have common sink state", __FILE_NAME__, __LINE__);
        }

        auto *materialize_sink_state = static_cast<MaterializeSinkState *>(task->sink_state_.get());

        if (result_table == nullptr) {
            result_table = Table::MakeResultTable(column_defs);
        }

        for (const auto &result_data_block : materialize_sink_state->data_block_array_) {
            result_table->Append(result_data_block);
        }
    }

    return result_table;
}

} // namespace infinity
