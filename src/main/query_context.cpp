// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include <sstream>
//#include "gperftools/profiler.h"

import stl;
import session;
import config;
import task_scheduler;
import storage;
import resource_manager;
import txn;
import parser;
import profiler;
import infinity_exception;
import logical_planner;
import logical_node_type;
import data_block;
import optimizer;
import physical_planner;
import fragment_builder;
import fragment_task;
import fragment_context;
import bind_context;
import logical_node;
import physical_operator;
import third_party;
import logger;
import query_result;
import status;
import session_manager;

module query_context;

namespace infinity {

QueryContext::QueryContext(BaseSession *session) : session_ptr_(session){};

QueryContext::~QueryContext() { UnInit(); }

void QueryContext::Init(Config *global_config_ptr,
                        TaskScheduler *scheduler_ptr,
                        Storage *storage_ptr,
                        ResourceManager *resource_manager_ptr,
                        SessionManager* session_manager) {
    global_config_ = global_config_ptr;
    scheduler_ = scheduler_ptr;
    storage_ = storage_ptr;
    resource_manager_ = resource_manager_ptr;
    session_manager_ = session_manager;

    initialized_ = true;
    cpu_number_limit_ = resource_manager_ptr->GetCpuResource();
    memory_size_limit_ = resource_manager_ptr->GetMemoryResource();


}

QueryResult QueryContext::Query(const String &query) {
    CreateQueryProfiler();

    StartProfile(QueryPhase::kParser);
    UniquePtr<ParserResult> parsed_result = MakeUnique<ParserResult>();
    UniquePtr<SQLParser> sql_parser = MakeUnique<SQLParser>();
    sql_parser->Parse(query, parsed_result.get());

    if (parsed_result->IsError()) {
        StopProfile(QueryPhase::kParser);
        Error<PlannerException>(parsed_result->error_message_);
    }

    if (parsed_result->statements_ptr_->size() != 1) {
        Error<PlannerException>("Only support single statement.");
    }
    StopProfile(QueryPhase::kParser);
    for (BaseStatement *statement : *parsed_result->statements_ptr_) {
        QueryResult query_result = QueryStatement(statement);
        return query_result;
    }

    Error<NetworkException>("Not reachable");
    return QueryResult::UnusedResult();
}

QueryResult QueryContext::QueryStatement(const BaseStatement *statement) {
    QueryResult query_result;
//    ProfilerStart("Query");
    try {
        this->CreateTxn();
        this->BeginTxn();
//        LOG_INFO(Format("created transaction, txn_id: {}, begin_ts: {}, statement: {}",
//                        session_ptr_->GetTxn()->TxnID(),
//                        session_ptr_->GetTxn()->BeginTS(),
//                        statement->ToString()));
        RecordQueryProfiler(statement->type_);

        // Build unoptimized logical plan for each SQL statement.
        StartProfile(QueryPhase::kLogicalPlan);
        SharedPtr<BindContext> bind_context;
        UniquePtr<LogicalPlanner> logical_planner = MakeUnique<LogicalPlanner>(this);
        auto state = logical_planner->Build(statement, bind_context);
        // FIXME
        if (!state.ok()) {
            Error<PlannerException>(state.message());
        }

        current_max_node_id_ = bind_context->GetNewLogicalNodeId();
        SharedPtr<LogicalNode> logical_plan = logical_planner->LogicalPlan();
        StopProfile(QueryPhase::kLogicalPlan);

        // Apply optimized rule to the logical plan
        StartProfile(QueryPhase::kOptimizer);
        if(NeedOptimize(logical_plan.get())) {
            UniquePtr<Optimizer> optimizer = MakeUnique<Optimizer>(this);
            optimizer->optimize(logical_plan);
        }
        StopProfile(QueryPhase::kOptimizer);

        // Build physical plan
        StartProfile(QueryPhase::kPhysicalPlan);
        UniquePtr<PhysicalPlanner> physical_planner = MakeUnique<PhysicalPlanner>(this);
        UniquePtr<PhysicalOperator> physical_plan = physical_planner->BuildPhysicalOperator(logical_plan);
        StopProfile(QueryPhase::kPhysicalPlan);

        StartProfile(QueryPhase::kPipelineBuild);
        // Fragment Builder, only for test now.
        UniquePtr<FragmentBuilder> fragment_builder = MakeUnique<FragmentBuilder>(this);
        auto plan_fragment = fragment_builder->BuildFragment(physical_plan.get());
        StopProfile(QueryPhase::kPipelineBuild);

        StartProfile(QueryPhase::kTaskBuild);
        Vector<FragmentTask *> tasks;

        FragmentContext::BuildTask(this, nullptr, plan_fragment.get(), tasks);
        StopProfile(QueryPhase::kTaskBuild);

        StartProfile(QueryPhase::kExecution);
        scheduler_->Schedule(this, tasks, plan_fragment.get());
        query_result.result_table_ = plan_fragment->GetResult();
        query_result.root_operator_type_ = logical_plan->operator_type();
        StopProfile(QueryPhase::kExecution);

        StartProfile(QueryPhase::kCommit);
        this->CommitTxn();
        StopProfile(QueryPhase::kCommit);
    } catch (const Exception &e) {
        StopProfile();
        StartProfile(QueryPhase::kRollback);
        this->RollbackTxn();
        StopProfile(QueryPhase::kRollback);
        query_result.result_table_ = nullptr;
        query_result.status_.Init(ErrorCode::kError, e.what());
    }
//    ProfilerStop();
    session_ptr_->IncreaseQueryCount();
    return query_result;
}

void QueryContext::CreateTxn() {
    if (session_ptr_->GetTxn() == nullptr) {
        Txn* new_txn = storage_->txn_manager()->CreateTxn();
        session_ptr_->SetTxn(new_txn);
    }
}

void QueryContext::BeginTxn() { session_ptr_->GetTxn()->Begin(); }

void QueryContext::CommitTxn() {
    Txn* txn = session_ptr_->GetTxn();
    storage_->txn_manager()->CommitTxn(txn);
    session_ptr_->SetTxn(nullptr);
}

void QueryContext::RollbackTxn() {
    Txn* txn = session_ptr_->GetTxn();
    storage_->txn_manager()->RollBackTxn(txn);
    session_ptr_->SetTxn(nullptr);
}

bool QueryContext::NeedOptimize(const LogicalNode* logical_node) {
    if(logical_node->left_node().get() == nullptr && logical_node->right_node().get() == nullptr) {
        switch(logical_node->operator_type()) {
            case LogicalNodeType::kInsert:
            case LogicalNodeType::kImport:
            case LogicalNodeType::kExport:
            case LogicalNodeType::kAlter:
            case LogicalNodeType::kCreateTable:
            case LogicalNodeType::kCreateIndex:
            case LogicalNodeType::kCreateCollection:
            case LogicalNodeType::kCreateSchema:
            case LogicalNodeType::kCreateView:
            case LogicalNodeType::kDropTable:
            case LogicalNodeType::kDropIndex:
            case LogicalNodeType::kDropCollection:
            case LogicalNodeType::kDropSchema:
            case LogicalNodeType::kDropView:
            case LogicalNodeType::kShow:
            case LogicalNodeType::kExplain:
            case LogicalNodeType::kCommand:
            case LogicalNodeType::kFlush:
            case LogicalNodeType::kOptimize: {
                return false;
            }
            default: {
                return true;
            }
        }
    }

    return true;
}

} // namespace infinity
