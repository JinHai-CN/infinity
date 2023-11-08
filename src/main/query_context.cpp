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

import stl;
import session;
import config;
import fragment_scheduler;
import storage;
import resource_manager;
import txn;
import parser;

import infinity_exception;
import logical_planner;
import logical_node_type;
import data_block;
import optimizer;
import physical_planner;
import fragment_builder;
import bind_context;
import logical_node;
import physical_operator;
import third_party;
import logger;
import query_result;
import status;

module query_context;

namespace infinity {

QueryContext::QueryContext(SessionBase *session) : session_ptr_(session){};

QueryContext::~QueryContext() { UnInit(); }

void QueryContext::Init(const Config *global_config_ptr,
                        FragmentScheduler *scheduler_ptr,
                        Storage *storage_ptr,
                        ResourceManager *resource_manager_ptr) {
    global_config_ = global_config_ptr;
    scheduler_ = scheduler_ptr;
    storage_ = storage_ptr;
    resource_manager_ = resource_manager_ptr;
    initialized_ = true;
    cpu_number_limit_ = resource_manager_ptr->GetCpuResource();
    memory_size_limit_ = resource_manager_ptr->GetMemoryResource();

    parser_ = MakeUnique<SQLParser>();
    logical_planner_ = MakeUnique<LogicalPlanner>(this);
    optimizer_ = MakeUnique<Optimizer>(this);
    physical_planner_ = MakeUnique<PhysicalPlanner>(this);
    fragment_builder_ = MakeUnique<FragmentBuilder>(this);
}

QueryResult QueryContext::Query(const String &query) {
    SharedPtr<ParserResult> parsed_result = MakeShared<ParserResult>();
    parser_->Parse(query, parsed_result);

    if (parsed_result->IsError()) {
        Error<PlannerException>(parsed_result->error_message_);
    }

    Assert<PlannerException>(parsed_result->statements_ptr_->size() == 1, "Only support single statement.");
    for (BaseStatement *statement : *parsed_result->statements_ptr_) {
        QueryResult query_result = QueryStatement(statement);
        return query_result;
    }

    Error<NetworkException>("Not reachable");
}

QueryResult QueryContext::QueryStatement(const BaseStatement *statement) {
    QueryResult query_result;
    try {
        this->CreateTxn();
        this->BeginTxn();
        LOG_INFO(Format("created transaction, txn_id: {}, begin_ts: {}, statement: {}",
                        session_ptr_->txn()->TxnID(),
                        session_ptr_->txn()->BeginTS(),
                        statement->ToString()));

        // Build unoptimized logical plan for each SQL statement.
        SharedPtr<BindContext> bind_context;
        logical_planner_->Build(statement, bind_context);
        current_max_node_id_ = bind_context->GetNewLogicalNodeId();

        SharedPtr<LogicalNode> unoptimized_plan = logical_planner_->LogicalPlan();

        // Apply optimized rule to the logical plan
        SharedPtr<LogicalNode> optimized_plan = optimizer_->optimize(unoptimized_plan);

        // Build physical plan
        SharedPtr<PhysicalOperator> physical_plan = physical_planner_->BuildPhysicalOperator(optimized_plan);

        // Fragment Builder, only for test now.
        // SharedPtr<PlanFragment> plan_fragment = fragment_builder.Build(physical_plan);
        auto plan_fragment = fragment_builder_->BuildFragment(physical_plan.get());

        scheduler_->Schedule(this, plan_fragment.get());
        query_result.result_table_ = plan_fragment->GetResult();
        query_result.root_operator_type_ = unoptimized_plan->operator_type();

        this->CommitTxn();
    } catch (const Exception &e) {
        this->RollbackTxn();
        query_result.result_table_ = nullptr;
        query_result.status_.Init(ErrorCode::kError, e.what());
    }
    return query_result;
}

void QueryContext::CreateTxn() {
    if (session_ptr_->txn() == nullptr) {
        session_ptr_->txn() = storage_->txn_manager()->CreateTxn();
    }
}

void QueryContext::BeginTxn() { session_ptr_->txn()->BeginTxn(); }

void QueryContext::CommitTxn() {
    session_ptr_->txn()->CommitTxn();
    session_ptr_->txn() = nullptr;
}

void QueryContext::RollbackTxn() {
    session_ptr_->txn()->RollbackTxn();
    session_ptr_->txn() = nullptr;
}

} // namespace infinity
