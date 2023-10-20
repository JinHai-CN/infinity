//
// Created by JinHai on 2022/9/26.
//

module;

import stl;
import parser;
import base_expression;
import query_context;
import logical_node;
import bind_context;
import expression_type;

import subquery_expression;
import value_expression;
import function_expression;

import logical_join;
import logical_limit;

import function_set;
import scalar_function;

import new_catalog;
import column_binding;
import third_party;

module subquery_unnest;

namespace infinity {

void SubqueryUnnest::UnnestSubqueries(SharedPtr<BaseExpression> &expr_ptr,
                                      SharedPtr<LogicalNode> &root,
                                      QueryContext *query_context,
                                      const SharedPtr<BindContext> &bind_context) {
    // 2. Call Unnest Subquery to resolve subquery
    if (expr_ptr->type() == ExpressionType::kSubQuery) {
        // Subquery, need to be unnested.
        UnnestSubquery(expr_ptr, root, query_context, bind_context);
    }
}

SharedPtr<BaseExpression> SubqueryUnnest::UnnestSubquery(SharedPtr<BaseExpression> &expr_ptr,
                                                         SharedPtr<LogicalNode> &root,
                                                         QueryContext *query_context,
                                                         const SharedPtr<BindContext> &bind_context) {
#if 0
    // 1. Check the subquery type: uncorrelated subquery or correlated subquery.
    auto subquery_expr = std::static_pointer_cast<SubqueryExpression>(expr_ptr);

    auto right = subquery_expr->bound_select_statement_ptr_->BuildPlan(query_context);
    // TODO: if the correlated information of the subquery should be stored in bind context.
    // Check the correlated information
    auto result = UnnestUncorrelated(subquery_expr.get(), root, right, query_context, bind_context);
    // If it isn't a correlated subquery

    // 2. Call different function to resolve uncorrelated subquery and correlated subquery.
    // 3. If the subquery also has another subquery nested, we need to resolve it recursively.

    return result;
#endif
}

SharedPtr<BaseExpression> SubqueryUnnest::UnnestUncorrelated(SubqueryExpression *expr_ptr,
                                                             SharedPtr<LogicalNode> &root,
                                                             SharedPtr<LogicalNode> &subquery_plan,
                                                             QueryContext *query_context,
                                                             const SharedPtr<BindContext> &bind_context) {
#if 0
    switch (expr_ptr->subquery_type_) {

        case SubqueryType::kScalar: {
            // Step1 Generate limit operator on the subquery
            SharedPtr<ValueExpression> limit_expression = MakeShared<ValueExpression>(Value::MakeBigInt(1));
            SharedPtr<ValueExpression> offset_expression = MakeShared<ValueExpression>(Value::MakeBigInt(0));
            SharedPtr<LogicalLimit> limit_node = MakeShared<LogicalLimit>(bind_context->GetNewLogicalNodeId(), limit_expression, offset_expression);

            limit_node->set_left_node(subquery_plan);
            // Step2 Generate aggregate first operator on the limit operator
            NewCatalog *catalog = query_context->storage()->catalog();
            SharedPtr<FunctionSet> function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "first");
            ColumnBinding limit_column_binding = limit_node->GetColumnBindings()[0];

            SharedPtr<ColumnExpression> argument = ColumnExpression::Make(expr_ptr->Type(),
                                                                          subquery_plan->name(),
                                                                          limit_column_binding.table_idx,
                                                                          "0",
                                                                          limit_column_binding.column_idx,
                                                                          0);

            auto aggregate_function_set_ptr = std::static_pointer_cast<AggregateFunctionSet>(function_set_ptr);
            AggregateFunction first_function = aggregate_function_set_ptr->GetMostMatchFunction(argument);

            Vector<SharedPtr<BaseExpression>> arguments;
            arguments.emplace_back(argument);
            auto first_function_expr = MakeShared<AggregateExpression>(first_function, arguments);

            Vector<SharedPtr<BaseExpression>> groups; // empty group by list
            Vector<SharedPtr<BaseExpression>> aggregates;
            aggregates.emplace_back(first_function_expr);
            u64 group_by_index = bind_context->GenerateTableIndex();
            u64 aggregate_index = bind_context->GenerateTableIndex();
            SharedPtr<LogicalAggregate> aggregate_node =
                MakeShared<LogicalAggregate>(bind_context->GetNewLogicalNodeId(), groups, group_by_index, aggregates, aggregate_index);

            aggregate_node->set_left_node(limit_node);

            // Step3 Generate cross product on the root and subquery plan
            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "cross_product" + ToStr(logical_node_id);
            SharedPtr<LogicalCrossProduct> cross_product_node = MakeShared<LogicalCrossProduct>(logical_node_id, alias, root, aggregate_node);

            root = cross_product_node;
            // Step4 Return the first column of the cross product as the result
            SharedPtr<ColumnExpression> result = ColumnExpression::Make(expr_ptr->Type(), "", aggregate_index, first_function_expr->Name(), 0, 0);

            return result;
        }
        case SubqueryType::kExists: {
            // Construct following plan tree:
            // CrossProduct
            // |-> left plan tree
            // |-> Project (count_star = 1)
            //     |-> Aggregate( count(*) as count_start)
            //         |-> Limit (1)
            //             |-> right plan tree
            PlannerError("Plan EXISTS uncorrelated subquery");
            break;
        }
        case SubqueryType::kNotExists: {
            PlannerError("Plan not EXISTS uncorrelated subquery");
            break;
        }
        case SubqueryType::kNotIn:
        case SubqueryType::kIn: {
            // 1. Generate right column expression
            ColumnBinding right_column_binding = subquery_plan->GetColumnBindings()[0];
            SharedPtr<ColumnExpression> right_column = ColumnExpression::Make(expr_ptr->left_->Type(),
                                                                              subquery_plan->name(),
                                                                              right_column_binding.table_idx,
                                                                              "0",
                                                                              right_column_binding.column_idx,
                                                                              0);

            // 2. Generate condition expression;
            Vector<SharedPtr<BaseExpression>> function_arguments;
            function_arguments.reserve(2);
            function_arguments.emplace_back(expr_ptr->left_);
            SharedPtr<BaseExpression> right_expr = CastExpression::AddCastToType(right_column, expr_ptr->left_->Type());
            function_arguments.emplace_back(right_expr);

            NewCatalog *catalog = query_context->storage()->catalog();
            SharedPtr<FunctionSet> function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "first");

            if (expr_ptr->subquery_type_ == SubqueryType::kIn) {
                function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "=");
            } else {
                function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "<>");
            }
            auto scalar_function_set_ptr = std::static_pointer_cast<ScalarFunctionSet>(function_set_ptr);
            ScalarFunction equi_function = scalar_function_set_ptr->GetMostMatchFunction(function_arguments);

            SharedPtr<FunctionExpression> function_expr_ptr = MakeShared<FunctionExpression>(equi_function, function_arguments);

            Vector<SharedPtr<BaseExpression>> conditions;
            conditions.emplace_back(function_expr_ptr);

            // 3. Generate mark join
            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "logical_join" + ToStr(logical_node_id);
            SharedPtr<LogicalJoin> join_node = MakeShared<LogicalJoin>(logical_node_id, JoinType::kMark, alias, conditions, root, subquery_plan);
            join_node->mark_index_ = bind_context->GenerateTableIndex();
            root = join_node;

            // 4. Generate output expression
            SharedPtr<ColumnExpression> result =
                ColumnExpression::Make(expr_ptr->Type(), function_expr_ptr->Name(), join_node->mark_index_, "0", 0, 0);

            return result;
        }
        case SubqueryType::kAny:
            PlannerError("Plan ANY uncorrelated subquery");
            break;
        default:
            PlannerError("Unknown subquery type.");
    }

    PlannerError("Not implement to unnest uncorrelated subquery.");
#endif
}

SharedPtr<BaseExpression> SubqueryUnnest::UnnestCorrelated(SubqueryExpression *expr_ptr,
                                                           SharedPtr<LogicalNode> &root,
                                                           SharedPtr<LogicalNode> &subquery_plan,
                                                           QueryContext *query_context,
                                                           const SharedPtr<BindContext> &bind_context) {
#if 0
    auto &correlated_columns = bind_context->correlated_column_exprs_;

    PlannerAssert(!correlated_columns.empty(), "No correlated column");

    // Valid the correlated columns are from one table.
    SizeT column_count = correlated_columns.size();
    SizeT table_index = correlated_columns[0]->binding().table_idx;
    for (SizeT idx = 1; idx < column_count; ++idx) {
        if (table_index != correlated_columns[idx]->binding().table_idx) {
            PlannerError("Correlated columns can be only from one table, now.")
        }
    }

    switch (expr_ptr->subquery_type_) {

        case SubqueryType::kExists: {
            DependentJoinFlattener dependent_join_flattener(bind_context, query_context);

            dependent_join_flattener.DetectCorrelatedExpressions(subquery_plan);

            // Push down the dependent join
            auto dependent_join = dependent_join_flattener.PushDependentJoin(subquery_plan);
            const Vector<ColumnBinding> &subplan_column_bindings = dependent_join->GetColumnBindings();

            // Generate inner join
            Vector<SharedPtr<BaseExpression>> join_conditions;
            SizeT correlated_base_index = dependent_join_flattener.CorrelatedColumnBaseIndex();

            GenerateJoinConditions(query_context, join_conditions, correlated_columns, subplan_column_bindings, correlated_base_index);

            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "logical_join" + ToStr(logical_node_id);
            SharedPtr<LogicalJoin> logical_join =
                MakeShared<LogicalJoin>(logical_node_id, JoinType::kMark, alias, join_conditions, root, dependent_join);
            logical_join->mark_index_ = bind_context->GenerateTableIndex();

            root = logical_join;
            // Generate result expression
            SharedPtr<Vector<String>> right_names = dependent_join->GetOutputNames();
            SharedPtr<ColumnExpression> result = ColumnExpression::Make(expr_ptr->Type(), alias, logical_join->mark_index_, right_names->at(0), 0, 0);
            return result;
        }
        case SubqueryType::kNotExists: {
            DependentJoinFlattener dependent_join_flattener(bind_context, query_context);

            dependent_join_flattener.DetectCorrelatedExpressions(subquery_plan);

            // Push down the dependent join
            auto dependent_join = dependent_join_flattener.PushDependentJoin(subquery_plan);
            const Vector<ColumnBinding> &subplan_column_bindings = dependent_join->GetColumnBindings();

            // Generate inner join
            Vector<SharedPtr<BaseExpression>> join_conditions;
            SizeT correlated_base_index = dependent_join_flattener.CorrelatedColumnBaseIndex();

            GenerateJoinConditions(query_context, join_conditions, correlated_columns, subplan_column_bindings, correlated_base_index);

            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "logical_join" + ToStr(logical_node_id);
            SharedPtr<LogicalJoin> logical_join =
                MakeShared<LogicalJoin>(logical_node_id, JoinType::kMark, alias, join_conditions, root, dependent_join);
            logical_join->mark_index_ = bind_context->GenerateTableIndex();

            root = logical_join;
            // Generate result expression
            SharedPtr<Vector<String>> right_names = dependent_join->GetOutputNames();
            SharedPtr<ColumnExpression> mark_column =
                ColumnExpression::Make(expr_ptr->Type(), alias, logical_join->mark_index_, right_names->at(0), 0, 0);

            // Add NOT function on the mark column
            NewCatalog *catalog = query_context->storage()->catalog();
            SharedPtr<FunctionSet> function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "not");
            Vector<SharedPtr<BaseExpression>> function_arguments;
            function_arguments.reserve(1);
            function_arguments.emplace_back(mark_column);
            auto scalar_function_set_ptr = std::static_pointer_cast<ScalarFunctionSet>(function_set_ptr);
            ScalarFunction equi_function = scalar_function_set_ptr->GetMostMatchFunction(function_arguments);

            SharedPtr<FunctionExpression> function_expr_ptr = MakeShared<FunctionExpression>(equi_function, function_arguments);
            return function_expr_ptr;
        }
        case SubqueryType::kNotIn:
        case SubqueryType::kIn: {
            DependentJoinFlattener dependent_join_flattener(bind_context, query_context);

            dependent_join_flattener.DetectCorrelatedExpressions(subquery_plan);

            // Push down the dependent join
            auto dependent_join = dependent_join_flattener.PushDependentJoin(subquery_plan);
            const Vector<ColumnBinding> &subplan_column_bindings = dependent_join->GetColumnBindings();
            const SharedPtr<Vector<String>> &subplan_column_names = dependent_join->GetOutputNames();
            const SharedPtr<Vector<SharedPtr<DataType>>> &subplan_column_types = dependent_join->GetOutputTypes();

            // Generate inner join
            Vector<SharedPtr<BaseExpression>> join_conditions;
            SizeT correlated_base_index = dependent_join_flattener.CorrelatedColumnBaseIndex();

            GenerateJoinConditions(query_context, join_conditions, correlated_columns, subplan_column_bindings, correlated_base_index);

            // IN comparison
            Vector<SharedPtr<BaseExpression>> in_arguments;

            SharedPtr<ColumnExpression> subquery_output_column = ColumnExpression::Make(*subplan_column_types->at(0),
                                                                                        "",
                                                                                        subplan_column_bindings[0].table_idx,
                                                                                        subplan_column_names->at(0),
                                                                                        subplan_column_bindings[0].column_idx,
                                                                                        0);

            if (expr_ptr->left_->Type() == *subplan_column_types->at(0)) {
                in_arguments.emplace_back(subquery_output_column);
            } else {
                BoundCastFunc cast = CastFunction::GetBoundFunc(*subplan_column_types->at(0), expr_ptr->left_->Type());
                SharedPtr<BaseExpression> cast_expr = MakeShared<CastExpression>(cast, subquery_output_column, expr_ptr->left_->Type());
                in_arguments.emplace_back(cast_expr);
            }

            InType in_type = expr_ptr->subquery_type_ == SubqueryType::kIn ? InType::kIn : InType::kNotIn;

            SharedPtr<InExpression> in_expression_ptr = MakeShared<InExpression>(in_type, expr_ptr->left_, in_arguments);

            join_conditions.emplace_back(in_expression_ptr);

            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "logical_join" + ToStr(logical_node_id);
            SharedPtr<LogicalJoin> logical_join =
                MakeShared<LogicalJoin>(logical_node_id, JoinType::kMark, alias, join_conditions, root, dependent_join);
            logical_join->mark_index_ = bind_context->GenerateTableIndex();

            root = logical_join;
            // Generate result expression
            SharedPtr<Vector<String>> right_names = dependent_join->GetOutputNames();
            SharedPtr<ColumnExpression> result = ColumnExpression::Make(expr_ptr->Type(), alias, logical_join->mark_index_, right_names->at(0), 0, 0);
            return result;
        }
        case SubqueryType::kScalar: {
            DependentJoinFlattener dependent_join_flattener(bind_context, query_context);

            dependent_join_flattener.DetectCorrelatedExpressions(subquery_plan);

            // Push down the dependent join
            auto dependent_join = dependent_join_flattener.PushDependentJoin(subquery_plan);
            const Vector<ColumnBinding> &subplan_column_bindings = dependent_join->GetColumnBindings();

            // Generate inner join
            Vector<SharedPtr<BaseExpression>> join_conditions;
            SizeT correlated_base_index = dependent_join_flattener.CorrelatedColumnBaseIndex();

            GenerateJoinConditions(query_context, join_conditions, correlated_columns, subplan_column_bindings, correlated_base_index);

            u64 logical_node_id = bind_context->GetNewLogicalNodeId();
            String alias = "logical_join" + ToStr(logical_node_id);
            SharedPtr<LogicalJoin> logical_join =
                MakeShared<LogicalJoin>(logical_node_id, JoinType::kInner, alias, join_conditions, root, dependent_join);
            root = logical_join;
            // Generate result expression
            SharedPtr<Vector<String>> right_names = dependent_join->GetOutputNames();
            ColumnBinding right_first_output_binding = subplan_column_bindings[0];
            SharedPtr<ColumnExpression> result = ColumnExpression::Make(expr_ptr->Type(),
                                                                        alias,
                                                                        right_first_output_binding.table_idx,
                                                                        right_names->at(0),
                                                                        right_first_output_binding.column_idx,
                                                                        0);
            return result;
        }
        case SubqueryType::kAny: {
            NotImplementError("Unnest correlated any subquery.");
        }
    }
    PlannerError("Unreachable")
#endif
}

void SubqueryUnnest::GenerateJoinConditions(QueryContext *query_context,
                                            Vector<SharedPtr<BaseExpression>> &join_conditions,
                                            const Vector<SharedPtr<ColumnExpression>> &correlated_columns,
                                            const Vector<ColumnBinding> &subplan_column_bindings,
                                            SizeT correlated_base_index) {
#if 0
    NewCatalog *catalog = query_context->storage()->catalog();
    SharedPtr<FunctionSet> function_set_ptr = NewCatalog::GetFunctionSetByName(catalog, "=");
    SizeT column_count = correlated_columns.size();
    for (SizeT idx = 0; idx < column_count; ++idx) {
        auto &left_column_expr = correlated_columns[idx];
        SizeT correlated_column_index = correlated_base_index + idx;
        if (correlated_column_index >= subplan_column_bindings.size()) {
            PlannerError(Format("Column index is out of range.{}/{}", correlated_column_index, subplan_column_bindings.size()))
        }

        // Generate new correlated column expression
        auto &right_column_binding = subplan_column_bindings[correlated_column_index];
        SharedPtr<ColumnExpression> right_column_expr = ColumnExpression::Make(left_column_expr->Type(),
                                                                               left_column_expr->table_name(),
                                                                               right_column_binding.table_idx,
                                                                               left_column_expr->column_name(),
                                                                               right_column_binding.column_idx,
                                                                               0);

        // Generate join condition expression
        Vector<SharedPtr<BaseExpression>> function_arguments;
        function_arguments.reserve(2);
        function_arguments.emplace_back(left_column_expr);
        function_arguments.emplace_back(right_column_expr);

        auto scalar_function_set_ptr = std::static_pointer_cast<ScalarFunctionSet>(function_set_ptr);
        ScalarFunction equi_function = scalar_function_set_ptr->GetMostMatchFunction(function_arguments);

        SharedPtr<FunctionExpression> function_expr_ptr = MakeShared<FunctionExpression>(equi_function, function_arguments);
        join_conditions.emplace_back(function_expr_ptr);
    }
#endif
}

} // namespace infinity