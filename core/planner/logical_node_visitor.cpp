//
// Created by jinhai on 23-2-16.
//

module;

#include <vector>
#include <memory>

import stl;
import base_expression;
import logical_node_type;
import infinity_exception;
import infinity_assert;
import third_party;
import expression_type;
import logical_node_type;
import logical_node;
import logical_aggregate;
import logical_join;
import logical_limit;
import logical_filter;
import logical_project;
import logical_sort;
import logical_insert;

import aggregate_expression;
import between_expression;
import case_expression;
import cast_expression;
import column_expression;
import function_expression;
import value_expression;
import in_expression;
import subquery_expression;
import knn_expression;
import conjunction_expression;
import logger;

module logical_node_visitor;

namespace infinity {

void LogicalNodeVisitor::VisitNodeChildren(LogicalNode &op) {
    if (op.left_node()) {
        VisitNode(*op.left_node());
    }
    if (op.right_node()) {
        VisitNode(*op.right_node());
    }
}

void LogicalNodeVisitor::VisitNodeExpression(LogicalNode &op) {
    switch (op.operator_type()) {
        case LogicalNodeType::kAggregate: {
            auto &node = (LogicalAggregate &)op;
            for (auto &expression : node.groups_) {
                VisitExpression(expression);
            }
            for (auto &expression : node.aggregates_) {
                VisitExpression(expression);
            }
            break;
        }
        case LogicalNodeType::kJoin: {
            auto &node = (LogicalJoin &)op;
            for (auto &expression : node.conditions_) {
                VisitExpression(expression);
            }
            break;
        }
        case LogicalNodeType::kLimit: {
            auto &node = (LogicalLimit &)op;
            if (node.limit_expression_ != nullptr) {
                VisitExpression(node.limit_expression_);
            }
            if (node.offset_expression_ != nullptr) {
                VisitExpression(node.offset_expression_);
            }
            break;
        }
        case LogicalNodeType::kFilter: {
            auto &node = (LogicalFilter &)op;
            VisitExpression(node.expression());
            break;
        }
        case LogicalNodeType::kProjection: {
            auto &node = (LogicalProject &)op;
            for (auto &expression : node.expressions_) {
                VisitExpression(expression);
            }
            break;
        }
        case LogicalNodeType::kSort: {
            auto &node = (LogicalSort &)op;
            for (auto &expression : node.expressions_) {
                VisitExpression(expression);
            }
            break;
        }
        case LogicalNodeType::kInsert: {
            auto &node = (LogicalInsert &)op;
            for (auto &value : node.value_list())
                for (auto &expression : value) {
                    VisitExpression(expression);
                }
            break;
        }
        default: {
            LOG_TRACE(Format("Visit logical node: {}", op.name()));
        }
    }
}

void LogicalNodeVisitor::VisitExpression(SharedPtr<BaseExpression> &expression) {
    SharedPtr<BaseExpression> result;
    switch (expression->type()) {

        case ExpressionType::kAggregate: {
            auto aggregate_expression = (AggregateExpression*)(expression.get());
            for (auto &argument : aggregate_expression->arguments()) {
                VisitExpression(argument);
            }

            result = VisitReplace(aggregate_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kCast: {
            auto cast_expression = (CastExpression*)(expression.get());
            for (auto &argument : cast_expression->arguments()) {
                VisitExpression(argument);
            }

            result = VisitReplace(cast_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kCase: {
            auto case_expression = (CaseExpression*)(expression.get());
            Assert<PlannerException>(case_expression->arguments().empty(), "Case expression shouldn't have arguments", __FILE_NAME__, __LINE__);
            for (auto &case_expr : case_expression->CaseExpr()) {
                VisitExpression(case_expr.then_expr_);
                VisitExpression(case_expr.when_expr_);
            }

            VisitExpression(case_expression->ElseExpr());

            result = VisitReplace(case_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kConjunction: {
            auto conjunction_expression = (ConjunctionExpression*)(expression.get());
            for (auto &argument : conjunction_expression->arguments()) {
                VisitExpression(argument);
            }

            result = VisitReplace(conjunction_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kColumn: {
            auto column_expression = (ColumnExpression*)(expression.get());
            Assert<PlannerException>(column_expression->arguments().empty(), "Column expression shouldn't have arguments", __FILE_NAME__, __LINE__);

            result = VisitReplace(column_expression);
            if (result.get() == nullptr) {
                Error<PlannerException>("Visit column expression will always rewrite the expression", __FILE_NAME__, __LINE__);
            }
            expression = result;
            break;
        }
        case ExpressionType::kFunction: {
            auto function_expression = (FunctionExpression*)(expression.get());
            for (auto &argument : function_expression->arguments()) {
                VisitExpression(argument);
            }

            result = VisitReplace(function_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kValue: {
            auto value_expression = (ValueExpression*)(expression.get());

            Assert<PlannerException>(value_expression->arguments().empty(), "Column expression shouldn't have arguments", __FILE_NAME__, __LINE__);

            result = VisitReplace(value_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kIn: {
            auto in_expression = (InExpression*)(expression.get());

            VisitExpression(in_expression->left_operand());
            for (auto &argument : in_expression->arguments()) {
                VisitExpression(argument);
            }

            result = VisitReplace(in_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        case ExpressionType::kSubQuery: {
            auto subquery_expression = (SubqueryExpression*)(expression.get());

            result = VisitReplace(subquery_expression);
            if (result.get() != nullptr) {
                Error<PlannerException>("Visit subquery expression will always rewrite the expression", __FILE_NAME__, __LINE__);
            }
            break;
        }
        case ExpressionType::kKnn: {
            auto knn_expression = (KnnExpression*)(expression.get());
            VisitExpression(knn_expression->arguments()[0]);

            result = VisitReplace(knn_expression);
            if (result.get() != nullptr) {
                expression = result;
            }
            break;
        }
        default: {
            Error<PlannerException>(Format("Unexpected expression type: {}", expression->Name()), __FILE_NAME__, __LINE__);
        }
    }
}

void LogicalNodeVisitor::VisitExpressionChildren(SharedPtr<BaseExpression> &expression) {
    switch (expression->type()) {

        case ExpressionType::kAggregate: {
            auto aggregate_expression = (AggregateExpression*)(expression.get());
            for (auto &argument : aggregate_expression->arguments()) {
                VisitExpression(argument);
            }
            break;
        }
        case ExpressionType::kGroupingFunction:
            break;
        case ExpressionType::kArithmetic:
            break;
        case ExpressionType::kCast: {
            auto cast_expression = (CastExpression*)(expression.get());
            for (auto &argument : cast_expression->arguments()) {
                VisitExpression(argument);
            }
            break;
        }
        case ExpressionType::kCase:
            break;
        case ExpressionType::kAnd:
            break;
        case ExpressionType::kConjunction:
            break;
        case ExpressionType::kOr:
            break;
        case ExpressionType::kNot:
            break;
        case ExpressionType::kColumn:
            break;
        case ExpressionType::kCorrelatedColumn:
            break;
        case ExpressionType::kExists:
            break;
        case ExpressionType::kExtract:
            break;
        case ExpressionType::kInterval:
            break;
        case ExpressionType::kFunction:
            break;
        case ExpressionType::kList:
            break;
        case ExpressionType::kLogical:
            break;
        case ExpressionType::kEqual:
            break;
        case ExpressionType::kNotEqual:
            break;
        case ExpressionType::kLessThan:
            break;
        case ExpressionType::kGreaterThan:
            break;
        case ExpressionType::kLessThanEqual:
            break;
        case ExpressionType::kGreaterThanEqual:
            break;
        case ExpressionType::kBetween:
            break;
        case ExpressionType::kNotBetween:
            break;
        case ExpressionType::kSubQuery:
            break;
        case ExpressionType::kUnaryMinus:
            break;
        case ExpressionType::kIsNull:
            break;
        case ExpressionType::kIsNotNull:
            break;
        case ExpressionType::kValue:
            break;
        case ExpressionType::kDefault:
            break;
        case ExpressionType::kParameter:
            break;
        case ExpressionType::kIn:
            break;
        case ExpressionType::kNotIn:
            break;
        case ExpressionType::kWindowRank:
            break;
        case ExpressionType::kWindowRowNumber:
            break;
        case ExpressionType::kDistinctFrom:
            break;
        case ExpressionType::kNotDistinctFrom:
            break;
        case ExpressionType::kPlaceholder:
            break;
        case ExpressionType::kPredicate:
            break;
        case ExpressionType::kRaw:
            break;
        default: {
            break;
        }
    }
}

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const AggregateExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const BetweenExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const CaseExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const CastExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const ColumnExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const ConjunctionExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const FunctionExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const ValueExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const InExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const SubqueryExpression* expression) { return nullptr; }

SharedPtr<BaseExpression> LogicalNodeVisitor::VisitReplace(const KnnExpression* expression) { return nullptr; }

} // namespace infinity
