//
// Created by JinHai on 2022/8/13.
//

module;

import stl;
import base_expression;
import parser;
import bind_context;
import column_expression;
import third_party;
import infinity_assert;
import infinity_exception;

module order_binder;

namespace infinity {

void OrderBinder::PushExtraExprToSelectList(ParsedExpr *expr, const SharedPtr<BindContext> &bind_context_ptr) {
    if (expr->type_ == ParsedExprType::kConstant) {
        ConstantExpr *order_by_index = (ConstantExpr *)expr;
        Assert<PlannerException>(order_by_index->literal_type_ == LiteralType::kInteger, "Error Order by expression", __FILE_NAME__, __LINE__);
        // Order by 1, means order by 1st select list item.
        return;
    }

    String expr_name = expr->GetName();

    if (bind_context_ptr->select_alias2index_.contains(expr_name)) {
        return;
    }

    if (bind_context_ptr->select_expr_name2index_.contains(expr_name)) {
        return;
    }

    bind_context_ptr->select_expr_name2index_[expr_name] = bind_context_ptr->select_expression_.size();
    bind_context_ptr->select_expression_.emplace_back(expr);
}

SharedPtr<BaseExpression> OrderBinder::BuildExpression(const ParsedExpr &expr, BindContext *bind_context_ptr, i64 depth, bool root) {
    if (expr.type_ == ParsedExprType::kKnn) {
        return ExpressionBinder::BuildKnnExpr((KnnExpr &)expr, bind_context_ptr, depth, root);
    }

    i64 column_id = -1;

    // If the expr is from projection, then create a column reference from projection.
    if (expr.type_ == ParsedExprType::kConstant) {
        ConstantExpr &const_expr = (ConstantExpr &)expr;
        if (const_expr.literal_type_ == LiteralType::kInteger) {
            column_id = const_expr.integer_value_;
            if (column_id >= bind_context_ptr->project_exprs_.size()) {
                Error<PlannerException>("Order by are going to use nonexistent column from select list.", __FILE_NAME__, __LINE__);
            }
            --column_id;
        } else {
            Error<PlannerException>("Order by non-integer constant value.", __FILE_NAME__, __LINE__);
        }
    } else {
        String expr_name = expr.GetName();

        if (bind_context_ptr->select_alias2index_.contains(expr_name)) {
            column_id = bind_context_ptr->select_alias2index_[expr_name];
        }

        if (bind_context_ptr->select_expr_name2index_.contains(expr_name)) {
            column_id = bind_context_ptr->select_expr_name2index_[expr_name];
        }

        if (column_id == -1) {
            Error<PlannerException>(Format("{} isn't found in project list.", expr_name), __FILE_NAME__, __LINE__);
        }
    }

    const SharedPtr<BaseExpression> &project_expr = bind_context_ptr->project_exprs_[column_id];

    SharedPtr<ColumnExpression> result = ColumnExpression::Make(project_expr->Type(),
                                                                bind_context_ptr->project_table_name_,
                                                                bind_context_ptr->project_table_index_,
                                                                ToStr(column_id),
                                                                column_id,
                                                                depth);
    result->source_position_ = SourcePosition(bind_context_ptr->binding_context_id_, ExprSourceType::kProjection);
    return result;
}

} // namespace infinity
