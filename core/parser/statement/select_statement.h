//
// Created by jinhai on 23-2-22.
//

#pragma once

#include "base_statement.h"
#include "expr/column_expr.h"
#include "expr/constant_expr.h"
#include "table_reference/base_table_reference.h"

namespace infinity {

class SelectStatement;

enum class SetOperatorType { kUnion, kUnionAll, kIntersect, kExcept };

struct WithExpr {
    ~WithExpr() {
        if (select_ != nullptr) {
            delete select_;
        }
    }
    std::string alias_{};
    BaseStatement *select_{};
};

enum OrderType { kAsc, kDesc };

std::string ToString(OrderType type);

struct OrderByExpr {
    ~OrderByExpr() { delete expr_; }
    ParsedExpr *expr_{};
    OrderType type_{OrderType::kAsc};
};

class SelectStatement final : public BaseStatement {
public:
    SelectStatement() : BaseStatement(StatementType::kSelect) {}

    ~SelectStatement() final;

    [[nodiscard]] std::string ToString() const final;

    BaseTableReference *table_ref_{nullptr};
    std::vector<ParsedExpr *> *select_list_{nullptr};
    bool select_distinct_{false};
    ParsedExpr *where_expr_{nullptr};
    std::vector<ParsedExpr *> *group_by_list_{nullptr};
    ParsedExpr *having_expr_{nullptr};
    std::vector<OrderByExpr *> *order_by_list{nullptr};
    ParsedExpr *limit_expr_{nullptr};
    ParsedExpr *offset_expr_{nullptr};
    std::vector<WithExpr *> *with_exprs_{nullptr};

    SetOperatorType set_op_{SetOperatorType::kUnion};
    SelectStatement *nested_select_{nullptr};
};

} // namespace infinity
