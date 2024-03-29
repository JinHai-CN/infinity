//
// Created by jinhai on 23-1-17.
//

#pragma once

#include "common/types/alias/smart_ptr.h"
#include "common/types/alias/strings.h"
#include "common/types/alias/containers.h"
#include "common/types/alias/primitives.h"
#include "parser/statement/select_statement.h"
#include "parser/parsed_expr/knn_expr.h"

namespace infinity {

class QueryContext;
class BindContext;
class BoundSelectStatement;
class SelectStatement;
class TableRef;
class TableReference;
class BaseTableReference;
class SubqueryReference;
class CrossProductReference;
class JoinReference;
class ParsedExpr;
class Binding;
class BindAliasProxy;

class QueryBinder : public std::enable_shared_from_this<QueryBinder> {
public:
    explicit QueryBinder(QueryContext *query_context, SharedPtr<BindContext> bind_context_ptr)
        : query_context_ptr_(std::move(query_context)), bind_context_ptr_(std::move(bind_context_ptr)) {}

    SharedPtr<BoundSelectStatement> BindSelect(const SelectStatement &statement);

    QueryContext *query_context_ptr_;

    SharedPtr<BindContext> bind_context_ptr_;

private:
    SharedPtr<TableRef> BuildFromClause(QueryContext *query_context, const BaseTableReference *from_table);

    SharedPtr<TableRef> BuildDummyTable(QueryContext *query_context);

    SharedPtr<TableRef> BuildTable(QueryContext *query_context, const TableReference *table_reference);

    SharedPtr<TableRef> BuildSubquery(QueryContext *query_context, const SubqueryReference *subquery_reference);

    SharedPtr<TableRef> BuildCTE(QueryContext *query_context, const String &name);

    SharedPtr<TableRef> BuildBaseTable(QueryContext *query_context, const TableReference *table_reference);

    SharedPtr<TableRef> BuildView(QueryContext *query_context, const TableReference *from_table);

    SharedPtr<TableRef> BuildCrossProduct(QueryContext *query_context, const CrossProductReference *cross_product);

    SharedPtr<TableRef> BuildJoin(QueryContext *query_context, const JoinReference *join_reference);

    void UnfoldStarExpression(QueryContext *query_context, const Vector<ParsedExpr *> &input_select_list, Vector<ParsedExpr *> &output_select_list);

    void GenerateColumns(const SharedPtr<Binding> &binding, const String &table_name, Vector<ParsedExpr *> &output_select_list);

    void BuildGroupBy(QueryContext *query_context,
                      const SelectStatement &select,
                      const SharedPtr<BindAliasProxy> &bind_alias_proxy,
                      SharedPtr<BoundSelectStatement> &select_statement);

    void BuildHaving(QueryContext *query_context,
                     const SelectStatement &select,
                     const SharedPtr<BindAliasProxy> &bind_alias_proxy,
                     SharedPtr<BoundSelectStatement> &select_statement);

    void PushOrderByToProject(QueryContext *query_context, const SelectStatement &statement);

    void BuildSelectList(QueryContext *query_context, SharedPtr<BoundSelectStatement> &select_statement);

    void BuildOrderBy(QueryContext *query_context, const SelectStatement &statement, SharedPtr<BoundSelectStatement> &bound_statement) const;

    void BuildLimit(QueryContext *query_context, const SelectStatement &statement, SharedPtr<BoundSelectStatement> &bound_statement) const;

    void PruneOutput(QueryContext *query_context, i64 select_column_count, SharedPtr<BoundSelectStatement> &bound_statement);

    static void CheckKnnAndOrderBy(KnnDistanceType distance_type, OrderType order_type);
};

} // namespace infinity
