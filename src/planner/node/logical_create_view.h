//
// Created by jinhai on 23-3-6.
//

#pragma once

#include <utility>

#include "parser/statement/extra/create_view_info.h"
#include "planner/bound_select_statement.h"
#include "planner/logical_node.h"

namespace infinity {

class LogicalCreateView : public LogicalNode {
public:
    static inline SharedPtr<LogicalCreateView> Make(u64 node_id,
                                                    const SharedPtr<Vector<String>> &names_ptr,
                                                    const SharedPtr<Vector<SharedPtr<DataType>>> &types_ptr,
                                                    const SharedPtr<CreateViewInfo> &create_view_info) {
        return MakeShared<LogicalCreateView>(node_id, names_ptr, types_ptr, create_view_info);
    }

public:
    LogicalCreateView(u64 node_id,
                      SharedPtr<Vector<String>> names_ptr,
                      SharedPtr<Vector<SharedPtr<DataType>>> types_ptr,
                      SharedPtr<CreateViewInfo> create_view_info)
        : LogicalNode(node_id, LogicalNodeType::kCreateView), names_ptr_(std::move(names_ptr)), types_ptr_(std::move(types_ptr)),
          create_view_info_(std::move(create_view_info)) {}

    [[nodiscard]] Vector<ColumnBinding> GetColumnBindings() const final;

    [[nodiscard]] SharedPtr<Vector<String>> GetOutputNames() const final;

    [[nodiscard]] SharedPtr<Vector<SharedPtr<DataType>>> GetOutputTypes() const final;

    String ToString(i64 &space) const final;

    inline String name() final { return "LogicalCreateView"; }

    [[nodiscard]] inline const SharedPtr<Vector<String>> &names_ptr() const { return names_ptr_; };

    [[nodiscard]] inline const SharedPtr<Vector<SharedPtr<DataType>>> &types_ptr() const { return types_ptr_; }

    [[nodiscard]] inline const SharedPtr<CreateViewInfo> &create_view_info() const { return create_view_info_; }

private:
    SharedPtr<Vector<String>> names_ptr_{};
    SharedPtr<Vector<SharedPtr<DataType>>> types_ptr_{};
    SharedPtr<CreateViewInfo> create_view_info_{nullptr};
};

} // namespace infinity
