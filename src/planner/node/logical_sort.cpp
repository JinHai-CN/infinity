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

import std;
import stl;
import column_binding;
import parser;
import base_expression;

module logical_sort;

namespace infinity {

Vector<ColumnBinding> LogicalSort::GetColumnBindings() const { return left_node_->GetColumnBindings(); }

SharedPtr<Vector<String>> LogicalSort::GetOutputNames() const { return left_node_->GetOutputNames(); }

SharedPtr<Vector<SharedPtr<DataType>>> LogicalSort::GetOutputTypes() const { return left_node_->GetOutputTypes(); }

String LogicalSort::ToString(i64 &space) const {

    std::stringstream ss;
    String arrow_str;
    if (space > 3) {
        space -= 4;
        arrow_str = "->  ";
    }
    ss << String(space, ' ') << arrow_str << "Order by: ";
    SizeT expression_count = expressions_.size();
    for (SizeT i = 0; i < expression_count - 1; ++i) {
        ss << expressions_[i]->Name() << " " << OrderBy2Str(order_by_types_[i]) << ", ";
    }
    ss << expressions_.back()->Name() << " " << OrderBy2Str(order_by_types_.back());
    space += arrow_str.size();

    return ss.str();
}

} // namespace infinity
