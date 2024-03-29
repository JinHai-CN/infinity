//
// Created by JinHai on 2022/12/15.
//

#pragma once

#include "common/types/alias/smart_ptr.h"

namespace infinity {

class ColumnVector;

struct CastParameters {
    bool strict{false};
};

using cast_function_t = bool (*)(const SharedPtr<ColumnVector> &source, SharedPtr<ColumnVector> &result, size_t count, CastParameters &parameters);

struct BoundCastFunc {
    explicit BoundCastFunc(cast_function_t func) : function(func){};

    cast_function_t function = nullptr;
};

} // namespace infinity
