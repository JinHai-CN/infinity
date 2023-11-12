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

import stl;
import column_vector;
import vector_buffer;
import bound_cast_func;
import parser;
import column_vector_cast;

import infinity_exception;
import third_party;

export module geography_cast;

namespace infinity {

export struct GeographyTryCastToVarlen;

export template <class SourceType>
inline BoundCastFunc BindGeographyCast(const DataType &source, DataType &target) {
    Assert<FunctionException>(source.type() != target.type(),
                              Format("Attempt to cast from {} to {}", source.ToString(), target.ToString()));
    switch (target.type()) {
        case LogicalType::kVarchar: {
            return BoundCastFunc(&ColumnVectorCast::TryCastColumnVectorToVarlen<SourceType, VarcharT, GeographyTryCastToVarlen>);
        }
        default: {
            Error<TypeException>(Format("Can't cast from geography type to {}", target.ToString()));
        }
    }
    return BoundCastFunc(nullptr);
}

struct GeographyTryCastToVarlen {
    template <typename SourceType, typename TargetType>
    static inline bool Run(const SourceType &source, TargetType &target, const SharedPtr<ColumnVector> &vector_ptr) {
        Error<FunctionException>(
                Format("Not support to cast from {} to {}", DataType::TypeToString<SourceType>(), DataType::TypeToString<TargetType>()));
        return false;
    }
};

template <>
inline bool GeographyTryCastToVarlen::Run(const PointT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const LineT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const LineSegT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const BoxT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const PathT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const PolygonT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

template <>
inline bool GeographyTryCastToVarlen::Run(const CircleT &source, VarcharT &target, const SharedPtr<ColumnVector> &vector_ptr) {
    Error<NotImplementException>("Not implemented");
    return false;
}

} // namespace infinity
