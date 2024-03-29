//
// Created by JinHai on 2022/12/16.
//

module;

import parser;
import stl;
//import infinity_assert;
//import infinity_exception;

export module null_value;

export namespace infinity {

template <typename ValueType>
inline ValueType NullValue() {
    TypeError(DataType::TypeToString<ValueType>() + " doesn't have null value.");
}

template <>
inline BooleanT NullValue() {
    return false;
}

template <>
inline TinyIntT NullValue() {
    return i8_inf;
}

template <>
inline SmallIntT NullValue() {
    return i16_inf;
}

template <>
inline IntegerT NullValue() {
    return i32_inf;
}

template <>
inline BigIntT NullValue() {
    return i64_inf;
}

template <>
inline HugeIntT NullValue() {
    HugeIntT value;
    value.lower = i64_inf;
    value.upper = i64_inf;
    return value;
}

template <>
inline FloatT NullValue() {
    return f32_inf;
}

template <>
inline DoubleT NullValue() {
    return f64_inf;
}

template <>
inline DecimalT NullValue() {
    return DecimalT(i64_inf, i64_inf);
}

template <>
inline VarcharT NullValue() {
    return VarcharT(nullptr, 0);
}

template <>
inline DateT NullValue() {
    return DateT(i32_inf);
}

template <>
inline TimeT NullValue() {
    return TimeT(i32_inf);
}

template <>
inline DateTimeT NullValue() {
    return DateTimeT(i32_inf, i32_inf);
}

template <>
inline TimestampT NullValue() {
    return TimestampT(i32_inf, i32_inf);
}

template <>
inline IntervalT NullValue() {
    IntervalT interval;
    interval.value = i32_inf;
    interval.unit = TimeUnit::kInvalidUnit;
    return interval;
}

template <>
inline PointT NullValue() {
    return PointT(f64_inf, f64_inf);
}

template <>
inline LineT NullValue() {
    return LineT(f64_inf, f64_inf, f64_inf);
}

template <>
inline LineSegT NullValue() {
    PointT p1(f64_inf, f64_inf);
    PointT p2(f64_inf, f64_inf);
    return LineSegT(p1, p2);
}

template <>
inline BoxT NullValue() {
    PointT p1(f64_inf, f64_inf);
    PointT p2(f64_inf, f64_inf);
    return BoxT(p1, p2);
}

template <>
inline PathT NullValue() {
    PathT path(u32_inf, i32_inf);
    path.ptr = ptr_inf;
    return path;
}

template <>
inline PolygonT NullValue() {
    ptr_t ptr{};
    u64 point_count{0}; // 65535 point are the max point count
    BoxT bounding_box{};
    PolygonT polygon;
    polygon.ptr = ptr_inf;
    polygon.point_count = u64_inf;
    PointT null_point = PointT(f64_inf, f64_inf);
    polygon.bounding_box.upper_left.x = f64_inf;
    polygon.bounding_box.upper_left.y = f64_inf;
    polygon.bounding_box.lower_right.x = f64_inf;
    polygon.bounding_box.lower_right.y = f64_inf;
    return polygon;
}

template <>
inline CircleT NullValue() {
    return CircleT(PointT(f64_inf, f64_inf), f64_inf);
}

template <>
inline BitmapT NullValue() {
    BitmapT bitmap;
    bitmap.ptr = u64_ptr_inf;
    bitmap.count = u64_inf;
    return bitmap;
}

template <>
inline UuidT NullValue() {
    UuidT uuid;
    return uuid;
}

template <>
inline BlobT NullValue() {
    BlobT blob;
    blob.size = u64_inf;
    blob.ptr = ptr_inf;
    return blob;
}

template <>
inline EmbeddingT NullValue() {
    ptr_t ptr = ptr_inf;
    EmbeddingT embedding(Move(ptr));
    return embedding;
}

template <>
inline RowT NullValue() {
    RowT row_id;
    row_id.segment_id_ = i32_inf;
    row_id.block_id_ = i32_inf;
    row_id.block_offset_ = i32_inf;
    return row_id;
}

template <>
inline MixedT NullValue() {
    auto null_value = MixedT::MakeNull();
    return null_value;
}

} // namespace infinity