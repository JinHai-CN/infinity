//
// Created by JinHai on 2022/10/27.
//

#include "data_type.h"
#include "type/info/decimal_info.h"
#include "type/info/embedding_info.h"
#include "type/info/varchar_info.h"
#include "type/logical_type.h"
#include "type/type_info.h"
#include "info/bitmap_info.h"
#include "spdlog/fmt/fmt.h"
#include <charconv>

namespace infinity {

DataType::DataType(LogicalType logical_type, std::shared_ptr<TypeInfo> type_info_ptr) : type_(logical_type), type_info_(std::move(type_info_ptr)) {
    switch (logical_type) {
        case kBoolean:
        case kTinyInt:
        case kSmallInt:
        case kInteger:
        case kBigInt:
        case kHugeInt:
        case kDecimal:
        case kFloat:
        case kDouble:
        case kDate:
        case kTime:
        case kDateTime:
        case kTimestamp:
        case kInterval:
        case kPoint:
        case kLine:
        case kLineSeg:
        case kBox:
        case kCircle:
        case kBitmap:
        case kUuid:
        case kEmbedding:
        case kRowID: {
            plain_type_ = true;
            break;
        }
        case kMixed:
        case kVarchar:
        case kArray:
        case kTuple:
        case kPath:
        case kPolygon:
        case kBlob: {
            plain_type_ = false;
            break;
        }
        case kNull:
        case kMissing: {
            plain_type_ = true;
            break;
        }
        case kInvalid:
            break;
    }
}

std::string DataType::ToString() const {
    if (type_ > kInvalid) {
        ParserError(fmt::format("Invalid logical data type {}.", int(type_)));
    }
    return LogicalType2Str(type_);
}

bool DataType::operator==(const DataType &other) const {
    if (this == &other)
        return true;
    if (type_ != other.type_)
        return false;
    if (plain_type_ != other.plain_type_)
        return false;
    if (this->type_info_ == nullptr && other.type_info_ == nullptr) {
        return true;
    }
    if (this->type_info_ != nullptr && other.type_info_ != nullptr) {
        if (*this->type_info_ != *other.type_info_)
            return false;
        else
            return true;
    } else {
        return false;
    }
}

bool DataType::operator!=(const DataType &other) const { return !operator==(other); }

size_t DataType::Size() const {
    if (type_ > kInvalid) {
        ParserError(fmt::format("Invalid logical data type {}.", int(type_)));
    }

    // embedding, varchar data can get data here.
    if (type_info_ != nullptr) {
        return type_info_->Size();
    }

    // StorageAssert(type_ != kEmbedding && type_ != kVarchar, "This ype should have type info");

    return LogicalTypeWidth(type_);
}

//int64_t DataType::CastRule(const DataType &from, const DataType &to) { return CastTable::instance().GetCastCost(from.type_, to.type_); }

void DataType::MaxDataType(const DataType &right) {
    if (*this == right) {
        return;
    }

    if (this->type_ == LogicalType::kInvalid) {
        *this = right;
        return;
    }

    if (right.type_ == LogicalType::kInvalid) {
        return;
    }

    if (this->IsNumeric() && right.IsNumeric()) {
        if (this->type_ > right.type_) {
            return;
        } else {
            *this = right;
            return;
        }
    }

    if (this->type_ == LogicalType::kVarchar) {
        return;
    }
    if (right.type_ == LogicalType::kVarchar) {
        *this = right;
        return;
    }

    if (this->type_ == LogicalType::kDateTime and right.type_ == LogicalType::kTimestamp) {
        *this = right;
        return;
    }

    if (this->type_ == LogicalType::kTimestamp and right.type_ == LogicalType::kDateTime) {
        return;
    }

    ParserError(fmt::format("Max type of left: {} and right: {}", this->ToString(), right.ToString()));
}

int32_t DataType::GetSizeInBytes() const {
    int32_t size = sizeof(LogicalType);
    if (this->type_info_ != nullptr) {
        switch (this->type_) {
            case LogicalType::kArray:
                ParserError("Array isn't implemented here.");
                break;
            case LogicalType::kBitmap:
                size += sizeof(int64_t);
                break;
            case LogicalType::kDecimal:
                size += sizeof(int64_t) * 2;
                break;
            case LogicalType::kEmbedding:
                size += sizeof(EmbeddingDataType);
                size += sizeof(int32_t);
                break;
            case LogicalType::kVarchar:
                size += sizeof(int32_t);
                break;
            default:
                ParserError(fmt::format("Unexpected type {} here.", int(this->type_)));
        }
    }
    return size;
}

template <>
std::string DataType::TypeToString<BooleanT>() {
    return "Boolean";
}

template <>
std::string DataType::TypeToString<TinyIntT>() {
    return "TinyInt";
}

template <>
std::string DataType::TypeToString<SmallIntT>() {
    return "SmallInt";
}

template <>
std::string DataType::TypeToString<IntegerT>() {
    return "Integer";
}

template <>
std::string DataType::TypeToString<BigIntT>() {
    return "BigInt";
}

template <>
std::string DataType::TypeToString<HugeIntT>() {
    return "HugeInt";
}

template <>
std::string DataType::TypeToString<FloatT>() {
    return "Float";
}

template <>
std::string DataType::TypeToString<DoubleT>() {
    return "Double";
}

template <>
std::string DataType::TypeToString<DecimalT>() {
    return "Decimal";
}

template <>
std::string DataType::TypeToString<VarcharT>() {
    return "Varchar";
}

template <>
std::string DataType::TypeToString<DateT>() {
    return "Date";
}

template <>
std::string DataType::TypeToString<TimeT>() {
    return "Time";
}

template <>
std::string DataType::TypeToString<DateTimeT>() {
    return "DateTime";
}

template <>
std::string DataType::TypeToString<TimestampT>() {
    return "Timestamp";
}

template <>
std::string DataType::TypeToString<IntervalT>() {
    return "Interval";
}

template <>
std::string DataType::TypeToString<ArrayT>() {
    return "Array";
}

// template <> std::string DataType::TypeToString<TupleT>() { return "Tuple"; }
template <>
std::string DataType::TypeToString<PointT>() {
    return "Point";
}

template <>
std::string DataType::TypeToString<LineT>() {
    return "Line";
}

template <>
std::string DataType::TypeToString<LineSegT>() {
    return "LineSegment";
}

template <>
std::string DataType::TypeToString<BoxT>() {
    return "Box";
}

template <>
std::string DataType::TypeToString<PathT>() {
    return "Path";
}

template <>
std::string DataType::TypeToString<PolygonT>() {
    return "Polygon";
}

template <>
std::string DataType::TypeToString<CircleT>() {
    return "Circle";
}

template <>
std::string DataType::TypeToString<BitmapT>() {
    return "Bitmap";
}

template <>
std::string DataType::TypeToString<UuidT>() {
    return "UUID";
}

template <>
std::string DataType::TypeToString<BlobT>() {
    return "Blob";
}

template <>
std::string DataType::TypeToString<EmbeddingT>() {
    return "Embedding";
}

template <>
std::string DataType::TypeToString<RowT>() {
    return "RowID";
}

template <>
std::string DataType::TypeToString<MixedT>() {
    return "Heterogeneous";
}

template <>
BooleanT DataType::StringToValue<BooleanT>(const std::string_view &str) {
    if (str.empty()) {
        return BooleanT{};
    }
    // TODO: should support True/False, maybe others
    std::string str_lower;
    for (char ch : str) {
        str_lower.push_back(std::tolower(ch));
    }
    ParserAssert(str_lower == "true" || str_lower == "false", "Boolean type should be true or false");
    return str_lower == "true";
}

template <>
TinyIntT DataType::StringToValue<TinyIntT>(const std::string_view &str) {
    if (str.empty()) {
        return TinyIntT{};
    }
    TinyIntT value{};
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse TinyInt error"); // TODO: throw error here
    return value;
}

template <>
SmallIntT DataType::StringToValue<SmallIntT>(const std::string_view &str) {
    if (str.empty()) {
        return SmallIntT{};
    }
    SmallIntT value{};
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse SmallInt error");
    return value;
}

template <>
IntegerT DataType::StringToValue<IntegerT>(const std::string_view &str) {
    if (str.empty()) {
        return IntegerT{};
    }
    IntegerT value{};
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse Integer error");
    return value;
}

template <>
BigIntT DataType::StringToValue<BigIntT>(const std::string_view &str) {
    if (str.empty()) {
        return BigIntT{};
    }
    BigIntT value{};
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse BigInt error");
    return value;
}

template <>
FloatT DataType::StringToValue<FloatT>(const std::string_view &str) {
    if (str.empty()) {
        return FloatT{};
    }
    FloatT value{};
#if defined(__APPLE__)
    auto ret = std::sscanf(str.data(), "%a", &value);
    ParserAssert(ret == str.size(), "Parse Float error");
#else
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse Float error");
#endif
    return value;
}

template <>
DoubleT DataType::StringToValue<DoubleT>(const std::string_view &str) {
    if (str.empty()) {
        return DoubleT{};
    }
    DoubleT value{};
#if defined(__APPLE__)
    auto ret = std::sscanf(str.data(), "%la", &value);
    ParserAssert(ret == str.size(), "Parse Double error");
#else
    auto res = std::from_chars(str.begin(), str.end(), value);
    ParserAssert(res.ptr == str.data() + str.size(), "Parse Double error");
#endif
    return value;
}
} // namespace infinity
