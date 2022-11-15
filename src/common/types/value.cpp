//
// Created by JinHai on 2022/11/7.
//

#include "value.h"
#include "common/utility/infinity_assert.h"

namespace infinity {

// Value maker

Value
Value::MakeBool(BooleanT input) {
    Value value(LogicalType::kBoolean);
    value.value_.boolean = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeTinyInt(TinyIntT input) {
    Value value(LogicalType::kTinyInt);
    value.value_.tiny_int = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeSmallInt(SmallIntT input) {
    Value value(LogicalType::kSmallInt);
    value.value_.small_int = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeInt(IntegerT input) {
    Value value(LogicalType::kInteger);
    value.value_.integer = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeBigInt(BigIntT input) {
    Value value(LogicalType::kBigInt);
    value.value_.big_int = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeHugeInt(HugeIntT input) {
    Value value(LogicalType::kHugeInt);
    value.value_.huge_int = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeFloat(FloatT input) {
    Value value(LogicalType::kFloat);
    value.value_.float32 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDouble(DoubleT input) {
    Value value(LogicalType::kDouble);
    value.value_.float64 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDecimal16(Decimal16T input) {
    Value value(LogicalType::kDecimal16);
    value.value_.decimal16 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDecimal32(Decimal32T input) {
    Value value(LogicalType::kDecimal32);
    value.value_.decimal32 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDecimal64(Decimal64T input) {
    Value value(LogicalType::kDecimal64);
    value.value_.decimal64 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDecimal128(Decimal128T input) {
    Value value(LogicalType::kDecimal128);
    value.value_.decimal128 = input;
    value.is_null_ = false;
    return value;
}


Value
Value::MakeVarchar(VarcharT& input) {
    Value value(LogicalType::kVarchar);
    value.value_.varchar = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeVarchar(const String& str) {
    Value value(LogicalType::kVarchar);
    value.value_.varchar.Initialize(str);
    value.is_null_ = false;
    return value;
}

Value
Value::MakeVarchar(const char* ptr) {
    Value value(LogicalType::kVarchar);
    value.value_.varchar.Initialize(ptr);
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar1(Char1T input) {
    Value value(LogicalType::kChar1);
    value.value_.char1 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar2(Char2T input) {
    Value value(LogicalType::kChar2);
    value.value_.char2 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar4(Char4T input) {
    Value value(LogicalType::kChar4);
    value.value_.char4 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar8(Char8T input) {
    Value value(LogicalType::kChar8);
    value.value_.char8 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar15(Char15T input) {
    Value value(LogicalType::kChar15);
    value.value_.char15 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar31(Char31T input) {
    Value value(LogicalType::kChar31);
    value.value_.char31 = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeChar63(Char63T input) {
    Value value(LogicalType::kChar63);
    value.value_.char63 = input;
    value.is_null_ = false;
    return value;
}


Value
Value::MakeDate(DateT input) {
    Value value(LogicalType::kDate);
    value.value_.date = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeTime(TimeT input) {
    Value value(LogicalType::kTime);
    value.value_.time = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeDateTime(DateTimeT input) {
    Value value(LogicalType::kDateTime);
    value.value_.datetime = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeTimestamp(TimestampT input) {
    Value value(LogicalType::kTimestamp);
    value.value_.timestamp = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeTimestampTz(TimestampTZT input) {
    Value value(LogicalType::kTimestampTZ);
    value.value_.timestamp_tz = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeInterval(IntervalT input) {
    Value value(LogicalType::kInterval);
    value.value_.interval = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeArray(ArrayT input) {
    Value value(LogicalType::kArray);
    value.value_.array = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakePoint(PointT input) {
    Value value(LogicalType::kPoint);
    value.value_.point = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeLine(LineT input) {
    Value value(LogicalType::kLine);
    value.value_.line = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeLineSegment(LineSegT input) {
    Value value(LogicalType::kLineSeg);
    value.value_.line_segment = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeBox(BoxT input) {
    Value value(LogicalType::kBox);
    value.value_.box = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakePath(PathT input) {
    Value value(LogicalType::kPath);
    value.value_.path = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakePolygon(PolygonT input) {
    Value value(LogicalType::kPolygon);
    value.value_.polygon = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeCircle(CircleT input) {
    Value value(LogicalType::kCircle);
    value.value_.circle = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeBitmap(BitmapT input) {
    Value value(LogicalType::kBitmap);
    value.value_.bitmap = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeUuid(UuidT input) {
    Value value(LogicalType::kUuid);
    value.value_.uuid = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeBlob(BlobT input) {
    Value value(LogicalType::kBlob);
    value.value_.blob = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeEmbedding(EmbeddingT input) {
    Value value(LogicalType::kEmbedding);
    value.value_.embedding = input;
    value.is_null_ = false;
    return value;
}

Value
Value::MakeMixedData(MixedT input) {
    Value value(LogicalType::kMixed);
    value.value_.mixed_value = input;
    value.is_null_ = false;
    return value;
}

// Value getter
template <> BooleanT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kBoolean, "Not matched type: " + type_.ToString());
    return value_.boolean;
}
template <> TinyIntT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kTinyInt, "Not matched type: " + type_.ToString());
    return value_.tiny_int;
}

template <> SmallIntT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kSmallInt, "Not matched type: " + type_.ToString());
    return value_.small_int;
}

template <> IntegerT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kInteger, "Not matched type: " + type_.ToString());
    return value_.integer;
}

template <> BigIntT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kBigInt, "Not matched type: " + type_.ToString());
    return value_.big_int;
}

template <> HugeIntT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kHugeInt, "Not matched type: " + type_.ToString());
    return value_.huge_int;
}

template <> FloatT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kFloat, "Not matched type: " + type_.ToString());
    return value_.float32;
}

template <> DoubleT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDouble, "Not matched type: " + type_.ToString());
    return value_.float64;
}

template <> Decimal16T
Value::GetValue() const {
TypeAssert(type_.type() == LogicalType::kDecimal16, "Not matched type: " + type_.ToString());
return value_.decimal16;
}

template <> Decimal32T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDecimal32, "Not matched type: " + type_.ToString());
    return value_.decimal32;
}

template <> Decimal64T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDecimal64, "Not matched type: " + type_.ToString());
    return value_.decimal64;
}

template <> Decimal128T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDecimal128, "Not matched type: " + type_.ToString());
    return value_.decimal128;
}

template <> VarcharT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kVarchar, "Not matched type: " + type_.ToString());
    return value_.varchar;
}

template <> Char1T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar1, "Not matched type: " + type_.ToString());
    return value_.char1;
}

template <> Char2T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar2, "Not matched type: " + type_.ToString());
    return value_.char2;
}

template <> Char4T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar4, "Not matched type: " + type_.ToString());
    return value_.char4;
}

template <> Char8T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar8, "Not matched type: " + type_.ToString());
    return value_.char8;
}

template <> Char15T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar15, "Not matched type: " + type_.ToString());
    return value_.char15;
}

template <> Char31T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar31, "Not matched type: " + type_.ToString());
    return value_.char31;
}

template <> Char63T
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kChar63, "Not matched type: " + type_.ToString());
    return value_.char63;
}

template <> DateT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDate, "Not matched type: " + type_.ToString());
    return value_.date;
}

template <> TimeT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kTime, "Not matched type: " + type_.ToString());
    return value_.time;
}

template <> DateTimeT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kDateTime, "Not matched type: " + type_.ToString());
    return value_.datetime;
}

template <> TimestampT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kTimestamp, "Not matched type: " + type_.ToString());
    return value_.timestamp;
}

template <> TimestampTZT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kTimestampTZ, "Not matched type: " + type_.ToString());
    return value_.timestamp_tz;
}

template <> IntervalT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kInterval, "Not matched type: " + type_.ToString());
    return value_.interval;
}

template <> ArrayT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kArray, "Not matched type: " + type_.ToString());
    return value_.array;
}

template <> PointT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kPoint, "Not matched type: " + type_.ToString());
    return value_.point;
}

template <> LineT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kLine, "Not matched type: " + type_.ToString());
    return value_.line;
}

template <> LineSegT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kLineSeg, "Not matched type: " + type_.ToString());
    return value_.line_segment;
}

template <> BoxT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kBox, "Not matched type: " + type_.ToString());
    return value_.box;
}

template <> PathT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kPath, "Not matched type: " + type_.ToString());
    return value_.path;
}

template <> PolygonT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kPolygon, "Not matched type: " + type_.ToString());
    return value_.polygon;
}

template <> CircleT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kCircle, "Not matched type: " + type_.ToString());
    return value_.circle;
}

template <> BitmapT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kBitmap, "Not matched type: " + type_.ToString());
    return value_.bitmap;
}

template <> UuidT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kUuid, "Not matched type: " + type_.ToString());
    return value_.uuid;
}

template <> BlobT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kBlob, "Not matched type: " + type_.ToString());
    return value_.blob;
}

template <> EmbeddingT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kEmbedding, "Not matched type: " + type_.ToString());
    return value_.embedding;
}

template <> MixedT
Value::GetValue() const {
    TypeAssert(type_.type() == LogicalType::kMixed, "Not matched type: " + type_.ToString());
    return value_.mixed_value;
}

template <> Value
Value::MakeValue(BooleanT input) {
    return MakeBool(input);
}

template <> Value
Value::MakeValue(TinyIntT input) {
    return MakeBool(input);
}

template <> Value
Value::MakeValue(SmallIntT input) {
    return MakeBool(input);
}

template <> Value
Value::MakeValue(IntegerT input) {
    return MakeBool(input);
}

template <> Value
Value::MakeValue(BigIntT input) {
    return MakeBool(input);
}

template <> Value
Value::MakeValue(HugeIntT input) {
    return MakeHugeInt(input);
}

template <> Value
Value::MakeValue(FloatT input) {
    return MakeFloat(input);
}

template <> Value
Value::MakeValue(DoubleT input) {
    return MakeDouble(input);
}

template <> Value
Value::MakeValue(Decimal16T input) {
    return MakeDecimal16(input);
}

template <> Value
Value::MakeValue(Decimal32T input) {
    return MakeDecimal32(input);
}

template <> Value
Value::MakeValue(Decimal64T input) {
    return MakeDecimal64(input);
}

template <> Value
Value::MakeValue(Decimal128T input) {
    return MakeDecimal128(input);
}

template <> Value
Value::MakeValue(VarcharT& input) {
    return MakeVarchar(input);
}

template <> Value
Value::MakeValue(const String& input_str_ref) {
    return MakeVarchar(input_str_ref);
}

template <> Value
Value::MakeValue(const char_t* input_ptr) {
    return MakeVarchar(input_ptr);
}

template <> Value Value::MakeValue(Char1T input) {
    return MakeChar1(input);
}

template <> Value Value::MakeValue(Char2T input) {
    return MakeChar2(input);
}

template <> Value Value::MakeValue(Char4T input) {
    return MakeChar4(input);
}

template <> Value Value::MakeValue(Char8T input) {
    return MakeChar8(input);
}

template <> Value Value::MakeValue(Char15T input) {
    return MakeChar15(input);
}

template <> Value Value::MakeValue(Char31T input) {
    return MakeChar31(input);
}

template <> Value Value::MakeValue(Char63T input) {
    return MakeChar63(input);
}

template <> Value
Value::MakeValue(DateT input) {
    return MakeDate(input);
}

template <> Value
Value::MakeValue(TimeT input) {
    return MakeTime(input);
}

template <> Value
Value::MakeValue(DateTimeT input) {
    return MakeDateTime(input);
}

template <> Value
Value::MakeValue(TimestampT input) {
    return MakeTimestamp(input);
}

template <> Value
Value::MakeValue(TimestampTZT input) {
    return MakeTimestampTz(input);
}

template <> Value
Value::MakeValue(IntervalT input) {
    return MakeInterval(input);
}

template <> Value
Value::MakeValue(ArrayT input) {
    return MakeArray(input);
}

template <> Value
Value::MakeValue(PointT input) {
    return MakePoint(input);
}

template <> Value
Value::MakeValue(LineT input) {
    return MakeLine(input);
}

template <> Value
Value::MakeValue(LineSegT input) {
    return MakeLineSegment(input);
}

template <> Value
Value::MakeValue(BoxT input) {
    return MakeBox(input);
}

template <> Value
Value::MakeValue(PathT input) {
    return MakePath(input);
}

template <> Value
Value::MakeValue(PolygonT input) {
    return MakePolygon(input);
}

template <> Value
Value::MakeValue(CircleT input) {
    return MakeCircle(input);
}

template <> Value
Value::MakeValue(BitmapT input) {
    return MakeBitmap(input);
}

template <> Value
Value::MakeValue(UuidT input) {
    return MakeUuid(input);
}

template <> Value
Value::MakeValue(BlobT input) {
    return MakeBlob(input);
}

template <> Value
Value::MakeValue(EmbeddingT input) {
    return MakeEmbedding(input);
}

template <> Value
Value::MakeValue(MixedT input) {
    return MakeMixedData(input);
}

Value::~Value() {
    switch(type_.type()) {
        case kVarchar: {
            value_.varchar.~VarcharType();
            break;
        }
        default: {

        }
    }
}

Value::Value(const Value& other) : type_(other.type_) {
    switch(type_.type()) {

        case kBoolean:
            value_.boolean = other.value_.boolean;
            break;
        case kTinyInt:
            break;
        case kSmallInt:
            break;
        case kInteger:
            break;
        case kBigInt:
            break;
        case kHugeInt:
            break;
        case kFloat:
            break;
        case kDouble:
            break;
        case kDecimal16:
            break;
        case kDecimal32:
            break;
        case kDecimal64:
            break;
        case kDecimal128:
            break;
        case kVarchar:
            break;
        case kChar1:
            break;
        case kChar2:
            break;
        case kChar4:
            break;
        case kChar8:
            break;
        case kChar15:
            break;
        case kChar31:
            break;
        case kChar63:
            break;
        case kDate:
            break;
        case kTime:
            break;
        case kDateTime:
            break;
        case kTimestamp:
            break;
        case kTimestampTZ:
            break;
        case kInterval:
            break;
        case kArray:
            break;
        case kTuple:
            break;
        case kPoint:
            break;
        case kLine:
            break;
        case kLineSeg:
            break;
        case kBox:
            break;
        case kPath:
            break;
        case kPolygon:
            break;
        case kCircle:
            break;
        case kBitmap:
            break;
        case kUuid:
            break;
        case kBlob:
            break;
        case kEmbedding:
            break;
        case kMixed:
            break;
        case kNull:
            break;
        case kMissing:
            break;
        case kInvalid:
            break;
    }
}

//Value::Value(Value&& other) noexcept :
//    type_(std::move(other.type_)),
//    value_(std::move(other.value_)),
//    is_null_(std::move(other.is_null_)) {
//}
//
//Value&
//Value::operator=(const Value& other) {
//    if(this == &other) return *this;
//    this->type_ = other.type_;
//    this->is_null_ = other.is_null_;
//    switch(this->type_.type()) {
//    }
//
//    return *this;
//}

Value&
Value::operator=(Value&& other)  noexcept {
    this->type_ = std::move(other.type_);
    this->is_null_ = std::move(other.is_null_);
//    this->value_ = std::move(other.value_);
    switch(this->type_.type()) {
        case kBoolean: {
            this->value_.boolean = std::move(other.value_.boolean);
            break;
        }
        case kTinyInt: {
            this->value_.tiny_int = std::move(other.value_.tiny_int);
            break;
        }
        case kSmallInt: {
            this->value_.small_int = std::move(other.value_.small_int);
            break;
        }
        case kInteger: {
            this->value_.integer = std::move(other.value_.integer);
            break;
        }
        case kBigInt: {
            this->value_.big_int = std::move(other.value_.big_int);
            break;
        }
        case kHugeInt: {
            this->value_.huge_int = std::move(other.value_.huge_int);
            break;
        }
        case kFloat: {
            this->value_.float32 = std::move(other.value_.float32);
            break;
        }
        case kDouble: {
            this->value_.float64 = std::move(other.value_.float64);
            break;
        }
        case kDecimal16: {
            this->value_.decimal16 = std::move(other.value_.decimal16);
            break;
        }
        case kDecimal32: {
            this->value_.decimal32 = std::move(other.value_.decimal32);
            break;
        }
        case kDecimal64: {
            this->value_.decimal64 = std::move(other.value_.decimal64);
            break;
        }
        case kDecimal128: {
            this->value_.decimal128 = std::move(other.value_.decimal128);
            break;
        }
        case kVarchar: {
            this->value_.varchar = std::move(other.value_.varchar);
            break;
        }
        case kChar1: {
            this->value_.char1.value = std::move(other.value_.char1.value);
            break;
        }
        case kChar2: {
            this->value_.char2 = std::move(other.value_.char2);
            break;
        }
        case kChar4: {
            this->value_.char4 = std::move(other.value_.char4);
            break;
        }
        case kChar8: {
            this->value_.char8 = std::move(other.value_.char8);
            break;
        }
        case kChar15: {
            this->value_.char15 = std::move(other.value_.char15);
            break;
        }
        case kChar31: {
            this->value_.char31 = std::move(other.value_.char31);
            break;
        }
        case kChar63: {
            this->value_.char63 = std::move(other.value_.char63);
            break;
        }
        case kDate: {
            this->value_.date = std::move(other.value_.date);
            break;
        }
        case kTime: {
            this->value_.time = std::move(other.value_.time);
            break;
        }
        case kDateTime: {
            this->value_.datetime = std::move(other.value_.datetime);
            break;
        }
        case kTimestamp: {
            this->value_.timestamp = std::move(other.value_.timestamp);
            break;
        }
        case kTimestampTZ: {
            this->value_.timestamp_tz = std::move(other.value_.timestamp_tz);
            break;
        }
        case kInterval: {
            this->value_.interval = std::move(other.value_.interval);
            break;
        }
        case kArray: {
            LOG_ERROR("Not implemented");
//            this->value_.array = std::move(other.value_.array);
            break;
        }
        case kTuple: {
            LOG_ERROR("Not implemented");
            break;
        }
        case kPoint: {
            this->value_.point = std::move(other.value_.point);
            break;
        }
        case kLine: {
            this->value_.line = std::move(other.value_.line);
            break;
        }
        case kLineSeg: {
            this->value_.line_segment = std::move(other.value_.line_segment);
            break;
        }
        case kBox: {
            this->value_.box = std::move(other.value_.box);
            break;
        }
        case kPath: {
            this->value_.path = std::move(other.value_.path);
            break;
        }
        case kPolygon: {
            // TODO
            break;
        }
        case kCircle: {
            this->value_.circle = std::move(other.value_.circle);
            break;
        }
        case kBitmap:
            break;
        case kUuid:
            break;
        case kBlob:
            break;
        case kEmbedding:
            break;
        case kMixed:
            break;
        case kNull:
            break;
        case kMissing:
            break;
        case kInvalid:
            break;
    }

    return *this;
}

//Value::Value(const Value& other) : type_(other.type_), value_(other.value_)  {
//    LOG_DEBUG("Value copy constructor");
//    switch(type_.type()) {
//        case kVarchar: {
//            value_.varchar.DeepCopy(other.value_.varchar);
//        }
//        default: {
//
//        }
//    }
//}
//
//Value::Value(const Value&& other) : type_(other.type_), value_(other.value_)  {
//    LOG_DEBUG("Value move constructor");
//    switch(type_.type()) {
//        case kVarchar: {
//            value_.varchar.DeepCopy(other.value_.varchar);
//        }
//        default: {
//
//        }
//    }
//}

String
Value::ToString() const {
    switch(type_.type()) {
        case kBoolean:
            return value_.boolean ? "true": "false";
        case kTinyInt:
            return std::to_string(value_.tiny_int);
        case kSmallInt:
            return std::to_string(value_.small_int);
        case kInteger:
            return std::to_string(value_.integer);
        case kBigInt:
            return std::to_string(value_.big_int);
        case kHugeInt:
            return value_.huge_int.ToString();
        case kFloat:
            return std::to_string(value_.float32);
        case kDouble:
            return std::to_string(value_.float64);
        case kDecimal16:
            break;
        case kDecimal32:
            break;
        case kDecimal64:
            break;
        case kDecimal128:
            break;
        case kVarchar:
            break;
        case kChar1:
            break;
        case kChar2:
            break;
        case kChar4:
            break;
        case kChar8:
            break;
        case kChar15:
            break;
        case kChar31:
            break;
        case kChar63:
            break;
        case kDate:
            break;
        case kTime:
            break;
        case kDateTime:
            break;
        case kTimestamp:
            break;
        case kTimestampTZ:
            break;
        case kInterval:
            break;
        case kArray:
            break;
        case kTuple:
            break;
        case kPoint:
            break;
        case kLine:
            break;
        case kLineSeg:
            break;
        case kBox:
            break;
        case kPath:
            break;
        case kPolygon:
            break;
        case kCircle:
            break;
        case kBitmap:
            break;
        case kUuid:
            break;
        case kBlob:
            break;
        case kEmbedding:
            break;
        case kMixed:
            break;
        case kNull:
            break;
        case kMissing:
            break;
        case kInvalid:
            break;
    }
    TypeError("Unexpected error.");
}

}