//
// Created by JinHai on 2022/9/14.
//

#include "function/aggregate/avg.h"
#include "function/aggregate_function_set.h"
#include "storage/meta/catalog.h"

namespace infinity {

template <typename ValueType, typename ResultType>
struct AvgState {
public:
    inline void Initialize() { NotImplementError("Initialize average state.") }

    inline void Update(const ValueType *__restrict input, SizeT idx) { NotImplementError("Update average state.") }

    inline void ConstantUpdate(const ValueType *__restrict input, SizeT idx, SizeT count) { NotImplementError("Constant update average state.") }

    inline ptr_t Finalize() { NotImplementError("Finalize average state.") }

    inline static SizeT Size(const DataType &data_type) { NotImplementError("Average state type size: " + data_type.ToString()) }
};

template <>
struct AvgState<TinyIntT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const TinyIntT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const TinyIntT *__restrict input, SizeT idx, SizeT count) {
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    [[nodiscard]] inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

template <>
struct AvgState<SmallIntT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const SmallIntT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const SmallIntT *__restrict input, SizeT idx, SizeT count) {
        // TODO: Need to check overflow.
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

template <>
struct AvgState<IntegerT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const IntegerT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const IntegerT *__restrict input, SizeT idx, SizeT count) {
        // TODO: Need to check overflow.
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

template <>
struct AvgState<BigIntT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const BigIntT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const BigIntT *__restrict input, SizeT idx, SizeT count) {
        // TODO: Need to check overflow.
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

template <>
struct AvgState<FloatT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const FloatT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const FloatT *__restrict input, SizeT idx, SizeT count) {
        // TODO: Need to check overflow.
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

template <>
struct AvgState<DoubleT, DoubleT> {
public:
    double value_{};
    i64 count_{};
    double result_{};

    inline void Initialize() {
        this->value_ = 0;
        this->count_ = 0;
    }

    inline void Update(const DoubleT *__restrict input, SizeT idx) {
        if (count_ == std::numeric_limits<i64>::max()) {
            ExecutorError(fmt::format("Data count exceeds: {}", count_));
        }
        this->count_++;
        value_ += input[idx];
    }

    inline void ConstantUpdate(const DoubleT *__restrict input, SizeT idx, SizeT count) {
        // TODO: Need to check overflow.
        this->count_ += count;
        value_ += (input[idx] * count);
    }

    inline ptr_t Finalize() {
        result_ = value_ / count_;
        return (ptr_t)&result_;
    }

    inline static SizeT Size(const DataType &data_type) { return sizeof(value_) + sizeof(count_) + sizeof(result_); }
};

void RegisterAvgFunction(const UniquePtr<NewCatalog> &catalog_ptr) {
    String func_name = "AVG";

    SharedPtr<AggregateFunctionSet> function_set_ptr = MakeShared<AggregateFunctionSet>(func_name);

    {
        AggregateFunction avg_function = UnaryAggregate<AvgState<TinyIntT, DoubleT>, TinyIntT, DoubleT>(func_name,
                                                                                                        DataType(LogicalType::kTinyInt),
                                                                                                        DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    {
        AggregateFunction avg_function = UnaryAggregate<AvgState<SmallIntT, DoubleT>, SmallIntT, DoubleT>(func_name,
                                                                                                          DataType(LogicalType::kSmallInt),
                                                                                                          DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    {
        AggregateFunction avg_function = UnaryAggregate<AvgState<IntegerT, DoubleT>, IntegerT, DoubleT>(func_name,
                                                                                                        DataType(LogicalType::kInteger),
                                                                                                        DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    {
        AggregateFunction avg_function =
            UnaryAggregate<AvgState<BigIntT, DoubleT>, BigIntT, DoubleT>(func_name, DataType(LogicalType::kBigInt), DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    {
        // HugeIntT average function isn't implemented
#if 0
        AggregateFunction avg_function
                = UnaryAggregate<AvgState<HugeIntT, DoubleT>, HugeIntT, DoubleT>(func_name,
                                                                               DataType(LogicalType::kHugeInt),
                                                                               DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
#endif
    }

    {
        // Decimal average function isn't implemented
#if 0
        AggregateFunction avg_function
                = UnaryAggregate<AvgState<DecimalT, DoubleT>, DecimalT, DoubleT>(func_name,
                                                                                     DataType(LogicalType::kDecimal),
                                                                                     DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
#endif
    }

    {
        AggregateFunction avg_function =
            UnaryAggregate<AvgState<FloatT, DoubleT>, FloatT, DoubleT>(func_name, DataType(LogicalType::kFloat), DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    {
        AggregateFunction avg_function =
            UnaryAggregate<AvgState<DoubleT, DoubleT>, DoubleT, DoubleT>(func_name, DataType(LogicalType::kDouble), DataType(LogicalType::kDouble));
        function_set_ptr->AddFunction(avg_function);
    }

    NewCatalog::AddFunctionSet(catalog_ptr.get(), function_set_ptr);
}

} // namespace infinity
