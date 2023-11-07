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

#include "type/info/array_info.h"
#include "sql_parser.h"
#include "type/number/float16.h"

#include "spdlog/details/registry.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/logger.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "cxxopts.hpp"
#include "json.hpp"

#include "toml.hpp"

#include "magic_enum.hpp"

#include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"

#include "faiss/Index.h"
#include "faiss/utils/distances.h"


#include "ctpl_stl.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <forward_list>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <experimental/source_location>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind/bind.hpp>

export module parser;

namespace infinity {

export using SQLParser = SQLParser;
export using ParserResult = ParserResult;

export using LogicalType = LogicalType;
export using ColumnDef = ColumnDef;
export using DataType = DataType;
export using RowID = RowID;

// Bool
export using BooleanT = BooleanT;

// Numeric
export using TinyIntT = TinyIntT;
export using SmallIntT = SmallIntT;
export using IntegerT = IntegerT;
export using BigIntT = BigIntT;
export using HugeIntT = HugeIntT;

export using FloatT = FloatT;
export using DoubleT = DoubleT;

export using DecimalT = DecimalT;

// std::string
export using VarcharT = VarcharType;

// Date and Time
export using DateT = DateType;
export using TimeT = TimeType;
export using DateTimeT = DateTimeType;
export using TimestampT = TimestampType;
export using IntervalT = IntervalType;

// Nest types
export using ArrayT = std::vector<Value>;
export using TupleT = std::vector<Value>;
// using ArrayT = ArrayType;
// using TupleT = TupleType;

// Geography
export using PointT = PointType;
export using LineT = LineType;
export using LineSegT = LineSegmentType;
export using BoxT = BoxType;
export using PathT = PathType;
export using PolygonT = PolygonType;
export using CircleT = CircleType;

// Other
export using BitmapT = BitmapType;
export using UuidT = UuidType;
export using BlobT = BlobType;
export using EmbeddingT = EmbeddingType;
export using RowT = RowID;

// Heterogeneous
export using MixedT = MixedType;

// TimeUnit
export using TimeUnit = TimeUnit;

export using float16_t = float16_t;

export using IntegerMixedType = IntegerMixedType;
export using FloatMixedType = FloatMixedType;
export using ArrayMixedType = ArrayMixedType;
export using BaseMixedType = BaseMixedType;
export using LongStrMixedType = LongStrMixedType;
export using MissingMixedType = MissingMixedType;
export using ShortStrMixedType = ShortStrMixedType;
export using MixedType = MixedType;
export using MixedValueType = MixedValueType;
export using TupleMixedType = TupleMixedType;

export using TypeInfo = TypeInfo;
export using EmbeddingDataType = EmbeddingDataType;
export using EmbeddingInfo = EmbeddingInfo;
export using DecimalInfo = DecimalInfo;
export using BitmapInfo = BitmapInfo;
export using VarcharInfo = VarcharInfo;
export using ArrayInfo = ArrayInfo;

export using TypeInfoType = TypeInfoType;

export template <typename T>
int32_t GetSizeInBytes(const T &value);

export template <>
int32_t GetSizeInBytes(const std::string &value);

export constexpr int64_t MAX_VARCHAR_SIZE = MAX_VARCHAR_SIZE_INTERNAL;
export constexpr int64_t EMBEDDING_LIMIT = EMBEDDING_LIMIT_INTERNAL;
export constexpr int64_t MAX_BITMAP_SIZE = MAX_BITMAP_SIZE_INTERNAL;

// Parser Exception
export using ParserException = ParserException;

export using StatementType = StatementType;
export using DDLType = DDLType;
export using ConflictType = ConflictType;
export using ConstraintType = ConstraintType;
export using KnnDistanceType = KnnDistanceType;
export using TableRefType = TableRefType;
export using ExplainType = ExplainType;
export using FlushType = FlushType;
export using EmbeddingDataType = EmbeddingDataType;

export using ExtraDDLInfo = ExtraDDLInfo;
export using CreateTableInfo = CreateTableInfo;
export using CreateIndexInfo = CreateIndexInfo;
export using CreateViewInfo = CreateViewInfo;
export using CreateCollectionInfo = CreateCollectionInfo;
export using CreateSchemaInfo = CreateSchemaInfo;

export using DropIndexInfo = DropIndexInfo;
export using DropTableInfo = DropTableInfo;
export using DropCollectionInfo = DropCollectionInfo;
export using DropSchemaInfo = DropSchemaInfo;
export using DropViewInfo = DropViewInfo;
export using CommandInfo = CommandInfo;
export using UseCmd = UseCmd;
export using CheckTable = CheckTable;

export using InitParameter = InitParameter;

export using BaseTableReference = BaseTableReference;
export using TableReference = TableReference;
export using JoinReference = JoinReference;
export using CrossProductReference = CrossProductReference;
export using SubqueryReference = SubqueryReference;

export using ShowStmtType = ShowStmtType;
export using CopyFileType = CopyFileType;
export using SetOperatorType = SetOperatorType;

export using BaseStatement = BaseStatement;
export using CreateStatement = CreateStatement;
export using SelectStatement = SelectStatement;
export using UpdateStatement = UpdateStatement;
export using DeleteStatement = DeleteStatement;
export using InsertStatement = InsertStatement;
export using DropStatement = DropStatement;
export using ShowStatement = ShowStatement;
export using CopyStatement = CopyStatement;
export using PrepareStatement = PrepareStatement;
export using ExecuteStatement = ExecuteStatement;
export using FlushStatement = FlushStatement;
export using AlterStatement = AlterStatement;
export using ExplainStatement = ExplainStatement;
export using CommandStatement = CommandStatement;

export using ParsedExprType = ParsedExprType;
export using OrderType = OrderType;
export using LiteralType = LiteralType;
export using SubqueryType = SubqueryType;
export using JoinType = JoinType;
export using KnnDistanceType = KnnDistanceType;
export using CommandType = CommandType;

export using ParsedExpr = ParsedExpr;
export using ColumnExpr = ColumnExpr;
export using ConstantExpr = ConstantExpr;
export using FunctionExpr = FunctionExpr;
export using KnnExpr = KnnExpr;
export using BetweenExpr = BetweenExpr;
export using SubqueryExpr = SubqueryExpr;
export using CaseExpr = CaseExpr;
export using WhenThen = WhenThen;
export using CastExpr = CastExpr;
export using WithExpr = WithExpr;
export using UpdateExpr = UpdateExpr;
export using InExpr = InExpr;
export using OrderByExpr = OrderByExpr;

export using ColumnDef = ColumnDef;
export using TableConstraint = TableConstraint;

export inline std::string ConflictType2Str(ConflictType type) {
    return ConflictTypeToStr(type);
}

export inline std::string OrderBy2Str(OrderType type) {
    return ToString(type);
}

export inline std::string JoinType2Str(JoinType type) {
    return ToString(type);
}

export inline std::string ConstrainType2String(ConstraintType type) {
    return ConstrainTypeToString(type);
}

//export template <typename T>
//T ReadBuf(char *const buf);
//
//export template <typename T>
//T ReadBufAdv(char *&buf);
//
//export template <>
//std::string ReadBuf<std::string>(char *const buf);
//
//export template <>
//std::string ReadBufAdv<std::string>(char *&buf);
//
//export template <typename T>
//void WriteBuf(char *const buf, const T &value);
//
//export template <typename T>
//void WriteBufAdv(char *&buf, const T &value);
//
//export template <>
//void WriteBuf<std::string>(char *const buf, const std::string &value);
//
//template <>
//void WriteBufAdv<std::string>(char *&buf, const std::string &value);

// spdlog
export enum class LogLevel { kTrace, kInfo, kWarning, kError, kFatal };

export std::string LogLevel2Str(LogLevel log_level) {
    switch (log_level) {

        case LogLevel::kTrace: {
            return "Trace";
        }
        case LogLevel::kInfo: {
            return "Info";
        }
        case LogLevel::kWarning: {
            return "Warning";
        }
        case LogLevel::kError: {
            return "Error";
        }
        case LogLevel::kFatal: {
            return "Fatal";
        }
    }
}

export template <typename... T>
FMT_INLINE void Printf(fmt::format_string<T...> fmt, T &&...args) {
    const auto &vargs = fmt::make_format_args(args...);
    return fmt::detail::is_utf8() ? vprint(fmt, vargs) : fmt::detail::vprint_mojibake(stdout, fmt, vargs);
}

export template <typename... T>
FMT_NODISCARD FMT_INLINE auto Format(fmt::format_string<T...> fmt, T &&...args) -> std::string {
    return vformat(fmt, fmt::make_format_args(args...));
}

export void RegisterLogger(const std::shared_ptr<spdlog::logger> &logger) { return spdlog::details::registry::instance().register_logger(logger); }

export void SetLogLevel(LogLevel log_level) {
    switch (log_level) {
        case LogLevel::kTrace:
            return spdlog::details::registry::instance().set_level(spdlog::level::level_enum::trace);
        case LogLevel::kInfo:
            return spdlog::details::registry::instance().set_level(spdlog::level::level_enum::info);
        case LogLevel::kWarning:
            return spdlog::details::registry::instance().set_level(spdlog::level::level_enum::warn);
        case LogLevel::kError:
            return spdlog::details::registry::instance().set_level(spdlog::level::level_enum::err);
        case LogLevel::kFatal:
            return spdlog::details::registry::instance().set_level(spdlog::level::level_enum::critical);
    }
}

export void ShutdownLogger() { spdlog::shutdown(); }

export using spd_sink_ptr = spdlog::sink_ptr;
export using spd_stdout_color_sink = spdlog::sinks::stdout_color_sink_mt;
export using spd_rotating_file_sink = spdlog::sinks::rotating_file_sink_mt;

export using spd_logger = spdlog::logger;
export using spd_log_level = spdlog::level::level_enum;

export template <typename T>
inline void spd_log(const T &msg, spd_log_level log_level) {
    switch (log_level) {
        case spdlog::level::trace: {
            return spdlog::default_logger_raw()->trace(msg);
        }
        case spdlog::level::info: {
            return spdlog::default_logger_raw()->info(msg);
        }
        case spdlog::level::warn: {
            return spdlog::default_logger_raw()->warn(msg);
        }
        case spdlog::level::err: {
            return spdlog::default_logger_raw()->error(msg);
        }
        case spdlog::level::critical: {
            return spdlog::default_logger_raw()->critical(msg);
        }
        default:
            assert(false);
    }
}

// cxxopts
export using CxxOptions = cxxopts::Options;

export template <typename T>
std::shared_ptr<cxxopts::Value> cxx_value() {
    return std::make_shared<cxxopts::values::standard_value<T>>();
}

export using ParseResult = cxxopts::ParseResult;

// Toml parser
export using TomlTable = toml::table;
//
export TomlTable TomlParseFile(const std::string &file_path) { return toml::parse_file(file_path); }

// Returns integer value from enum value.
export template <typename E>
constexpr auto EnumInteger(E value) noexcept -> magic_enum::detail::enable_if_t<E, magic_enum::underlying_type_t<E>> {
    return static_cast<magic_enum::underlying_type_t<E>>(value);
}

// Json Parser
export using Json = nlohmann::json;

// ConcurrentQueue

export template<typename T>
using ConcurrentQueue = moodycamel::ConcurrentQueue<T>;

export template<typename T>
using BlockingConcurrentQueue = moodycamel::BlockingConcurrentQueue<T>;

// Faiss
export using FaissIndex = faiss::Index;

export inline float fvec_inner_product(const float* x, const float* y, size_t d) {
    return faiss::fvec_inner_product(x, y, d);
}

export inline float fvec_L2sqr(const float* x, const float* y, size_t d) {
    return faiss::fvec_L2sqr(x, y, d);
}

export constexpr int faiss_distance_compute_blas_threshold = 20;
export constexpr int faiss_distance_compute_blas_query_bs = 4096;
export constexpr int faiss_distance_compute_blas_database_bs = 1024;
export constexpr int faiss_distance_compute_min_k_reservoir = 100;

}

export namespace std {

using std::source_location;
//using std::stringstream;

}

namespace infinity {

export {

// Containers

template <typename T1, typename T2>
using Pair = std::pair<T1, T2>;

template <typename T, std::size_t N>
using Array = std::array<T, N>;

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Deque = std::deque<T>;

template <typename T>
using List = std::list<T>;

template <typename T>
using Queue = std::queue<T>;

template <typename S, typename T>
using Map = std::map<S, T>;

template <typename T>
using Set = std::set<T>;

template <typename S, typename T>
using HashMap = std::unordered_map<S, T>;

template <typename S>
using HashSet = std::unordered_set<S>;

template <typename T>
using MaxHeap = std::priority_queue<T>;

template <typename T, typename C>
using Heap = std::priority_queue<T, std::vector<T>, C>;

template <typename T>
using Optional = std::optional<T>;

using StdOfStream = std::ofstream;

// String

using String = std::basic_string<char>;

using StringView = std::string_view;

inline bool IsEqual(const String &s1, const String &s2) { return s1 == s2; }

inline bool IsEqual(const String &s1, const char *s2) { return s1 == s2; }

inline String TrimPath(const String &path) {
    const auto pos = path.find("/src/");
    if (pos == String::npos)
        return path;
    return path.substr(pos + 1);
}

void ToUpper(String & str) { std::transform(str.begin(), str.end(), str.begin(), ::toupper); }
int ToUpper(int c) { return ::toupper(c); }

void ToLower(String & str) { std::transform(str.begin(), str.end(), str.begin(), ::tolower); }
int ToLower(int c) { return ::tolower(c); }

inline void StringToLower(String & str) {
    std::transform(str.begin(), str.end(), str.begin(), [](const auto c) { return std::tolower(c); });
}

const char *FromChars(const char *first, const char *last, unsigned long &value) {
    auto res = std::from_chars(first, last, value);
    if (res.ec == std::errc()) {
        return res.ptr;
    } else {
        return nullptr;
    }
}

template <typename T>
inline const T &Min(const T &a, const T &b) {
    return std::min(a, b);
}

template <typename T>
inline const T &Max(const T &a, const T &b) {
    return std::max(a, b);
}

// ToStr()

template <typename T>
inline String ToStr(T value) {
    return std::to_string(value);
}

// stoi
inline int StrToInt(const std::string &str, size_t *idx = 0, int base = 10) { return std::stoi(str, idx, base); }

// StrToL
inline long StrToL(const char *__restrict nptr, char **__restrict endptr, int base) { return std::strtol(nptr, endptr, base); }

// StrToF
inline float StrToF(const char *__restrict nptr, char **__restrict endptr) { return std::strtof(nptr, endptr); }

// StrToD
inline double StrToD(const char *__restrict nptr, char **__restrict endptr) { return std::strtod(nptr, endptr); }

// Primitives

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using idx_t = u64;

using f32 = float;
using f64 = double;

using offset_t = int64_t;

using ptr_t = char *;
using const_ptr_t = const char *;
using char_t = char;
using SizeT = size_t;
using StreamSize = std::streamsize;

using TxnTimeStamp = u64;

// Concurrency

using RWMutex = std::shared_mutex;
using ThreadPool = ctpl::thread_pool;

using Thread = std::thread;

using atomic_u32 = std::atomic_uint32_t;
using au64 = std::atomic_uint64_t;
using ai64 = std::atomic_int64_t;
using aptr = std::atomic_uintptr_t;
using atomic_bool = std::atomic_bool;

constexpr u64 u64_min = std::numeric_limits<u64>::min();
constexpr i64 i64_min = std::numeric_limits<i64>::min();
constexpr u32 u32_min = std::numeric_limits<u32>::min();
constexpr i32 i32_min = std::numeric_limits<i32>::min();
constexpr i16 i16_min = std::numeric_limits<i16>::min();
constexpr u16 u16_min = std::numeric_limits<u16>::min();
constexpr i8 i8_min = std::numeric_limits<i8>::min();
constexpr u8 u8_min = std::numeric_limits<u8>::min();

constexpr u64 u64_max = std::numeric_limits<u64>::max();
constexpr i64 i64_max = std::numeric_limits<i64>::max();
constexpr u32 u32_max = std::numeric_limits<u32>::max();
constexpr i32 i32_max = std::numeric_limits<i32>::max();
constexpr i16 i16_max = std::numeric_limits<i16>::max();
constexpr u16 u16_max = std::numeric_limits<u16>::max();
constexpr i8 i8_max = std::numeric_limits<i8>::max();
constexpr u8 u8_max = std::numeric_limits<u8>::max();

constexpr f32 f32_inf = std::numeric_limits<f32>::infinity();
constexpr f32 f32_min = std::numeric_limits<f32>::min();
constexpr f32 f32_max = std::numeric_limits<f32>::max();
constexpr f64 f64_inf = std::numeric_limits<f64>::infinity();
constexpr f64 f64_min = std::numeric_limits<f64>::min();
constexpr f64 f64_max = std::numeric_limits<f64>::max();

constexpr u64 u64_inf = std::numeric_limits<u64>::infinity();
constexpr i64 i64_inf = std::numeric_limits<i64>::infinity();
constexpr u32 u32_inf = std::numeric_limits<u32>::infinity();
constexpr i32 i32_inf = std::numeric_limits<i32>::infinity();
constexpr i16 i16_inf = std::numeric_limits<i16>::infinity();
constexpr u16 u16_inf = std::numeric_limits<u16>::infinity();
constexpr i8 i8_inf = std::numeric_limits<i8>::infinity();
constexpr u8 u8_inf = std::numeric_limits<u8>::infinity();

constexpr ptr_t ptr_inf = std::numeric_limits<ptr_t>::infinity();
constexpr u64 *u64_ptr_inf = std::numeric_limits<u64 *>::infinity();

template <typename T>
using Atomic = std::atomic<T>;

// Smart ptr

template <typename T>
using SharedPtr = std::shared_ptr<T>;

template <typename T, typename... Args>
inline SharedPtr<T> MakeShared(Args && ...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
using UniquePtr = std::unique_ptr<T>;

template <typename T, typename... Args>
inline UniquePtr<T> MakeUnique(Args && ...args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename U>
inline constexpr Pair<T, U> MakePair(T&& first, U&& second) {
    return std::make_pair<T, U>(std::forward<T>(first), std::forward<U>(second));
}

// DB Type

using ColumnID = u32;

// Exception

using StdException = std::exception;

// Move
template <typename T>
[[nodiscard]] constexpr typename std::remove_reference<T>::type &&Move(T && value) noexcept {
    return static_cast<typename std::remove_reference<T>::type &&>(value);
}

// Forward
template <typename T>
[[nodiscard]] constexpr T &&Forward(typename std::remove_reference<T>::type & value) noexcept {
    return static_cast<T &&>(value);
}

// Chrono
using Clock = std::chrono::high_resolution_clock;

template <typename T>
using TimePoint = std::chrono::time_point<T, std::chrono::nanoseconds>;

using NanoSeconds = std::chrono::nanoseconds;
using MicroSeconds = std::chrono::microseconds;
using MilliSeconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

inline NanoSeconds ElapsedFromStart(const TimePoint<Clock> &end, const TimePoint<Clock> &start) { return end - start; }

template <typename T>
T ChronoCast(const NanoSeconds &nano_seconds) {
    return std::chrono::duration_cast<T>(nano_seconds);
}

// Memcpy
void *Memcpy(void *__restrict dest, const void *__restrict src, size_t n) { return memcpy(dest, src, n); }
void *Memset(void *__restrict dest, int value, size_t n) { return memset(dest, value, n); }

// Memcmp
int Memcmp(const void *__restrict s1, const void *__restrict s2, size_t n) { return memcmp(s1, s2, n); }

// IsStandLayout
template <typename T>
concept IsStandLayout = std::is_standard_layout_v<T>;

template <typename T>
concept IsTrivial = std::is_trivial_v<T>;

// Mutex
template <typename T>
using SharedLock = std::shared_lock<T>;

template <typename T>
using UniqueLock = std::unique_lock<T>;

template <typename T>
using LockGuard = std::lock_guard<T>;

constexpr std::memory_order MemoryOrderRelax = std::memory_order::relaxed;
constexpr std::memory_order MemoryOrderConsume = std::memory_order::consume;
constexpr std::memory_order MemoryOrderRelease = std::memory_order::release;
constexpr std::memory_order MemoryOrderAcquire = std::memory_order::acquire;
constexpr std::memory_order MemoryOrderAcqrel = std::memory_order::acq_rel;
constexpr std::memory_order MemoryOrderSeqcst = std::memory_order::seq_cst;

using CondVar = std::condition_variable;

// Stringstream
using StringStream = std::basic_stringstream<char>;

// Dir
using Path = std::filesystem::path;
using DirEntry = std::filesystem::directory_entry;

inline Vector<String> GetFilesFromDir(const String &path) {
    Vector<String> result;
    for (auto &i : std::filesystem::directory_iterator(path)) {
        result.emplace_back(i.path().string());
    }
    return result;
}

// typeid
//    using TypeID = std::typeid();

// std::function
template <typename T>
using StdFunction = std::function<T>;

// SharedPtr
template <typename T>
using EnableSharedFromThis = std::enable_shared_from_this<T>;

using Mutex = std::mutex;

float HugeValf() { return HUGE_VALF; }

template <typename T, typename Allocator = std::allocator<T>>
using ForwardList = std::forward_list<T, Allocator>;
}

} // namespace infinity

namespace infinity {

export using BoostErrorCode = boost::system::error_code;
export using AsioIpAddr = boost::asio::ip::address;
export using AsioAcceptor = boost::asio::ip::tcp::acceptor;
export using AsioIOService = boost::asio::io_service;
export using AsioEndPoint = boost::asio::ip::tcp::endpoint;
export using AsioSocket = boost::asio::ip::tcp::socket;

export AsioIpAddr asio_make_address(const std::string &str, boost::system::error_code &ec) BOOST_ASIO_NOEXCEPT {
return boost::asio::ip::make_address(str.c_str(), ec);
}

} // namespace infinity





