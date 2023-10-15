module;

#include "cxxopts.hpp"
#include "spdlog/details/registry.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// #include "toml.hpp"

export module third_party;

namespace infinity {

// spdlog
export enum class LogLevel { kTrace, kInfo, kWarning, kError, kFatal };

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

export void ShutdownLogger() {
    spdlog::shutdown();
}

export using spd_sink_ptr = spdlog::sink_ptr;
export using spd_stdout_color_sink = spdlog::sinks::stdout_color_sink_mt;
export using spd_rotating_file_sink = spdlog::sinks::rotating_file_sink_mt;

export using spd_logger = spdlog::logger;
export using spd_log_level = spdlog::level::level_enum;

// cxxopts
export using CxxOptions = cxxopts::Options;

export template <typename T>
std::shared_ptr<cxxopts::Value> cxx_value() {
    return std::make_shared<cxxopts::values::standard_value<T>>();
}

export using ParseResult = cxxopts::ParseResult;

// Toml parser
// using TomlParseResult = toml::parse_result;
//
// export TomlParseResult TomlParseFile(const std::string &file_path) { return toml::parse_file(file_path); }

} // namespace infinity
