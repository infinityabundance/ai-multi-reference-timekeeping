#include "aimrt/logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

namespace aimrt {

namespace {

struct LoggingState {
    LogLevel level = LogLevel::kInfo;
    bool json = true;
    std::optional<std::string> log_file = std::nullopt;
    int max_bytes = 0;
    int backup_count = 0;
    std::unique_ptr<std::ostream> file_stream;
    std::mutex mutex;
};

LoggingState& state() {
    static LoggingState instance;
    return instance;
}

LogLevel parse_level(const std::string& level) {
    if (level == "DEBUG") {
        return LogLevel::kDebug;
    }
    if (level == "WARN") {
        return LogLevel::kWarn;
    }
    if (level == "ERROR") {
        return LogLevel::kError;
    }
    return LogLevel::kInfo;
}

bool allow_log(LogLevel configured, LogLevel incoming) {
    return static_cast<int>(incoming) >= static_cast<int>(configured);
}

std::string level_name(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug:
            return "DEBUG";
        case LogLevel::kInfo:
            return "INFO";
        case LogLevel::kWarn:
            return "WARN";
        case LogLevel::kError:
            return "ERROR";
    }
    return "INFO";
}

void rotate_logs(LoggingState& log_state) {
    if (!log_state.log_file.has_value()) {
        return;
    }
    if (log_state.max_bytes <= 0 || log_state.backup_count <= 0) {
        return;
    }

    const std::filesystem::path base_path(*log_state.log_file);
    std::error_code ec;
    if (!std::filesystem::exists(base_path, ec)) {
        return;
    }

    const auto size = std::filesystem::file_size(base_path, ec);
    if (ec || size < static_cast<std::uintmax_t>(log_state.max_bytes)) {
        return;
    }

    log_state.file_stream.reset();

    for (int index = log_state.backup_count - 1; index >= 1; --index) {
        std::filesystem::path source = base_path;
        source += "." + std::to_string(index);
        if (!std::filesystem::exists(source, ec)) {
            continue;
        }
        std::filesystem::path destination = base_path;
        destination += "." + std::to_string(index + 1);
        std::filesystem::rename(source, destination, ec);
    }

    std::filesystem::path rotated = base_path;
    rotated += ".1";
    std::filesystem::rename(base_path, rotated, ec);
    log_state.file_stream = std::make_unique<std::ofstream>(base_path, std::ios::app);
}

}  // namespace

Logger::Logger(std::string name) : name_(std::move(name)) {}

void Logger::log(LogLevel level, const std::string& message,
                 const std::map<std::string, std::string>& extra) const {
    auto& log_state = state();
    if (!allow_log(log_state.level, level)) {
        return;
    }
    std::lock_guard<std::mutex> guard(log_state.mutex);
    rotate_logs(log_state);
    std::ostream* output = &std::cout;
    if (log_state.file_stream) {
        output = log_state.file_stream.get();
    }

    if (log_state.json) {
        std::ostringstream payload;
        payload << "{\"level\":\"" << level_name(level) << "\",\"name\":\"" << name_
                << "\",\"message\":\"" << message << "\"";
        for (const auto& [key, value] : extra) {
            payload << ",\"" << key << "\":\"" << value << "\"";
        }
        payload << "}";
        *output << payload.str() << '\n';
    } else {
        *output << level_name(level) << " " << name_ << " " << message;
        if (!extra.empty()) {
            *output << " |";
            for (const auto& [key, value] : extra) {
                *output << " " << key << "=" << value;
            }
        }
        *output << '\n';
    }
}

void Logger::debug(const std::string& message, const std::map<std::string, std::string>& extra) const {
    log(LogLevel::kDebug, message, extra);
}

void Logger::info(const std::string& message, const std::map<std::string, std::string>& extra) const {
    log(LogLevel::kInfo, message, extra);
}

void Logger::warn(const std::string& message, const std::map<std::string, std::string>& extra) const {
    log(LogLevel::kWarn, message, extra);
}

void Logger::error(const std::string& message, const std::map<std::string, std::string>& extra) const {
    log(LogLevel::kError, message, extra);
}

void configure_logging(const LoggingConfig& config) {
    auto& log_state = state();
    log_state.level = parse_level(config.level);
    log_state.json = config.json;
    log_state.log_file = config.log_file;
    log_state.max_bytes = config.max_bytes;
    log_state.backup_count = config.backup_count;
    log_state.file_stream.reset();

    if (config.log_file.has_value()) {
        auto stream = std::make_unique<std::ofstream>(*config.log_file, std::ios::app);
        if (stream->is_open()) {
            log_state.file_stream = std::move(stream);
        }
    }
}

Logger get_logger(const std::string& name) {
    return Logger(name);
}

}  // namespace aimrt
