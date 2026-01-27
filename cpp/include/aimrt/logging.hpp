#ifndef AIMRT_LOGGING_HPP
#define AIMRT_LOGGING_HPP

#include <map>
#include <optional>
#include <ostream>
#include <string>

#include "aimrt/config.hpp"

namespace aimrt {

enum class LogLevel {
    kDebug,
    kInfo,
    kWarn,
    kError,
};

class Logger {
public:
    explicit Logger(std::string name);

    void log(LogLevel level, const std::string& message,
             const std::map<std::string, std::string>& extra = {}) const;

    void debug(const std::string& message,
               const std::map<std::string, std::string>& extra = {}) const;
    void info(const std::string& message,
              const std::map<std::string, std::string>& extra = {}) const;
    void warn(const std::string& message,
              const std::map<std::string, std::string>& extra = {}) const;
    void error(const std::string& message,
               const std::map<std::string, std::string>& extra = {}) const;

private:
    std::string name_;
};

void configure_logging(const LoggingConfig& config);
Logger get_logger(const std::string& name);

}  // namespace aimrt

#endif  // AIMRT_LOGGING_HPP
