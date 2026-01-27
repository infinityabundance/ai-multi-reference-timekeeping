#ifndef AIMRT_COMMON_HPP
#define AIMRT_COMMON_HPP

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aimrt {

inline std::string ltrim(std::string value) {
    auto it = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    value.erase(value.begin(), it);
    return value;
}

inline std::string rtrim(std::string value) {
    auto it = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    value.erase(it.base(), value.end());
    return value;
}

inline std::string trim(std::string value) {
    return rtrim(ltrim(std::move(value)));
}

inline std::vector<std::string> split(std::string_view value, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : value) {
        if (ch == delimiter) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    parts.push_back(current);
    return parts;
}

inline std::string strip_quotes(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

inline double seconds_since_epoch() {
    using clock = std::chrono::system_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

}  // namespace aimrt

#endif  // AIMRT_COMMON_HPP
