#include "aimrt/config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include "aimrt/common.hpp"

namespace aimrt {

namespace {

bool parse_bool(const std::string& value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error("invalid boolean: " + value);
}

std::optional<std::string> parse_optional_string(const std::string& value) {
    auto stripped = strip_quotes(trim(value));
    if (stripped == "null" || stripped == "none") {
        return std::nullopt;
    }
    return stripped;
}

}  // namespace

TimeServerSettings TimeServerSettings::from_toml(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("unable to open config file: " + path);
    }

    TimeServerSettings settings;
    std::string current_section;
    std::string line;

    while (std::getline(file, line)) {
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            current_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }
        auto key = trim(line.substr(0, eq_pos));
        auto value = trim(line.substr(eq_pos + 1));

        if (current_section == "logging") {
            if (key == "level") {
                settings.logging.level = strip_quotes(value);
            } else if (key == "json") {
                settings.logging.json = parse_bool(value);
            } else if (key == "log_file") {
                settings.logging.log_file = parse_optional_string(value);
            } else if (key == "max_bytes") {
                settings.logging.max_bytes = std::stoi(value);
            } else if (key == "backup_count") {
                settings.logging.backup_count = std::stoi(value);
            }
        } else if (current_section == "metrics") {
            if (key == "enabled") {
                settings.metrics.enabled = parse_bool(value);
            } else if (key == "host") {
                settings.metrics.host = strip_quotes(value);
            } else if (key == "port") {
                settings.metrics.port = std::stoi(value);
            } else if (key == "window_size") {
                settings.metrics.window_size = std::stoi(value);
            }
        } else if (current_section == "security") {
            if (key == "divergence_threshold") {
                settings.security.divergence_threshold = std::stod(value);
            } else if (key == "grid_frequency") {
                auto parsed = parse_optional_string(value);
                if (parsed.has_value()) {
                    settings.security.grid_frequency = std::stod(parsed.value());
                } else {
                    settings.security.grid_frequency = std::nullopt;
                }
            } else if (key == "grid_tolerance_hz") {
                settings.security.grid_tolerance_hz = std::stod(value);
            } else if (key == "max_measurement_rate") {
                settings.security.max_measurement_rate = std::stod(value);
            }
        } else if (current_section == "learning") {
            if (key == "enabled") {
                settings.learning.enabled = parse_bool(value);
            } else if (key == "fusion_model_path") {
                settings.learning.fusion_model_path = parse_optional_string(value);
            } else if (key == "max_offset_delta_ns") {
                settings.learning.max_offset_delta_ns = std::stod(value);
            } else if (key == "min_variance") {
                settings.learning.min_variance = std::stod(value);
            } else if (key == "max_variance") {
                settings.learning.max_variance = std::stod(value);
            }
        } else if (current_section.empty()) {
            if (key == "chrony_shm_path") {
                settings.chrony_shm_path = strip_quotes(value);
            }
        }
    }

    return settings;
}

}  // namespace aimrt
