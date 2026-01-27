#ifndef AIMRT_CONFIG_HPP
#define AIMRT_CONFIG_HPP

#include <optional>
#include <string>

namespace aimrt {

struct LoggingConfig {
    std::string level = "INFO";
    bool json = true;
    std::optional<std::string> log_file = std::nullopt;
    int max_bytes = 1'000'000;
    int backup_count = 3;
};

struct MetricsConfig {
    bool enabled = true;
    std::string host = "0.0.0.0";
    int port = 8000;
    int window_size = 60;
};

struct SecurityConfig {
    double divergence_threshold = 0.005;
    std::optional<double> grid_frequency = 60.0;
    double grid_tolerance_hz = 0.5;
    double max_measurement_rate = 5.0;
};

struct LearningConfig {
    bool enabled = false;
    std::optional<std::string> fusion_model_path = std::nullopt;
    double max_offset_delta_ns = 1e6;
    double min_variance = 1e-18;
    double max_variance = 1e-6;
};

struct TimeServerSettings {
    LoggingConfig logging{};
    MetricsConfig metrics{};
    SecurityConfig security{};
    LearningConfig learning{};
    std::string chrony_shm_path = "/dev/shm/chrony_shm0";

    static TimeServerSettings from_toml(const std::string& path);
};

}  // namespace aimrt

#endif  // AIMRT_CONFIG_HPP
