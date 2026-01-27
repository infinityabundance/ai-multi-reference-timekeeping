#ifndef AIMRT_OBSERVABILITY_HPP
#define AIMRT_OBSERVABILITY_HPP

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace aimrt {

struct HealthStatus {
    double last_update = 0.0;
    std::optional<double> shm_last_write = std::nullopt;
    bool ok = false;
};

class HealthMonitor {
public:
    explicit HealthMonitor(double freshness_window = 10.0);

    void mark_update();
    void mark_shm_write();
    HealthStatus status() const;

private:
    double freshness_window_ = 10.0;
    mutable std::mutex mutex_;
    std::optional<double> last_update_;
    std::optional<double> shm_last_write_;
};

class MetricsTracker {
public:
    explicit MetricsTracker(int window_size = 60);

    void update(double offset);
    std::vector<std::pair<std::string, double>> metrics() const;

private:
    int window_size_ = 60;
    std::vector<double> offsets_;
};

class MetricsExporter {
public:
    MetricsExporter(MetricsTracker& tracker, HealthMonitor& health);
    ~MetricsExporter();

    void start(const std::string& host, int port);
    void stop();

private:
    void serve(const std::string& host, int port);

    MetricsTracker& tracker_;
    HealthMonitor& health_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    int server_fd_ = -1;
};

}  // namespace aimrt

#endif  // AIMRT_OBSERVABILITY_HPP
