#ifndef AIMRT_DAEMON_HPP
#define AIMRT_DAEMON_HPP

#include <atomic>

#include "aimrt/api.hpp"
#include "aimrt/chrony.hpp"

namespace aimrt {

struct DaemonConfig {
    double step_interval_s = 1.0;
    bool chrony_enabled = true;
};

class TimeServerDaemon {
public:
    TimeServerDaemon(TimeServerRuntime runtime, DaemonConfig config);

    void run();
    void stop();

private:
    TimeServerRuntime runtime_;
    DaemonConfig config_;
    Logger logger_;
    std::optional<ChronyShmWriter> chrony_;
    std::atomic<bool> running_{false};
};

void run_daemon(std::shared_ptr<TimeServerSettings> settings = nullptr);

}  // namespace aimrt

#endif  // AIMRT_DAEMON_HPP
