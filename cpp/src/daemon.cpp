#include "aimrt/daemon.hpp"

#include <thread>

namespace aimrt {

TimeServerDaemon::TimeServerDaemon(TimeServerRuntime runtime, DaemonConfig config)
    : runtime_(std::move(runtime)), config_(config), logger_(get_logger("TimeServerDaemon")) {
    if (config_.chrony_enabled) {
        chrony_.emplace();
    }
}

void TimeServerDaemon::run() {
    running_ = true;
    logger_.info("daemon_started", {{"interval_s", std::to_string(config_.step_interval_s)}});
    while (running_) {
        auto [update, frame, drift, drift_hint] = runtime_.server->step(config_.step_interval_s);
        runtime_.metrics->update(update.fused_offset);
        runtime_.health->mark_update();
        if (chrony_.has_value()) {
            chrony_->write(ChronyShmSample{update.fused_offset, 0.0, 0, 0, 1});
            runtime_.health->mark_shm_write();
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(config_.step_interval_s));
    }
}

void TimeServerDaemon::stop() {
    running_ = false;
}

void run_daemon(std::shared_ptr<TimeServerSettings> settings) {
    auto runtime = build_time_server({}, {}, nullptr, settings);
    TimeServerDaemon daemon(runtime, DaemonConfig{});
    daemon.run();
}

}  // namespace aimrt
