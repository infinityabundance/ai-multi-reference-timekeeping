#include "aimrt/api.hpp"
#include "aimrt/config.hpp"
#include "aimrt/daemon.hpp"
#include "aimrt/logging.hpp"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

aimrt::TimeServerDaemon* g_daemon = nullptr;

void handle_signal(int) {
    if (g_daemon) {
        g_daemon->stop();
    }
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config <path> [--no-chrony]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    bool chrony_enabled = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            config_path = argv[++i];
        } else if (arg == "--no-chrony") {
            chrony_enabled = false;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        auto settings = aimrt::TimeServerSettings::from_toml(config_path);
        auto references = aimrt::build_references_from_settings(settings);
        auto sensors = aimrt::build_sensors_from_settings(settings);
        auto runtime = aimrt::build_time_server(references, sensors, nullptr,
                                                std::make_shared<aimrt::TimeServerSettings>(settings));
        aimrt::DaemonConfig config;
        config.step_interval_s = settings.daemon_step_interval_s;
        config.chrony_enabled = chrony_enabled;

        aimrt::TimeServerDaemon daemon(runtime, config);
        g_daemon = &daemon;
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        daemon.run();
    } catch (const std::exception& exc) {
        std::cerr << "aimrtd error: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
