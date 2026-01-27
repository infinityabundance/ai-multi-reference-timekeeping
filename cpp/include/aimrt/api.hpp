#ifndef AIMRT_API_HPP
#define AIMRT_API_HPP

#include <memory>
#include <vector>

#include "aimrt/config.hpp"
#include "aimrt/logging.hpp"
#include "aimrt/observability.hpp"
#include "aimrt/partitioning.hpp"
#include "aimrt/safety.hpp"
#include "aimrt/time_server.hpp"

namespace aimrt {

struct TimeServerRuntime {
    std::shared_ptr<TimeServer> server;
    std::shared_ptr<HealthMonitor> health;
    std::shared_ptr<MetricsTracker> metrics;
    std::shared_ptr<MetricsExporter> exporter;
};

TimeServerRuntime build_time_server(const std::vector<std::shared_ptr<ReferenceInput>>& references,
                                    const std::vector<SensorInput*>& sensors,
                                    std::shared_ptr<InferenceModel> inference = nullptr,
                                    std::shared_ptr<TimeServerSettings> settings = nullptr,
                                    Logger logger = get_logger("aimrt"),
                                    std::shared_ptr<SafetyCase> safety_case = nullptr);

}  // namespace aimrt

#endif  // AIMRT_API_HPP
