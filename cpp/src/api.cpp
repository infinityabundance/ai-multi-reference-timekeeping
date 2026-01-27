#include "aimrt/api.hpp"

#include "aimrt/fusion.hpp"
#include "aimrt/kalman.hpp"
#include "aimrt/observability.hpp"
#include "aimrt/time_server.hpp"

namespace aimrt {

TimeServerRuntime build_time_server(const std::vector<std::shared_ptr<ReferenceInput>>& references,
                                    const std::vector<SensorInput*>& sensors,
                                    std::shared_ptr<InferenceModel> inference,
                                    std::shared_ptr<TimeServerSettings> settings,
                                    Logger logger,
                                    std::shared_ptr<SafetyCase> safety_case) {
    auto effective_settings = settings ? std::move(settings) : std::make_shared<TimeServerSettings>();
    configure_logging(effective_settings->logging);

    ClockKalmanFilter kalman(ClockState{0.0, 0.0}, ClockCovariance{1.0, 0.0, 0.0, 1.0}, 1e-4, 1e-6);
    auto fusion = std::make_unique<HeuristicFusion>();
    VirtualClock clock(std::move(kalman), std::move(fusion));

    auto partition_supervisor = std::make_shared<PartitionSupervisor>();
    auto aggregator = std::make_shared<SensorAggregator>(sensors, nullptr, logger, partition_supervisor);
    auto server = std::make_shared<TimeServer>(
        std::move(clock), references, aggregator,
        inference ? std::move(inference) : std::make_shared<LightweightInferenceModel>(),
        std::make_shared<SlewDriftDetector>(), std::nullopt, nullptr, logger, safety_case, partition_supervisor);

    auto health = std::make_shared<HealthMonitor>();
    auto metrics = std::make_shared<MetricsTracker>(effective_settings->metrics.window_size);
    auto exporter = std::make_shared<MetricsExporter>(*metrics, *health);
    if (effective_settings->metrics.enabled) {
        exporter->start(effective_settings->metrics.host, effective_settings->metrics.port);
    }

    return TimeServerRuntime{server, health, metrics, exporter};
}

}  // namespace aimrt
