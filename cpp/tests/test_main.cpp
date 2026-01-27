#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aimrt/characterization.hpp"
#include "aimrt/fusion.hpp"
#include "aimrt/kalman.hpp"
#include "aimrt/metrics.hpp"
#include "aimrt/partitioning.hpp"
#include "aimrt/references.hpp"
#include "aimrt/safety.hpp"
#include "aimrt/time_server.hpp"

namespace {

int failures = 0;

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        failures += 1;
    }
}

void expect_near(double value, double expected, double tolerance, const std::string& message) {
    if (std::fabs(value - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " (got " << value << ", expected " << expected << ")\n";
        failures += 1;
    }
}

void test_kalman() {
    aimrt::ClockKalmanFilter filter(aimrt::ClockState{0.0, 0.0}, aimrt::ClockCovariance{1.0, 0.0, 0.0, 1.0},
                                    1e-4, 1e-6);
    filter.predict(1.0);
    expect_near(filter.state().offset, 0.0, 1e-9, "kalman predict offset");
    filter.update(0.5, 0.1);
    expect_true(filter.state().offset > 0.0, "kalman update adjusts offset");
}

void test_fusion() {
    aimrt::ReferenceFusion fusion;
    std::vector<aimrt::Measurement> measurements = {
        {"ref1", 1.0, 1.0, false, 1.0},
        {"ref2", 2.0, 1.0, false, 1.0},
    };
    auto [offset, variance, weights] = fusion.fuse(measurements);
    expect_near(offset, 1.5, 1e-9, "fusion offset");
    expect_near(variance, 0.5, 1e-9, "fusion variance");
    expect_near(weights["ref1"], 0.5, 1e-9, "fusion weight ref1");
}

void test_metrics() {
    std::vector<double> offsets = {0.1, 0.2, 0.15, 0.05, -0.1, -0.2};
    auto tdev_value = aimrt::tdev(offsets, 1);
    auto mtie_value = aimrt::mtie(offsets, 2);
    expect_true(tdev_value >= 0.0, "tdev non-negative");
    expect_true(mtie_value >= 0.0, "mtie non-negative");
    auto stats = aimrt::holdover_stats(offsets, 1.0);
    expect_true(stats.max_offset > 0.0, "holdover max offset");
}

void test_safety() {
    aimrt::SafetyCase safety;
    aimrt::Hazard hazard{"TEST", "Test hazard", 2, "C", "Mitigation"};
    safety.register_hazard(hazard);
    safety.record("TEST", "Triggered", 0.0);
    auto summary = safety.risk_summary();
    expect_true(summary["TEST"] > 0, "risk summary recorded");
}

void test_partitioning() {
    aimrt::PartitionSupervisor supervisor(2, 0.0);
    supervisor.record_failure("sensor");
    supervisor.record_failure("sensor");
    expect_true(supervisor.should_reboot("sensor"), "partition reboot scheduled");
    supervisor.reboot("sensor");
    expect_true(!supervisor.should_reboot("sensor"), "partition reboot clears schedule");
}

void test_references() {
    aimrt::ReferenceSource source{"ref", 0.01, 0.0, 0.0};
    auto measurement = source.measure(0.5, 1.0);
    expect_true(measurement.variance > 0.0, "reference measurement variance");

    aimrt::DeterministicReference deterministic("det", 0.1, [](double t) { return t; });
    auto deterministic_measurement = deterministic.measure(2.0);
    expect_near(deterministic_measurement.offset, 2.0, 1e-9, "deterministic reference");
}

void test_sensor_validator() {
    aimrt::SensorValidator validator(100.0);
    aimrt::SensorValues values;
    values["temperature_c"] = 20.0;
    values["humidity_pct"] = 50.0;
    values["gps_lock"] = true;
    auto validated = validator.validate(values);
    expect_true(validated.find("temperature_c") != validated.end(), "validator keeps temperature");
    expect_true(validated.find("gps_lock") != validated.end(), "validator keeps gps lock");
}

}  // namespace

int main() {
    try {
        test_kalman();
        test_fusion();
        test_metrics();
        test_safety();
        test_partitioning();
        test_references();
        test_sensor_validator();
    } catch (const std::exception& exc) {
        std::cerr << "Unhandled exception: " << exc.what() << "\n";
        return 1;
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
