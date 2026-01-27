#include "aimrt/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace aimrt {

std::tuple<double, double, std::map<std::string, double>>
ReferenceFusion::fuse(const std::vector<Measurement>& measurements) const {
    if (measurements.empty()) {
        throw std::runtime_error("At least one measurement is required for fusion");
    }

    std::map<std::string, double> weights;
    double weighted_sum = 0.0;
    double weight_total = 0.0;

    for (const auto& measurement : measurements) {
        if (measurement.variance <= 0.0) {
            throw std::runtime_error("Measurement variance must be positive for " + measurement.name);
        }
        const double weight = 1.0 / measurement.variance;
        weights[measurement.name] = weight;
        weight_total += weight;
        weighted_sum += weight * measurement.offset;
    }

    const double fused_offset = weighted_sum / weight_total;
    const double fused_variance = 1.0 / weight_total;

    std::map<std::string, double> normalized_weights;
    for (const auto& [name, weight] : weights) {
        normalized_weights[name] = weight / weight_total;
    }

    return {fused_offset, fused_variance, normalized_weights};
}

std::tuple<double, double, std::map<std::string, double>>
HeuristicFusion::fuse(const std::vector<Measurement>& measurements) const {
    if (measurements.empty()) {
        throw std::runtime_error("At least one measurement is required for fusion");
    }

    std::map<std::string, double> weights;
    double weighted_sum = 0.0;
    double weight_total = 0.0;

    for (const auto& measurement : measurements) {
        if (measurement.variance <= 0.0) {
            throw std::runtime_error("Measurement variance must be positive for " + measurement.name);
        }
        double quality = measurement.has_quality ? measurement.quality : 1.0;
        quality = std::clamp(quality, 0.0, 1.0);
        if (quality <= 0.0) {
            continue;
        }
        const double weight = (1.0 / measurement.variance) * quality;
        weights[measurement.name] = weight;
        weight_total += weight;
        weighted_sum += weight * measurement.offset;
    }

    if (weight_total <= 0.0) {
        throw std::runtime_error("No valid measurements available after applying quality scores");
    }

    const double fused_offset = weighted_sum / weight_total;
    const double fused_variance = 1.0 / weight_total;

    std::map<std::string, double> normalized_weights;
    for (const auto& [name, weight] : weights) {
        normalized_weights[name] = weight / weight_total;
    }

    return {fused_offset, fused_variance, normalized_weights};
}

VirtualClock::VirtualClock(ClockKalmanFilter kalman, std::unique_ptr<ReferenceFusion> fusion)
    : kalman_(std::move(kalman)),
      fusion_(fusion ? std::move(fusion) : std::make_unique<ReferenceFusion>()) {}

const ClockState& VirtualClock::state() const {
    return kalman_.state();
}

ClockUpdate VirtualClock::step(double dt, const std::vector<Measurement>& measurements) {
    kalman_.predict(dt);
    auto [fused_offset, fused_variance, weights] = fusion_->fuse(measurements);
    kalman_.update(fused_offset, fused_variance);

    return ClockUpdate{fused_offset, fused_variance, kalman_.state(), weights};
}

std::vector<std::pair<double, double>> VirtualClock::history() const {
    return {{kalman_.state().offset, kalman_.state().drift}};
}

}  // namespace aimrt
