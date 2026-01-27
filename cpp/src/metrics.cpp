#include "aimrt/metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace aimrt {

double tdev(const std::vector<double>& offsets, int tau) {
    if (tau <= 0) {
        throw std::runtime_error("tau must be positive");
    }
    if (offsets.size() < static_cast<size_t>(3 * tau)) {
        throw std::runtime_error("Not enough samples for TDEV");
    }

    std::vector<double> diffs;
    diffs.reserve(offsets.size() - static_cast<size_t>(3 * tau) + 1);
    for (size_t i = 0; i + static_cast<size_t>(3 * tau) <= offsets.size(); ++i) {
        double avg1 = 0.0;
        double avg2 = 0.0;
        double avg3 = 0.0;
        for (int j = 0; j < tau; ++j) {
            avg1 += offsets[i + j];
            avg2 += offsets[i + static_cast<size_t>(tau) + j];
            avg3 += offsets[i + static_cast<size_t>(2 * tau) + j];
        }
        avg1 /= tau;
        avg2 /= tau;
        avg3 /= tau;
        const double diff = avg1 - 2.0 * avg2 + avg3;
        diffs.push_back(diff);
    }

    double variance = 0.0;
    for (double diff : diffs) {
        variance += diff * diff;
    }
    variance /= (2.0 * diffs.size());
    return std::sqrt(variance);
}

double mtie(const std::vector<double>& offsets, int window) {
    if (window <= 0) {
        throw std::runtime_error("window must be positive");
    }
    if (offsets.size() < static_cast<size_t>(window)) {
        throw std::runtime_error("Not enough samples for MTIE");
    }

    double max_tie = 0.0;
    for (size_t i = 0; i + static_cast<size_t>(window) <= offsets.size(); ++i) {
        double min_value = offsets[i];
        double max_value = offsets[i];
        for (int j = 1; j < window; ++j) {
            const double value = offsets[i + static_cast<size_t>(j)];
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        max_tie = std::max(max_tie, max_value - min_value);
    }
    return max_tie;
}

HoldoverStats holdover_stats(const std::vector<double>& offsets, double sample_interval) {
    if (sample_interval <= 0.0) {
        throw std::runtime_error("sample_interval must be positive");
    }
    if (offsets.empty()) {
        throw std::runtime_error("offsets must be non-empty");
    }

    double max_offset = 0.0;
    double sum_sq = 0.0;
    for (double value : offsets) {
        max_offset = std::max(max_offset, std::abs(value));
        sum_sq += value * value;
    }

    const double rms_offset = std::sqrt(sum_sq / offsets.size());
    const double duration = sample_interval * static_cast<double>(offsets.size() - 1);
    return HoldoverStats{max_offset, rms_offset, duration};
}

}  // namespace aimrt
