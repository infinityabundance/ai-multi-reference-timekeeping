#include "aimrt/characterization.hpp"

#include <cmath>

namespace aimrt {

void RunningStats::update(double value) {
    count += 1;
    const double delta = value - mean;
    mean += delta / count;
    const double delta2 = value - mean;
    m2 += delta * delta2;
}

double RunningStats::variance() const {
    if (count < 2) {
        return 0.0;
    }
    return m2 / (count - 1);
}

double RunningStats::stddev() const {
    return std::sqrt(variance());
}

void SensorCharacterization::update(const std::string& name, double residual) {
    stats_[name].update(residual);
}

double SensorCharacterization::z_score(const std::string& name, double residual) const {
    auto it = stats_.find(name);
    if (it == stats_.end()) {
        return 0.0;
    }
    const auto& stats = it->second;
    if (stats.count < 2 || stats.stddev() == 0.0) {
        return 0.0;
    }
    return (residual - stats.mean) / stats.stddev();
}

}  // namespace aimrt
