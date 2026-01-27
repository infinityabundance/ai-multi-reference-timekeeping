#ifndef AIMRT_FUSION_HPP
#define AIMRT_FUSION_HPP

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "aimrt/kalman.hpp"

namespace aimrt {

struct Measurement {
    std::string name;
    double offset = 0.0;
    double variance = 1.0;
    bool has_quality = false;
    double quality = 1.0;
};

struct ClockUpdate {
    double fused_offset = 0.0;
    double fused_variance = 0.0;
    ClockState state{};
    std::map<std::string, double> reference_weights;
};

class ReferenceFusion {
public:
    virtual ~ReferenceFusion() = default;
    virtual std::tuple<double, double, std::map<std::string, double>>
    fuse(const std::vector<Measurement>& measurements) const;
};

class HeuristicFusion : public ReferenceFusion {
public:
    std::tuple<double, double, std::map<std::string, double>>
    fuse(const std::vector<Measurement>& measurements) const override;
};

class VirtualClock {
public:
    explicit VirtualClock(ClockKalmanFilter kalman, std::unique_ptr<ReferenceFusion> fusion = nullptr);

    const ClockState& state() const;

    ClockUpdate step(double dt, const std::vector<Measurement>& measurements);
    std::vector<std::pair<double, double>> history() const;

private:
    ClockKalmanFilter kalman_;
    std::unique_ptr<ReferenceFusion> fusion_;
};

}  // namespace aimrt

#endif  // AIMRT_FUSION_HPP
