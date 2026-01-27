#ifndef AIMRT_REFERENCES_HPP
#define AIMRT_REFERENCES_HPP

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "aimrt/fusion.hpp"

namespace aimrt {

struct ReferenceSource {
    std::string name;
    double noise_std = 0.0;
    double bias = 0.0;
    double drift_bias = 0.0;

    Measurement measure(double true_offset, double elapsed) const;
};

class ReferenceEnsemble {
public:
    explicit ReferenceEnsemble(std::vector<ReferenceSource> sources);

    std::vector<Measurement> snapshot(double true_offset, double elapsed) const;

private:
    std::vector<ReferenceSource> sources_;
};

class DeterministicReference {
public:
    DeterministicReference(std::string name, double variance, std::function<double(double)> generator);

    Measurement measure(double elapsed) const;

private:
    std::string name_;
    double variance_ = 0.0;
    std::function<double(double)> generator_;
};

}  // namespace aimrt

#endif  // AIMRT_REFERENCES_HPP
