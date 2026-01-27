#include "aimrt/references.hpp"

#include <stdexcept>

namespace aimrt {

Measurement ReferenceSource::measure(double true_offset, double elapsed) const {
    if (noise_std <= 0.0) {
        throw std::runtime_error("noise_std must be positive");
    }
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, noise_std);
    const double noise = dist(rng);
    const double biased_offset = true_offset + bias + drift_bias * elapsed + noise;
    return Measurement{name, biased_offset, noise_std * noise_std, false, 1.0};
}

ReferenceEnsemble::ReferenceEnsemble(std::vector<ReferenceSource> sources) : sources_(std::move(sources)) {
    if (sources_.empty()) {
        throw std::runtime_error("At least one reference source is required");
    }
}

std::vector<Measurement> ReferenceEnsemble::snapshot(double true_offset, double elapsed) const {
    std::vector<Measurement> output;
    output.reserve(sources_.size());
    for (const auto& source : sources_) {
        output.push_back(source.measure(true_offset, elapsed));
    }
    return output;
}

DeterministicReference::DeterministicReference(std::string name, double variance,
                                               std::function<double(double)> generator)
    : name_(std::move(name)), variance_(variance), generator_(std::move(generator)) {
    if (variance_ <= 0.0) {
        throw std::runtime_error("variance must be positive");
    }
}

Measurement DeterministicReference::measure(double elapsed) const {
    return Measurement{name_, generator_(elapsed), variance_, false, 1.0};
}

}  // namespace aimrt
