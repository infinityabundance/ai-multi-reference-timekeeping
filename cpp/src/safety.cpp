#include "aimrt/safety.hpp"

#include <stdexcept>

namespace aimrt {

namespace {

int likelihood_score(const std::string& likelihood) {
    if (likelihood == "A") {
        return 5;
    }
    if (likelihood == "B") {
        return 4;
    }
    if (likelihood == "C") {
        return 3;
    }
    if (likelihood == "D") {
        return 2;
    }
    if (likelihood == "E") {
        return 1;
    }
    throw std::runtime_error("likelihood must be A..E");
}

}  // namespace

int RiskMatrix::score(int severity, const std::string& likelihood) {
    if (severity < 1 || severity > 4) {
        throw std::runtime_error("severity must be 1..4");
    }
    return (5 - severity) * likelihood_score(likelihood);
}

bool RiskMatrix::acceptable(int severity, const std::string& likelihood, int threshold) {
    return score(severity, likelihood) <= threshold;
}

void SafetyCase::register_hazard(const Hazard& hazard) {
    hazards_[hazard.code] = hazard;
}

void SafetyCase::record(const std::string& code, const std::string& detail, double timestamp) {
    if (hazards_.find(code) == hazards_.end()) {
        throw std::runtime_error("Unknown hazard code " + code);
    }
    occurrences_.push_back(HazardOccurrence{code, detail, timestamp});
}

std::map<std::string, int> SafetyCase::risk_summary() const {
    std::map<std::string, int> summary;
    for (const auto& [code, hazard] : hazards_) {
        summary[code] = RiskMatrix::score(hazard.severity, hazard.likelihood);
    }
    return summary;
}

std::vector<Hazard> SafetyCase::unacceptable_hazards(int threshold) const {
    std::vector<Hazard> output;
    for (const auto& [code, hazard] : hazards_) {
        if (!RiskMatrix::acceptable(hazard.severity, hazard.likelihood, threshold)) {
            output.push_back(hazard);
        }
    }
    return output;
}

}  // namespace aimrt
