#ifndef AIMRT_SAFETY_HPP
#define AIMRT_SAFETY_HPP

#include <map>
#include <string>
#include <vector>

namespace aimrt {

struct Hazard {
    std::string code;
    std::string description;
    int severity = 4;
    std::string likelihood;
    std::string mitigation;
};

struct HazardOccurrence {
    std::string code;
    std::string detail;
    double timestamp = 0.0;
};

class RiskMatrix {
public:
    static int score(int severity, const std::string& likelihood);
    static bool acceptable(int severity, const std::string& likelihood, int threshold = 6);
};

class SafetyCase {
public:
    void register_hazard(const Hazard& hazard);
    void record(const std::string& code, const std::string& detail, double timestamp);
    std::map<std::string, int> risk_summary() const;
    std::vector<Hazard> unacceptable_hazards(int threshold = 6) const;

private:
    std::map<std::string, Hazard> hazards_;
    std::vector<HazardOccurrence> occurrences_;
};

}  // namespace aimrt

#endif  // AIMRT_SAFETY_HPP
