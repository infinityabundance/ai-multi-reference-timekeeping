#ifndef AIMRT_CHARACTERIZATION_HPP
#define AIMRT_CHARACTERIZATION_HPP

#include <map>
#include <string>

namespace aimrt {

struct RunningStats {
    double mean = 0.0;
    double m2 = 0.0;
    int count = 0;

    void update(double value);
    double variance() const;
    double stddev() const;
};

class SensorCharacterization {
public:
    void update(const std::string& name, double residual);
    double z_score(const std::string& name, double residual) const;

private:
    std::map<std::string, RunningStats> stats_;
};

}  // namespace aimrt

#endif  // AIMRT_CHARACTERIZATION_HPP
