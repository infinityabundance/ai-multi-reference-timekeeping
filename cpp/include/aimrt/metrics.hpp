#ifndef AIMRT_METRICS_HPP
#define AIMRT_METRICS_HPP

#include <vector>

namespace aimrt {

struct HoldoverStats {
    double max_offset = 0.0;
    double rms_offset = 0.0;
    double duration = 0.0;
};

double tdev(const std::vector<double>& offsets, int tau);
double mtie(const std::vector<double>& offsets, int window);
HoldoverStats holdover_stats(const std::vector<double>& offsets, double sample_interval);

}  // namespace aimrt

#endif  // AIMRT_METRICS_HPP
