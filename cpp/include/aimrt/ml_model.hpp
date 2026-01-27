#ifndef AIMRT_ML_MODEL_HPP
#define AIMRT_ML_MODEL_HPP

#include <map>
#include <string>
#include <vector>

#include "aimrt/time_server.hpp"

namespace aimrt {

struct TrainingSample {
    SensorFrame frame{};
    double target_scale = 1.0;
};

struct LinearVarianceModel {
    std::map<std::string, double> weights;
    double bias = 0.0;

    double predict_scale(const SensorFrame& frame) const;
    void train(const std::vector<TrainingSample>& samples, double learning_rate = 1e-3, int epochs = 100);
};

}  // namespace aimrt

#endif  // AIMRT_ML_MODEL_HPP
