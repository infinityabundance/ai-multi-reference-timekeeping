#include "aimrt/ml_model.hpp"

#include <algorithm>

namespace aimrt {

double LinearVarianceModel::predict_scale(const SensorFrame& frame) const {
    double score = bias;
    for (const auto& [feature, weight] : weights) {
        score += weight * feature_value(frame, feature);
    }
    return std::clamp(1.0 + score, 0.1, 10.0);
}

void LinearVarianceModel::train(const std::vector<TrainingSample>& samples, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : samples) {
            const double prediction = predict_scale(sample.frame);
            const double error = prediction - sample.target_scale;
            bias -= learning_rate * error;
            for (auto& [feature, weight] : weights) {
                const double grad = error * feature_value(sample.frame, feature);
                weight -= learning_rate * grad;
            }
        }
    }
}

}  // namespace aimrt
