#ifndef AIMRT_KALMAN_HPP
#define AIMRT_KALMAN_HPP

#include <tuple>

namespace aimrt {

struct ClockState {
    double offset = 0.0;
    double drift = 0.0;
};

struct ClockCovariance {
    double p00 = 0.0;
    double p01 = 0.0;
    double p10 = 0.0;
    double p11 = 0.0;

    std::tuple<std::tuple<double, double>, std::tuple<double, double>> as_matrix() const {
        return {{p00, p01}, {p10, p11}};
    }
};

class ClockKalmanFilter {
public:
    ClockKalmanFilter(ClockState state, ClockCovariance covariance,
                      double process_noise_offset, double process_noise_drift);

    const ClockState& state() const;
    const ClockCovariance& covariance() const;

    void predict(double dt);
    void update(double measured_offset, double measurement_variance);

private:
    ClockState state_{};
    ClockCovariance cov_{};
    double q_offset_ = 0.0;
    double q_drift_ = 0.0;
};

}  // namespace aimrt

#endif  // AIMRT_KALMAN_HPP
