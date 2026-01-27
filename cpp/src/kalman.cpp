#include "aimrt/kalman.hpp"

#include <stdexcept>

namespace aimrt {

ClockKalmanFilter::ClockKalmanFilter(ClockState state, ClockCovariance covariance,
                                     double process_noise_offset, double process_noise_drift)
    : state_(state), cov_(covariance), q_offset_(process_noise_offset), q_drift_(process_noise_drift) {}

const ClockState& ClockKalmanFilter::state() const {
    return state_;
}

const ClockCovariance& ClockKalmanFilter::covariance() const {
    return cov_;
}

void ClockKalmanFilter::predict(double dt) {
    if (dt <= 0.0) {
        throw std::runtime_error("dt must be positive");
    }

    const double new_offset = state_.offset + state_.drift * dt;
    const double new_drift = state_.drift;
    state_ = ClockState{new_offset, new_drift};

    double p00 = cov_.p00 + dt * (cov_.p10 + cov_.p01) + dt * dt * cov_.p11;
    double p01 = cov_.p01 + dt * cov_.p11;
    double p10 = cov_.p10 + dt * cov_.p11;
    double p11 = cov_.p11;

    p00 += q_offset_ * dt;
    p11 += q_drift_ * dt;

    cov_ = ClockCovariance{p00, p01, p10, p11};
}

void ClockKalmanFilter::update(double measured_offset, double measurement_variance) {
    if (measurement_variance <= 0.0) {
        throw std::runtime_error("measurement_variance must be positive");
    }

    const double innovation_cov = cov_.p00 + measurement_variance;
    const double kalman_gain_offset = cov_.p00 / innovation_cov;
    const double kalman_gain_drift = cov_.p10 / innovation_cov;
    const double residual = measured_offset - state_.offset;

    const double updated_offset = state_.offset + kalman_gain_offset * residual;
    const double updated_drift = state_.drift + kalman_gain_drift * residual;
    state_ = ClockState{updated_offset, updated_drift};

    const double p00 = (1.0 - kalman_gain_offset) * cov_.p00;
    const double p01 = (1.0 - kalman_gain_offset) * cov_.p01;
    const double p10 = cov_.p10 - kalman_gain_drift * cov_.p00;
    const double p11 = cov_.p11 - kalman_gain_drift * cov_.p01;
    cov_ = ClockCovariance{p00, p01, p10, p11};
}

}  // namespace aimrt
