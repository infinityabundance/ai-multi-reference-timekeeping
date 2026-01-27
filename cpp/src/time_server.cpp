#include "aimrt/time_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

#include "aimrt/common.hpp"

namespace aimrt {

namespace {

constexpr double kNtpEpoch = 2208988800.0;
constexpr double kPi = 3.14159265358979323846;

std::optional<double> get_optional_double(const SensorValues& values, const std::string& key) {
    auto it = values.find(key);
    if (it == values.end()) {
        return std::nullopt;
    }
    if (std::holds_alternative<double>(it->second)) {
        return std::get<double>(it->second);
    }
    return std::nullopt;
}

std::optional<bool> get_optional_bool(const SensorValues& values, const std::string& key) {
    auto it = values.find(key);
    if (it == values.end()) {
        return std::nullopt;
    }
    if (std::holds_alternative<bool>(it->second)) {
        return std::get<bool>(it->second);
    }
    return std::nullopt;
}

double ntp_to_unix(const uint8_t* data) {
    const uint32_t seconds = (static_cast<uint32_t>(data[0]) << 24) |
                             (static_cast<uint32_t>(data[1]) << 16) |
                             (static_cast<uint32_t>(data[2]) << 8) |
                             static_cast<uint32_t>(data[3]);
    const uint32_t fraction = (static_cast<uint32_t>(data[4]) << 24) |
                              (static_cast<uint32_t>(data[5]) << 16) |
                              (static_cast<uint32_t>(data[6]) << 8) |
                              static_cast<uint32_t>(data[7]);
    return static_cast<double>(seconds) - kNtpEpoch + static_cast<double>(fraction) / std::pow(2.0, 32.0);
}

Measurement parse_rmc(const std::string& name, const std::string& line, double variance) {
    auto fields = split(line, ',');
    if (fields.size() < 10 || fields[2] != "A") {
        throw std::runtime_error("Invalid RMC data");
    }
    const std::string& time_str = fields[1];
    const std::string& date_str = fields[9];
    if (time_str.size() < 6 || date_str.size() != 6) {
        throw std::runtime_error("Invalid RMC time/date");
    }
    const int hh = std::stoi(time_str.substr(0, 2));
    const int mm = std::stoi(time_str.substr(2, 2));
    const int ss = std::stoi(time_str.substr(4, 2));
    const int day = std::stoi(date_str.substr(0, 2));
    const int month = std::stoi(date_str.substr(2, 2));
    const int year = std::stoi(date_str.substr(4, 2)) + 2000;

    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hh;
    tm.tm_min = mm;
    tm.tm_sec = ss;
    tm.tm_isdst = 0;
    const double gps_timestamp = timegm(&tm);
    const double offset = gps_timestamp - seconds_since_epoch();
    return Measurement{name, offset, variance, false, 1.0};
}

std::string locate_hwclock() {
    const char* path_env = std::getenv("PATH");
    if (!path_env) {
        throw std::runtime_error("PATH not set for hwclock lookup");
    }
    std::string path(path_env);
    auto parts = split(path, ':');
    for (const auto& part : parts) {
        std::string candidate = part + "/hwclock";
        if (::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
    }
    throw std::runtime_error("hwclock not available for RTC reference");
}

double goertzel(const std::vector<double>& samples, int sample_rate, double frequency) {
    if (sample_rate <= 0) {
        throw std::runtime_error("sample_rate must be positive");
    }
    if (frequency <= 0.0) {
        return 0.0;
    }
    const int k = static_cast<int>(0.5 + (samples.size() * frequency) / sample_rate);
    const double omega = (2.0 * kPi * k) / samples.size();
    const double coeff = 2.0 * std::cos(omega);
    double s_prev = 0.0;
    double s_prev2 = 0.0;
    for (double sample : samples) {
        const double s = sample + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;
}

std::complex<double> goertzel_complex(const std::vector<double>& samples, int sample_rate, double frequency) {
    if (sample_rate <= 0) {
        throw std::runtime_error("sample_rate must be positive");
    }
    if (frequency <= 0.0) {
        return {0.0, 0.0};
    }
    const int k = static_cast<int>(0.5 + (samples.size() * frequency) / sample_rate);
    const double omega = (2.0 * kPi * k) / samples.size();
    const double coeff = 2.0 * std::cos(omega);
    double s_prev = 0.0;
    double s_prev2 = 0.0;
    for (double sample : samples) {
        const double s = sample + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    const double real = s_prev - s_prev2 * std::cos(omega);
    const double imag = s_prev2 * std::sin(omega);
    return {real, imag};
}

double band_energy(const std::vector<double>& samples, int sample_rate, std::pair<double, double> band) {
    const auto [low, high] = band;
    if (low <= 0.0 || high <= low) {
        return 0.0;
    }
    const double step = std::max(1.0, (high - low) / 5.0);
    double energy = 0.0;
    for (double freq = low; freq <= high; freq += step) {
        energy += goertzel(samples, sample_rate, freq);
    }
    return energy;
}

std::tuple<std::optional<double>, std::optional<double>, std::optional<double>> estimate_ac_hum(
    const std::vector<double>& samples, int sample_rate, const std::vector<double>& candidates) {
    if (samples.empty() || sample_rate <= 0) {
        return {std::nullopt, std::nullopt, std::nullopt};
    }
    std::optional<double> best_frequency;
    double best_power = 0.0;
    double total_power = 0.0;
    for (double sample : samples) {
        total_power += sample * sample;
    }
    total_power /= samples.size();

    std::vector<double> search_frequencies;
    for (double candidate : candidates) {
        for (double offset : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
            search_frequencies.push_back(candidate + offset);
        }
    }

    for (double frequency : search_frequencies) {
        const double power = goertzel(samples, sample_rate, frequency);
        if (power > best_power) {
            best_power = power;
            best_frequency = frequency;
        }
    }

    if (!best_frequency.has_value()) {
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    const auto phase = goertzel_complex(samples, sample_rate, *best_frequency);
    const double phase_rad = std::atan2(phase.imag(), phase.real());
    const double snr = best_power / std::max(total_power, 1e-12);
    const double uncertainty = 1.0 / (1.0 + snr);
    return {best_frequency, phase_rad, uncertainty};
}

}  // namespace

SensorValidator::SensorValidator(double max_samples_per_sec, std::map<std::string, std::pair<double, double>> ranges,
                                 Logger logger)
    : max_samples_per_sec_(max_samples_per_sec), logger_(std::move(logger)) {
    if (ranges.empty()) {
        ranges = {
            {"temperature_c", {-40.0, 85.0}},
            {"humidity_pct", {0.0, 100.0}},
            {"pressure_hpa", {900.0, 1100.0}},
            {"ac_hum_hz", {49.5, 60.5}},
            {"radio_snr_db", {0.0, 60.0}},
            {"geiger_cpm", {0.0, 20000.0}},
            {"ambient_audio_db", {-120.0, 0.0}},
            {"bird_activity", {0.0, 1e9}},
            {"traffic_activity", {0.0, 1e9}},
        };
    }
    ranges_ = std::move(ranges);
}

SensorValues SensorValidator::validate(const SensorValues& values) {
    const double now = seconds_since_epoch();
    if (max_samples_per_sec_ > 0.0) {
        if (last_sample_.has_value() && now - last_sample_.value() < 1.0 / max_samples_per_sec_) {
            logger_.warn("sensor_rate_limited", {{"max_samples_per_sec", std::to_string(max_samples_per_sec_)}});
            return {};
        }
    }
    last_sample_ = now;

    SensorValues filtered;
    for (const auto& [key, value] : values) {
        if (std::holds_alternative<std::monostate>(value)) {
            filtered[key] = value;
            continue;
        }
        if (std::holds_alternative<bool>(value)) {
            filtered[key] = value;
            continue;
        }
        if (!std::holds_alternative<double>(value)) {
            continue;
        }
        const double numeric_value = std::get<double>(value);
        auto range_it = ranges_.find(key);
        if (range_it != ranges_.end()) {
            const auto [min_value, max_value] = range_it->second;
            if (numeric_value < min_value || numeric_value > max_value) {
                logger_.warn("sensor_value_out_of_range",
                             {{"field", key}, {"value", std::to_string(numeric_value)}});
                continue;
            }
        }
        filtered[key] = value;
    }
    return filtered;
}

SensorAggregator::SensorAggregator(std::vector<SensorInput*> inputs,
                                   std::shared_ptr<SensorValidator> validator,
                                   Logger logger,
                                   std::shared_ptr<PartitionSupervisor> partitions)
    : inputs_(std::move(inputs)),
      validator_(validator ? std::move(validator) : std::make_shared<SensorValidator>()),
      logger_(std::move(logger)),
      partitions_(partitions ? std::move(partitions) : std::make_shared<PartitionSupervisor>()) {}

SensorFrame SensorAggregator::sample() {
    SensorValues updates;
    for (auto* sensor : inputs_) {
        const std::string partition_name = typeid(*sensor).name();
        try {
            if (partitions_->should_reboot(partition_name)) {
                partitions_->reboot(partition_name);
                logger_.info("partition_rebooted", {{"partition", partition_name}});
            }
            auto values = sensor->sample();
            auto validated = validator_->validate(values);
            updates.insert(validated.begin(), validated.end());
            partitions_->record_success(partition_name);
        } catch (const std::exception& exc) {
            logger_.error("sensor_sample_failed", {{"sensor", partition_name}, {"error", exc.what()}});
            const auto& state = partitions_->record_failure(partition_name);
            logger_.warn("partition_fault",
                         {{"partition", partition_name},
                          {"failures", std::to_string(state.failures)},
                          {"healthy", state.healthy ? "true" : "false"}});
        }
    }

    return frame_from_values(seconds_since_epoch(), updates);
}

LightweightInferenceModel::LightweightInferenceModel(double nominal_hum_hz, double temperature_coeff,
                                                     double humidity_coeff, double pressure_coeff,
                                                     double ac_hum_coeff, double audio_coeff,
                                                     double radiation_coeff, double radio_coeff,
                                                     double gps_lock_penalty)
    : nominal_hum_hz_(nominal_hum_hz),
      temperature_coeff_(temperature_coeff),
      humidity_coeff_(humidity_coeff),
      pressure_coeff_(pressure_coeff),
      ac_hum_coeff_(ac_hum_coeff),
      audio_coeff_(audio_coeff),
      radiation_coeff_(radiation_coeff),
      radio_coeff_(radio_coeff),
      gps_lock_penalty_(gps_lock_penalty) {}

double LightweightInferenceModel::adjusted_variance(double base_variance, const SensorFrame& frame,
                                                    const std::string& reference_name) {
    if (base_variance <= 0.0) {
        throw std::runtime_error("base_variance must be positive");
    }
    double penalty = 1.0;
    if (frame.ac_hum_hz.has_value()) {
        const double deviation = std::abs(frame.ac_hum_hz.value() - nominal_hum_hz_) / nominal_hum_hz_;
        penalty += deviation * ac_hum_coeff_;
    }
    if (frame.temperature_c.has_value() && last_frame_.has_value()) {
        const double previous = last_frame_->temperature_c.value_or(frame.temperature_c.value());
        penalty += std::abs(frame.temperature_c.value() - previous) * temperature_coeff_;
    }
    if (frame.humidity_pct.has_value()) {
        penalty += std::abs(frame.humidity_pct.value()) * humidity_coeff_ / 100.0;
    }
    if (frame.pressure_hpa.has_value()) {
        penalty += std::abs(frame.pressure_hpa.value() - 1013.25) * pressure_coeff_ / 100.0;
    }
    if (frame.ambient_audio_db.has_value()) {
        penalty += frame.ambient_audio_db.value() * audio_coeff_ / 100.0;
    }
    if (frame.geiger_cpm.has_value()) {
        penalty += frame.geiger_cpm.value() * radiation_coeff_ / 100.0;
    }
    if (frame.radio_snr_db.has_value()) {
        penalty += std::max(0.0, 30.0 - frame.radio_snr_db.value()) * radio_coeff_ / 30.0;
    }
    if (frame.gps_lock.has_value() && !frame.gps_lock.value()) {
        penalty *= gps_lock_penalty_;
    }
    if (!reference_name.empty()) {
        penalty *= 1.0;
    }

    return base_variance * penalty;
}

double LightweightInferenceModel::drift_hint(const SensorFrame& frame) {
    double hint = 0.0;
    if (frame.temperature_c.has_value() && last_frame_.has_value()) {
        const double previous = last_frame_->temperature_c.value_or(frame.temperature_c.value());
        hint += (frame.temperature_c.value() - previous) * 1e-9;
    }
    if (frame.humidity_pct.has_value()) {
        hint += (frame.humidity_pct.value() - 50.0) * 5e-10;
    }
    if (frame.pressure_hpa.has_value()) {
        hint += (frame.pressure_hpa.value() - 1013.25) * 1e-10;
    }
    return hint;
}

void LightweightInferenceModel::record_frame(const SensorFrame& frame) {
    last_frame_ = frame;
}

void LightweightInferenceModel::update(const SensorFrame&, const std::vector<Measurement>&, double,
                                       std::optional<double>) {}

LinearInferenceModel::LinearInferenceModel(std::map<std::string, double> feature_weights,
                                           std::map<std::string, double> drift_weights,
                                           std::map<std::string, double> reference_bias,
                                           double bias)
    : feature_weights_(std::move(feature_weights)),
      drift_weights_(std::move(drift_weights)),
      reference_bias_(std::move(reference_bias)),
      bias_(bias) {}

double LinearInferenceModel::adjusted_variance(double base_variance, const SensorFrame& frame,
                                               const std::string& reference_name) {
    if (base_variance <= 0.0) {
        throw std::runtime_error("base_variance must be positive");
    }
    double score = bias_ + reference_bias_[reference_name];
    for (const auto& [feature, weight] : feature_weights_) {
        score += weight * feature_value(frame, feature);
    }
    const double scale = 1.0 + 1.0 / (1.0 + std::exp(-score));
    return base_variance * scale;
}

double LinearInferenceModel::drift_hint(const SensorFrame& frame) {
    double hint = 0.0;
    for (const auto& [feature, weight] : drift_weights_) {
        hint += weight * feature_value(frame, feature);
    }
    return hint;
}

void LinearInferenceModel::record_frame(const SensorFrame& frame) {
    last_frame_ = frame;
}

void LinearInferenceModel::update(const SensorFrame&, const std::vector<Measurement>&, double,
                                  std::optional<double>) {}

MlVarianceModel::MlVarianceModel(std::map<std::string, double> feature_weights,
                                 std::map<std::string, double> reference_bias, double bias,
                                 double learning_rate, double min_scale, double max_scale,
                                 std::shared_ptr<SensorCharacterization> characterization)
    : feature_weights_(std::move(feature_weights)),
      reference_bias_(std::move(reference_bias)),
      bias_(bias),
      learning_rate_(learning_rate),
      min_scale_(min_scale),
      max_scale_(max_scale),
      characterization_(characterization ? std::move(characterization)
                                         : std::make_shared<SensorCharacterization>()) {}

double MlVarianceModel::adjusted_variance(double base_variance, const SensorFrame& frame,
                                          const std::string& reference_name) {
    if (base_variance <= 0.0) {
        throw std::runtime_error("base_variance must be positive");
    }
    double score = bias_ + reference_bias_[reference_name];
    for (const auto& [feature, weight] : feature_weights_) {
        score += weight * feature_value(frame, feature);
    }
    double scale = 1.0 + std::tanh(score);
    scale = std::clamp(scale, min_scale_, max_scale_);
    return base_variance * scale;
}

double MlVarianceModel::drift_hint(const SensorFrame& frame) {
    double hint = 0.0;
    if (last_frame_.has_value() && frame.temperature_c.has_value()) {
        const double previous = last_frame_->temperature_c.value_or(frame.temperature_c.value());
        hint += (frame.temperature_c.value() - previous) * 5e-10;
    }
    return hint;
}

void MlVarianceModel::record_frame(const SensorFrame& frame) {
    last_frame_ = frame;
}

void MlVarianceModel::update(const SensorFrame& frame, const std::vector<Measurement>& measurements,
                             double fused_offset, std::optional<double> ground_truth_offset) {
    (void)frame;
    const double target_offset = ground_truth_offset.value_or(fused_offset);
    for (const auto& measurement : measurements) {
        const double residual = measurement.offset - target_offset;
        const double expected = std::sqrt(std::max(measurement.variance, 1e-12));
        const double error = (std::abs(residual) - expected) / std::max(expected, 1e-6);
        characterization_->update(measurement.name, residual);
        if (std::abs(error) > 3.0 || std::abs(characterization_->z_score(measurement.name, residual)) > 3.0) {
            continue;
        }
        const double step = std::clamp(learning_rate_ * error, -learning_rate_, learning_rate_);
        bias_ += step;
        reference_bias_[measurement.name] += step;
    }
}

SlewDriftDetector::SlewDriftDetector(int window) : window_(window) {
    if (window_ < 2) {
        throw std::runtime_error("window must be >= 2");
    }
}

SlewDriftEstimate SlewDriftDetector::update(double timestamp, double offset) {
    history_.push_back({timestamp, offset});
    if (static_cast<int>(history_.size()) > window_) {
        history_.erase(history_.begin());
    }
    if (history_.size() < 2) {
        return SlewDriftEstimate{0.0, 0.0, static_cast<int>(history_.size())};
    }

    double t_mean = 0.0;
    double o_mean = 0.0;
    for (const auto& [t, o] : history_) {
        t_mean += t;
        o_mean += o;
    }
    t_mean /= history_.size();
    o_mean /= history_.size();

    double numerator = 0.0;
    double denominator = 0.0;
    for (const auto& [t, o] : history_) {
        numerator += (t - t_mean) * (o - o_mean);
        denominator += (t - t_mean) * (t - t_mean);
    }
    if (denominator == 0.0) {
        denominator = 1.0;
    }
    const double drift = numerator / denominator;
    const double slew = history_.back().second - history_.front().second;
    return SlewDriftEstimate{slew, drift, static_cast<int>(history_.size())};
}

RateLimiter::RateLimiter(double max_samples_per_sec) : max_samples_per_sec_(max_samples_per_sec) {}

bool RateLimiter::allow() {
    const double now = seconds_since_epoch();
    if (max_samples_per_sec_ <= 0.0) {
        last_sample_ = now;
        return true;
    }
    if (last_sample_.has_value() && now - last_sample_.value() < 1.0 / max_samples_per_sec_) {
        return false;
    }
    last_sample_ = now;
    return true;
}

SecurityMonitor::SecurityMonitor(double divergence_threshold, std::optional<double> grid_frequency,
                                 double grid_tolerance_hz, double max_measurement_rate, Logger logger,
                                 std::shared_ptr<SafetyCase> safety_case)
    : divergence_threshold_(divergence_threshold),
      grid_frequency_(grid_frequency),
      grid_tolerance_hz_(grid_tolerance_hz),
      rate_limit_(max_measurement_rate),
      logger_(std::move(logger)),
      safety_(std::move(safety_case)) {}

std::vector<SecurityAlert> SecurityMonitor::evaluate_frame(const SensorFrame& frame) {
    std::vector<SecurityAlert> alerts;
    if (frame.ac_hum_hz.has_value() && grid_frequency_.has_value()) {
        if (std::abs(frame.ac_hum_hz.value() - grid_frequency_.value()) > grid_tolerance_hz_) {
            SecurityAlert alert{"grid_frequency_out_of_range",
                                "AC hum " + std::to_string(frame.ac_hum_hz.value()) +
                                    " Hz outside expected grid",
                                "warn"};
            logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
            if (safety_) {
                safety_->record("AC_HUM_INJECTION", alert.detail, seconds_since_epoch());
            }
            alerts.push_back(alert);
        }
    }
    return alerts;
}

std::pair<Measurement, std::vector<SecurityAlert>> SecurityMonitor::evaluate_measurement(
    const Measurement& measurement, const SensorFrame& frame) {
    std::vector<SecurityAlert> alerts;
    Measurement adjusted = measurement;

    if (!rate_limit_.allow()) {
        SecurityAlert alert{"rate_limited", measurement.name + " rate limited", "warn"};
        logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
        if (safety_) {
            safety_->record("SENSOR_FLOODING", alert.detail, seconds_since_epoch());
        }
        alerts.push_back(alert);
        adjusted.variance *= 10.0;
        return {adjusted, alerts};
    }

    for (const auto& [name, offset] : last_measurements_) {
        if (name == measurement.name) {
            continue;
        }
        if (std::abs(offset - measurement.offset) > divergence_threshold_) {
            SecurityAlert alert{"reference_divergence", measurement.name + " diverged from " + name, "warn"};
            logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
            if (safety_) {
                safety_->record("REFERENCE_DIVERGENCE", alert.detail, seconds_since_epoch());
            }
            alerts.push_back(alert);
        }
    }

    if (measurement.name.rfind("gps", 0) == 0 && frame.ac_hum_hz.has_value()) {
        if (std::abs(measurement.offset) > divergence_threshold_) {
            SecurityAlert alert{"gps_spoofing_suspected", "GPS offset spike detected", "warn"};
            logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
            if (safety_) {
                safety_->record("GPS_SPOOFING", alert.detail, seconds_since_epoch());
            }
            alerts.push_back(alert);
        }
    }

    last_measurements_[measurement.name] = measurement.offset;
    return {measurement, alerts};
}

TimeServer::TimeServer(VirtualClock clock, std::vector<std::shared_ptr<ReferenceInput>> references,
                       std::shared_ptr<SensorAggregator> sensors, std::shared_ptr<InferenceModel> inference,
                       std::shared_ptr<SlewDriftDetector> drift_detector,
                       std::optional<std::string> ground_truth_reference,
                       std::shared_ptr<SecurityMonitor> security_monitor, Logger logger,
                       std::shared_ptr<SafetyCase> safety_case,
                       std::shared_ptr<PartitionSupervisor> partition_supervisor)
    : clock_(std::move(clock)),
      references_(std::move(references)),
      sensors_(sensors ? std::move(sensors)
                       : std::make_shared<SensorAggregator>(std::vector<SensorInput*>{}, nullptr, logger,
                                                            partition_supervisor)),
      inference_(inference ? std::move(inference) : std::make_shared<LightweightInferenceModel>()),
      drift_detector_(drift_detector ? std::move(drift_detector) : std::make_shared<SlewDriftDetector>()),
      ground_truth_reference_(std::move(ground_truth_reference)),
      safety_(std::move(safety_case)),
      security_(security_monitor ? std::move(security_monitor)
                                 : std::make_shared<SecurityMonitor>(0.005, 60.0, 0.5, 5.0, logger, safety_)),
      logger_(std::move(logger)) {}

TimeServerStep TimeServer::step_with_context(double dt) {
    std::lock_guard<std::mutex> guard(lock_);
    const SensorFrame frame = sensors_->sample();
    std::vector<Measurement> measurements;
    std::optional<double> ground_truth_offset;

    for (const auto& alert : security_->evaluate_frame(frame)) {
        logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
    }

    for (const auto& reference : references_) {
        try {
            Measurement measurement = reference->sample(frame);
            auto [evaluated, alerts] = security_->evaluate_measurement(measurement, frame);
            for (const auto& alert : alerts) {
                logger_.warn("security_alert", {{"code", alert.code}, {"detail", alert.detail}});
            }
            const double adjusted_variance = inference_->adjusted_variance(evaluated.variance, frame, evaluated.name);
            Measurement enriched = evaluated;
            enriched.variance = adjusted_variance;
            enriched.has_quality = true;
            enriched.quality = 1.0 / (1.0 + adjusted_variance);
            if (ground_truth_reference_.has_value() && evaluated.name == ground_truth_reference_.value()) {
                ground_truth_offset = evaluated.offset;
            }
            measurements.push_back(enriched);
        } catch (const std::exception& exc) {
            logger_.error("reference_sample_failed", {{"reference", "Reference"}, {"error", exc.what()}});
        }
    }

    if (measurements.empty()) {
        throw std::runtime_error("No valid measurements available after sampling references");
    }

    ClockUpdate update = clock_.step(dt, measurements);
    SlewDriftEstimate drift_estimate = drift_detector_->update(frame.timestamp, update.state.offset);
    const double drift_hint = inference_->drift_hint(frame);
    inference_->record_frame(frame);
    inference_->update(frame, measurements, update.fused_offset, ground_truth_offset);

    return TimeServerStep{update, frame, drift_estimate, drift_hint, measurements, safety_};
}

std::tuple<ClockUpdate, SensorFrame, SlewDriftEstimate, double> TimeServer::step(double dt) {
    auto result = step_with_context(dt);
    return {result.update, result.frame, result.drift, result.drift_hint};
}

const ClockState& TimeServer::state() const {
    return clock_.state();
}

NtpReference::NtpReference(std::string name, std::string host, int port, double base_variance, double timeout)
    : name_(std::move(name)), host_(std::move(host)), port_(port), base_variance_(base_variance), timeout_(timeout) {}

Measurement NtpReference::sample(const SensorFrame&) {
    const double t1 = seconds_since_epoch();
    uint8_t packet[48] = {0};
    packet[0] = 0x1b;

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Unable to create NTP socket");
    }

    timeval tv{};
    tv.tv_sec = static_cast<int>(timeout_);
    tv.tv_usec = static_cast<int>((timeout_ - tv.tv_sec) * 1e6);
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
        ::close(sock);
        throw std::runtime_error("Invalid NTP host address");
    }

    if (::sendto(sock, packet, sizeof(packet), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sock);
        throw std::runtime_error("Failed to send NTP request");
    }

    uint8_t buffer[48];
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    const ssize_t received = ::recvfrom(sock, buffer, sizeof(buffer), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
    const double t4 = seconds_since_epoch();
    ::close(sock);

    if (received < 48) {
        throw std::runtime_error("Invalid NTP response length");
    }

    const double t2 = ntp_to_unix(buffer + 32);
    const double t3 = ntp_to_unix(buffer + 40);
    const double offset = ((t2 - t1) + (t3 - t4)) / 2.0;
    return Measurement{name_, offset, base_variance_, false, 1.0};
}

RtcReference::RtcReference(std::string name, double base_variance)
    : name_(std::move(name)), base_variance_(base_variance), hwclock_path_(locate_hwclock()) {}

Measurement RtcReference::sample(const SensorFrame&) {
    std::string command = hwclock_path_ + " --get --utc --format '%Y-%m-%d %H:%M:%S'";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to run hwclock");
    }
    char buffer[128];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    pclose(pipe);
    output = trim(output);

    std::tm tm{};
    std::istringstream stream(output);
    stream >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (stream.fail()) {
        throw std::runtime_error("Failed to parse hwclock output");
    }

    const double rtc_timestamp = timegm(&tm);
    const double system_timestamp = seconds_since_epoch();
    return Measurement{name_, rtc_timestamp - system_timestamp, base_variance_, false, 1.0};
}

NmeaGpsReference::NmeaGpsReference(std::string name, std::function<std::string()> line_source,
                                   double base_variance)
    : name_(std::move(name)), line_source_(std::move(line_source)), base_variance_(base_variance) {}

Measurement NmeaGpsReference::sample(const SensorFrame&) {
    for (int i = 0; i < 5; ++i) {
        const std::string line = trim(line_source_());
        if (line.rfind("$GPRMC", 0) == 0 || line.rfind("$GNRMC", 0) == 0) {
            return parse_rmc(name_, line, base_variance_);
        }
    }
    throw std::runtime_error("No valid GPS NMEA sentences found");
}

SerialLineSensor::SerialLineSensor(std::function<SensorValues(const std::string&)> parser,
                                   std::function<std::string()> line_source)
    : parser_(std::move(parser)), line_source_(std::move(line_source)) {}

SensorValues SerialLineSensor::sample() {
    return parser_(line_source_());
}

EnvironmentalSensor::EnvironmentalSensor(
    std::function<std::tuple<std::optional<double>, std::optional<double>, std::optional<double>>()> reader)
    : reader_(std::move(reader)) {}

SensorValues EnvironmentalSensor::sample() {
    auto [temperature_c, humidity_pct, pressure_hpa] = reader_();
    return {
        {"temperature_c", temperature_c.has_value() ? SensorValue{temperature_c.value()} : SensorValue{}},
        {"humidity_pct", humidity_pct.has_value() ? SensorValue{humidity_pct.value()} : SensorValue{}},
        {"pressure_hpa", pressure_hpa.has_value() ? SensorValue{pressure_hpa.value()} : SensorValue{}},
    };
}

I2CEnvironmentalSensor::I2CEnvironmentalSensor(
    std::function<std::tuple<std::optional<double>, std::optional<double>, std::optional<double>>(int, int)> reader,
    int bus, int address)
    : reader_(std::move(reader)), bus_(bus), address_(address) {}

SensorValues I2CEnvironmentalSensor::sample() {
    auto [temperature_c, humidity_pct, pressure_hpa] = reader_(bus_, address_);
    return {
        {"temperature_c", temperature_c.has_value() ? SensorValue{temperature_c.value()} : SensorValue{}},
        {"humidity_pct", humidity_pct.has_value() ? SensorValue{humidity_pct.value()} : SensorValue{}},
        {"pressure_hpa", pressure_hpa.has_value() ? SensorValue{pressure_hpa.value()} : SensorValue{}},
    };
}

GeigerCounterSensor::GeigerCounterSensor(std::function<std::optional<double>()> reader)
    : reader_(std::move(reader)) {}

SensorValues GeigerCounterSensor::sample() {
    auto value = reader_();
    return {{"geiger_cpm", value.has_value() ? SensorValue{value.value()} : SensorValue{}}};
}

RadioSnrSensor::RadioSnrSensor(std::function<std::optional<double>()> reader) : reader_(std::move(reader)) {}

SensorValues RadioSnrSensor::sample() {
    auto value = reader_();
    return {{"radio_snr_db", value.has_value() ? SensorValue{value.value()} : SensorValue{}}};
}

GpsLockSensor::GpsLockSensor(std::function<std::optional<bool>()> reader) : reader_(std::move(reader)) {}

SensorValues GpsLockSensor::sample() {
    auto value = reader_();
    return {{"gps_lock", value.has_value() ? SensorValue{value.value()} : SensorValue{}}};
}

SerialReference::SerialReference(std::function<Measurement(const std::string&)> parser,
                                 std::function<std::string()> line_source)
    : parser_(std::move(parser)), line_source_(std::move(line_source)) {}

Measurement SerialReference::sample(const SensorFrame&) {
    return parser_(line_source_());
}

GpioPulseSensor::GpioPulseSensor(std::function<int()> read_level, std::string field_name, double min_period)
    : read_level_(std::move(read_level)), field_name_(std::move(field_name)), min_period_(min_period) {}

SensorValues GpioPulseSensor::sample() {
    const int level = read_level_();
    const double now = seconds_since_epoch();
    std::optional<double> frequency;
    if (last_level_.has_value() && level != last_level_.value()) {
        if (last_edge_.has_value()) {
            const double period = now - last_edge_.value();
            if (period >= min_period_) {
                frequency = 1.0 / period;
            }
        }
        last_edge_ = now;
    }
    last_level_ = level;
    return {{field_name_, frequency.has_value() ? SensorValue{frequency.value()} : SensorValue{}}};
}

AudioFeatureSensor::AudioFeatureSensor(AudioSampleSource& sample_source, std::vector<double> hum_candidates,
                                       std::pair<double, double> bird_band,
                                       std::pair<double, double> traffic_band)
    : source_(sample_source),
      hum_candidates_(std::move(hum_candidates)),
      bird_band_(bird_band),
      traffic_band_(traffic_band) {}

SensorValues AudioFeatureSensor::sample() {
    auto [samples, sample_rate] = source_.sample();
    if (samples.empty()) {
        return {
            {"ambient_audio_db", SensorValue{}},
            {"ac_hum_hz", SensorValue{}},
            {"ac_hum_phase_rad", SensorValue{}},
            {"ac_hum_uncertainty", SensorValue{}},
            {"bird_activity", SensorValue{}},
            {"traffic_activity", SensorValue{}},
        };
    }

    double sum_sq = 0.0;
    for (double sample : samples) {
        sum_sq += sample * sample;
    }
    const double rms = std::sqrt(sum_sq / samples.size());
    const double ambient_db = 20.0 * std::log10(std::max(rms, 1e-12));

    auto [ac_hum_hz, ac_hum_phase, ac_hum_uncertainty] =
        estimate_ac_hum(samples, sample_rate, hum_candidates_);
    const double bird_energy = band_energy(samples, sample_rate, bird_band_);
    const double traffic_energy = band_energy(samples, sample_rate, traffic_band_);

    return {
        {"ambient_audio_db", ambient_db},
        {"ac_hum_hz", ac_hum_hz.has_value() ? SensorValue{ac_hum_hz.value()} : SensorValue{}},
        {"ac_hum_phase_rad", ac_hum_phase.has_value() ? SensorValue{ac_hum_phase.value()} : SensorValue{}},
        {"ac_hum_uncertainty",
         ac_hum_uncertainty.has_value() ? SensorValue{ac_hum_uncertainty.value()} : SensorValue{}},
        {"bird_activity", bird_energy},
        {"traffic_activity", traffic_energy},
    };
}

SensorFrame frame_from_values(double timestamp, const SensorValues& values) {
    SensorFrame frame{};
    frame.timestamp = timestamp;
    frame.temperature_c = get_optional_double(values, "temperature_c");
    frame.humidity_pct = get_optional_double(values, "humidity_pct");
    frame.pressure_hpa = get_optional_double(values, "pressure_hpa");
    frame.ac_hum_hz = get_optional_double(values, "ac_hum_hz");
    frame.ac_hum_phase_rad = get_optional_double(values, "ac_hum_phase_rad");
    frame.ac_hum_uncertainty = get_optional_double(values, "ac_hum_uncertainty");
    frame.radio_snr_db = get_optional_double(values, "radio_snr_db");
    frame.geiger_cpm = get_optional_double(values, "geiger_cpm");
    frame.ambient_audio_db = get_optional_double(values, "ambient_audio_db");
    frame.bird_activity = get_optional_double(values, "bird_activity");
    frame.traffic_activity = get_optional_double(values, "traffic_activity");
    frame.gps_lock = get_optional_bool(values, "gps_lock");
    return frame;
}

std::function<std::string()> open_line_source(const std::string& path) {
    auto stream = std::make_shared<std::ifstream>(path);
    if (!stream->is_open()) {
        throw std::runtime_error("Unable to open line source: " + path);
    }
    return [stream]() {
        std::string line;
        std::getline(*stream, line);
        return line;
    };
}

double feature_value(const SensorFrame& frame, const std::string& name) {
    if (name == "temperature_c" && frame.temperature_c.has_value()) {
        return frame.temperature_c.value();
    }
    if (name == "humidity_pct" && frame.humidity_pct.has_value()) {
        return frame.humidity_pct.value();
    }
    if (name == "pressure_hpa" && frame.pressure_hpa.has_value()) {
        return frame.pressure_hpa.value();
    }
    if (name == "ac_hum_hz" && frame.ac_hum_hz.has_value()) {
        return frame.ac_hum_hz.value();
    }
    if (name == "ac_hum_phase_rad" && frame.ac_hum_phase_rad.has_value()) {
        return frame.ac_hum_phase_rad.value();
    }
    if (name == "ac_hum_uncertainty" && frame.ac_hum_uncertainty.has_value()) {
        return frame.ac_hum_uncertainty.value();
    }
    if (name == "radio_snr_db" && frame.radio_snr_db.has_value()) {
        return frame.radio_snr_db.value();
    }
    if (name == "geiger_cpm" && frame.geiger_cpm.has_value()) {
        return frame.geiger_cpm.value();
    }
    if (name == "ambient_audio_db" && frame.ambient_audio_db.has_value()) {
        return frame.ambient_audio_db.value();
    }
    if (name == "bird_activity" && frame.bird_activity.has_value()) {
        return frame.bird_activity.value();
    }
    if (name == "traffic_activity" && frame.traffic_activity.has_value()) {
        return frame.traffic_activity.value();
    }
    if (name == "gps_lock" && frame.gps_lock.has_value()) {
        return frame.gps_lock.value() ? 1.0 : 0.0;
    }
    return 0.0;
}

}  // namespace aimrt
