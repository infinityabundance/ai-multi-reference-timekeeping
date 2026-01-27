#ifndef AIMRT_TIME_SERVER_HPP
#define AIMRT_TIME_SERVER_HPP

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "aimrt/characterization.hpp"
#include "aimrt/fusion.hpp"
#include "aimrt/logging.hpp"
#include "aimrt/partitioning.hpp"
#include "aimrt/safety.hpp"

namespace aimrt {

struct SensorFrame {
    double timestamp = 0.0;
    std::optional<double> temperature_c;
    std::optional<double> humidity_pct;
    std::optional<double> pressure_hpa;
    std::optional<double> ac_hum_hz;
    std::optional<double> ac_hum_phase_rad;
    std::optional<double> ac_hum_uncertainty;
    std::optional<double> radio_snr_db;
    std::optional<double> geiger_cpm;
    std::optional<double> ambient_audio_db;
    std::optional<double> bird_activity;
    std::optional<double> traffic_activity;
    std::optional<bool> gps_lock;
};

using SensorValue = std::variant<std::monostate, double, bool>;
using SensorValues = std::map<std::string, SensorValue>;

class SensorInput {
public:
    virtual ~SensorInput() = default;
    virtual SensorValues sample() = 0;
};

class AudioSampleSource {
public:
    virtual ~AudioSampleSource() = default;
    virtual std::pair<std::vector<double>, int> sample() = 0;
};

class SensorValidator {
public:
    explicit SensorValidator(double max_samples_per_sec = 5.0,
                             std::map<std::string, std::pair<double, double>> ranges = {},
                             Logger logger = get_logger("SensorValidator"));

    SensorValues validate(const SensorValues& values);

private:
    double max_samples_per_sec_ = 5.0;
    std::map<std::string, std::pair<double, double>> ranges_;
    std::optional<double> last_sample_;
    Logger logger_;
};

class SensorAggregator {
public:
    explicit SensorAggregator(std::vector<SensorInput*> inputs,
                              std::shared_ptr<SensorValidator> validator = nullptr,
                              Logger logger = get_logger("SensorAggregator"),
                              std::shared_ptr<PartitionSupervisor> partitions = nullptr);

    SensorFrame sample();

private:
    std::vector<SensorInput*> inputs_;
    std::shared_ptr<SensorValidator> validator_;
    Logger logger_;
    std::shared_ptr<PartitionSupervisor> partitions_;
};

class ReferenceInput {
public:
    virtual ~ReferenceInput() = default;
    virtual Measurement sample(const SensorFrame& frame) = 0;
};

class InferenceModel {
public:
    virtual ~InferenceModel() = default;
    virtual double adjusted_variance(double base_variance, const SensorFrame& frame,
                                     const std::string& reference_name) = 0;
    virtual double drift_hint(const SensorFrame& frame) = 0;
    virtual void record_frame(const SensorFrame& frame) = 0;
    virtual void update(const SensorFrame& frame, const std::vector<Measurement>& measurements,
                        double fused_offset, std::optional<double> ground_truth_offset) = 0;
};

class LightweightInferenceModel : public InferenceModel {
public:
    explicit LightweightInferenceModel(double nominal_hum_hz = 60.0, double temperature_coeff = 0.01,
                                       double humidity_coeff = 0.005, double pressure_coeff = 0.002,
                                       double ac_hum_coeff = 0.1, double audio_coeff = 0.01,
                                       double radiation_coeff = 0.003, double radio_coeff = 0.004,
                                       double gps_lock_penalty = 2.0);

    double adjusted_variance(double base_variance, const SensorFrame& frame,
                             const std::string& reference_name) override;
    double drift_hint(const SensorFrame& frame) override;
    void record_frame(const SensorFrame& frame) override;
    void update(const SensorFrame& frame, const std::vector<Measurement>& measurements,
                double fused_offset, std::optional<double> ground_truth_offset) override;

private:
    double nominal_hum_hz_ = 60.0;
    double temperature_coeff_ = 0.01;
    double humidity_coeff_ = 0.005;
    double pressure_coeff_ = 0.002;
    double ac_hum_coeff_ = 0.1;
    double audio_coeff_ = 0.01;
    double radiation_coeff_ = 0.003;
    double radio_coeff_ = 0.004;
    double gps_lock_penalty_ = 2.0;
    std::optional<SensorFrame> last_frame_;
};

class LinearInferenceModel : public InferenceModel {
public:
    LinearInferenceModel(std::map<std::string, double> feature_weights = {},
                         std::map<std::string, double> drift_weights = {},
                         std::map<std::string, double> reference_bias = {},
                         double bias = 0.0);

    double adjusted_variance(double base_variance, const SensorFrame& frame,
                             const std::string& reference_name) override;
    double drift_hint(const SensorFrame& frame) override;
    void record_frame(const SensorFrame& frame) override;
    void update(const SensorFrame& frame, const std::vector<Measurement>& measurements,
                double fused_offset, std::optional<double> ground_truth_offset) override;

private:
    std::map<std::string, double> feature_weights_;
    std::map<std::string, double> drift_weights_;
    std::map<std::string, double> reference_bias_;
    double bias_ = 0.0;
    std::optional<SensorFrame> last_frame_;
};

class MlVarianceModel : public InferenceModel {
public:
    MlVarianceModel(std::map<std::string, double> feature_weights = {},
                    std::map<std::string, double> reference_bias = {}, double bias = 0.0,
                    double learning_rate = 1e-3, double min_scale = 0.5, double max_scale = 5.0,
                    std::shared_ptr<SensorCharacterization> characterization = nullptr);

    double adjusted_variance(double base_variance, const SensorFrame& frame,
                             const std::string& reference_name) override;
    double drift_hint(const SensorFrame& frame) override;
    void record_frame(const SensorFrame& frame) override;
    void update(const SensorFrame& frame, const std::vector<Measurement>& measurements,
                double fused_offset, std::optional<double> ground_truth_offset) override;

private:
    std::map<std::string, double> feature_weights_;
    std::map<std::string, double> reference_bias_;
    double bias_ = 0.0;
    double learning_rate_ = 1e-3;
    double min_scale_ = 0.5;
    double max_scale_ = 5.0;
    std::optional<SensorFrame> last_frame_;
    std::shared_ptr<SensorCharacterization> characterization_;
};

struct SlewDriftEstimate {
    double slew = 0.0;
    double drift = 0.0;
    int samples = 0;
};

class SlewDriftDetector {
public:
    explicit SlewDriftDetector(int window = 10);
    SlewDriftEstimate update(double timestamp, double offset);

private:
    int window_ = 10;
    std::vector<std::pair<double, double>> history_;
};

struct SecurityAlert {
    std::string code;
    std::string detail;
    std::string severity = "warn";
};

class RateLimiter {
public:
    explicit RateLimiter(double max_samples_per_sec);
    bool allow();

private:
    double max_samples_per_sec_ = 0.0;
    std::optional<double> last_sample_;
};

class SecurityMonitor {
public:
    SecurityMonitor(double divergence_threshold = 0.005,
                    std::optional<double> grid_frequency = 60.0,
                    double grid_tolerance_hz = 0.5,
                    double max_measurement_rate = 5.0,
                    Logger logger = get_logger("SecurityMonitor"),
                    std::shared_ptr<SafetyCase> safety_case = nullptr);

    std::vector<SecurityAlert> evaluate_frame(const SensorFrame& frame);
    std::pair<Measurement, std::vector<SecurityAlert>> evaluate_measurement(const Measurement& measurement,
                                                                            const SensorFrame& frame);

private:
    double divergence_threshold_ = 0.005;
    std::optional<double> grid_frequency_ = 60.0;
    double grid_tolerance_hz_ = 0.5;
    RateLimiter rate_limit_;
    std::map<std::string, double> last_measurements_;
    Logger logger_;
    std::shared_ptr<SafetyCase> safety_;
};

struct TimeServerStep {
    ClockUpdate update{};
    SensorFrame frame{};
    SlewDriftEstimate drift{};
    double drift_hint = 0.0;
    std::vector<Measurement> measurements;
    std::shared_ptr<SafetyCase> safety_case;
};

class TimeServer {
public:
    TimeServer(VirtualClock clock, std::vector<std::shared_ptr<ReferenceInput>> references,
               std::shared_ptr<SensorAggregator> sensors = nullptr,
               std::shared_ptr<InferenceModel> inference = nullptr,
               std::shared_ptr<SlewDriftDetector> drift_detector = nullptr,
               std::optional<std::string> ground_truth_reference = std::nullopt,
               std::shared_ptr<SecurityMonitor> security_monitor = nullptr,
               Logger logger = get_logger("TimeServer"),
               std::shared_ptr<SafetyCase> safety_case = nullptr,
               std::shared_ptr<PartitionSupervisor> partition_supervisor = nullptr);

    TimeServerStep step_with_context(double dt);
    std::tuple<ClockUpdate, SensorFrame, SlewDriftEstimate, double> step(double dt);
    const ClockState& state() const;

private:
    VirtualClock clock_;
    std::vector<std::shared_ptr<ReferenceInput>> references_;
    std::shared_ptr<SensorAggregator> sensors_;
    std::shared_ptr<InferenceModel> inference_;
    std::shared_ptr<SlewDriftDetector> drift_detector_;
    std::optional<std::string> ground_truth_reference_;
    std::shared_ptr<SafetyCase> safety_;
    std::shared_ptr<SecurityMonitor> security_;
    Logger logger_;
    std::mutex lock_;
};

class NtpReference : public ReferenceInput {
public:
    NtpReference(std::string name, std::string host = "time.nist.gov", int port = 123,
                 double base_variance = 1e-3, double timeout = 1.0);

    Measurement sample(const SensorFrame& frame) override;

private:
    std::string name_;
    std::string host_;
    int port_ = 123;
    double base_variance_ = 1e-3;
    double timeout_ = 1.0;
};

class RtcReference : public ReferenceInput {
public:
    explicit RtcReference(std::string name, double base_variance = 5e-3);

    Measurement sample(const SensorFrame& frame) override;

private:
    std::string name_;
    double base_variance_ = 5e-3;
    std::string hwclock_path_;
};

class NmeaGpsReference : public ReferenceInput {
public:
    NmeaGpsReference(std::string name, std::function<std::string()> line_source, double base_variance = 1e-4);

    Measurement sample(const SensorFrame& frame) override;

private:
    std::string name_;
    std::function<std::string()> line_source_;
    double base_variance_ = 1e-4;
};

class SerialLineSensor : public SensorInput {
public:
    SerialLineSensor(std::function<SensorValues(const std::string&)> parser,
                     std::function<std::string()> line_source);

    SensorValues sample() override;

private:
    std::function<SensorValues(const std::string&)> parser_;
    std::function<std::string()> line_source_;
};

class EnvironmentalSensor : public SensorInput {
public:
    explicit EnvironmentalSensor(std::function<std::tuple<std::optional<double>, std::optional<double>,
                                                         std::optional<double>>()> reader);

    SensorValues sample() override;

private:
    std::function<std::tuple<std::optional<double>, std::optional<double>, std::optional<double>>()> reader_;
};

class I2CEnvironmentalSensor : public SensorInput {
public:
    I2CEnvironmentalSensor(
        std::function<std::tuple<std::optional<double>, std::optional<double>, std::optional<double>>(int, int)>
            reader,
        int bus, int address);

    SensorValues sample() override;

private:
    std::function<std::tuple<std::optional<double>, std::optional<double>, std::optional<double>>(int, int)> reader_;
    int bus_ = 0;
    int address_ = 0;
};

class GeigerCounterSensor : public SensorInput {
public:
    explicit GeigerCounterSensor(std::function<std::optional<double>()> reader);
    SensorValues sample() override;

private:
    std::function<std::optional<double>()> reader_;
};

class RadioSnrSensor : public SensorInput {
public:
    explicit RadioSnrSensor(std::function<std::optional<double>()> reader);
    SensorValues sample() override;

private:
    std::function<std::optional<double>()> reader_;
};

class GpsLockSensor : public SensorInput {
public:
    explicit GpsLockSensor(std::function<std::optional<bool>()> reader);
    SensorValues sample() override;

private:
    std::function<std::optional<bool>()> reader_;
};

class SerialReference : public ReferenceInput {
public:
    SerialReference(std::function<Measurement(const std::string&)> parser, std::function<std::string()> line_source);

    Measurement sample(const SensorFrame& frame) override;

private:
    std::function<Measurement(const std::string&)> parser_;
    std::function<std::string()> line_source_;
};

class GpioPulseSensor : public SensorInput {
public:
    GpioPulseSensor(std::function<int()> read_level, std::string field_name, double min_period = 0.01);

    SensorValues sample() override;

private:
    std::function<int()> read_level_;
    std::string field_name_;
    double min_period_ = 0.01;
    std::optional<int> last_level_;
    std::optional<double> last_edge_;
};

class AudioFeatureSensor : public SensorInput {
public:
    AudioFeatureSensor(AudioSampleSource& sample_source,
                       std::vector<double> hum_candidates = {50.0, 60.0},
                       std::pair<double, double> bird_band = {2000.0, 8000.0},
                       std::pair<double, double> traffic_band = {50.0, 300.0});

    SensorValues sample() override;

private:
    AudioSampleSource& source_;
    std::vector<double> hum_candidates_;
    std::pair<double, double> bird_band_;
    std::pair<double, double> traffic_band_;
};

SensorFrame frame_from_values(double timestamp, const SensorValues& values);

std::function<std::string()> open_line_source(const std::string& path);

double feature_value(const SensorFrame& frame, const std::string& name);

}  // namespace aimrt

#endif  // AIMRT_TIME_SERVER_HPP
