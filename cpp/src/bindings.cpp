#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aimrt/api.hpp"
#include "aimrt/fusion.hpp"
#include "aimrt/kalman.hpp"
#include "aimrt/time_server.hpp"

namespace py = pybind11;

namespace {

class PyReferenceInput : public aimrt::ReferenceInput {
public:
    using aimrt::ReferenceInput::ReferenceInput;

    aimrt::Measurement sample(const aimrt::SensorFrame& frame) override {
        PYBIND11_OVERRIDE_PURE(aimrt::Measurement, aimrt::ReferenceInput, sample, frame);
    }
};

class PySensorInput : public aimrt::SensorInput {
public:
    using aimrt::SensorInput::SensorInput;

    aimrt::SensorValues sample() override {
        PYBIND11_OVERRIDE_PURE(aimrt::SensorValues, aimrt::SensorInput, sample);
    }
};

}  // namespace

PYBIND11_MODULE(aimrt_python, m) {
    m.doc() = "Pybind11 bindings for the AIMRT C++ core.";

    py::class_<aimrt::ClockState>(m, "ClockState")
        .def(py::init<>())
        .def_readwrite("offset", &aimrt::ClockState::offset)
        .def_readwrite("drift", &aimrt::ClockState::drift);

    py::class_<aimrt::ClockCovariance>(m, "ClockCovariance")
        .def(py::init<>())
        .def_readwrite("p00", &aimrt::ClockCovariance::p00)
        .def_readwrite("p01", &aimrt::ClockCovariance::p01)
        .def_readwrite("p10", &aimrt::ClockCovariance::p10)
        .def_readwrite("p11", &aimrt::ClockCovariance::p11)
        .def("as_matrix", &aimrt::ClockCovariance::as_matrix);

    py::class_<aimrt::ClockKalmanFilter>(m, "ClockKalmanFilter")
        .def(py::init<aimrt::ClockState, aimrt::ClockCovariance, double, double>())
        .def("predict", &aimrt::ClockKalmanFilter::predict)
        .def("update", &aimrt::ClockKalmanFilter::update)
        .def_property_readonly("state", &aimrt::ClockKalmanFilter::state)
        .def_property_readonly("covariance", &aimrt::ClockKalmanFilter::covariance);

    py::class_<aimrt::Measurement>(m, "Measurement")
        .def(py::init<>())
        .def_readwrite("name", &aimrt::Measurement::name)
        .def_readwrite("offset", &aimrt::Measurement::offset)
        .def_readwrite("variance", &aimrt::Measurement::variance)
        .def_readwrite("has_quality", &aimrt::Measurement::has_quality)
        .def_readwrite("quality", &aimrt::Measurement::quality);

    py::class_<aimrt::ClockUpdate>(m, "ClockUpdate")
        .def_readwrite("fused_offset", &aimrt::ClockUpdate::fused_offset)
        .def_readwrite("fused_variance", &aimrt::ClockUpdate::fused_variance)
        .def_readwrite("state", &aimrt::ClockUpdate::state)
        .def_readwrite("reference_weights", &aimrt::ClockUpdate::reference_weights);

    py::class_<aimrt::SensorFrame>(m, "SensorFrame")
        .def(py::init<>())
        .def_readwrite("timestamp", &aimrt::SensorFrame::timestamp)
        .def_readwrite("temperature_c", &aimrt::SensorFrame::temperature_c)
        .def_readwrite("humidity_pct", &aimrt::SensorFrame::humidity_pct)
        .def_readwrite("pressure_hpa", &aimrt::SensorFrame::pressure_hpa)
        .def_readwrite("ac_hum_hz", &aimrt::SensorFrame::ac_hum_hz)
        .def_readwrite("ac_hum_phase_rad", &aimrt::SensorFrame::ac_hum_phase_rad)
        .def_readwrite("ac_hum_uncertainty", &aimrt::SensorFrame::ac_hum_uncertainty)
        .def_readwrite("radio_snr_db", &aimrt::SensorFrame::radio_snr_db)
        .def_readwrite("geiger_cpm", &aimrt::SensorFrame::geiger_cpm)
        .def_readwrite("ambient_audio_db", &aimrt::SensorFrame::ambient_audio_db)
        .def_readwrite("bird_activity", &aimrt::SensorFrame::bird_activity)
        .def_readwrite("traffic_activity", &aimrt::SensorFrame::traffic_activity)
        .def_readwrite("gps_lock", &aimrt::SensorFrame::gps_lock);

    py::class_<aimrt::SlewDriftEstimate>(m, "SlewDriftEstimate")
        .def_readwrite("slew", &aimrt::SlewDriftEstimate::slew)
        .def_readwrite("drift", &aimrt::SlewDriftEstimate::drift)
        .def_readwrite("samples", &aimrt::SlewDriftEstimate::samples);

    py::class_<aimrt::ReferenceInput, PyReferenceInput, std::shared_ptr<aimrt::ReferenceInput>>(m, "ReferenceInput")
        .def(py::init<>());

    py::class_<aimrt::SensorInput, PySensorInput, std::shared_ptr<aimrt::SensorInput>>(m, "SensorInput")
        .def(py::init<>());

    py::class_<aimrt::TimeServer, std::shared_ptr<aimrt::TimeServer>>(m, "TimeServer")
        .def("step", &aimrt::TimeServer::step)
        .def_property_readonly("state", &aimrt::TimeServer::state);

    py::class_<aimrt::TimeServerRuntime>(m, "TimeServerRuntime")
        .def_readonly("server", &aimrt::TimeServerRuntime::server)
        .def_readonly("health", &aimrt::TimeServerRuntime::health)
        .def_readonly("metrics", &aimrt::TimeServerRuntime::metrics)
        .def_readonly("exporter", &aimrt::TimeServerRuntime::exporter);

    m.def("build_time_server_py",
          [](const std::vector<std::shared_ptr<aimrt::ReferenceInput>>& references,
             const std::vector<std::shared_ptr<aimrt::SensorInput>>& sensors,
             std::shared_ptr<aimrt::TimeServerSettings> settings) {
              return aimrt::build_time_server(references, sensors, nullptr, std::move(settings));
          },
          py::arg("references"), py::arg("sensors"),
          py::arg("settings") = nullptr);
}
