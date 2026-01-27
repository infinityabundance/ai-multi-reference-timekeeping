#include "aimrt/observability.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

#include "aimrt/common.hpp"
#include "aimrt/metrics.hpp"

namespace aimrt {

HealthMonitor::HealthMonitor(double freshness_window) : freshness_window_(freshness_window) {}

void HealthMonitor::mark_update() {
    std::lock_guard<std::mutex> guard(mutex_);
    last_update_ = seconds_since_epoch();
}

void HealthMonitor::mark_shm_write() {
    std::lock_guard<std::mutex> guard(mutex_);
    shm_last_write_ = seconds_since_epoch();
}

HealthStatus HealthMonitor::status() const {
    std::lock_guard<std::mutex> guard(mutex_);
    const double now = seconds_since_epoch();
    const double last_update = last_update_.value_or(0.0);
    const bool ok = last_update_.has_value() && (now - last_update <= freshness_window_);
    return HealthStatus{last_update, shm_last_write_, ok};
}

MetricsTracker::MetricsTracker(int window_size) : window_size_(window_size) {}

void MetricsTracker::update(double offset) {
    offsets_.push_back(offset);
    if (static_cast<int>(offsets_.size()) > window_size_) {
        offsets_.erase(offsets_.begin());
    }
}

std::vector<std::pair<std::string, double>> MetricsTracker::metrics() const {
    if (offsets_.size() < 3) {
        return {};
    }
    std::vector<std::pair<std::string, double>> output;
    output.emplace_back("tdev_tau1", tdev(offsets_, 1));
    output.emplace_back("mtie_window2", mtie(offsets_, 2));
    return output;
}

MetricsExporter::MetricsExporter(MetricsTracker& tracker, HealthMonitor& health)
    : tracker_(tracker), health_(health) {}

MetricsExporter::~MetricsExporter() {
    stop();
}

void MetricsExporter::start(const std::string& host, int port) {
    if (running_.exchange(true)) {
        return;
    }
    thread_ = std::thread(&MetricsExporter::serve, this, host, port);
}

void MetricsExporter::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void MetricsExporter::serve(const std::string& host, int port) {
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        return;
    }

    int opt = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        return;
    }
    if (::listen(server_fd_, 4) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        return;
    }

    while (running_) {
        sockaddr_in client{};
        socklen_t len = sizeof(client);
        int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client), &len);
        if (client_fd < 0) {
            continue;
        }

        char buffer[1024] = {0};
        const ssize_t read_bytes = ::read(client_fd, buffer, sizeof(buffer) - 1);
        if (read_bytes <= 0) {
            ::close(client_fd);
            continue;
        }

        std::string request(buffer, static_cast<size_t>(read_bytes));
        auto first_line_end = request.find("\r\n");
        std::string first_line = first_line_end == std::string::npos ? request : request.substr(0, first_line_end);
        auto parts = split(first_line, ' ');
        std::string path = parts.size() >= 2 ? parts[1] : "/";

        std::string body;
        std::string status = "200 OK";
        std::string content_type = "text/plain";

        if (path == "/metrics") {
            std::ostringstream response;
            for (const auto& [key, value] : tracker_.metrics()) {
                response << key << " " << value << "\n";
            }
            body = response.str();
        } else if (path == "/health") {
            auto status_info = health_.status();
            std::ostringstream response;
            response << "{\"last_update\":" << status_info.last_update
                     << ",\"ok\":" << (status_info.ok ? "true" : "false");
            if (status_info.shm_last_write.has_value()) {
                response << ",\"shm_last_write\":" << status_info.shm_last_write.value();
            }
            response << "}";
            body = response.str();
            content_type = "application/json";
            if (!status_info.ok) {
                status = "503 Service Unavailable";
            }
        } else {
            status = "404 Not Found";
            body = "";
        }

        std::ostringstream header;
        header << "HTTP/1.1 " << status << "\r\n"
               << "Content-Type: " << content_type << "\r\n"
               << "Content-Length: " << body.size() << "\r\n\r\n";

        const std::string response = header.str() + body;
        ::send(client_fd, response.c_str(), response.size(), 0);
        ::close(client_fd);
    }

    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

}  // namespace aimrt
