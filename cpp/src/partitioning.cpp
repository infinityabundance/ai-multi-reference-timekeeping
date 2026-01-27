#include "aimrt/partitioning.hpp"

#include "aimrt/common.hpp"

namespace aimrt {

PartitionSupervisor::PartitionSupervisor(int max_failures, double reboot_delay)
    : max_failures_(max_failures), reboot_delay_(reboot_delay) {}

PartitionState& PartitionSupervisor::state(const std::string& name) {
    auto it = partitions_.find(name);
    if (it == partitions_.end()) {
        it = partitions_.emplace(name, PartitionState{name}).first;
    }
    return it->second;
}

PartitionState& PartitionSupervisor::record_failure(const std::string& name) {
    auto& current = state(name);
    current.failures += 1;
    current.last_failure = seconds_since_epoch();
    if (current.failures >= max_failures_) {
        current.healthy = false;
        current.reboot_at = seconds_since_epoch() + reboot_delay_;
    }
    return current;
}

PartitionState& PartitionSupervisor::record_success(const std::string& name) {
    auto& current = state(name);
    current.failures = 0;
    current.last_failure = std::nullopt;
    current.healthy = true;
    current.reboot_at = std::nullopt;
    return current;
}

bool PartitionSupervisor::should_reboot(const std::string& name) {
    auto& current = state(name);
    if (!current.reboot_at.has_value()) {
        return false;
    }
    return seconds_since_epoch() >= current.reboot_at.value();
}

PartitionState& PartitionSupervisor::reboot(const std::string& name) {
    auto& current = state(name);
    current.failures = 0;
    current.last_failure = std::nullopt;
    current.healthy = true;
    current.reboot_at = std::nullopt;
    return current;
}

}  // namespace aimrt
