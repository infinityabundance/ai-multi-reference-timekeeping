#ifndef AIMRT_PARTITIONING_HPP
#define AIMRT_PARTITIONING_HPP

#include <map>
#include <optional>
#include <string>

namespace aimrt {

struct PartitionState {
    std::string name;
    int failures = 0;
    std::optional<double> last_failure;
    bool healthy = true;
    std::optional<double> reboot_at;
};

class PartitionSupervisor {
public:
    explicit PartitionSupervisor(int max_failures = 3, double reboot_delay = 5.0);

    PartitionState& state(const std::string& name);
    PartitionState& record_failure(const std::string& name);
    PartitionState& record_success(const std::string& name);
    bool should_reboot(const std::string& name);
    PartitionState& reboot(const std::string& name);

private:
    int max_failures_ = 3;
    double reboot_delay_ = 5.0;
    std::map<std::string, PartitionState> partitions_;
};

}  // namespace aimrt

#endif  // AIMRT_PARTITIONING_HPP
