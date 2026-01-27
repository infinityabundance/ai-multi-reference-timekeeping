#ifndef AIMRT_CHRONY_HPP
#define AIMRT_CHRONY_HPP

#include <optional>
#include <string>

namespace aimrt {

struct ChronyShmSample {
    double offset = 0.0;
    double delay = 0.0;
    int leap = 0;
    int status = 0;
    int mode = 1;
};

class ChronyShmWriter {
public:
    explicit ChronyShmWriter(std::string path = "/dev/shm/chrony_shm0");

    void write(const ChronyShmSample& sample);
    std::optional<double> last_write() const;
    void close();

private:
    void* ensure_mmap();

    std::string path_;
    size_t size_ = 100;
    int fd_ = -1;
    void* mmap_ = nullptr;
    std::optional<double> last_write_;
};

}  // namespace aimrt

#endif  // AIMRT_CHRONY_HPP
