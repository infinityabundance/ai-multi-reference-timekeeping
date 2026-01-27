#include "aimrt/chrony.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "aimrt/common.hpp"

namespace aimrt {

ChronyShmWriter::ChronyShmWriter(std::string path) : path_(std::move(path)) {}

void* ChronyShmWriter::ensure_mmap() {
    if (mmap_) {
        return mmap_;
    }

    fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("Unable to open chrony shm file");
    }
    if (ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Unable to size chrony shm file");
    }
    mmap_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mmap_ == MAP_FAILED) {
        mmap_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Unable to mmap chrony shm file");
    }
    return mmap_;
}

void ChronyShmWriter::write(const ChronyShmSample& sample) {
    void* map = ensure_mmap();
    const double now = seconds_since_epoch();
    const uint32_t seconds = static_cast<uint32_t>(now);
    const uint32_t nanos = static_cast<uint32_t>((now - seconds) * 1e9);

    struct ChronyPayload {
        uint32_t mode;
        uint32_t count;
        uint32_t valid;
        double clock_time_sec;
        double clock_time_nsec;
        double offset;
        double delay;
        double dispersion;
        int32_t leap;
        int32_t status;
        int32_t mode2;
        uint32_t checksum;
    } payload{};

    payload.mode = 0;
    payload.count = 1;
    payload.valid = 0;
    payload.clock_time_sec = static_cast<double>(seconds);
    payload.clock_time_nsec = static_cast<double>(nanos);
    payload.offset = sample.offset;
    payload.delay = sample.delay;
    payload.dispersion = 0.0;
    payload.leap = sample.leap;
    payload.status = sample.status;
    payload.mode2 = sample.mode;
    payload.checksum = static_cast<uint32_t>(sample.offset * 1e6) ^ static_cast<uint32_t>(sample.delay * 1e6);

    std::memset(map, 0, size_);
    std::memcpy(map, &payload, sizeof(payload));
    last_write_ = seconds_since_epoch();
}

std::optional<double> ChronyShmWriter::last_write() const {
    return last_write_;
}

void ChronyShmWriter::close() {
    if (mmap_) {
        ::munmap(mmap_, size_);
        mmap_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace aimrt
