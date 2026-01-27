#include "aimrt/chrony.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdint>
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

    auto crc32 = [](const uint8_t* data, size_t length) {
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < length; ++i) {
            crc ^= data[i];
            for (int bit = 0; bit < 8; ++bit) {
                const uint32_t mask = -(crc & 1u);
                crc = (crc >> 1) ^ (0xEDB88320u & mask);
            }
        }
        return ~crc;
    };

    auto encode_be_double = [](double value, std::array<uint8_t, 8>& out) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            out[i] = static_cast<uint8_t>((bits >> (56 - i * 8)) & 0xFFu);
        }
    };

    std::array<uint8_t, 16> crc_payload{};
    std::array<uint8_t, 8> offset_bytes{};
    std::array<uint8_t, 8> delay_bytes{};
    encode_be_double(sample.offset, offset_bytes);
    encode_be_double(sample.delay, delay_bytes);
    std::memcpy(crc_payload.data(), offset_bytes.data(), offset_bytes.size());
    std::memcpy(crc_payload.data() + offset_bytes.size(), delay_bytes.data(), delay_bytes.size());
    const uint32_t checksum = crc32(crc_payload.data(), crc_payload.size());

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
    payload.checksum = checksum;

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
