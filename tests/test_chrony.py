import mmap
import struct
import zlib

from ai_multi_reference_timekeeping.chrony import ChronyShmSample, ChronyShmWriter


def test_chrony_shm_writer_crc(tmp_path) -> None:
    shm_path = tmp_path / "chrony_shm0"
    writer = ChronyShmWriter(path=str(shm_path))
    sample = ChronyShmSample(offset=0.001, delay=0.0001)
    writer.write(sample)
    writer.close()

    with shm_path.open("r+b") as handle:
        mm = mmap.mmap(handle.fileno(), 0)
        data = mm.read(100)
        mm.close()

    fmt = "!IIIdddddiiiI"
    unpacked = struct.unpack(fmt, data[: struct.calcsize(fmt)])
    offset = unpacked[5]
    delay = unpacked[6]
    checksum = unpacked[-1]
    expected = zlib.crc32(struct.pack("!dd", offset, delay))
    assert checksum == expected
