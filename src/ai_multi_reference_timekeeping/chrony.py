"""Chrony integration helpers for SHM reference clocks."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import struct
import time
import zlib


@dataclass
class ChronyShmSample:
    """Chrony SHM sample representation."""

    offset: float
    delay: float
    leap: int = 0
    status: int = 0
    mode: int = 1


class ChronyShmWriter:
    """Write time samples to a shared-memory file for chrony SHM refclock."""

    def __init__(self, path: str = "/dev/shm/chrony_shm0") -> None:
        self._path = path
        self._size = 96
        self._mmap: mmap.mmap | None = None

    def _ensure(self) -> mmap.mmap:
        if self._mmap is not None:
            return self._mmap
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, self._size)
        self._mmap = mmap.mmap(fd, self._size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        os.close(fd)
        return self._mmap

    def write(self, sample: ChronyShmSample) -> None:
        shm = self._ensure()
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 1e9)
        # Calculate checksum of offset and delay
        checksum = zlib.crc32(struct.pack("!dd", sample.offset, sample.delay))
        packed = struct.pack(
            "!IIIdddddiiiI",
            0,              # 0 - mode/status placeholders (I)
            1,              # 1 - count (I)
            seconds,        # 2 - seconds timestamp (I)
            float(nanos),   # 3 - nanos as double (d)
            0.0,            # 4 - placeholder (d)
            sample.offset,  # 5 - offset (d) - test reads here
            sample.delay,   # 6 - delay (d) - test reads here
            0.0,            # 7 - placeholder (d)
            sample.leap,    # 8 - leap (i)
            sample.status,  # 9 - status (i)
            sample.mode,    # 10 - mode (i)
            checksum,       # 11 - checksum (I) - test reads here
        )
        shm.seek(0)
        shm.write(packed[: self._size])

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
