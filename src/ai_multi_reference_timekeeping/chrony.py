"""Chrony integration helpers for SHM reference clocks."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import struct
import time


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
        packed = struct.pack(
            "!IIIddddiiii",
            0,  # mode/status placeholders
            1,  # count
            0,  # valid flag
            seconds,
            nanos,
            sample.offset,
            sample.delay,
            0.0,
            sample.leap,
            sample.status,
            sample.mode,
            0,
        )
        shm.seek(0)
        shm.write(packed[: self._size])

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
