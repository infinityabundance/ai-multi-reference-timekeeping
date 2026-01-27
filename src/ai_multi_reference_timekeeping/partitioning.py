"""Partitioning and fault containment inspired by STANAG 4626.

This module provides lightweight time/space partitioning controls that isolate
sensor failures and allow controlled recovery. The goal is modularity and
fault containment, not full hypervisor-level isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class PartitionState:
    """Track partition health and reboot timing."""

    name: str
    failures: int = 0
    last_failure: float | None = None
    healthy: bool = True
    reboot_at: float | None = None


class PartitionSupervisor:
    """Supervise partitions and trigger reboots on repeated failures."""

    def __init__(self, max_failures: int = 3, reboot_delay: float = 5.0) -> None:
        self._max_failures = max_failures
        self._reboot_delay = reboot_delay
        self._partitions: dict[str, PartitionState] = {}

    def state(self, name: str) -> PartitionState:
        return self._partitions.setdefault(name, PartitionState(name=name))

    def record_failure(self, name: str) -> PartitionState:
        state = self.state(name)
        state.failures += 1
        state.last_failure = time.time()
        if state.failures >= self._max_failures:
            state.healthy = False
            state.reboot_at = time.time() + self._reboot_delay
        return state

    def record_success(self, name: str) -> PartitionState:
        state = self.state(name)
        state.failures = 0
        state.last_failure = None
        state.healthy = True
        state.reboot_at = None
        return state

    def should_reboot(self, name: str) -> bool:
        state = self.state(name)
        if state.reboot_at is None:
            return False
        return time.time() >= state.reboot_at

    def reboot(self, name: str) -> PartitionState:
        state = self.state(name)
        state.failures = 0
        state.last_failure = None
        state.healthy = True
        state.reboot_at = None
        return state
