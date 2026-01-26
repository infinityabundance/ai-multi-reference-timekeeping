from ai_multi_reference_timekeeping.partitioning import PartitionSupervisor


def test_partition_reboot_cycle() -> None:
    supervisor = PartitionSupervisor(max_failures=1, reboot_delay=0.0)
    state = supervisor.record_failure("sensor")
    assert not state.healthy
    assert supervisor.should_reboot("sensor")
    supervisor.reboot("sensor")
    assert supervisor.state("sensor").healthy
