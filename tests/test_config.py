from ai_multi_reference_timekeeping.config import TimeServerSettings


def test_settings_load_defaults() -> None:
    settings = TimeServerSettings()
    assert settings.logging.level
    assert settings.metrics.port
