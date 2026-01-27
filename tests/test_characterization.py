from ai_multi_reference_timekeeping.characterization import SensorCharacterization


def test_sensor_characterization_z_score() -> None:
    characterization = SensorCharacterization()
    for value in [0.0, 0.1, -0.1, 0.05]:
        characterization.update("ref", value)
    z = characterization.z_score("ref", 0.2)
    assert z != 0.0
