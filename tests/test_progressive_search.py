"""Tests for the progressive search and the datasheet related calculations."""

# Standard Library Imports
import json
from unittest.mock import MagicMock

# Third-Party Imports
import pytest

import chopper_tune
from chopper_tune import ChopperTune, MeasurementMode, pick_preferred


def effective_hysteresis(params: dict) -> int:
    """Return the effective hysteresis start value of the given registers."""
    return params["hstrt"] + params["hend"] - 2


def default_landscape(params: dict) -> float:
    """Simulate a motor that gets smooth at an effective hysteresis of 4.

    Above that, the vibrations go down very slowly, like a real plateau, which
    makes the highest hysteresis the lowest vibration result.
    """
    hysteresis = effective_hysteresis(params)
    score = 500.0 + 150.0 * max(4 - hysteresis, 0) - 1.0 * hysteresis
    return score + 2.0 * abs(params["toff"] - 3) + 1.0 * abs(params["tbl"] - 2)


class SimulatedTuner:
    """Create a ChopperTune without a printer, using a simulated motor."""

    def __new__(cls, driver="2209", landscape=default_landscape, **overrides):
        """Create the ChopperTune instance."""
        tuner = ChopperTune.__new__(ChopperTune)
        tuner.gcode = MagicMock()
        tuner.reactor = MagicMock()
        tuner.reactor.monotonic.return_value = 0.0
        tuner.printer = MagicMock()
        tuner.printer.command_error = RuntimeError
        tuner.printer.is_shutdown.return_value = False
        tuner.debug = False
        tuner.fclk = 12
        tuner.plateau_tolerance = 0.05
        tuner.min_chopper_frequency = 22_000
        tuner.iterations = 1
        tuner.measurement_mode = MeasurementMode.Vibrations
        tuner.driver = driver
        tuner.kinematics = "corexy"
        tuner.steppers = ("stepper_x", "stepper_y")
        tuner.driver_settings = {"stepper_x": {}}
        tuner.travel_distance = 100
        tuner.coord_generator = None
        tuner.static_acceleration_vector = (0, 0, 0)
        tuner.raw_folder = None
        tuner.toff_range_requested = False
        tuner.reset_sample_data()
        tuner.reset_registers()
        tuner.registers["stepper_count"] = 1
        tuner.current_min = tuner.current_max = 800
        tuner.tbl_min, tuner.tbl_max = 0, 3
        tuner.toff_min, tuner.toff_max = 1, 8
        tuner.hstrt_min, tuner.hstrt_max = 0, 7
        tuner.hend_min, tuner.hend_max = 2, 15
        tuner.hstrt_hend_max = 16
        tuner.tpfd_min, tuner.tpfd_max = 0, 15
        tuner.min_speed = tuner.max_speed = 63.0
        for name, value in overrides.items():
            setattr(tuner, name, value)

        tuner.measured = []

        def measure(*_args):
            params = {
                key: tuner.registers[key]
                for key in ("tbl", "toff", "hstrt", "hend", "tpfd")
            }
            tuner.measured.append(params)
            return landscape(params)

        tuner.measure_vibrations = measure
        tuner.calc_magnitude = lambda samples, static_data: samples
        tuner.search_method = chopper_tune.SearchMethod.Progressive
        return tuner


def test_pick_preferred_uses_the_preference_among_similar_scores():
    """The most preferred candidate within the tolerance is picked."""
    assert pick_preferred({0: 104.0, 1: 100.0, 2: 103.0}, [2, 0, 1], 0.05) == 2


def test_pick_preferred_skips_candidates_out_of_tolerance():
    """A preferred candidate with much higher vibrations is not picked."""
    assert pick_preferred({0: 200.0, 1: 100.0}, [0, 1], 0.05) == 1


def test_pick_preferred_ignores_invalid_candidates():
    """Candidates penalized with an infinite score are never picked."""
    assert pick_preferred({0: float("inf"), 1: 100.0}, [0, 1], 0.05) == 1


def test_pick_preferred_no_valid_candidates():
    """None is returned when there are no valid candidates."""
    assert pick_preferred({0: float("inf")}, [0], 0.05) is None


def test_pick_preferred_reference_limits_the_tolerance():
    """The tolerance is relative to the lowest score seen so far."""
    assert pick_preferred({0: 104.0, 1: 102.0}, [0, 1], 0.05, reference=98.0) == 1


@pytest.mark.parametrize(
    ("driver", "tbl", "expected_clocks"),
    [
        ("5160", 2, 36),
        ("5160", 3, 54),
        ("2209", 2, 32),
        ("2209", 3, 40),
    ],
)
def test_calculate_frequency_uses_datasheet_clocks(driver, tbl, expected_clocks):
    """The upper limit is calculated with NCLK = 24 + 32 * TOFF and the TBL table."""
    tuner = SimulatedTuner(driver=driver)
    expected = 12_000_000 / (2 * (24 + 32 * 3) + 2 * expected_clocks)
    assert tuner.calculate_frequency(tbl, 3) == pytest.approx(expected)


def test_estimate_chopper_frequency_matches_datasheet_example():
    """As in the datasheets, TOFF=3 at 12 MHz is for a 25 kHz chopper frequency."""
    assert SimulatedTuner().estimate_chopper_frequency(3) == pytest.approx(25_000)


@pytest.mark.parametrize("tbl", [0, 1])
def test_toff_1_requires_tbl_2(tbl):
    """TOFF=1 with TBL < 2 is rejected without moving the motor."""
    tuner = SimulatedTuner()
    score = tuner.objective_function([800, tbl, 1, 0, 6, 0, 6300])
    assert score == float("inf")
    assert tuner.measured == []


def test_toff_1_with_tbl_2_is_allowed():
    """TOFF=1 with TBL=2 is measured."""
    tuner = SimulatedTuner()
    assert tuner.objective_function([800, 2, 1, 0, 6, 0, 6300]) != float("inf")


def test_progressive_finds_the_lowest_smooth_hysteresis():
    """The start of the plateau is selected, not the highest hysteresis."""
    tuner = SimulatedTuner()
    params = tuner.perform_progressive_search()
    registers = {"hstrt": params[3], "hend": params[4]}
    assert effective_hysteresis(registers) == 4


def test_progressive_prefers_hysteresis_decrement():
    """The hysteresis start value is mostly moved to HSTRT if it makes no difference."""
    tuner = SimulatedTuner()
    params = tuner.perform_progressive_search()
    assert params[4] == tuner.hend_min
    assert params[3] == 4


def test_progressive_only_tests_recommended_values_by_default():
    """TOFF stays in the inaudible band and TBL at recommended values."""
    tuner = SimulatedTuner()
    tuner.perform_progressive_search()
    assert {sample["toff"] for sample in tuner.measured} == {2, 3}
    assert {sample["tbl"] for sample in tuner.measured} == {1, 2}


def test_progressive_stays_within_the_requested_ranges():
    """No sample is taken out of the requested ranges."""
    tuner = SimulatedTuner(
        tbl_min=0,
        tbl_max=0,
        toff_min=4,
        toff_max=4,
        hstrt_min=1,
        hstrt_max=2,
        hend_min=3,
        hend_max=5,
    )
    tuner.perform_progressive_search()
    assert tuner.measured
    for sample in tuner.measured:
        assert sample["tbl"] == 0
        assert sample["toff"] == 4
        assert 1 <= sample["hstrt"] <= 2
        assert 3 <= sample["hend"] <= 5


def test_progressive_honors_explicit_out_of_band_toff():
    """An explicitly requested TOFF range out of the band is still measured."""
    tuner = SimulatedTuner(toff_min=8, toff_max=8)
    params = tuner.perform_progressive_search()
    assert params[2] == 8


def test_progressive_reports_the_skipped_toff_values():
    """The TOFF values out of the band are listed with their chopper frequency."""
    tuner = SimulatedTuner()
    tuner.perform_progressive_search()
    messages = [call.args[0] for call in tuner.gcode.respond_info.call_args_list]
    assert (
        "Skipping TOFF 1 (53.6kHz), 4 (19.7kHz), 5 (16.3kHz), 6 (13.9kHz), "
        "7 (12.1kHz), 8 (10.7kHz), they are outside the 22-35kHz chopper frequency "
        "band (see min_chopper_frequency). Set TOFF_MIN and TOFF_MAX to test them "
        "anyway." in messages
    )
    assert (
        "Testing TOFF 2 (34.1kHz), 3 (25.0kHz) (estimated chopper frequency)"
        in messages
    )


def test_progressive_tests_the_whole_requested_toff_range():
    """A TOFF range given by the user is tested completely."""
    tuner = SimulatedTuner(toff_range_requested=True)
    tuner.perform_progressive_search()
    assert {sample["toff"] for sample in tuner.measured} == set(range(1, 9))
    messages = [call.args[0] for call in tuner.gcode.respond_info.call_args_list]
    assert not any(message.startswith("Skipping TOFF") for message in messages)
    assert any(
        message.startswith("Testing TOFF 1 (53.6kHz), 4 (19.7kHz), 5 (16.3kHz)")
        and message.endswith("Low frequencies can cause an audible whine.")
        for message in messages
    )


def test_progressive_toff_preference_still_applies_to_a_requested_range():
    """A slightly lower score out of the band doesn't beat the preferred TOFF."""

    def landscape(params):
        # TOFF=8 is about 2% lower than TOFF=3, within the 5% tolerance.
        return default_landscape(params) - 4.0 * params["toff"]

    tuner = SimulatedTuner(toff_range_requested=True, landscape=landscape)
    assert tuner.perform_progressive_search()[2] == 3


def test_progressive_min_chopper_frequency_is_configurable():
    """A lower minimum chopper frequency allows the longer off times."""
    tuner = SimulatedTuner(min_chopper_frequency=15_000)
    tuner.perform_progressive_search()
    assert {sample["toff"] for sample in tuner.measured} == {2, 3, 4, 5}


def low_hysteresis_landscape(gain: float):
    """Simulate a motor that vibrates less with a lower hysteresis."""

    def landscape(params):
        hysteresis = effective_hysteresis(params)
        score = 500.0 + gain * hysteresis
        return score + 2.0 * abs(params["toff"] - 3) + 1.0 * abs(params["tbl"] - 2)

    return landscape


def test_progressive_prefers_the_default_hysteresis_among_similar_results():
    """A slightly lower vibration doesn't move the hysteresis away from 4."""
    # hysteresis 0 is about 2% lower than hysteresis 4
    tuner = SimulatedTuner(landscape=low_hysteresis_landscape(2.5))
    params = tuner.perform_progressive_search()
    assert params[1:5] == [2, 3, 4, 2]


def test_progressive_uses_a_clearly_better_hysteresis():
    """A much lower vibration overrides the default hysteresis."""
    # hysteresis 0 is about 24% lower than hysteresis 4
    tuner = SimulatedTuner(landscape=low_hysteresis_landscape(40.0))
    params = tuner.perform_progressive_search()
    assert effective_hysteresis({"hstrt": params[3], "hend": params[4]}) == 0


def test_progressive_hysteresis_sweep_reaches_the_default():
    """The sweep doesn't stop before the default hysteresis is compared."""

    def landscape(params):
        hysteresis = effective_hysteresis(params)
        score = {0: 500.0, 4: 510.0}.get(hysteresis, 600.0)
        return score + 2.0 * abs(params["toff"] - 3) + 1.0 * abs(params["tbl"] - 2)

    tuner = SimulatedTuner(landscape=landscape)
    params = tuner.perform_progressive_search()
    assert effective_hysteresis({"hstrt": params[3], "hend": params[4]}) == 4


def test_progressive_reports_the_sample_count_as_an_upper_limit():
    """The sweeps stop early, so the total number of samples is a maximum."""
    tuner = SimulatedTuner()
    tuner.perform_progressive_search()
    messages = [call.args[0] for call in tuner.gcode.respond_info.call_args_list]
    progress = [message for message in messages if message.startswith("Sample")]
    assert progress
    assert all("(up to " in message and "R: up to " in message for message in progress)


def test_progress_report_without_upper_limit_shows_percentage():
    """Other search methods keep reporting the percentage."""
    tuner = SimulatedTuner()
    tuner.total_expected_samples = 10
    tuner.number_of_samples = 5
    tuner.progress_report(100.0)
    assert "5/10 (50.0%)" in tuner.gcode.respond_info.call_args_list[0].args[0]


def test_progressive_stops_the_hysteresis_sweep_on_the_plateau():
    """Higher hysteresis values are not measured after the plateau is reached."""
    tuner = SimulatedTuner()
    tuner.perform_progressive_search()
    assert max(effective_hysteresis(sample) for sample in tuner.measured) < 10


def test_progressive_final_score_is_within_tolerance_of_the_lowest():
    """The tolerance does not add up over the sweeps."""
    tuner = SimulatedTuner(driver="5160")
    params = tuner.perform_progressive_search()
    name = tuner.generate_sample_name(*params[:6], params[6] / 100)
    assert tuner.samples[name] <= tuner.best_result * 1.05


def test_progressive_tpfd_prefers_the_lowest_value():
    """TPFD stays at 0 unless a higher value is measurably better."""
    tuner = SimulatedTuner(driver="5160")
    assert tuner.perform_progressive_search()[5] == 0


def test_progressive_tpfd_is_increased_if_it_helps():
    """A TPFD value that reduces mid-range resonances is selected."""

    def landscape(params):
        return default_landscape(params) + 300.0 * max(4 - params["tpfd"], 0)

    tuner = SimulatedTuner(driver="5160", landscape=landscape)
    assert tuner.perform_progressive_search()[5] == 4


def test_progressive_does_not_sweep_tpfd_on_unsupported_drivers():
    """TPFD is only tested on the drivers that support it."""
    tuner = SimulatedTuner(driver="2209")
    tuner.perform_progressive_search()
    assert tuner.registers["tpfd"] == 0
    assert all(
        call.args[0].find("TPFD") == -1
        for call in (tuner.gcode.respond_info.call_args_list)
    )


def test_progressive_best_speed_matches_the_sample_name():
    """The returned speed gives the same sample name as the measurement."""
    tuner = SimulatedTuner(min_speed=63.125, max_speed=63.125)
    params = tuner.perform_progressive_search()
    assert params[6] == 6312
    name = tuner.generate_sample_name(*params[:6], params[6] / 100)
    assert name in tuner.samples


def test_progressive_warns_about_ignored_speed_and_current_ranges():
    """The user is told that the ranges are not used."""
    tuner = SimulatedTuner(max_speed=100.0, current_max=900)
    tuner.perform_progressive_search()
    messages = [call.args[0] for call in tuner.gcode.respond_info.call_args_list]
    assert any("ignoring MAX_SPEED" in message for message in messages)
    assert any("ignoring CURRENT_MAX_MA" in message for message in messages)


def test_current_is_applied_to_all_steppers():
    """All of the steppers of an axis get the new current."""
    tuner = SimulatedTuner()
    tuner.registers["stepper_count"] = 2
    tuner.gcode.run_script_from_command = MagicMock()
    tuner.apply_registers(tuner.steppers, "curr", 800)
    scripts = [
        call.args[0] for call in tuner.gcode.run_script_from_command.call_args_list
    ]
    assert scripts == [
        "SET_TMC_CURRENT STEPPER=stepper_x CURRENT=0.8",
        "SET_TMC_CURRENT STEPPER=stepper_x1 CURRENT=0.8",
    ]


@pytest.mark.parametrize(
    ("threshold", "kinematics", "axes", "raises"),
    [
        (None, "corexy", ("x", "y"), False),
        (0, "corexy", ("x", "y"), False),
        (999999, "corexy", ("x", "y"), True),
        # 63 mm/s on the diagonal is 89.1 mm/s on the stepper.
        (80, "corexy", ("x", "y"), False),
        (100, "corexy", ("x", "y"), True),
        (80, "cartesian", ("x", "y"), True),
        (80, "corexy", ("z", "x"), True),
    ],
)
def test_validate_chopper_mode(threshold, kinematics, axes, raises):
    """Tuning is stopped if the driver would be in StealthChop."""
    tuner = SimulatedTuner(kinematics=kinematics)
    tuner.driver_settings["stepper_x"]["stealthchop_threshold"] = threshold
    if raises:
        with pytest.raises(RuntimeError, match="StealthChop"):
            tuner.validate_chopper_mode(axes, tuner.steppers)
    else:
        tuner.validate_chopper_mode(axes, tuner.steppers)


def test_store_data_creates_the_results_folder(tmp_path, monkeypatch):
    """The sample data is stored even if the plots are not generated first."""
    results_folder = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(chopper_tune, "RESULTS_FOLDER", str(results_folder))
    tuner = SimulatedTuner(accel_chip_name="adxl345", sense_resistor=0.11)
    tuner.samples = {"sample": 1.0}
    tuner.store_data(date_stamp="now")
    stored = next(results_folder.iterdir())
    assert json.loads(stored.read_text()) == {"sample": 1.0}


@pytest.mark.parametrize(
    ("given", "expected"),
    [((0, 0), (1, 1)), ((0, 8), (1, 8)), ((1, 8), (1, 8)), ((3, 3), (3, 3))],
)
def test_validate_toff_values_never_disables_the_driver(given, expected):
    """TOFF=0 disables the driver, so it is replaced with the lowest valid value."""
    assert SimulatedTuner().validate_toff_values(*given) == expected


class FakeCommand:
    """A G-Code command with the given parameters."""

    def __init__(self, **params):
        """Store the parameters."""
        self.params = params

    def get(self, name, default=None):
        """Return the parameter value or the default value."""
        return self.params.get(name, default)

    get_int = get_float = get


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, (1, 8, False)),
        ({"TOFF_MIN": 1}, (1, 8, True)),
        ({"TOFF_MAX": 8}, (1, 8, True)),
        ({"TOFF_MIN": 2, "TOFF_MAX": 6}, (2, 6, True)),
    ],
)
def test_cmd_chopper_tune_detects_a_requested_toff_range(params, expected):
    """Giving any TOFF limit, even the default value, requests the whole range."""
    tuner = SimulatedTuner()
    tuner.chopper_tune = MagicMock()
    tuner.cmd_chopper_tune(FakeCommand(**params))
    kwargs = tuner.chopper_tune.call_args.kwargs
    assert (
        kwargs["toff_min"],
        kwargs["toff_max"],
        kwargs["toff_range_requested"],
    ) == expected


def test_cmd_chopper_tune_runs_before_returning():
    """The command is done when it returns, so that the next commands wait for it."""
    tuner = SimulatedTuner()
    tuner.chopper_tune = MagicMock()
    tuner.cmd_chopper_tune(FakeCommand())
    tuner.chopper_tune.assert_called_once()
    tuner.reactor.register_callback.assert_not_called()


def test_cmd_chopper_tune_passes_command_errors_as_is():
    """Command errors are already handled by Klipper."""
    tuner = SimulatedTuner()
    tuner.chopper_tune = MagicMock(side_effect=RuntimeError("the message"))
    with pytest.raises(RuntimeError, match="^the message$"):
        tuner.cmd_chopper_tune(FakeCommand())


def test_cmd_chopper_tune_converts_other_exceptions_to_command_errors():
    """Klipper would shut down the printer for any other exception."""
    tuner = SimulatedTuner()
    tuner.chopper_tune = MagicMock(side_effect=ValueError("the problem"))
    with pytest.raises(RuntimeError, match="Chopper tune failed: the problem"):
        tuner.cmd_chopper_tune(FakeCommand())
    assert "ValueError" in tuner.gcode.respond_info.call_args.args[0]


def test_tuning_stops_after_an_emergency_stop():
    """No more samples are taken and the G-Code mutex gets released."""
    tuner = SimulatedTuner()
    tuner.printer.is_shutdown.return_value = True
    with pytest.raises(RuntimeError, match="Printer is shutdown"):
        tuner.perform_progressive_search()
    assert tuner.measured == []


@pytest.mark.parametrize(
    ("driver", "configured", "expected"),
    [
        ("2130", None, 13.2),
        ("2208", None, 12),
        ("2209", None, 12),
        ("2240", None, 12.5),
        ("5160", None, 12),
        ("5160", 16, 16),
    ],
)
def test_get_fclk(driver, configured, expected):
    """The clock is selected per driver and can be overridden with the config."""
    assert SimulatedTuner(driver=driver, fclk=configured).get_fclk() == expected


def test_get_current_range_represents_every_milliamp():
    """Float errors in the configured current don't lose a milliamp."""
    tuner = SimulatedTuner()
    for milliamps in range(100, 3001):
        tuner.driver_settings["stepper_x"]["run_current"] = milliamps / 1000
        current_range = tuner.get_current_range(
            MeasurementMode.Vibrations, None, None, tuner.steppers
        )
        assert current_range == (milliamps, milliamps)


def test_save_configs_keeps_milliamp_resolution():
    """The current is saved with three decimals."""
    tuner = SimulatedTuner(configfile=MagicMock())
    tuner.save_configs({"current": 1234})
    tuner.configfile.set.assert_called_once_with(
        "tmc2209 stepper_x", "run_current", "1.234"
    )


def test_raw_samples_are_not_saved_by_default(tmp_path, monkeypatch):
    """Nothing is written unless SAVE_RAW is used."""
    monkeypatch.setattr(chopper_tune, "RESULTS_FOLDER", str(tmp_path))
    tuner = SimulatedTuner()
    tuner.raw_folder = None
    tuner.save_raw_samples("sample", [])
    assert list(tmp_path.iterdir()) == []


class FakeSample:
    """An accelerometer sample."""

    def __init__(self, time, accel_x, accel_y, accel_z):
        """Create the sample."""
        self.time = time
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z


def test_raw_samples_are_saved_with_metadata(tmp_path, monkeypatch):
    """The samples and the metadata can be loaded back."""
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(chopper_tune, "RESULTS_FOLDER", str(tmp_path))
    tuner = SimulatedTuner()
    tuner.start_raw_capture({"steppers": ("stepper_x", "stepper_y"), "speed": 63})
    samples = [FakeSample(1000.0 + i / 3200, i, -i, 9806.0) for i in range(5)]
    tuner.save_raw_samples("tbl=2_iteration=1", samples, speed=63.0, score=12.5)

    folder = next((tmp_path / "raw").iterdir())
    assert folder.name.endswith("_stepper_x")
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata == {"steppers": ["stepper_x", "stepper_y"], "speed": 63}
    data = np.load(folder / "tbl=2_iteration=1.npz")
    assert data["start_time"] == 1000.0
    assert data["time"][-1] == pytest.approx(4 / 3200)
    assert data["accel"].shape == (5, 3)
    assert data["accel"][4].tolist() == [4.0, -4.0, 9806.0]
    assert data["score"] == 12.5


def test_raw_sample_save_errors_do_not_stop_the_tuning(tmp_path):
    """Saving is disabled with a warning if a file can not be written."""
    tuner = SimulatedTuner()
    tuner.raw_folder = str(tmp_path / "missing")
    tuner.save_raw_samples("sample", [FakeSample(0.0, 0, 0, 0)])
    assert tuner.raw_folder is None
    assert "Stopped saving" in tuner.gcode.respond_info.call_args.args[0]


def test_raw_samples_are_saved_for_each_iteration(tmp_path):
    """Every measurement is saved with its score."""
    tuner = SimulatedTuner(iterations=2)
    tuner.measure_vibrations = lambda *_args: [FakeSample(0.0, 1, 2, 3)]
    tuner.calc_magnitude = lambda samples, static_data: 7.0
    tuner.raw_folder = str(tmp_path)
    tuner.objective_function([800, 2, 3, 4, 2, 0, 6300])
    assert sorted(path.name.split("_iteration=")[1] for path in tmp_path.iterdir()) == [
        "1.npz",
        "2.npz",
    ]
