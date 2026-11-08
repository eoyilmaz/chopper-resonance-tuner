"""Chopper Tune extension for Klipper.

TMC drivers registers calibration tool.

Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>

This file may be distributed under the terms of the GNU GPLv3 license.
"""

# Standard Library Imports
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING

# Third Party Imports
import numpy as np
from scipy import signal
from scipy.optimize import brute, differential_evolution

# Klipper Imports
import gcode

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    from configfile import ConfigWrapper
    from extras.adxl345 import ADXL345, Accel_Measurement
    from extras.lis2dw import LIS2DW
    from gcode import GCodeCommand, GCodeDispatch
    from klippy import Printer
    from reactor import PollReactor
    from toolhead import ToolHead

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


DEFAULT_ACCEL_CHIP = "adxl345"
RESULTS_FOLDER = os.path.expanduser("~/printer_data/config/chopper_magnitude")
DATA_FOLDER = os.path.expanduser("~/printer_data/config/chopper_magnitude/tmp")

# The internal clock of the drivers in MHz, same as the values used by Klipper.
DRIVER_FCLK = {"2130": 13.2, "2240": 12.5}
DEFAULT_FCLK = 12
CUTOFF_RANGE = 5
# Graphs generation
COLORS = [
    "",
    "#2F4F4F",
    "#12B57F",
    "#9DB512",
    "#DF8816",
    "#1297B5",
    "#5912B5",
    "#B51284",
    "#127D0C",
]

SUPPORTED_DRIVERS = ["2130", "2208", "2209", "2660", "2240", "5160"]
DRIVERS_SUPPORTING_TPFD = ["2240", "5160"]

DEFAULT_REGISTER_VALUES = {
    "TBL": (0, 3),
    "TOFF": (1, 8),
    "HSTRT": (0, 7),
    "HEND": (2, 15),
    "HSTRT_HEND_MAX": 16,
    "TPFD": (0, 15),
}

# Comparator blank time in clocks per TBL value. TMC2208/TMC2209 datasheet:
# 16, 24, 32, 40. TMC5160 datasheet (and the other supported drivers): 16, 24,
# 36, 54.
TBL_CLOCKS = {"2208": (16, 24, 32, 40), "2209": (16, 24, 32, 40)}
DEFAULT_TBL_CLOCKS = (16, 24, 36, 54)

# The datasheets recommend a 16-30 kHz chopper frequency and calculate the
# starting TOFF for 25 kHz, assuming that the two slow decay phases take 50% of
# the chopper cycle. The real frequency also depends on the motor and the
# hysteresis.
CHOPPER_FREQ_TARGET = 25_000  # Hz
# The datasheets recommend 16-30 kHz, but the lower end of that range can be
# audible: a chopper whine at standstill was heard at an estimated 19.7 kHz and
# not at 25 kHz. An accelerometer can't sense it, so the minimum frequency is a
# hard limit, which can be changed with `min_chopper_frequency`.
DEFAULT_MIN_CHOPPER_FREQ = 22  # kHz
# Slightly over the recommended 30 kHz, as the frequency is only an estimate.
MAX_CHOPPER_FREQ = 35_000  # Hz
SLOW_DECAY_DUTY = 0.5

# Blank time recommended by the datasheets for most applications.
RECOMMENDED_TBL = (1, 2)
DEFAULT_TBL = 2
# "choosing a low to medium default value for the hysteresis (for example,
# effective hysteresis = 4) normally fits most applications."
DEFAULT_EFFECTIVE_HYSTERESIS = 4
# HSTRT adds (register + 1) and HEND adds (register - 3), so the effective
# hysteresis start value is: HSTRT + HEND - 2
HYSTERESIS_REGISTER_OFFSET = 2
# Stop an ascending sweep after this many samples without a real improvement.
PROGRESSIVE_PATIENCE = 3

# Indices of the parameters in the list passed to the objective function.
PARAM_TBL = 1
PARAM_TOFF = 2
PARAM_HSTRT = 3
PARAM_HEND = 4
PARAM_TPFD = 5


def pick_preferred(
    scores: dict[int, float],
    preference: list[int],
    tolerance: float,
    reference: float = float("inf"),
) -> None | int:
    """Pick the most preferred candidate that is close enough to the lowest score.

    An accelerometer can not sense the audible chopper noise caused by a low
    chopper frequency or a too high hysteresis. So, the lowest vibration alone
    doesn't give the quietest values, and among the candidates that have similar
    vibrations the one that the datasheet favors should be used.

    Args:
        scores (dict[int, float]): The measured score of each candidate.
        preference (list[int]): The candidates from most to least preferred.
        tolerance (float): How much higher than the lowest score a candidate can
            be, as a ratio.
        reference (float): The lowest score seen outside of these candidates.
            Using it prevents the small differences allowed by the tolerance
            from adding up over successive sweeps.

    Returns:
        None | int: The picked candidate, or None if no candidate is valid.
    """
    valid = {key: score for key, score in scores.items() if score != float("inf")}
    if not valid:
        return None
    lowest_key = min(valid, key=valid.get)
    limit = min(valid[lowest_key], reference) * (1 + tolerance)
    for key in preference:
        if key in valid and valid[key] <= limit:
            return key
    return lowest_key


class MeasurementMode(IntEnum):
    """Integer enumerator to specify the current measurement mode."""

    Resonances = 0
    Vibrations = 1

    def __repr__(self) -> str:
        """Return the enum name for str().

        Returns:
            str: The name as the string representation of this MeasurementMode.
        """
        return self.name

    __str__ = __repr__

    @classmethod
    def to_mode(cls, mode: int | str | MeasurementMode) -> MeasurementMode:
        """Convert the given mode value to a MeasurementMode enum.

        Args:
            mode (int | str | MeasurementMode]): The value to convert to a
                MeasurementMode.

        Raises:
            TypeError: Input value type is invalid.
            ValueError: Input value is invalid.

        Returns:
            MeasurementMode: The enum.
        """
        if not isinstance(mode, (int, str, MeasurementMode)):
            raise TypeError(
                "mode should be a MeasurementMode enum value or one of "
                f"{[m.name for m in cls] + [m.value for m in cls]}, "
                f"not {mode.__class__.__name__}: '{mode}'"
            )
        if isinstance(mode, str):
            mode_name_lut = {m.name.lower(): m.name for m in cls}
            mode_name_lut.update({m.value: m.name for m in cls})
            mode_lower_case = mode.lower()
            if mode_lower_case not in mode_name_lut:
                raise ValueError(
                    "mode should be a MeasurementMode enum value or one of "
                    f"{[m.name for m in cls] + [m.value for m in cls]}, "
                    f"not '{mode}'"
                )

            return cls.__members__[mode_name_lut[mode_lower_case]]

        return mode


class SearchMethod(IntEnum):
    """Integer enumerator to specify the current search method."""

    BruteForce = 0
    Adaptive = 1
    Progressive = 2

    def __repr__(self) -> str:
        """Return the enum name for str().

        Returns:
            str: The name as the string representation of this SearchMethod.
        """
        return self.name

    __str__ = __repr__

    @classmethod
    def to_method(cls, method: int | str | SearchMethod) -> SearchMethod:
        """Convert the given method value to a SearchMethod enum.

        Args:
            method (int | str | SearchMethod]): The value to convert to a
                SearchMethod.

        Raises:
            TypeError: Input value type is invalid.
            ValueError: Input value is invalid.

        Returns:
            SearchMethod: The enum.
        """
        if not isinstance(method, (int, str, SearchMethod)):
            raise TypeError(
                "method should be a SearchMethod enum value or one of "
                f"{[m.name for m in cls] + [m.value for m in cls]}, "
                f"not {method.__class__.__name__}: '{method}'"
            )
        if isinstance(method, str):
            method_name_lut = {m.name.lower(): m.name for m in cls}
            method_name_lut.update({m.value: m.name for m in cls})
            method_lower_case = method.lower()
            if method_lower_case.replace("_", "") not in method_name_lut:
                raise ValueError(
                    "method should be a SearchMethod enum value or one of "
                    f"{[m.name for m in cls] + [m.value for m in cls]}, "
                    f"not '{method}'"
                )

            return cls.__members__[method_name_lut[method_lower_case.replace("_", "")]]

        return method


class AccelerometerMeasure:
    """A context manager that helps with accelerometer measurement."""

    def __init__(
        self,
        accelerometer: ADXL345 | LIS2DW,
    ) -> None:
        self.accelerometer = accelerometer
        self.bg_client = None
        self.samples: None | list[Accel_Measurement] = None

    def __enter__(self) -> Self:
        """Enter to the context."""
        if self.bg_client is None:
            self.bg_client = self.accelerometer.start_internal_client()
        return self

    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        tb: None | TracebackType,
    ) -> None:
        """Exit the context.

        Ignore the exceptions, if any, Klipper will handle it.
        """
        self.bg_client.finish_measurements()
        # retrieve data
        self.samples = self.bg_client.samples or self.bg_client.get_samples()


class Coord(list):
    """Custom "list" class for coordinates - add easy access to x, y, z components.

    The difference between the gcode.Coord is that, this class allows attribute
    setting.

    Args:
        t (Coord | list | tuple): Another Coord instance or a list or a tuple.
    """

    __slots__ = ()

    def __new__(cls, t: Coord | list | tuple) -> Self:
        """Create a new Coord instance."""
        if len(t) < 4:
            t = list(tuple(t) + (0,) * (3 - len(t)))
        return list.__new__(cls, t)

    @property
    def x(self) -> float:
        """Return the x component.

        Returns:
            float: The x component.
        """
        return self[0]

    @x.setter
    def x(self, x: float) -> None:
        """Set the x component.

        Args:
            x (float): The x component value.
        """
        self[0] = x

    @property
    def y(self) -> float:
        """Return the y component.

        Returns:
            float: The y component.
        """
        return self[1]

    @y.setter
    def y(self, y: float) -> None:
        """Set the y component.

        Args:
            y (float): The y component value.
        """
        self[1] = y

    @property
    def z(self) -> float:
        """Return the z component.

        Returns:
            float: The z component.
        """
        return self[2]

    @z.setter
    def z(self, z: float) -> None:
        """Set the z component.

        Args:
            z (float): The z component value.
        """
        self[2] = z

    def length(self) -> float:
        """Return the vector length.

        Returns:
            float: The vector length.
        """
        return float(reduce(lambda x, y: x + y**2, [0, *self]) ** 0.5)

    def unitize(self) -> Self:
        """Make self unit vector."""
        other = self / self.length()
        for i in range(len(self)):
            self[i] = other[i]
        return self

    def __add__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the + operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        if isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        return super().__add__(other)

    def __iadd__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the += operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Self: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        if isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        return super().__iadd__(other)

    def __sub__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the - operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x - other[0], self.y - other[1], self.z - other[2]))
        if isinstance(other, (int, float)):
            return Coord([i - other for i in self])
        return super().__sub__(other)

    def __mul__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the * operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the multiplication.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        if isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        return super().__mul__(other)

    def __imul__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the *= operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Self: The result of the multiplication.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        if isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        return super().__imul__(other)

    def __truediv__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the / operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the division.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        if isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        return super().__truediv__(other)

    def __itruediv__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the /= operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the division.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        if isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        return super().__itruediv__(other)


class CoordGenerator:
    """A class to generate coordinates/positions for chopper tuning.

    Args:
        direction (Coord | list | tuple): The initial direction.
        start_coord (None | Coord): The starting coordinate.
    """

    def __init__(
        self, direction: Coord | list | tuple, start_coord: None | Coord = None
    ) -> None:
        self._direction = None
        self.direction = direction
        if start_coord is None:
            start_coord = Coord((0, 0, 0))
        self.current_coord = start_coord

    @property
    def direction(self) -> Coord:
        """Return the direction of this CoordGenerator.

        Returns:
            Coord: The direction of this CoordGenerator.
        """
        return self._direction

    @direction.setter
    def direction(self, direction: Coord | list | float) -> None:
        """Set the direction attribute value.

        Args:
            direction (Coord | list | float): The direction value.

        Raises:
            TypeError: If the direction is not a list, tuple or Coord instance.
        """
        if not isinstance(direction, (list, tuple, Coord)):
            raise TypeError(
                f"direction should be a list, tuple or a Coord instance, not "
                f"{direction.__class__.__name__}: '{direction}'"
            )

        if isinstance(direction, (list, tuple)):
            direction = Coord(direction)

        if direction.length() == 0:
            raise ValueError("direction length cannot be zero.")

        self._direction = direction.unitize()

    def switch_direction(self) -> None:
        """Switch direction."""
        self.direction *= (-1, -1, -1)

    def next(self, travel_distance: float) -> float:
        """Get the next position.

        Args:
            travel_distance (float): The travel distance.

        Returns:
            float: The next position.
        """
        self.current_coord += self.direction * travel_distance
        return self.current_coord


class ChopperTune:
    """The main class to handle the chopper tune functionality.

    Args:
        config (ConfigWrapper): The configuration wrapper.
    """

    def __init__(self, config: ConfigWrapper) -> None:
        self.printer: Printer = config.get_printer()
        self.gcode: GCodeDispatch = self.printer.lookup_object("gcode")
        self.configfile = self.printer.lookup_object("configfile")
        self.toolhead: None | ToolHead = None
        self.accelerometer: None | ADXL345 | LIS2DW = None
        self.settings = None
        self.reactor: PollReactor = self.printer.get_reactor()
        self.driver_settings = {}
        self.stepper_settings = {}
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "hend": -1,
            "hstrt": -1,
            "tpfd": -1,
            "curr": -1,
        }

        # config values
        self.debug = config.getboolean("debug", False)
        self.inset = config.getfloat("inset", 10)
        self.current_change_step = config.getint("current_change_step", 25)
        self.measure_time = config.getint("measure_time", 1250)
        self.required_rpm = config.getfloatlist("required_rpm", [37.5, 150, 1.5])
        self.delay = config.getfloat("delay", 500)
        self.fclk = config.getfloat("fclk", None)
        self.plateau_tolerance = config.getfloat("plateau_tolerance", 0.05)
        self.min_chopper_frequency = (
            config.getfloat(
                "min_chopper_frequency", DEFAULT_MIN_CHOPPER_FREQ, minval=0.0
            )
            * 1000
        )

        self.kinematics = config.getsection("printer").get("kinematics")

        # runtime variables
        self.driver = None
        self.resistor = None
        self.samples = {}  # store all samples
        self.total_expected_samples = -1  # the expected number of samples
        # whether total_expected_samples is the upper limit, not an estimate
        self.total_expected_samples_is_max = False
        self.number_of_samples = 0  # current number of samples taken
        self.number_of_real_samples = 0  # current number of real samples taken
        self.best_result = 999_999_999  # store the best result
        self.speed_vs_vibrations = []  # store speed vs vibrations
        self.global_start_time = 0  # store the global start time
        self.raw_folder = None  # where the raw samples are saved, if requested
        # whether the TOFF range was given by the user, if so all of it is tested
        self.toff_range_requested = False

        # Calculated values
        self.search_method = None
        self.measurement_mode = MeasurementMode.Vibrations
        self.travel_speed = None
        self.travel_distance = None
        self.coord_generator = None
        self.accel_chip_name = None
        self.steppers = None
        self.current = None
        self.static_acceleration_vector = None
        self.static_acceleration_magnitude = None
        self.initial_position = None
        self.initial_direction = None

        # Bounds
        self.bounds = []
        self.current_min = None
        self.current_max = None
        self.tbl_min = None
        self.tbl_max = None
        self.toff_min = None
        self.toff_max = None
        self.hstrt_min = None
        self.hstrt_max = None
        self.hstrt_hend_max = None
        self.hend_min = None
        self.hend_max = None
        self.tpfd_min = None
        self.tpfd_max = None
        self.min_speed = None
        self.max_speed = None
        self.speed_change_step = None

        self.register_commands()
        self.printer.register_event_handler("klippy:connect", self._connect)

    def register_commands(self) -> None:
        """Register GCode commands."""
        self.gcode.register_command("CHOPPER_TUNE", self.cmd_chopper_tune)
        self.gcode.register_command("CHOPPER_TUNE_DEBUG", self.cmd_chopper_tune_debug)

    def _connect(self) -> None:
        """Handle printer connect event."""
        self.settings = self.configfile.get_status(None)["settings"]
        self.toolhead = self.printer.lookup_object("toolhead")
        for axis in "xyz":
            driver, _ = self.detect_driver(axis)
            self.driver_settings[f"stepper_{axis}"] = self.settings.get(
                f"tmc{driver} stepper_{axis}", {}
            )
            self.stepper_settings[f"stepper_{axis}"] = self.settings.get(
                f"stepper_{axis}", {}
            )

    def detect_driver(self, stepper: str) -> None | tuple[str, str]:
        """Detect the driver of the selected stepper.

        Args:
            stepper (str): The stepper name.

        Returns:
            tuple[None, None] | tuple[str, str]: A tuple containing the stepper
                driver model and sense resistor, or None if the driver thus the
                sense resistor cannot be detected.
        """
        drivers = SUPPORTED_DRIVERS
        stepper = f"stepper_{stepper}"
        resistor = None
        for driver in drivers:
            if "run_current" not in self.settings.get(f"tmc{driver} {stepper}", {}):
                continue
            self.gcode.respond_info(f"Selected tmc{driver} for {stepper}")
            if driver != "2240":
                resistor = self.settings[f"tmc{driver} {stepper}"]["sense_resistor"]
            else:
                resistor = self.settings[f"tmc{driver} {stepper}"]["rref"]
            return driver, resistor

        return None, None

    def get_axes_and_steppers(
        self, axis: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get main and secondary axis / stepper.

        Args:
            axis (str): The to be tuned.

        Returns:
            tuple[tuple[str, ...], tuple[str, ...]]: A tuple containing:
                - axes (tuple[str, ...]): The main and secondary axis.
                - steppers (tuple[str, ...]): The main and secondary stepper.
        """
        if axis not in ("x", "y", "z"):
            raise self.printer.command_error(f"WARNING!!! Incorrect axis: {axis}")

        if self.kinematics not in ("corexy", "cartesian"):
            raise self.printer.command_error(
                f"WARNING!!! Unsupported kinematics: {self.kinematics}"
            )

        if self.kinematics == "corexy":
            if axis in ("x", "y"):
                if axis == "x":
                    axes = ("x", "y")
                    steppers = ("stepper_x", "stepper_y")
                elif axis == "y":
                    axes = ("y", "x")
                    steppers = ("stepper_y", "stepper_x")
            elif axis == "z":
                axes = ("z", "x")
                steppers = ("stepper_z",)
        elif self.kinematics == "cartesian":
            if axis == "x":
                axes = ("x", "y")
                steppers = ("stepper_x",)
            elif axis == "y":
                axes = ("y", "x")
                steppers = ("stepper_y",)
            elif axis == "z":
                axes = ("z", "x")
                steppers = ("stepper_z",)

        return axes, steppers

    def get_axis_limits(self, axes: list[str]) -> tuple[float, float, float, float]:
        """Select main and secondary axis / stepper.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - a_axis_min (float): The minimum position of the main axis.
                - a_axis_max (float): The maximum position of the main axis.
                - a_axis_mid (float): The middle position of the main axis.
                - b_axis_mid (float): The middle position of the secondary axis.
        """
        if axes[0] == "z":
            a_axis_min = (
                max(self.stepper_settings[f"stepper_{axes[0]}"]["position_min"], 0)
                + self.inset
            )
        else:
            a_axis_min = (
                self.stepper_settings[f"stepper_{axes[0]}"]["position_min"] + self.inset
            )

        a_axis_max = (
            self.stepper_settings[f"stepper_{axes[0]}"]["position_max"] - self.inset
        )
        a_axis_mid = (
            self.stepper_settings[f"stepper_{axes[0]}"]["position_max"]
            + self.stepper_settings[f"stepper_{axes[0]}"]["position_min"]
        ) / 2
        b_axis_mid = (
            self.stepper_settings[f"stepper_{axes[1]}"]["position_max"]
            + self.stepper_settings[f"stepper_{axes[1]}"]["position_min"]
        ) / 2
        return a_axis_min, a_axis_max, a_axis_mid, b_axis_mid

    def get_travel_speed_and_acceleration(self, axes: list[str]) -> tuple[float, float]:
        """Select main and secondary axis / stepper.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            tuple[float, float]: A tuple containing:
                - acceleration (float): The acceleration for the movement.
                - travel_speed (float): The travel speed for idle movements.
        """
        if axes[0] == "z":
            acceleration = self.settings["printer"]["max_z_accel"]
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_z_velocity", 0) / 2
        else:
            acceleration = self.settings["printer"].get("max_accel")
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_velocity") / 2

        return acceleration, travel_speed

    def validate_tpfd_values(
        self,
        driver: str,
        tpfd_min: int,
        tpfd_max: int,
    ) -> tuple[int, int]:
        """Validate the TPFD min and max values.

        Args:
            driver (str): The stepper driver model.
            tpfd_min (int): The TPFD min value.
            tpfd_max (int): The TPFD max value.

        Returns:
            tuple[int, int]: Validated tpfd_min and tpfd_max values.
        """
        if driver in DRIVERS_SUPPORTING_TPFD:
            # limit the TPFD value to valid range
            tpfd_min = max(
                DEFAULT_REGISTER_VALUES["TPFD"][0],
                min(tpfd_min, DEFAULT_REGISTER_VALUES["TPFD"][1]),
            )
            tpfd_max = min(
                DEFAULT_REGISTER_VALUES["TPFD"][1],
                max(tpfd_max, DEFAULT_REGISTER_VALUES["TPFD"][0]),
            )
            if tpfd_min > tpfd_max:
                # reverse values if min is greater than max after validation
                temp = tpfd_min
                tpfd_min = tpfd_max
                tpfd_max = temp
        else:
            # ignore TPFD if the driver doesn't support it
            tpfd_min = tpfd_max = DEFAULT_REGISTER_VALUES["TPFD"][0]

        return tpfd_min, tpfd_max

    def validate_toff_values(self, toff_min: int, toff_max: int) -> tuple[int, int]:
        """Validate the TOFF min and max values.

        TOFF=0 disables the driver and the motor would free-wheel during the move.

        Args:
            toff_min (int): The TOFF min value.
            toff_max (int): The TOFF max value.

        Returns:
            tuple[int, int]: Validated toff_min and toff_max values.
        """
        toff_min = max(toff_min, 1)
        return toff_min, max(toff_max, toff_min)

    def validate_chopper_mode(self, axes: list[str], steppers: list[str]) -> None:
        """Check if the driver will use SpreadCycle at the measurement speed.

        TBL, HSTRT, HEND and TPFD have no effect in StealthChop, so the results of
        a tuning done in StealthChop would be meaningless.

        Args:
            axes (list[str]): The main and secondary axis.
            steppers (list[str]): The main and secondary stepper.

        Raises:
            CommandError: If the driver is in StealthChop at the measurement speed.
        """
        threshold = self.driver_settings[steppers[0]].get("stealthchop_threshold")
        if not threshold:
            return
        stepper_speed = self.min_speed
        if self.kinematics == "corexy" and axes[0] != "z":
            # The measurement moves are done on the diagonals.
            stepper_speed *= 2**0.5
        if stepper_speed < threshold:
            raise self.printer.command_error(
                f"WARNING!!! {steppers[0]} will be in StealthChop at "
                f"{stepper_speed:.1f} mm/s (stealthchop_threshold: {threshold}), "
                "chopper parameters have no effect in that mode. Please remove or "
                "lower stealthchop_threshold in printer.cfg"
            )

    def reset_sample_data(self) -> None:
        """Reset sample data."""
        self.total_expected_samples = -1
        self.total_expected_samples_is_max = False
        self.number_of_samples = 0
        self.number_of_real_samples = 0
        self.best_result = 999_999_999
        self.samples = {}
        self.speed_vs_vibrations = []
        self.global_start_time = 0

    def reset_registers(self) -> None:
        """Reset registers to default values."""
        # Reset register values
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "hend": -1,
            "hstrt": -1,
            "tpfd": -1,
            "curr": -1,
        }

    def apply_registers(
        self,
        steppers: list[str],
        field: str,
        value: int,
    ) -> None:
        """Apply registers.

        Args:
            steppers (list[str]): The name of the steppers to set the register
                field values of.
            field (str): The name of the field to set the value of.
            value (int): The value to set to.
        """
        if field is None or value is None:
            return

        stepper = steppers[0]  # just update the main stepper
        for stepper_index in range(self.registers["stepper_count"]):
            # stepper_x,
            # stepper_y,
            # stepper_z, stepper_z1, stepper_z2, stepper_z3, ...
            # don't add index for the first stepper
            stepper_index = str(stepper_index) if stepper_index > 0 else ""
            if self.debug:
                self.gcode.respond_info(
                    f"Setting {field.lower()} "
                    f"from {self.registers[field]} to {value} "
                    f"on {stepper}{stepper_index}"
                )

            if field.lower() == "curr":
                if self.registers[field.lower()] != value:
                    self.gcode.run_script_from_command(
                        f"SET_TMC_CURRENT STEPPER={stepper}{stepper_index} "
                        f"CURRENT={value / 1000}"
                    )
            elif (
                not (field == "tpfd" and self.driver not in DRIVERS_SUPPORTING_TPFD)
                and self.registers[field.lower()] != value
            ):
                self.gcode.run_script_from_command(
                    "SET_TMC_FIELD "
                    f"STEPPER={stepper}{stepper_index} "
                    f"FIELD={field} VALUE={value}"
                )
        # store the last applied value
        self.registers[field.lower()] = value

    def get_stepper_count(self, axis: str) -> int:
        """Get the stepper count of the given axis.

        Args:
            axis (str): One of ["x", "y", "z"].

        Returns:
            int: The stepper count of the requested axis.
        """
        axis_steppers = [
            key for key in self.settings if key.startswith(f"stepper_{axis}")
        ]
        return len(axis_steppers)

    def get_accelerometer_chip(self, accel_chip_name: str, axis: None | str) -> str:
        """Get accelerometer chip.

        Args:
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            axis (None | str): The selected axis.
        """
        # Select accelerometer
        if accel_chip_name == "default":
            resonance_tester = self.settings.get("resonance_tester", {})
            if "accel_chip" in resonance_tester:
                accel_chip_name = resonance_tester["accel_chip"]
            elif axis is not None and f"accel_chip_{axis}" in resonance_tester:
                accel_chip_name = resonance_tester[f"accel_chip_{axis}"]
            else:
                # Use Default accelerometer
                accel_chip_name = DEFAULT_ACCEL_CHIP

        self.gcode.respond_info(f"Selected {accel_chip_name} for accelerometer")
        return accel_chip_name

    def get_current_range(
        self,
        measurement_mode: MeasurementMode,
        current_min: None | int,
        current_max: None | int,
        steppers: list[str],
    ) -> tuple[int, int]:
        """Get run current.

        Args:
            measurement_mode (MeasurementMode): The measurement mode.
            current_min (None | int): The minimum current value.
            current_max (None | int): The maximum current value.
            steppers (list[str]): The main and secondary stepper.

        Returns:
            tuple[int, int]: The minimum and maximum current values.
        """
        # Select run_current
        run_current = round(
            float(self.driver_settings[steppers[0]].get("run_current")) * 1000
        )
        if current_min is None:
            current_min = run_current
            if self.debug:
                self.gcode.respond_info(
                    f"Set default run_current: {current_min} mA to run_current_min"
                )
        else:
            current_min = int(current_min)

        if current_max is None:
            current_max = run_current
            if self.debug:
                self.gcode.respond_info(
                    f"Set default run_current: {current_max} mA to run_current_max"
                )
        else:
            current_max = int(current_max)

        if measurement_mode == MeasurementMode.Resonances:
            current_max = current_min

        return current_min, current_max

    def get_default_stepper_parameters(
        self,
        steppers: list[str],
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Return default stepper parameters.

        Args:
            steppers (list[str]): The main and secondary stepper.

        Returns:
            tuple[int, int, int, int, int, int, int, int]: The default stepper
                parameters.
        """
        tbl_max = tbl_min = self.driver_settings[steppers[0]].get("driver_tbl")
        toff_max = toff_min = self.driver_settings[steppers[0]].get("driver_toff")
        hstrt_max = hstrt_min = self.driver_settings[steppers[0]].get("driver_hstrt")
        hend_max = hend_min = self.driver_settings[steppers[0]].get("driver_hend")
        if self.driver in DRIVERS_SUPPORTING_TPFD:
            tpfd_max = tpfd_min = self.driver_settings[steppers[0]].get("driver_tpfd")
        else:
            tpfd_max = tpfd_min = DEFAULT_REGISTER_VALUES["TPFD"][0]

        return (
            tbl_min,
            tbl_max,
            toff_min,
            toff_max,
            hstrt_min,
            hstrt_max,
            hend_min,
            hend_max,
            tpfd_min,
            tpfd_max,
        )

    def configure_speed_limits(
        self,
        min_speed: None | float,
        max_speed: None | float,
        speed_change_step: None | float,
        measure_time: float,
        axes: list[str],
        steppers: list[str],
        a_axis_min: float,
        a_axis_max: float,
        acceleration: float,
    ) -> tuple[float, float, float]:
        """Configure speed limits.

        Args:
            min_speed (float | str): The in speed value, or can be set to None
                to auto calculate the value over the required RPM value.
            max_speed (float | str): The max speed value, or can be set to None
                to auto calculate the value over the required RPM value.
            speed_change_step (float | str): The step in each iteration the
                speed will be increased to.
            measure_time (float): The measurement time in seconds.
            axes (list[str]): The main and secondary axis.
            steppers (list[str]): The main and secondary stepper.
            a_axis_min (float): The minimum position of the main axis.
            a_axis_max (float): The maximum position of the main axis.
            acceleration (float): The acceleration for the movement.

        Returns:
            tuple[float, float, float]: The minimum speed, maximum speed and
                speed change step.
        """
        # In vibration measurement mode,
        # search and take registers from printer.cfg,
        # to set the speed range
        if self.measurement_mode == MeasurementMode.Resonances:
            rotation_dist = self.stepper_settings[steppers[0]].get("rotation_distance")
            # get gear ratio
            gear_ratio = self.stepper_settings[steppers[0]].get("gear_ratio")
            if not gear_ratio:  # can be () or None
                gear_ratio = ((1, 1),)
            gear_ratio = tuple(float(r) for r in gear_ratio[0])
            full_steps_per_rotation = self.stepper_settings[steppers[0]].get(
                "full_steps_per_rotation", 200
            )

            steps_multiplier = (
                full_steps_per_rotation
                / 200
                / (float(gear_ratio[0]) / float(gear_ratio[1]))
                * rotation_dist
                / 60  # to convert to mm/s from mm/min
            )

            if min_speed is None:
                min_speed = float(self.required_rpm[0] * steps_multiplier)
            else:
                min_speed = float(min_speed)

            if max_speed is None:
                max_required_speed = float(self.required_rpm[1] * steps_multiplier)
                max_speed = min(
                    (
                        (
                            -acceleration * measure_time
                            + (
                                (acceleration * measure_time) ** 2
                                + 4 * acceleration * (a_axis_max - a_axis_min)
                            )
                            ** 0.5
                        )
                        / 2
                    ),
                    max_required_speed,
                )
            else:
                max_speed = float(max_speed)

            if speed_change_step is None:
                speed_change_step = self.required_rpm[2] * steps_multiplier
            else:
                speed_change_step = float(speed_change_step)
        else:
            # Protect not defined speed & converting str -> float
            if min_speed is None or max_speed is None:
                raise self.printer.command_error(
                    "WARNING!!! Resonance speed must be defined"
                )
            min_speed, max_speed = float(min_speed), float(max_speed)
            speed_change_step = 1

        # Check speed limit
        if axes[0] == "z":
            max_velocity = self.settings["printer"].get("max_z_velocity", 0)
        else:
            max_velocity = self.settings["printer"].get("max_velocity")

        if max_speed > max_velocity:
            raise self.printer.command_error(
                f"WARNING!!! Required speed ({max_speed} mm/s) on axis ({axes[0]}) "
                f"is faster than kinematics allow ({max_velocity}), "
                f"please lower speed or increase speed limit in printer.cfg"
            )

        return min_speed, max_speed, speed_change_step

    def calculate_travel_distance(
        self,
        axes: list[str],
        a_axis_min: float,
        a_axis_max: float,
        max_speed: float,
        acceleration: float,
        measure_time: float,
        travel_distance: float | str,
    ) -> float:
        """Calculate travel distance.

        Args:
            axes (list[str]): The main and secondary axis.
            a_axis_min (float): The minimum position of the main axis.
            a_axis_max (float): The maximum position of the main axis.
            max_speed (float): The maximum speed of the main axis.
            acceleration (float): The acceleration of the main axis.
            measure_time (float): The measurement time in seconds.
            travel_distance (None | float): The travel distance, or can be set
                to None to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.

        Returns:
            float: The calculated travel distance.
        """
        # Calculate min required toolhead travel distance from speed,
        # acceleration and time
        accel_decel_distance = max_speed**2 / acceleration
        auto_travel_distance = accel_decel_distance + (max_speed * measure_time)
        if self.debug:
            self.gcode.respond_info(
                f"Acceleration & deceleration zone = {accel_decel_distance} mm"
            )
            self.gcode.respond_info(
                "Auto calculated min required travel distance = "
                f"{auto_travel_distance} mm"
            )

        # Protect exceeding axis limits & calculate travel distance
        if travel_distance is None:
            if a_axis_min + auto_travel_distance > a_axis_max:
                raise self.printer.command_error(
                    f"WARNING!!! Required travel distance on axis ({axes[0]}) "
                    f"({auto_travel_distance:.1f} mm) is longer than kinematics "
                    "allows, please lower speed or increase acceleration"
                )

            travel_distance = auto_travel_distance
        else:
            travel_distance = int(travel_distance)
            if a_axis_min + travel_distance > a_axis_max:
                travel_distance = a_axis_max - a_axis_min
                if travel_distance < auto_travel_distance:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) is "
                        "less than it should be, please increase acceleration "
                        "or lower speed"
                    )
                if travel_distance > auto_travel_distance:
                    self.gcode.respond_info(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        "is longer than kinematics allows, lowering..."
                    )
            elif travel_distance < auto_travel_distance:
                travel_distance = auto_travel_distance
                if a_axis_min + auto_travel_distance > a_axis_max:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        f"is less than required ({auto_travel_distance:.1f} mm), "
                        "and longer than kinematics allows, please lower "
                        "speed or increase acceleration"
                    )

        return travel_distance

    def display_process_summary(
        self,
        current_min: int,
        current_max: int,
        tbl_min: int,
        tbl_max: int,
        toff_min: int,
        toff_max: int,
        hstrt_min: int,
        hstrt_max: int,
        hend_min: int,
        hend_max: int,
        tpfd_min: int,
        tpfd_max: int,
        min_speed: float,
        max_speed: float,
        speed_change_step: float,
        iterations: int,
        search_method: SearchMethod,
        a_axis_min: float,
        travel_distance: float,
    ) -> None:
        """Display process information.

        Args:
            current_min (int): The minimum current value.
            current_max (int): The maximum current value.
            tbl_min (int): The minimum TBL value.
            tbl_max (int): The maximum TBL value.
            toff_min (int): The minimum TOFF value.
            toff_max (int): The maximum TOFF value.
            hstrt_min (int): The minimum HSTRT value.
            hstrt_max (int): The maximum HSTRT value.
            hend_min (int): The minimum HEND value.
            hend_max (int): The maximum HEND value.
            tpfd_min (int): The minimum TPFD value.
            tpfd_max (int): The maximum TPFD value.
            min_speed (float): The minimum speed value.
            max_speed (float): The maximum speed value.
            speed_change_step (float): The speed change step value.
            iterations (int): The number of iterations.
            search_method (SearchMethod): The search method.
            a_axis_min (float): The minimum position of the main axis.
            travel_distance (float): The travel distance value.
        """
        if self.measurement_mode == MeasurementMode.Resonances:
            # Resonance measurement mode uses the minimum values for registers.
            self.gcode.respond_info(
                f"Final max travel distance = {travel_distance:.1f} mm, "
                f"position min = {a_axis_min:.1f}, "
                f"traveling = {a_axis_min:.1f} --> {travel_distance + a_axis_min:.1f}"
            )
            self.gcode.respond_info(
                f"Start find resonances mode\n"
                f"Method     : {search_method}\n"
                f"speed      : {min_speed:.2f} --> {max_speed:.2f} mm/s with "
                f"{speed_change_step:.2f} mm/s step\n"
                f"current    : {current_min} mA\n"
                f"TBL        : {tbl_min}\n"
                f"TOFF       : {toff_min}\n"
                f"HSTRT      : {hstrt_min}\n"
                f"HEND       : {hend_min}"
            )
        else:
            self.gcode.respond_info(
                f"Final travel distance = {travel_distance:.1f} mm, "
                f"position min = {a_axis_min:.1f}, "
                f"traveling = {a_axis_min:.1f} --> {travel_distance + a_axis_min:.1f}"
            )
            self.gcode.respond_info(
                "Start of register enumeration mode\n"
                f"Method     : {search_method}\n"
                f"speed      : {min_speed:.2f} --> {max_speed:.2f} mm/s\n"
                f"current    : {current_min} --> {current_max} mA\n"
                f"iterations : {iterations}\n"
                f"TBL        : {tbl_min} --> {tbl_max}\n"
                f"TOFF       : {toff_min} --> {toff_max}\n"
                f"HSTRT      : {hstrt_min} --> {hstrt_max}\n"
                f"HEND       : {hend_min} --> {hend_max}\n"
                f"TPFD       : {tpfd_min} --> {tpfd_max}"
            )

    def home(self) -> None:
        """Home."""
        self.gcode.run_script_from_command("G28 X Y Z")
        self.toolhead.wait_moves()

    def get_static_acceleration(self) -> list[Accel_Measurement]:
        """Get static accelerometer samples.

        This will help removing the effect of gravity + static vibrations from
        real vibration data.

        Returns:
            list[Accel_Measurement]: The measurement data samples.
        """
        self.toolhead.wait_moves()
        with AccelerometerMeasure(self.accelerometer) as accelerometer_measurement:
            self.toolhead.dwell(5.0)
        return accelerometer_measurement.samples

    def measure_vibrations(
        self,
        coord_generator: CoordGenerator,
        travel_distance: float,
        speed: float,
    ) -> list[Accel_Measurement]:
        """Perform vibration measurement.

        Args:
            coord_generator (CoordGenerator): The coordinate generator.
            travel_distance (float): The travel distance.
            speed (float): The speed for the measurement.
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            name (str): The name of the measurement.

        Returns:
            list[Accel_Measurement]: The measurement data samples.
        """
        # Start accel_chip data collection
        with AccelerometerMeasure(
            accelerometer=self.accelerometer
        ) as accelerometer_measurement:
            next_coord = coord_generator.next(travel_distance)
            self.toolhead.manual_move(next_coord, speed)
        # Move to the initial position
        self.toolhead.manual_move(self.initial_position, self.travel_speed)
        coord_generator.current_coord.x = self.initial_position.x
        coord_generator.current_coord.y = self.initial_position.y
        coord_generator.current_coord.z = self.initial_position.z
        self.toolhead.wait_moves()
        self.number_of_real_samples += 1
        return accelerometer_measurement.samples

    def get_fclk(self) -> float:
        """Return the driver clock, preferring the value set in the config.

        Returns:
            float: The driver clock in MHz.
        """
        if self.fclk is not None:
            return self.fclk
        return DRIVER_FCLK.get(self.driver, DEFAULT_FCLK)

    def calculate_frequency(self, tbl: int, toff: int) -> float:
        """Calculate the upper limit of the chopper frequency.

        As per the datasheets, the two slow decay phases and the two blank times
        per chopper cycle put an upper limit to the chopper frequency. The actual
        frequency is lower, as it also includes the on and fast decay phases.

        Args:
            tbl (int): The TBL value.
            toff (int): The TOFF value.

        Returns:
            float: The calculated frequency in Hz.
        """
        tbl_clocks = TBL_CLOCKS.get(self.driver, DEFAULT_TBL_CLOCKS)[tbl]
        return self.get_fclk() * 1_000_000 / (2 * (24 + 32 * toff) + 2 * tbl_clocks)

    def estimate_chopper_frequency(self, toff: int) -> float:
        """Estimate the chopper frequency the way the datasheets do.

        Args:
            toff (int): The TOFF value.

        Returns:
            float: The estimated chopper frequency in Hz.
        """
        slow_decay_clocks = 2 * (24 + 32 * toff)
        return self.get_fclk() * 1_000_000 * SLOW_DECAY_DUTY / slow_decay_clocks

    def get_initial_direction(self, axes: list[str]) -> Coord:
        """Return the initial direction based on the axes and kinematics.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            Coord: The initial direction.
        """
        initial_direction = Coord((1, 0, 0))
        if axes[0] == "x":
            if self.kinematics == "corexy":
                initial_direction = Coord((1, 1, 0)).unitize()
            else:
                initial_direction = Coord((1, 0, 0))
        elif axes[0] == "y":
            if self.kinematics == "corexy":
                initial_direction = Coord((-1, 1, 0)).unitize()
            else:
                initial_direction = Coord((0, 1, 0))
        elif axes[0] == "z":
            initial_direction = Coord((0, 0, 1))
        return initial_direction

    def wait_for_file_write(self, data_path: str) -> None:
        """Wait for file write to complete.

        Args:
            data_path (str): The path to the CSV file.
        """
        start_time = self.reactor.monotonic()
        max_wait_time = 10  # seconds
        prev_size = -1
        curr_size = 0

        while True:
            curr_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
            if curr_size > 0 and curr_size == prev_size:
                break

            prev_size = curr_size

            # Yield control back to Klipper for 100ms
            self.reactor.pause(self.reactor.monotonic() + 0.1)

            if self.reactor.monotonic() - start_time > max_wait_time:
                self.gcode.respond_info(f"Timeout waiting for file: {data_path}")
                break

    def start_raw_capture(self, metadata: dict) -> None:
        """Create the folder for the raw samples and save the metadata into it.

        Args:
            metadata (dict): The information needed to analyse the raw samples.
        """
        date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_folder = os.path.join(
            RESULTS_FOLDER, "raw", f"{date_stamp}_{metadata['steppers'][0]}"
        )
        try:
            os.makedirs(self.raw_folder, exist_ok=True)
            with open(os.path.join(self.raw_folder, "metadata.json"), "w") as file:
                json.dump(metadata, file, indent=4, default=str)
        except OSError as e:
            self.gcode.respond_info(f"WARNING!!! Not saving raw samples: {e}")
            self.raw_folder = None
            return
        self.gcode.respond_info(f"Raw samples will be saved to: {self.raw_folder}")

    def save_raw_samples(
        self,
        file_name: str,
        samples: list[Accel_Measurement],
        **extra: float,
    ) -> None:
        """Save the raw accelerometer samples, if requested.

        A failure to save doesn't stop the tuning, as the samples are only for
        offline analysis.

        Args:
            file_name (str): The file name without the extension.
            samples (list[Accel_Measurement]): The accelerometer samples.
            **extra (float): Other values to save with the samples.
        """
        if self.raw_folder is None:
            return
        times = np.array([sample.time for sample in samples], dtype=np.float64)
        accel = np.array(
            [(sample.accel_x, sample.accel_y, sample.accel_z) for sample in samples],
            dtype=np.float32,
        ).reshape(-1, 3)
        start_time = times[0] if times.size else np.nan
        try:
            np.savez_compressed(
                os.path.join(self.raw_folder, f"{file_name}.npz"),
                start_time=start_time,
                time=(times - start_time).astype(np.float32),
                accel=accel,
                **extra,
            )
        except OSError as e:
            self.gcode.respond_info(f"WARNING!!! Stopped saving raw samples: {e}")
            self.raw_folder = None

    def calc_static_acceleration_magnitude(
        self, samples: list[Accel_Measurement]
    ) -> tuple[float, float, float]:
        """Calculate static acceleration data from CSV file.

        Args:
            samples (list[Accel_Measurement]): The list of acceleration data
                samples.

        Returns:
            tuple[float, float, float]: Mean static acceleration vector.
        """
        accel_x = np.array([sample.accel_x for sample in samples])
        accel_y = np.array([sample.accel_y for sample in samples])
        accel_z = np.array([sample.accel_z for sample in samples])
        # Return the mean of each axis as the baseline vector
        return (
            float(np.mean(accel_x)),
            float(np.mean(accel_y)),
            float(np.mean(accel_z)),
        )

    def calc_magnitude(
        self, samples: list[Accel_Measurement], static_data: tuple[float, float, float]
    ) -> float:
        """Calculate median magnitude of acceleration data from CSV file.

        Args:
            samples (list[Accel_Measurement]): The list of acceleration data samples.
            static_data (tuple[float, float, float]): Mean static acceleration
                vector.

        Returns:
            float: Median magnitude of acceleration data.
        """
        accel_x = np.array([sample.accel_x for sample in samples]) - static_data[0]
        accel_y = np.array([sample.accel_y for sample in samples]) - static_data[1]
        accel_z = np.array([sample.accel_z for sample in samples]) - static_data[2]

        # calculate the sample rate
        duration = samples[-1].time - samples[0].time
        number_of_samples = len(samples)
        sample_rate = number_of_samples / duration

        magnitudes = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Create a 4th order Butterworth filter
        cutoff_freq = 300  # Hz
        nyquist_freq = sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist_freq

        b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
        filtered_magnitudes = signal.filtfilt(b, a, magnitudes)

        # Percentile Trimming (The "Middle 60%")
        lower_bound = np.percentile(filtered_magnitudes, 20)
        upper_bound = np.percentile(filtered_magnitudes, 80)
        trimmed_magnitudes = filtered_magnitudes[
            (filtered_magnitudes >= lower_bound) & (filtered_magnitudes <= upper_bound)
        ]
        return float(np.median(trimmed_magnitudes))

    def chopper_tune(
        self,
        axis: str,
        current_min: None | int = None,
        current_max: None | int = None,
        tbl_min: int = DEFAULT_REGISTER_VALUES["TBL"][0],
        tbl_max: int = DEFAULT_REGISTER_VALUES["TBL"][1],
        toff_min: int = DEFAULT_REGISTER_VALUES["TOFF"][0],
        toff_max: int = DEFAULT_REGISTER_VALUES["TOFF"][1],
        hstrt_hend_max: int = DEFAULT_REGISTER_VALUES["HSTRT_HEND_MAX"],
        hstrt_min: int = DEFAULT_REGISTER_VALUES["HSTRT"][0],
        hstrt_max: int = DEFAULT_REGISTER_VALUES["HSTRT"][1],
        hend_min: int = DEFAULT_REGISTER_VALUES["HEND"][0],
        hend_max: int = DEFAULT_REGISTER_VALUES["HEND"][1],
        tpfd_min: int = DEFAULT_REGISTER_VALUES["TPFD"][0],
        tpfd_max: int = DEFAULT_REGISTER_VALUES["TPFD"][1],
        min_speed: None | int = None,
        max_speed: None | int = None,
        speed_change_step: None | int = None,
        search_method: SearchMethod = SearchMethod.BruteForce,
        travel_distance: None | int = None,
        direction: int = 1,
        accel_chip_name: str = "default",
        run_plotter: bool = True,
        compare_with: None | str = None,
        save_raw: bool = False,
        toff_range_requested: bool = False,
    ) -> None | dict:
        """Measure vibrations and tune stepper motors for low noise.

        Args:
            axis (str): Axis to tune. Should be one of ["x", "y", "z"].
            current_min (None | int): Minimum steeper current in mA, or use
                None to set the current to the `run_current` value.
            current_max (None | int): Maximum steeper current in mA, or use
                None to set the current to the `run_current` value.
            tbl_min (int): The min TBL value.
            tbl_max (int): The max TBL value.
            toff_min (int): The min TOFF value.
            toff_max (int): The max TOFF value.
            hstrt_hend_max (int): The max HSTRT_HEND value
            hstrt_min (int): The min HSTRT value.
            hstrt_max (int): The max HSTRT value.
            hend_min (int): The min HEND value.
            hend_max (int): The max HEND value.
            tpfd_min (int): The min TPFD value.
            tpfd_max (int): The max TPFD value.
            min_speed (None | int): The in speed value, or can be set to
                None to auto calculate the value over the required RPM
                value.
            max_speed (None | int): The max speed value, or can be set to
                None to auto calculate the value over the required RPM
                value.
            speed_change_step (None | int): The step in each iteration the speed
                will be increased to.
            search_method (SearchMethod): The search method, can be one of
                [SearchMethod.BruteForce, SearchMethod.Adaptive], default value
                is SearchMethod.BruteForce.
            travel_distance (None | int): The travel distance, or can be set to
                None to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.
            direction (int): The movement direction, can be 1 or -1.
                1 means starting from the minimum position to maximum position,
                -1 means starting from the maximum position to minimum position.
            accel_chip (str): The name of the acceleration chip.
            run_plotter (bool): If set to True, the magnitude graphs will be
                generated after the vibration measurements are completed.
            compare_with (None | str): The name of the previous sample set to
                compare with.
            save_raw (bool): If set to True, the raw accelerometer samples of each
                measurement are saved for offline analysis.
            toff_range_requested (bool): If set to True, the progressive search
                tests all TOFF values from `toff_min` to `toff_max`, otherwise only
                the ones with a recommended chopper frequency.

        Returns:
            None | dict: The best parameters found, or None if tuning was not
                done using the adaptive method.
        """
        # reset previous run data
        self.reset_sample_data()
        self.reset_registers()
        self.raw_folder = None
        self.toff_range_requested = toff_range_requested

        self.search_method = search_method

        # Force brute_force in vibration measurement mode
        if self.measurement_mode == MeasurementMode.Resonances:
            self.search_method = SearchMethod.BruteForce

        self.gcode.respond_info(f"Selected {self.search_method} as search method")

        measure_time = self.measure_time / 1000
        # Find the steppers count of the main axis
        self.registers["stepper_count"] = self.get_stepper_count(axis)

        self.driver, self.sense_resistor = self.detect_driver(stepper=axis)
        tpfd_min, tpfd_max = self.validate_tpfd_values(self.driver, tpfd_min, tpfd_max)
        toff_min, toff_max = self.validate_toff_values(toff_min, toff_max)

        axes, steppers = self.get_axes_and_steppers(axis)

        a_axis_min, a_axis_max, a_axis_mid, b_axis_mid = self.get_axis_limits(axes)
        acceleration, self.travel_speed = self.get_travel_speed_and_acceleration(axes)
        accel_chip_name = self.get_accelerometer_chip(accel_chip_name, axis)
        self.accelerometer = self.printer.lookup_object(accel_chip_name)

        current_min, current_max = self.get_current_range(
            self.measurement_mode, current_min, current_max, steppers
        )

        if self.measurement_mode == MeasurementMode.Resonances:
            # In vibration measurement mode,
            # search and take registers from printer.cfg
            (
                tbl_min,
                tbl_max,
                toff_min,
                toff_max,
                hstrt_min,
                hstrt_max,
                hend_min,
                hend_max,
                tpfd_min,
                tpfd_max,
            ) = self.get_default_stepper_parameters(steppers)

        (min_speed, max_speed, speed_change_step) = self.configure_speed_limits(
            min_speed,
            max_speed,
            speed_change_step,
            measure_time,
            axes,
            steppers,
            a_axis_min,
            a_axis_max,
            acceleration,
        )

        self.min_speed = min_speed
        if self.measurement_mode == MeasurementMode.Vibrations:
            self.validate_chopper_mode(axes, steppers)

        travel_distance = self.calculate_travel_distance(
            axes,
            a_axis_min,
            a_axis_max,
            max_speed,
            acceleration,
            measure_time,
            travel_distance,
        )

        # Info message
        self.display_process_summary(
            current_min,
            current_max,
            tbl_min,
            tbl_max,
            toff_min,
            toff_max,
            hstrt_min,
            hstrt_max,
            hend_min,
            hend_max,
            tpfd_min,
            tpfd_max,
            min_speed,
            max_speed,
            speed_change_step,
            self.iterations,
            self.search_method,
            a_axis_min,
            travel_distance,
        )

        # Home regardless of previous homing state
        self.home()
        home_pos = Coord(self.toolhead.get_position())

        # Get initial position and direction
        self.initial_position = {
            "x": Coord((a_axis_mid, b_axis_mid, home_pos.z)),
            "y": Coord((b_axis_mid, a_axis_mid, home_pos.z)),
            "z": Coord((b_axis_mid, home_pos.y, a_axis_mid)),
        }[axes[0]]
        center_position = self.initial_position
        self.initial_direction = self.get_initial_direction(axes)
        if direction == -1:
            self.initial_direction = self.initial_direction * -1

        # try:
        #     self.toolhead.set_max_velocities(None, acceleration, None, None)
        # except AttributeError:
        # This is either an older Klipper version or Kalico
        self.gcode.run_script_from_command(
            f"SET_VELOCITY_LIMIT ACCEL={acceleration}\n"
            f"SET_VELOCITY_LIMIT ACCEL_TO_DECEL={acceleration}"
        )
        self.toolhead.wait_moves()

        # move away from the middle exactly half or a travel distance
        self.initial_position -= self.initial_direction * (travel_distance / 2)
        self.toolhead.manual_move(self.initial_position, self.travel_speed)
        self.toolhead.wait_moves()

        # Measure accelerometer noise
        samples = self.get_static_acceleration()
        self.static_acceleration_vector = self.calc_static_acceleration_magnitude(
            samples
        )
        self.static_acceleration_magnitude = float(
            np.linalg.norm(self.static_acceleration_vector)
        )

        if save_raw:
            accel_chip_settings = self.settings.get(accel_chip_name, {})
            self.start_raw_capture(
                {
                    "axis": axis,
                    "kinematics": self.kinematics,
                    "steppers": steppers,
                    "driver": self.driver,
                    "fclk": self.get_fclk(),
                    "sense_resistor": self.sense_resistor,
                    "stepper_settings": self.stepper_settings[steppers[0]],
                    "driver_settings": self.driver_settings[steppers[0]],
                    "accelerometer": accel_chip_name,
                    "accelerometer_rate": accel_chip_settings.get("rate"),
                    "measurement_mode": str(self.measurement_mode),
                    "search_method": str(self.search_method),
                    "iterations": self.iterations,
                    "direction": list(self.initial_direction),
                    "acceleration": acceleration,
                    "travel_distance": travel_distance,
                    "measure_time": measure_time,
                    "static_acceleration_vector": self.static_acceleration_vector,
                }
            )
            self.save_raw_samples("static", samples)

        # set the global start time here
        self.global_start_time = self.reactor.monotonic()

        nv = self.static_acceleration_vector
        self.gcode.respond_info(
            f"Static acceleration vector    = {nv[0]:0.1f} {nv[1]:0.1f} {nv[2]:0.1f} mm/s²\n"
            f"Static acceleration magnitude = {self.static_acceleration_magnitude:.1f} mm/s²\n"
            "(HINT: this should be close to earth's gravity of 9806 mm/s²)"
        )

        # Create the coordinate generator
        coord_generator = CoordGenerator(
            direction=self.initial_direction,
            start_coord=self.initial_position,
        )

        # calculated vars
        self.travel_distance = travel_distance
        self.coord_generator = coord_generator
        self.accel_chip_name = accel_chip_name
        self.steppers = steppers
        self.current = current_min

        # bounds
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_change_step = speed_change_step
        self.current_min = current_min
        self.current_max = current_max

        self.tbl_min = tbl_min
        self.tbl_max = tbl_max
        self.toff_min = toff_min
        self.toff_max = toff_max
        self.hstrt_min = hstrt_min
        self.hstrt_max = hstrt_max
        self.hstrt_hend_max = hstrt_hend_max
        self.hend_min = hend_min
        self.hend_max = hend_max
        self.tpfd_min = tpfd_min
        self.tpfd_max = tpfd_max

        self.bounds = [
            (self.current_min, self.current_max),
            (self.tbl_min, self.tbl_max),
            (self.toff_min, self.toff_max),
            (self.hstrt_min, self.hstrt_max),
            (self.hend_min, self.hend_max),
            (self.tpfd_min, self.tpfd_max),
        ]
        best_parameters = self.search_best_parameters()
        if self.measurement_mode == MeasurementMode.Resonances:
            max_vibrations_and_speed = sorted(
                self.speed_vs_vibrations, key=lambda x: x[1]
            )[-1]
            self.gcode.respond_info(
                "Finding resonances took "
                f"{self.convert_seconds_to_hms(self.get_time_elapsed())}\n"
                "Max vibrations seems to be at "
                f"{max_vibrations_and_speed[0]:0.1f} mm/s"
            )

        self.toolhead.dwell(0.5)
        self.toolhead.manual_move(center_position, self.travel_speed)
        self.toolhead.wait_moves()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_plotter:
            self.gcode.respond_info("Magnitude graphs generation...")
            self.gcode.respond_info("This may take a while, please wait")
            self.plot_data(date_stamp=now)

        # Store data
        self.store_data(date_stamp=now)

        if self.measurement_mode != MeasurementMode.Resonances:
            # Save Config
            self.save_configs(best_parameters)

            # Compare best result with previous samples
            if best_parameters and compare_with is not None:
                sample_name = self.generate_sample_name(
                    best_parameters["current"],
                    best_parameters["tbl"],
                    best_parameters["toff"],
                    best_parameters["hstrt"],
                    best_parameters["hend"],
                    best_parameters["tpfd"],
                    best_parameters["speed"],
                )
                self.compare_results(
                    sample_name,
                    self.samples[sample_name],
                    compare_with,
                )

        # reset samples related data
        self.total_expected_samples = -1
        self.total_expected_samples_is_max = False
        self.number_of_samples = 0
        self.number_of_real_samples = 0
        self.samples = {}

        return best_parameters

    def generate_sample_name(
        self,
        current: int,
        tbl: int,
        toff: int,
        hstrt: int,
        hend: int,
        tpfd: int,
        speed: float,
    ) -> str:
        """Generate a sample name based on the parameters.

        Args:
            current (int): The current value.
            tbl (int): The TBL value.
            toff (int): The TOFF value.
            hstrt (int): The HSTRT value.
            hend (int): The HEND value.
            tpfd (int): The TPFD value.
            speed (float): The speed value.

        Returns:
            str: The generated sample name.
        """
        freq = self.calculate_frequency(tbl, toff)
        return (
            f"current={current}_"
            f"tbl={tbl}_"
            f"toff={toff}_"
            f"hstrt={hstrt}_"
            f"hend={hend}_"
            f"tpfd={tpfd}_"
            f"speed={speed:.2f}_"
            f"freq={freq / 1000:.2f}kHz"
        )

    def compare_results(
        self,
        sample_name: str,
        sample_result: float,
        previous_samples_name: str,
    ) -> tuple[None | float, None | float]:
        """Compare current sample result with previous samples.

        Args:
            sample_name (str): The name of the current sample.
            sample_result (float): The result of the current sample.
            previous_samples_name (str): The name of the previous samples file.

        Returns:
            tuple[None | float, None | float]: The percentile of the given
                value within the previous samples, and the previously measured
                value for the sample name.
        """
        previous_sample_path = os.path.join(
            RESULTS_FOLDER, f"{previous_samples_name}.json"
        )
        if not os.path.exists(previous_sample_path):
            return None, None

        with open(previous_sample_path) as f:
            previous_samples = json.load(f)

        previous_values = np.array(list(previous_samples.values()))
        previous_sample_result = previous_samples.get(sample_name)
        percentile = float((previous_values <= sample_result).mean())

        # report comparison results
        message = (
            "Comparison with previous samples:\n---------------------------------\n"
        )
        if percentile is not None:
            message += f"Sample    : {sample_name}\nPercentile: {percentile:.2%}\n"
        if previous_sample_result is not None:
            message += (
                f"Previous value\n"
                f"for the same\n"
                f"sample    : {previous_sample_result:.1f} mm/s²"
            )
        if percentile is None and previous_sample_result is None:
            message += "No previous sample data found for comparison!!!\n"
        self.gcode.respond_info(message)
        return percentile, previous_sample_result

    def get_time_elapsed(self) -> float:
        """Get elapsed time since the start of the process.

        Returns:
            float: The time elapsed in seconds.
        """
        return self.reactor.monotonic() - self.global_start_time

    def get_remaining_time(self) -> float:
        """Estimate remaining time for the process.

        Returns:
            float: The estimated remaining time in seconds.
        """
        time_passed = self.get_time_elapsed()
        total_estimated_process_time = (
            time_passed / self.number_of_samples * self.total_expected_samples
        )
        return max(total_estimated_process_time - time_passed, 0)

    def convert_seconds_to_hms(self, seconds: float) -> str:
        """Convert seconds to hours, minutes, and seconds.

        Args:
            seconds (float): The time in seconds.

        Returns:
            str: The time in "HHh MMm SSs" format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        # do not include hours or minutes if zero
        hours_str = f"{hours}h" if hours > 0 else ""
        minutes_str = f"{minutes}m" if minutes > 0 or hours > 0 else ""

        return f"{hours_str}{minutes_str}{secs}s"

    def progress_report(
        self, measured_vibrations: float = 0.0, iteration: int = 0, cached: bool = False
    ) -> None:
        """Report progress.

        Args:
            measured_vibrations (float): The measured vibrations.
            iteration (int): The current iteration. Only report progress on the
                first iteration.
            cached (bool): Whether the vibrations value is from a cached sample.
        """
        if iteration == 0:
            if self.total_expected_samples > 0 and self.total_expected_samples_is_max:
                self.gcode.respond_info(
                    f"Sample {'(cached)' if cached else '        '}: "
                    f"{self.number_of_samples} (up to "
                    f"{self.total_expected_samples}) | "
                    f"E: {self.convert_seconds_to_hms(self.get_time_elapsed())} | "
                    f"R: up to "
                    f"{self.convert_seconds_to_hms(self.get_remaining_time())}"
                )
            elif self.total_expected_samples > 0:
                percent_complete = (
                    self.number_of_samples / self.total_expected_samples
                ) * 100
                self.gcode.respond_info(
                    f"Sample {'(cached)' if cached else '        '}: "
                    f"{self.number_of_samples}/"
                    f"{self.total_expected_samples} "
                    f"({percent_complete:0.1f}%) | "
                    f"E: {self.convert_seconds_to_hms(self.get_time_elapsed())} | "
                    f"R: {self.convert_seconds_to_hms(self.get_remaining_time())}"
                )
            else:
                self.gcode.respond_info(
                    f"Sample         : {self.number_of_samples} | "
                    f"E: {self.convert_seconds_to_hms(self.get_time_elapsed())}"
                )
        self.gcode.respond_info(
            f"Iteration {iteration + 1:<5d}: {measured_vibrations:0.1f} mm/s²"
        )

    def objective_function(self, params: list[float]) -> float:
        """Objective function for optimization.

        Args:
            params (list[float]): The parameters to optimize.

        Returns:
            float: The average measured vibrations.
        """
        # This command holds the G-Code mutex, which FIRMWARE_RESTART needs. So,
        # stop right away after an emergency stop.
        if self.printer.is_shutdown():
            raise self.printer.command_error("Printer is shutdown")

        self.gcode.respond_info("-------------------------------")
        current, tbl, toff, hstrt, hend, tpfd, speed = [round(p) for p in params]

        # convert speed back to the correct range
        speed = float(speed) / 100

        # Dump TMC settings
        self.gcode.respond_info(
            f"tbl={tbl} "
            f"toff={toff} "
            f"hstrt={hstrt} "
            f"hend={hend} "
            f"tpfd={tpfd} "
            f"current={current} "
            f"speed={speed:.2f}"
        )

        # As per the datasheets, TOFF=1 should only be used with TBL >= 2
        if (
            self.measurement_mode == MeasurementMode.Vibrations
            and toff == 1
            and tbl < 2
        ):
            self.gcode.respond_info("Penalizing toff=1 with tbl < 2: inf mm/s²")
            self.number_of_samples += 1  # consider this as a sample
            return float("inf")

        # penalize hstrt + hend > hstrt_hend_max
        if hstrt + hend > self.hstrt_hend_max:
            self.gcode.respond_info(
                f"Penalizing hstrt + hend > {self.hstrt_hend_max}: inf mm/s²"
            )
            self.number_of_samples += 1  # consider this as a sample
            return float("inf")

        # Set tbl, toff, hend, and hstrt values
        self.apply_registers(self.steppers, "curr", current)
        self.apply_registers(self.steppers, "tbl", tbl)
        self.apply_registers(self.steppers, "toff", toff)
        self.apply_registers(self.steppers, "hstrt", hstrt)
        self.apply_registers(self.steppers, "hend", hend)
        self.apply_registers(self.steppers, "tpfd", tpfd)
        sample_name = self.generate_sample_name(
            current=current,
            tbl=tbl,
            toff=toff,
            hstrt=hstrt,
            hend=hend,
            tpfd=tpfd,
            speed=speed,
        )
        if sample_name in self.samples:
            cached_vibrations = self.samples[sample_name]
            self.number_of_samples += 1  # consider this as a sample
            self.progress_report(cached_vibrations, cached=True)
            return cached_vibrations

        total_vibrations = 0
        self.number_of_samples += 1
        travel_distance = self.travel_distance
        if self.measurement_mode == MeasurementMode.Resonances:
            travel_distance = self.travel_distance * (speed / self.max_speed)

        for iteration in range(self.iterations):
            samples = self.measure_vibrations(
                self.coord_generator,
                travel_distance,
                speed,
            )
            measured_vibrations = self.calc_magnitude(
                samples=samples,
                static_data=self.static_acceleration_vector,
            )

            self.save_raw_samples(
                f"{sample_name}_iteration={iteration + 1}",
                samples,
                speed=speed,
                score=measured_vibrations,
            )

            total_vibrations += measured_vibrations
            self.progress_report(measured_vibrations, iteration)

            # Allow other Klipper tasks to run between iterations
            self.reactor.pause(self.reactor.monotonic() + 0.05)

        total_vibrations /= self.iterations
        if self.iterations > 1:
            self.gcode.respond_info(f"Mean vibrations: {total_vibrations:0.1f} mm/s²")
        # store best result for the last report
        self.best_result = min(self.best_result, total_vibrations)

        self.samples[sample_name] = total_vibrations
        self.speed_vs_vibrations.append((speed, total_vibrations))

        # Allow other Klipper tasks to run between iterations
        self.reactor.pause(self.reactor.monotonic() + 0.05)

        return total_vibrations

    def progressive_sweep(
        self,
        best_params: list[int],
        candidates: list[dict[int, int]],
        preference: None | list[int] = None,
        patience: None | int = None,
        measure_at_least: int = 0,
    ) -> list[int]:
        """Measure the given candidates and return the parameters of the picked one.

        Args:
            best_params (list[int]): The current best parameters.
            candidates (list[dict[int, int]]): The parameter index and value pairs
                to change for each candidate, in measurement order.
            preference (None | list[int]): The candidate indices from most to least
                preferred, among the ones with similar vibrations. By default, the
                earlier the candidate is measured, the more it is preferred.
            patience (None | int): If given, stop after this many consecutive
                candidates without a real improvement.
            measure_at_least (int): The number of candidates to measure before
                `patience` can stop the sweep.

        Returns:
            list[int]: The new best parameters, which are the same as the given
                ones if no candidate is valid.
        """
        # Taken before measuring, to compare this sweep against the earlier ones.
        reference = self.best_result
        scores = {}
        lowest = float("inf")
        stale = 0
        for index, candidate in enumerate(candidates):
            params = list(best_params)
            for param_index, value in candidate.items():
                params[param_index] = value
            scores[index] = self.objective_function(params)
            if scores[index] < lowest * (1 - self.plateau_tolerance):
                stale = 0
            else:
                stale += 1
            lowest = min(lowest, scores[index])
            if (
                patience is not None
                and stale >= patience
                and index + 1 >= measure_at_least
            ):
                break

        picked = pick_preferred(
            scores,
            list(range(len(candidates))) if preference is None else preference,
            self.plateau_tolerance,
            reference,
        )
        if picked is None:
            return best_params

        params = list(best_params)
        for param_index, value in candidates[picked].items():
            params[param_index] = value
        return params

    def format_toff_values(self, toff_values: list[int]) -> str:
        """Format TOFF values with their estimated chopper frequency.

        Args:
            toff_values (list[int]): The TOFF values.

        Returns:
            str: The values, e.g. "3 (25.0kHz), 4 (19.7kHz)".
        """
        return ", ".join(
            f"{toff} ({self.estimate_chopper_frequency(toff) / 1000:.1f}kHz)"
            for toff in toff_values
        )

    def get_progressive_toff_values(self) -> list[int]:
        """Return the TOFF values to test, from most to least preferred.

        Returns:
            list[int]: All the requested values if the TOFF range was given by the
                user, otherwise the ones that give an inaudible chopper frequency
                that is not too high, or all of them if none does.
        """
        requested = list(range(self.toff_min, self.toff_max + 1))
        band_min, band_max = self.min_chopper_frequency, MAX_CHOPPER_FREQ
        band_text = f"{band_min / 1000:g}-{band_max / 1000:g}kHz"
        in_band = [
            toff
            for toff in requested
            if band_min <= self.estimate_chopper_frequency(toff) <= band_max
        ]
        out_of_band = [toff for toff in requested if toff not in in_band]
        if self.toff_range_requested or not in_band:
            values = requested
            if out_of_band:
                reason = (
                    "they were requested"
                    if self.toff_range_requested
                    else "none of the requested values is in that band"
                )
                self.gcode.respond_info(
                    f"Testing TOFF {self.format_toff_values(out_of_band)} although "
                    f"they are outside the {band_text} chopper frequency band, as "
                    f"{reason}. Low frequencies can cause an audible whine."
                )
        else:
            values = in_band
            if out_of_band:
                self.gcode.respond_info(
                    f"Skipping TOFF {self.format_toff_values(out_of_band)}, they are "
                    f"outside the {band_text} chopper frequency band (see "
                    "min_chopper_frequency). Set TOFF_MIN and TOFF_MAX to test "
                    "them anyway."
                )
        self.gcode.respond_info(
            f"Testing TOFF {self.format_toff_values(values)} "
            "(estimated chopper frequency)"
        )
        return sorted(
            values,
            key=lambda toff: abs(
                self.estimate_chopper_frequency(toff) - CHOPPER_FREQ_TARGET
            ),
        )

    def perform_brute_force_search(self) -> list[int]:
        """Perform brute-force search for optimal parameters.

        Returns:
            list[int]: The best parameters found.
        """
        # Brute-force search
        bounds = [
            slice(self.current_min, self.current_max + 1, self.current_change_step),
            slice(self.tbl_min, self.tbl_max + 1, 1),
            slice(self.toff_min, self.toff_max + 1, 1),
            slice(self.hstrt_min, self.hstrt_max + 1, 1),
            slice(self.hend_min, self.hend_max + 1, 1),
            slice(self.tpfd_min, self.tpfd_max + 1, 1),
            slice(
                int(self.min_speed * 100),
                int(self.max_speed * 100) + 1,
                int(self.speed_change_step * 100),
            ),
        ]

        # update total expected samples
        total_current_steps = (
            self.current_max - self.current_min
        ) // self.current_change_step + 1
        total_tbl_steps = self.tbl_max - self.tbl_min + 1
        total_toff_steps = self.toff_max - self.toff_min + 1
        total_hstrt_steps = self.hstrt_max - self.hstrt_min + 1
        total_hend_steps = self.hend_max - self.hend_min + 1
        total_tpfd_steps = self.tpfd_max - self.tpfd_min + 1
        total_speed_steps = (
            int(self.max_speed * 100) - int(self.min_speed * 100)
        ) // int(self.speed_change_step * 100) + 1
        self.total_expected_samples = (
            total_current_steps
            * total_tbl_steps
            * total_toff_steps
            * total_hstrt_steps
            * total_hend_steps
            * total_tpfd_steps
            * total_speed_steps
        )

        result = brute(
            self.objective_function,
            bounds,
            finish=None,  # disable local optimization at the end
        )

        return [round(p) for p in result]

    def perform_adaptive_search(self) -> list[int]:
        """Perform adaptive search for optimal parameters.

        Returns:
            list[int]: The best parameters found.
        """
        # Adaptive search
        # 'strategy' and 'popsize' are tuned to reduce total measurements
        # 'tol' can be higher since our parameters are discrete

        bounds = [
            (self.current_min, self.current_max),
            (self.tbl_min, self.tbl_max),
            (self.toff_min, self.toff_max),
            (self.hstrt_min, self.hstrt_max),
            (self.hend_min, self.hend_max),
            (self.tpfd_min, self.tpfd_max),
            (self.min_speed * 100, self.max_speed * 100),
        ]

        number_of_changing_params = 0
        for bound in bounds:
            if bound[0] != bound[1]:
                number_of_changing_params += 1
        number_of_changing_params += 1  # add one for safety

        maxiter = 10
        popsize = 5
        self.total_expected_samples = maxiter * popsize * number_of_changing_params
        result = differential_evolution(
            self.objective_function,
            bounds,
            init="sobol",
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,  # Total evaluations = maxiter * popsize * N_params
            tol=0.05,  # Higher tolerance for discrete parameters
            mutation=(0.3, 0.8),
            recombination=0.9,  # Increased for faster parameter mixing
            polish=False,  # Polish uses local minimize, which we avoid for discrete
            updating="immediate",  # Uses best results immediately
        )

        return [round(p) for p in result.x]

    def perform_progressive_search(self) -> list[int]:
        """Perform progressive search for optimal parameters.

        This follows the SpreadCycle chapter of the TMC2209 and TMC5160 datasheets:
        start from the recommended TOFF and TBL with a medium hysteresis, then
        optimize the hysteresis. A wrong hysteresis or a too low chopper frequency
        doesn't necessarily increase the vibrations, but does increase the chopper
        noise that an accelerometer can not sense. So, TOFF values that give a
        chopper frequency below `min_chopper_frequency` are not tested, and among
        similar results, the values favored by the datasheets are used.

        Returns:
            list[int]: The best parameters found.
        """
        self.gcode.respond_info(
            "Starting Progressive (Trinamic datasheet) Optimization..."
        )
        if self.min_speed != self.max_speed:
            self.gcode.respond_info(
                f"Progressive search uses a single speed: {self.min_speed:.2f} mm/s, "
                "ignoring MAX_SPEED."
            )
        if self.current_min != self.current_max:
            self.gcode.respond_info(
                f"Progressive search uses a single current: {self.current_min} mA, "
                "ignoring CURRENT_MAX_MA."
            )

        toff_values = self.get_progressive_toff_values()
        tbl_values = [
            tbl
            for tbl in range(self.tbl_min, self.tbl_max + 1)
            if tbl in RECOMMENDED_TBL
        ] or list(range(self.tbl_min, self.tbl_max + 1))
        tbl_values.sort(key=lambda tbl: abs(tbl - DEFAULT_TBL))

        # Start with the recommended values, within the requested limits.
        start_hstrt = self.hstrt_min
        start_hend = min(
            max(
                DEFAULT_EFFECTIVE_HYSTERESIS + HYSTERESIS_REGISTER_OFFSET - start_hstrt,
                self.hend_min,
            ),
            self.hend_max,
        )
        best_params = [
            self.current_min,
            tbl_values[0],
            toff_values[0],
            start_hstrt,
            start_hend,
            self.tpfd_min,
            round(self.min_speed * 100),
        ]
        hend_values = list(range(self.hend_min, self.hend_max + 1))
        hstrt_values = list(range(self.hstrt_min, self.hstrt_max + 1))
        tpfd_values = list(range(self.tpfd_min, self.tpfd_max + 1))
        self.total_expected_samples = (
            2 * len(toff_values)
            + len(tbl_values)
            + len(hend_values)
            + len(hstrt_values)
            + (len(tpfd_values) if self.driver in DRIVERS_SUPPORTING_TPFD else 0)
        )
        # Sweeps stop early when there is no improvement.
        self.total_expected_samples_is_max = True

        def sweep_toff(params: list[int]) -> list[int]:
            ascending = sorted(toff_values)
            return self.progressive_sweep(
                params,
                [{PARAM_TOFF: toff} for toff in ascending],
                preference=[ascending.index(toff) for toff in toff_values],
            )

        # Step 1: TOFF sets the upper limit of the chopper frequency.
        self.gcode.respond_info("Step 1: Optimizing TOFF...")
        best_params = sweep_toff(best_params)
        self.gcode.respond_info(
            f"-> Best TOFF found: {self.format_toff_values([best_params[PARAM_TOFF]])}"
        )

        # Step 2: TBL needs to safely cover the switching event and the ringing.
        self.gcode.respond_info("Step 2: Optimizing TBL...")
        ascending_tbl = sorted(tbl_values)
        best_params = self.progressive_sweep(
            best_params,
            [{PARAM_TBL: tbl} for tbl in ascending_tbl],
            preference=[ascending_tbl.index(tbl) for tbl in tbl_values],
        )
        self.gcode.respond_info(f"-> Best TBL found: {best_params[PARAM_TBL]}")

        # Step 3: Increase the hysteresis until the motor runs smoothly. A too low
        # value reduces the microstep accuracy, and a too high value increases the
        # chopper noise without improving the wave shape. Neither is necessarily
        # sensed by an accelerometer, so the datasheet default is preferred over
        # similar results, and the sweep doesn't stop before measuring it.
        self.gcode.respond_info("Step 3: Optimizing the hysteresis...")
        hstrt = best_params[PARAM_HSTRT]

        def hysteresis_of(hend: int) -> int:
            return hstrt + hend - HYSTERESIS_REGISTER_OFFSET

        hysteresis_preference = sorted(
            range(len(hend_values)),
            key=lambda index: (
                abs(hysteresis_of(hend_values[index]) - DEFAULT_EFFECTIVE_HYSTERESIS),
                hend_values[index],
            ),
        )
        best_params = self.progressive_sweep(
            best_params,
            [{PARAM_HEND: hend} for hend in hend_values],
            preference=hysteresis_preference,
            patience=PROGRESSIVE_PATIENCE,
            measure_at_least=hysteresis_preference[0] + 1,
        )
        self.gcode.respond_info(
            f"-> Best HEND found: {best_params[PARAM_HEND]} (effective hysteresis: "
            f"{hysteresis_of(best_params[PARAM_HEND])})"
        )

        # Step 4: Split the same hysteresis start value between HSTRT and HEND. A
        # lower HEND gives more hysteresis decrement, which stabilizes the chopper
        # frequency.
        self.gcode.respond_info("Step 4: Optimizing the HSTRT and HEND split...")
        register_sum = best_params[PARAM_HSTRT] + best_params[PARAM_HEND]
        splits = [
            {PARAM_HSTRT: hstrt, PARAM_HEND: register_sum - hstrt}
            for hstrt in hstrt_values
            if self.hend_min <= register_sum - hstrt <= self.hend_max
        ]
        best_params = self.progressive_sweep(
            best_params,
            splits,
            preference=list(reversed(range(len(splits)))),
        )
        self.gcode.respond_info(
            f"-> Best HSTRT found: {best_params[PARAM_HSTRT]}, "
            f"HEND: {best_params[PARAM_HEND]}"
        )

        # Step 5: The optimum hysteresis and TOFF depend on each other, so check
        # TOFF again with the final hysteresis.
        self.gcode.respond_info("Step 5: Checking TOFF with the final hysteresis...")
        best_params = sweep_toff(best_params)
        self.gcode.respond_info(
            f"-> Best TOFF found: {self.format_toff_values([best_params[PARAM_TOFF]])}"
        )

        # Step 6: "Starting from 0, increase value, in case the motor suffers from
        # mid-range resonances."
        if self.driver in DRIVERS_SUPPORTING_TPFD:
            self.gcode.respond_info("Step 6: Optimizing TPFD...")
            best_params = self.progressive_sweep(
                best_params,
                [{PARAM_TPFD: tpfd} for tpfd in tpfd_values],
                patience=PROGRESSIVE_PATIENCE,
            )
            self.gcode.respond_info(f"-> Best TPFD found: {best_params[PARAM_TPFD]}")

        return best_params

    def search_best_parameters(self) -> list[int]:
        """Run the parameter search process.

        This can use either brute-force or adaptive optimization methods depending
        on the selected search method.

        Returns:
            dict: The best parameters found.
        """
        self.gcode.respond_info("Starting optimization...")
        start_time = self.reactor.monotonic()

        if self.measurement_mode == MeasurementMode.Resonances:
            # Dump TMC settings
            self.gcode.respond_info(
                f"tbl={self.tbl_min} "
                f"toff={self.toff_min} "
                f"hstrt={self.hstrt_min} "
                f"hend={self.hend_min} "
                f"tpfd={self.tpfd_min} "
                f"current={self.current_min} "
                f"speed={self.min_speed:.2f} --> {self.max_speed:.2f}"
            )

        # set initial values
        self.apply_registers(self.steppers, "curr", self.current_min)
        initial_tbl = self.tbl_min
        if self.measurement_mode == MeasurementMode.Vibrations and self.toff_min == 1:
            # TOFF=1 should only be used with TBL >= 2
            initial_tbl = max(initial_tbl, min(2, self.tbl_max))
        self.apply_registers(self.steppers, "tbl", initial_tbl)
        self.apply_registers(self.steppers, "toff", self.toff_min)
        self.apply_registers(self.steppers, "hend", self.hend_min)
        self.apply_registers(self.steppers, "hstrt", self.hstrt_min)
        self.apply_registers(self.steppers, "tpfd", self.tpfd_min)

        if self.search_method == SearchMethod.BruteForce:
            best_params = self.perform_brute_force_search()
        elif self.search_method == SearchMethod.Adaptive:
            best_params = self.perform_adaptive_search()
        elif self.search_method == SearchMethod.Progressive:
            best_params = self.perform_progressive_search()

        # update overall best params with final results
        overall_best_params = {
            "current": best_params[0],
            "tbl": best_params[1],
            "toff": best_params[2],
            "hstrt": best_params[3],
            "hend": best_params[4],
            "tpfd": best_params[5],
            "speed": float(best_params[6]) / 100,
        }

        duration = self.reactor.monotonic() - start_time
        selected_score = self.samples.get(
            self.generate_sample_name(**overall_best_params), float("inf")
        )

        result_message = (
            f"Optimization Completed in {self.convert_seconds_to_hms(duration)}\n"
            f"Number of samples : {self.number_of_samples}\n"
            f"Lowest Score      : {self.best_result:.1f} mm/s²\n"
            f"Selected Score    : {selected_score:.1f} mm/s²\n\n"
            "Parameters\n"
            "----------\n"
            f"speed        : {overall_best_params['speed']:.2f} mm/s\n"
            f"current      : {overall_best_params['current']} mA\n"
            f"driver_tbl   : {overall_best_params['tbl']}\n"
            f"driver_toff  : {overall_best_params['toff']}\n"
            f"driver_hstrt : {overall_best_params['hstrt']}\n"
            f"driver_hend  : {overall_best_params['hend']}\n"
        )
        if self.driver in DRIVERS_SUPPORTING_TPFD:
            result_message += f"driver_tpfd : {overall_best_params['tpfd']}"
        self.gcode.respond_info(result_message)

        # Apply best parameters
        self.apply_registers(self.steppers, "curr", overall_best_params["current"])
        self.apply_registers(self.steppers, "tbl", overall_best_params["tbl"])
        self.apply_registers(self.steppers, "toff", overall_best_params["toff"])
        self.apply_registers(self.steppers, "hend", overall_best_params["hend"])
        self.apply_registers(self.steppers, "hstrt", overall_best_params["hstrt"])
        self.apply_registers(self.steppers, "tpfd", overall_best_params["tpfd"])

        return overall_best_params

    def save_configs(self, best_parameters: dict | None) -> None:
        """Save the best parameters to printer.cfg.

        Args:
            best_parameters (dict | None): The best parameters found.
        """
        if best_parameters is None:
            return

        for stepper_index in range(self.registers["stepper_count"]):
            stepper_index = str(stepper_index) if stepper_index > 0 else ""
            for field, value in best_parameters.items():
                if field == "speed":
                    continue  # skip speed field

                if field == "tpfd" and (
                    value == -1 or self.driver not in DRIVERS_SUPPORTING_TPFD
                ):
                    continue  # skip tpfd for unsupported drivers and if the value is -1

                field_name = f"driver_{field}" if field != "current" else "run_current"
                value = str(value) if field != "current" else f"{value / 1000:0.3f}"

                self.configfile.set(
                    f"tmc{self.driver} {self.steppers[0]}{stepper_index}",
                    field_name,
                    value,
                )

        self.gcode.respond_info(
            "Best parameters saved to printer.cfg, run SAVE_CONFIG to apply."
        )

    def plot_data(self, date_stamp: None | str = None) -> None:
        """Plot the collected vibration data.

        Args:
            date_stamp (None | str): The timestamp string to use in filenames.
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        if date_stamp is None:
            date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        params = [
            reversed(list(self.samples.items())),
            sorted(self.samples.items(), key=lambda x: x[1]),
        ]
        names = ["as_measured", "sorted"]
        plot_html_paths = []
        for param, name in zip(params, names):
            fig = go.Figure()
            for entry in param:
                toff = int(entry[0].split("_")[2].split("=")[1])
                color = COLORS[toff if toff <= 8 else toff - 8]
                fig.add_trace(
                    go.Bar(
                        x=[entry[1]],
                        y=[entry[0]],
                        marker_color=color,
                        orientation="h",
                        showlegend=False,
                    )
                )
            fig.update_layout(
                title="Median Magnitude vs Parameters",
                xaxis_title="Median Magnitude (mm/s²)",
                yaxis_title="Parameters",
                coloraxis_showscale=True,
            )
            plot_html_path = os.path.join(
                RESULTS_FOLDER,
                f"{date_stamp}"
                f"_interactive_plot_{name}"
                f"_{self.accel_chip_name}"
                f"_tmc{self.driver}"
                f"_{self.steppers[0]}"
                f"_{self.sense_resistor}"
                ".html",
            )
            plot_html_paths.append(plot_html_path)
            # check if the RESULTS_FOLDER exists before writing
            if not os.path.exists(RESULTS_FOLDER):
                os.makedirs(RESULTS_FOLDER)

            pio.write_html(fig, plot_html_path, auto_open=False)
            speed1 = params[1][0][0].split("_")[6].split("=")[1]
            speed2 = params[1][1][0].split("_")[6].split("=")[1]
            if speed1 != speed2:
                break

        # Export Info
        self.gcode.respond_info("Access to interactive plot at:")
        for plot_html_path in plot_html_paths:
            self.gcode.respond_info(f"{plot_html_path}")

    def store_data(self, date_stamp: None | str = None) -> None:
        """Store collected sample data to a JSON file.

        Args:
            date_stamp (None | str): The timestamp string to use in filenames.
        """
        if date_stamp is None:
            date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(
            RESULTS_FOLDER,
            f"{date_stamp}"
            f"_sample_data"
            f"_{self.accel_chip_name}"
            f"_tmc{self.driver}"
            f"_{self.steppers[0]}"
            f"_{self.sense_resistor}"
            ".json",
        )
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        with open(json_path, mode="w") as file:
            json.dump(self.samples, file, indent=4)
        self.gcode.respond_info(f"Sample data saved to: {json_path}")

    def cmd_chopper_tune(self, gcmd: GCodeCommand) -> None:
        """Collect data from G-Code command and run optimization.

        This runs as a regular G-Code command, so it holds the G-Code mutex until
        it is done and the following commands wait for it. Emergency stop still
        works immediately as it doesn't need the mutex, the same as with the other
        long running Klipper commands.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Raises:
            CommandError: If the tuning fails.
        """
        try:
            axis = gcmd.get("AXIS", "x").lower()
            direction = gcmd.get_int("DIRECTION", 1)
            # search_method can be brute_force, adaptive or progressive
            search_method = SearchMethod.to_method(
                gcmd.get("SEARCH_METHOD", "progressive").lower()
            )
            current_min = gcmd.get_float("CURRENT_MIN_MA", None)
            current_max = gcmd.get_float("CURRENT_MAX_MA", None)
            tbl_min = gcmd.get_int("TBL_MIN", DEFAULT_REGISTER_VALUES["TBL"][0])
            tbl_max = gcmd.get_int("TBL_MAX", DEFAULT_REGISTER_VALUES["TBL"][1])
            toff_min = gcmd.get_int("TOFF_MIN", None)
            toff_max = gcmd.get_int("TOFF_MAX", None)
            toff_range_requested = toff_min is not None or toff_max is not None
            if toff_min is None:
                toff_min = DEFAULT_REGISTER_VALUES["TOFF"][0]
            if toff_max is None:
                toff_max = DEFAULT_REGISTER_VALUES["TOFF"][1]
            hstrt_hend_max = gcmd.get_int(
                "HSTRT_HEND_MAX", DEFAULT_REGISTER_VALUES["HSTRT_HEND_MAX"]
            )
            hstrt_min = gcmd.get_int("HSTRT_MIN", DEFAULT_REGISTER_VALUES["HSTRT"][0])
            hstrt_max = gcmd.get_int("HSTRT_MAX", DEFAULT_REGISTER_VALUES["HSTRT"][1])
            hend_min = gcmd.get_int("HEND_MIN", DEFAULT_REGISTER_VALUES["HEND"][0])
            hend_max = gcmd.get_int("HEND_MAX", DEFAULT_REGISTER_VALUES["HEND"][1])
            tpfd_min = gcmd.get_int("TPFD_MIN", DEFAULT_REGISTER_VALUES["TPFD"][0])
            tpfd_max = gcmd.get_int("TPFD_MAX", DEFAULT_REGISTER_VALUES["TPFD"][1])
            min_speed = gcmd.get_float("MIN_SPEED", None)
            max_speed = gcmd.get_float("MAX_SPEED", None)
            speed_change_step = gcmd.get_float("SPEED_CHANGE_STEP", None)
            self.iterations = gcmd.get_int("ITERATIONS", 1)
            travel_distance = gcmd.get_float("TRAVEL_DISTANCE", None)
            accel_chip_name = gcmd.get("ACCELEROMETER", "default").lower()
            compare_with = gcmd.get("COMPARE_WITH", None)

            self.measurement_mode = {
                "0": MeasurementMode.Vibrations,
                "1": MeasurementMode.Resonances,
                "false": MeasurementMode.Vibrations,
                "true": MeasurementMode.Resonances,
            }.get(
                gcmd.get("FIND_RESONANCES", "false").lower(), MeasurementMode.Resonances
            )
            run_plotter = {
                "0": False,
                "1": True,
                "false": False,
                "true": True,
            }.get(gcmd.get("RUN_PLOTTER", "true").lower(), True)
            save_raw = gcmd.get("SAVE_RAW", "false").lower() in ("1", "true")

            self.chopper_tune(
                axis=axis,
                current_min=current_min,
                current_max=current_max,
                tbl_min=tbl_min,
                tbl_max=tbl_max,
                toff_min=toff_min,
                toff_max=toff_max,
                hstrt_hend_max=hstrt_hend_max,
                hstrt_min=hstrt_min,
                hstrt_max=hstrt_max,
                hend_min=hend_min,
                hend_max=hend_max,
                tpfd_min=tpfd_min,
                tpfd_max=tpfd_max,
                min_speed=min_speed,
                max_speed=max_speed,
                speed_change_step=speed_change_step,
                search_method=search_method,
                travel_distance=travel_distance,
                direction=direction,
                accel_chip_name=accel_chip_name,
                run_plotter=run_plotter,
                compare_with=compare_with,
                save_raw=save_raw,
                toff_range_requested=toff_range_requested,
            )
        except self.printer.command_error:
            raise
        except Exception as e:
            # Klipper shuts down on any exception other than a command error. A
            # problem in this tool shouldn't do that.
            self.gcode.respond_info(traceback.format_exc())
            raise self.printer.command_error(f"Chopper tune failed: {e}") from e

    def cmd_chopper_tune_debug(self, gcmd: GCodeCommand) -> bool:
        """Development debug tool.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Returns:
            bool: True if command completed successfully, False otherwise.
        """
        try:
            self.gcode.respond_info(f"x stepper count: {self.get_stepper_count('x')}")
            self.gcode.respond_info(f"y stepper count: {self.get_stepper_count('y')}")
            self.gcode.respond_info(f"z stepper count: {self.get_stepper_count('z')}")
        except Exception:
            self.gcode.respond_info(traceback.format_exc())


def load_config(config: ConfigWrapper) -> ChopperTune:
    """Load the ChopperTune config prefix.

    Args:
        config (ConfigWrapper): The config wrapper.

    Returns:
        ChopperTune: The ChopperTune instance.
    """
    return ChopperTune(config)


def load_config_prefix(config: ConfigWrapper) -> ChopperTune:
    """Load the ChopperTune config prefix.

    Args:
        config (ConfigWrapper): The config wrapper.

    Returns:
        ChopperTune: The ChopperTune instance.
    """
    return ChopperTune(config)
