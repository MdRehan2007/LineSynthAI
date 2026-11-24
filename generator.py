"""Synthetic sensor data generation for LineSynth AI.

Every machine run is built in four steps:

1. Baseline "drivers" (relative motor load, speed, pressure, vibration) are
   created as slowly varying mean-reverting processes plus a gentle cycle.
2. Fault events are scheduled and their effects are applied to the drivers
   (see ``fault_simulator.py``).
3. Motor current and temperature are *derived* from the drivers, which keeps
   the sensors physically related (more load -> more current -> more heat,
   with a thermal lag; vibration grows with speed deviation).
4. Measurement noise is added, values are clipped to plausible limits and
   rounded.

DISCLAIMER: all operating ranges below are hypothetical values chosen to make
believable synthetic data.  They are NOT certified industrial limits and must
not be used for real safety or quality decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fault_simulator import (
    CONDITION_OPTIONS,
    FAULT,
    NORMAL,
    WARNING,
    apply_fault_effects,
    build_plan,
    label_records,
    plan_events,
)

SENSOR_COLUMNS: List[str] = [
    "temperature",
    "pressure",
    "vibration",
    "motor_speed",
    "motor_current",
]
SENSOR_LABELS: Dict[str, str] = {
    "temperature": "Temperature (°C)",
    "pressure": "Pressure (bar)",
    "vibration": "Vibration (mm/s)",
    "motor_speed": "Motor Speed (RPM)",
    "motor_current": "Motor Current (A)",
}
SENSOR_UNITS: Dict[str, str] = {
    "temperature": "°C",
    "pressure": "bar",
    "vibration": "mm/s",
    "motor_speed": "RPM",
    "motor_current": "A",
}
SENSOR_DECIMALS: Dict[str, int] = {
    "temperature": 2,
    "pressure": 3,
    "vibration": 3,
    "motor_speed": 1,
    "motor_current": 2,
}
ID_COLUMNS = ["timestamp", "machine_id", "machine_type"]
LABEL_COLUMNS = ["condition", "fault_type", "fault_label"]
DATASET_COLUMNS: List[str] = ID_COLUMNS + SENSOR_COLUMNS + LABEL_COLUMNS

RECORD_OPTIONS = [100, 500, 1000, 5000, 10000]
MIN_RECORDS = 10
MAX_RECORDS = 100_000
MAX_MACHINES = 10
MAX_SEED = 2**32 - 1

WARNING_RULES = {
    "Warning counts as normal (label 0)": False,
    "Warning counts as fault (label 1)": True,
}

# Relative measurement noise (sd as a fraction of the nominal value).
NOISE_FRACTION = {
    "temperature": 0.004,
    "pressure": 0.008,
    "vibration": 0.06,
    "motor_speed": 0.0015,
    "motor_current": 0.012,
}
THERMAL_LAG_ALPHA = 0.30  # first-order lag, per sample


@dataclass(frozen=True)
class MachineProfile:
    """Hypothetical operating profile of a machine type."""

    name: str
    prefix: str
    description: str
    nominal: Dict[str, float]
    normal_range: Dict[str, Tuple[float, float]]
    temp_rise: float  # deg C rise of a fully developed overheating fault


MACHINE_PROFILES: Dict[str, MachineProfile] = {
    "Milk Processing Machine": MachineProfile(
        name="Milk Processing Machine",
        prefix="MILK",
        description="Pasteuriser / homogeniser skid (hypothetical values).",
        nominal={"temperature": 72.0, "pressure": 2.5, "vibration": 1.8,
                 "motor_speed": 1450.0, "motor_current": 12.0},
        normal_range={"temperature": (68.0, 76.0), "pressure": (2.0, 3.0),
                      "vibration": (0.5, 2.8), "motor_speed": (1420.0, 1480.0),
                      "motor_current": (9.0, 14.5)},
        temp_rise=12.0,
    ),
    "Beverage Filling Machine": MachineProfile(
        name="Beverage Filling Machine",
        prefix="BEV",
        description="Rotary bottle filler (hypothetical values).",
        nominal={"temperature": 38.0, "pressure": 4.0, "vibration": 2.0,
                 "motor_speed": 1750.0, "motor_current": 15.0},
        normal_range={"temperature": (34.0, 42.0), "pressure": (3.5, 4.5),
                      "vibration": (0.6, 3.0), "motor_speed": (1720.0, 1780.0),
                      "motor_current": (11.0, 17.5)},
        temp_rise=14.0,
    ),
    "Food Packaging Machine": MachineProfile(
        name="Food Packaging Machine",
        prefix="PACK",
        description="Flow-wrap / sealing line (hypothetical values).",
        nominal={"temperature": 45.0, "pressure": 6.0, "vibration": 2.4,
                 "motor_speed": 1200.0, "motor_current": 8.0},
        normal_range={"temperature": (40.0, 50.0), "pressure": (5.5, 6.5),
                      "vibration": (0.8, 3.5), "motor_speed": (1170.0, 1230.0),
                      "motor_current": (6.0, 9.8)},
        temp_rise=15.0,
    ),
    "Biscuit Production Machine": MachineProfile(
        name="Biscuit Production Machine",
        prefix="BISC",
        description="Dough mixer / oven zone drive (hypothetical values).",
        nominal={"temperature": 180.0, "pressure": 1.2, "vibration": 3.0,
                 "motor_speed": 900.0, "motor_current": 22.0},
        normal_range={"temperature": (170.0, 190.0), "pressure": (1.0, 1.5),
                      "vibration": (1.0, 4.5), "motor_speed": (870.0, 930.0),
                      "motor_current": (17.0, 26.0)},
        temp_rise=35.0,
    ),
}
MACHINE_TYPES: List[str] = list(MACHINE_PROFILES)


class ConfigError(ValueError):
    """Raised for invalid generator settings; ``errors`` lists every problem."""

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


@dataclass
class GeneratorConfig:
    """All user-selectable generation settings."""

    machine_type: str = MACHINE_TYPES[0]
    condition: str = "Mixed Conditions"
    n_records: int = 1000
    sensors: Tuple[str, ...] = tuple(SENSOR_COLUMNS)
    seed: int = 42
    n_machines: int = 1
    fault_pct: float = 15.0
    fault_duration: int = 30
    sampling_seconds: int = 60
    start_time: datetime = datetime(2025, 1, 1)
    warning_as_fault: bool = False

    def validate(self) -> None:
        errors: List[str] = []
        if self.machine_type not in MACHINE_PROFILES:
            errors.append(f"Unknown machine type: {self.machine_type!r}.")
        if self.condition not in CONDITION_OPTIONS:
            errors.append(f"Unknown machine condition: {self.condition!r}.")

        records_ok = _is_int(self.n_records) and MIN_RECORDS <= self.n_records <= MAX_RECORDS
        if not records_ok:
            errors.append(
                f"Number of records must be a whole number between {MIN_RECORDS:,} "
                f"and {MAX_RECORDS:,}."
            )
        machines_ok = _is_int(self.n_machines) and 1 <= self.n_machines <= MAX_MACHINES
        if not machines_ok:
            errors.append(f"Number of machines must be between 1 and {MAX_MACHINES}.")
        if records_ok and machines_ok and self.n_records // self.n_machines < MIN_RECORDS:
            errors.append(
                f"Each machine needs at least {MIN_RECORDS} records; "
                "reduce the number of machines or increase the record count."
            )

        sensors = list(self.sensors) if self.sensors is not None else []
        if not sensors:
            errors.append("Select at least one sensor parameter.")
        elif any(s not in SENSOR_COLUMNS for s in sensors):
            errors.append("Unknown sensor parameter selected.")
        elif len(set(sensors)) != len(sensors):
            errors.append("Sensor parameters must not be repeated.")

        if not _is_int(self.seed) or not 0 <= self.seed <= MAX_SEED:
            errors.append(f"Random seed must be a whole number between 0 and {MAX_SEED}.")

        if self.condition != NORMAL:
            if not _is_number(self.fault_pct) or not 0 < self.fault_pct <= 50:
                errors.append("Fault percentage must be greater than 0 and at most 50.")
            if not _is_int(self.fault_duration) or not 3 <= self.fault_duration <= 500:
                errors.append("Fault duration must be a whole number between 3 and 500 records.")
        if not _is_int(self.sampling_seconds) or not 1 <= self.sampling_seconds <= 86400:
            errors.append("Sampling interval must be a whole number of seconds (1 to 86,400).")
        try:
            pd.Timestamp(self.start_time)
        except Exception:
            errors.append("Start time is not a valid date.")
        if errors:
            raise ConfigError(errors)


def _ou(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance mean-reverting (Ornstein-Uhlenbeck style) process."""
    a = 1.0 - theta
    scale = np.sqrt(1.0 - a * a)
    shocks = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = a * x[i - 1] + scale * shocks[i]
    return x


def _baseline_drivers(profile: MachineProfile, n: int, dt: int,
                      rng: np.random.Generator) -> Dict[str, np.ndarray]:
    nom = profile.nominal
    t = np.arange(n) * float(dt)
    phase = rng.uniform(0, 2 * np.pi, size=2)
    cycle_a = np.sin(2 * np.pi * t / (3 * 3600.0) + phase[0])   # ~3 h production cycle
    cycle_b = np.sin(2 * np.pi * t / (5 * 3600.0) + phase[1])
    load = 1.0 + 0.05 * _ou(n, 0.03, rng) + 0.025 * cycle_a
    speed = nom["motor_speed"] * (1.0 + 0.002 * _ou(n, 0.05, rng) - 0.02 * (load - 1.0))
    pressure = nom["pressure"] * (1.0 + 0.012 * _ou(n, 0.04, rng) + 0.005 * cycle_b)
    vibration = nom["vibration"] * (1.0 + 0.08 * _ou(n, 0.05, rng))
    return {"load": load, "speed": speed, "pressure": pressure,
            "vibration": vibration, "heat_extra": np.zeros(n)}


def _derive_sensors(drivers: Dict[str, np.ndarray], profile: MachineProfile,
                    rng: np.random.Generator) -> Dict[str, np.ndarray]:
    nom = profile.nominal
    n = len(drivers["load"])
    z = {s: rng.standard_normal(n) for s in SENSOR_COLUMNS}

    current = nom["motor_current"] * drivers["load"] + nom["motor_current"] * NOISE_FRACTION["motor_current"] * z["motor_current"]

    target = (nom["temperature"]
              + 0.8 * profile.temp_rise * (current / nom["motor_current"] - 1.0)
              + drivers["heat_extra"])
    filtered = np.empty(n)
    filtered[0] = target[0]
    for i in range(1, n):  # thermal inertia
        filtered[i] = filtered[i - 1] + THERMAL_LAG_ALPHA * (target[i] - filtered[i - 1])
    temperature = filtered + nom["temperature"] * NOISE_FRACTION["temperature"] * z["temperature"]

    speed = drivers["speed"] + nom["motor_speed"] * NOISE_FRACTION["motor_speed"] * z["motor_speed"]
    pressure = drivers["pressure"] * (1.0 + NOISE_FRACTION["pressure"] * z["pressure"])
    speed_dev = np.abs(speed / nom["motor_speed"] - 1.0)
    vibration = drivers["vibration"] * (1.0 + 2.0 * speed_dev) * (1.0 + NOISE_FRACTION["vibration"] * z["vibration"])

    return {
        "temperature": np.clip(temperature, 5.0, None),
        "pressure": np.clip(pressure, 0.05, None),
        "vibration": np.clip(vibration, 0.05, None),
        "motor_speed": np.clip(speed, 0.0, None),
        "motor_current": np.clip(current, 0.1, None),
    }


def _generate_machine_run(cfg: GeneratorConfig, profile: MachineProfile, machine_id: str,
                          n: int, rng: np.random.Generator) -> pd.DataFrame:
    dt = int(cfg.sampling_seconds)
    events = plan_events(n, cfg.condition, float(cfg.fault_pct), int(cfg.fault_duration), rng)
    plan = build_plan(n, events)
    drivers = _baseline_drivers(profile, n, dt, rng)
    drivers = apply_fault_effects(drivers, profile, plan, rng)
    sensors = _derive_sensors(drivers, profile, rng)
    condition, fault_type, label = label_records(plan, cfg.warning_as_fault)

    timestamps = pd.Timestamp(cfg.start_time) + pd.to_timedelta(np.arange(n) * dt, unit="s")
    data = {"timestamp": timestamps, "machine_id": machine_id, "machine_type": cfg.machine_type}
    for s in SENSOR_COLUMNS:
        data[s] = np.round(sensors[s], SENSOR_DECIMALS[s])
    data["condition"] = condition.astype(str)
    data["fault_type"] = fault_type.astype(str)
    data["fault_label"] = label.astype(np.int64)
    return pd.DataFrame(data)


def generate_dataset(cfg: GeneratorConfig) -> pd.DataFrame:
    """Generate a synthetic dataset with exactly ``cfg.n_records`` rows.

    Records are split as evenly as possible over ``cfg.n_machines`` independent
    machine runs (each with its own random stream derived from the seed), then
    sorted by timestamp and machine id.  Columns of sensors that were not
    selected are omitted.
    """
    cfg.validate()
    profile = MACHINE_PROFILES[cfg.machine_type]
    base, extra = divmod(int(cfg.n_records), int(cfg.n_machines))
    counts = [base + (1 if i < extra else 0) for i in range(cfg.n_machines)]
    seeds = np.random.SeedSequence(int(cfg.seed)).spawn(cfg.n_machines)

    frames = []
    for i, (count, seed_seq) in enumerate(zip(counts, seeds), start=1):
        rng = np.random.default_rng(seed_seq)
        frames.append(_generate_machine_run(cfg, profile, f"{profile.prefix}-{i:02d}", count, rng))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "machine_id"], kind="stable").reset_index(drop=True)

    keep = ID_COLUMNS + [s for s in SENSOR_COLUMNS if s in cfg.sensors] + LABEL_COLUMNS
    return df[keep]


# --------------------------------------------------------------------------- #
# Dataset helpers shared by the app, the ML module and the tests
# --------------------------------------------------------------------------- #
REQUIRED_BASE_COLUMNS = ["timestamp", "machine_id", "condition", "fault_label"]


def available_sensors(df: pd.DataFrame) -> List[str]:
    return [s for s in SENSOR_COLUMNS if s in df.columns]


def dataset_problems(df: Optional[pd.DataFrame]) -> List[str]:
    """Return a list of human-readable problems (empty list = usable dataset)."""
    if df is None:
        return ["No dataset has been generated yet."]
    if not isinstance(df, pd.DataFrame):
        return ["The stored dataset is not a table."]
    if df.empty:
        return ["The dataset is empty."]
    problems = []
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing required columns: " + ", ".join(missing) + ".")
    if not available_sensors(df):
        problems.append("The dataset has no sensor columns.")
    return problems


def summarize_dataset(df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Counts used by the dashboard cards (all computed from the data)."""
    if dataset_problems(df):
        return {"total": 0, "normal": 0, "warning": 0, "fault": 0, "labelled_fault": 0, "machines": 0}
    cond = df["condition"]
    return {
        "total": int(len(df)),
        "normal": int((cond == NORMAL).sum()),
        "warning": int((cond == WARNING).sum()),
        "fault": int((cond == FAULT).sum()),
        "labelled_fault": int((df["fault_label"] == 1).sum()),
        "machines": int(df["machine_id"].nunique()),
    }


def machine_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per machine: record counts per condition and latest status."""
    if dataset_problems(df):
        return pd.DataFrame()
    counts = (pd.crosstab(df["machine_id"], df["condition"])
              .reindex(columns=[NORMAL, WARNING, FAULT], fill_value=0))
    grouped = df.sort_values("timestamp", kind="stable").groupby("machine_id")
    out = pd.DataFrame({
        "machine_type": grouped["machine_type"].first() if "machine_type" in df.columns else "",
        "records": grouped.size(),
        "normal": counts[NORMAL],
        "warning": counts[WARNING],
        "fault": counts[FAULT],
        "fault_label_rate_%": (grouped["fault_label"].mean() * 100).round(1),
        "latest_condition": grouped["condition"].last(),
    })
    return out.reset_index()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def build_csv_filename(config: Optional[dict], n_rows: int, now: Optional[datetime] = None) -> str:
    """Descriptive file name, e.g. linesynth_milk_processing_machine_mixed_conditions_1000rows_seed42_20250101_120000.csv"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if config:
        return (f"linesynth_{_slug(config.get('machine_type', 'machine'))}_"
                f"{_slug(config.get('condition', 'data'))}_{n_rows}rows_"
                f"seed{config.get('seed', 'x')}_{stamp}.csv")
    return f"linesynth_dataset_{n_rows}rows_{stamp}.csv"


def dataset_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise the full dataset as UTF-8 CSV (ISO timestamps, no index)."""
    if df is None or len(df) == 0:
        raise ValueError("There is no dataset to export.")
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_csv(index=False).encode("utf-8")
