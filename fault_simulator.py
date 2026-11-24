"""Fault planning and fault-effect simulation for LineSynth AI.

A fault is modelled as an *event*: a contiguous run of records in which a
fault develops over time.  Inside an event a severity value ramps from
``MIN_SEVERITY`` up to 1.0 and then holds until the event ends (think of the
end of the event as a maintenance reset).  Severity drives how strongly each
sensor is disturbed, so faults appear as realistic sequences and never as
independent random outliers.

Labelling rule (documented, used everywhere in the project)
-----------------------------------------------------------
* severity == 0                          -> condition "Normal",  fault_label 0
* 0 < severity < WARNING_SEVERITY_THRESHOLD -> condition "Warning", fault_label 0
                                            (1 if ``warning_as_fault`` is on)
* severity >= WARNING_SEVERITY_THRESHOLD -> condition "Fault",   fault_label 1

``fault_type`` carries the name of the developing fault for both Warning and
Fault rows, and "None" for Normal rows.

All numeric effects below are HYPOTHETICAL and chosen only to produce
plausible-looking synthetic data.  They are not engineering or safety limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

NORMAL = "Normal"
WARNING = "Warning"
FAULT = "Fault"
CONDITIONS = [NORMAL, WARNING, FAULT]
NO_FAULT = "None"

OVERHEATING = "Overheating"
EXCESSIVE_VIBRATION = "Excessive Vibration"
MOTOR_OVERLOAD = "Motor Overload"
PRESSURE_ABNORMALITY = "Pressure Abnormality"
MIXED = "Mixed Conditions"

FAULT_TYPES: List[str] = [
    OVERHEATING,
    EXCESSIVE_VIBRATION,
    MOTOR_OVERLOAD,
    PRESSURE_ABNORMALITY,
]
CONDITION_OPTIONS: List[str] = [NORMAL, *FAULT_TYPES, MIXED]

MIN_SEVERITY = 0.2
WARNING_SEVERITY_THRESHOLD = 0.5
MIN_FAULT_DURATION = 3
MAX_FAULT_PCT = 50.0

# Effect strengths at severity 1.0 (hypothetical).
OVERHEAT_CURRENT_GAIN = 0.12      # +12 % motor load
VIBRATION_GAIN = 2.5              # +250 % of nominal vibration
VIBRATION_IRREGULARITY = 0.35     # relative irregular fluctuation
VIBRATION_SPEED_JITTER = 0.010    # 1 % RPM jitter
OVERLOAD_CURRENT_GAIN = 0.60      # +60 % motor current
OVERLOAD_SPEED_DROP = 0.08        # -8 % RPM
OVERLOAD_HEAT_FRACTION = 0.35     # extra heating as fraction of temp_rise
PRESSURE_DRIFT = 0.30             # +/-30 % of nominal pressure
PRESSURE_WAVE = 0.15              # oscillation amplitude (fraction of nominal)
PRESSURE_NOISE = 0.08

FAULT_DESCRIPTIONS: Dict[str, str] = {
    OVERHEATING: (
        "Temperature rises gradually; motor current creeps up slightly and "
        "vibration increases a little."
    ),
    EXCESSIVE_VIBRATION: (
        "Vibration grows with irregular fluctuations; motor speed becomes "
        "unstable; temperature and current rise slightly."
    ),
    MOTOR_OVERLOAD: (
        "Motor current increases strongly, motor speed drops, and "
        "temperature climbs as the extra current heats the machine."
    ),
    PRESSURE_ABNORMALITY: (
        "Pressure drifts away from its normal band with oscillating, "
        "unexpected fluctuations; pump current changes slightly."
    ),
}


@dataclass(frozen=True)
class FaultEvent:
    """One fault sequence inside a machine run."""

    event_id: int
    fault_type: str
    start: int
    length: int
    direction: int  # +1 / -1, used by pressure abnormalities


@dataclass
class FaultPlan:
    """Per-record view of all fault events of one machine run."""

    n: int
    events: List[FaultEvent]
    severity: np.ndarray
    fault_type: np.ndarray
    direction: np.ndarray
    position: np.ndarray
    event_id: np.ndarray


def _assign_types(condition: str, count: int, rng: np.random.Generator) -> List[str]:
    if condition in FAULT_TYPES:
        return [condition] * count
    order: List[str] = []
    while len(order) < count:
        order.extend(str(x) for x in rng.permutation(FAULT_TYPES))
    return order[:count]


def plan_events(
    n: int,
    condition: str,
    fault_pct: float,
    duration: int,
    rng: np.random.Generator,
) -> List[FaultEvent]:
    """Schedule non-overlapping fault events for a run of ``n`` records.

    The timeline is cut into equally long slots (one per event) and each event
    is placed at a random position inside its slot.  Events are therefore
    spread over the whole run, are always separated by normal records, and
    never overlap.  ``fault_pct`` is approximate: the number of fault-affected
    records is ``round(n * fault_pct / 100 / duration) * duration``.
    """
    if condition not in CONDITION_OPTIONS:
        raise ValueError(f"Unknown machine condition: {condition!r}")
    if condition == NORMAL:
        return []
    if not 0 < fault_pct <= MAX_FAULT_PCT:
        raise ValueError(f"Fault percentage must be in (0, {MAX_FAULT_PCT:g}].")
    if duration < MIN_FAULT_DURATION:
        raise ValueError(f"Fault duration must be at least {MIN_FAULT_DURATION} records.")

    dur = int(min(duration, max(5, n // 4)))
    min_slot = dur + 2
    if n < min_slot:
        return []
    target = n * fault_pct / 100.0
    n_events = max(1, int(round(target / dur)))
    n_events = min(n_events, max(1, n // min_slot))
    slot = n // n_events
    types = _assign_types(condition, n_events, rng)

    events: List[FaultEvent] = []
    for i in range(n_events):
        slot_end = n if i == n_events - 1 else (i + 1) * slot
        lo = i * slot + 1
        hi = max(lo, slot_end - dur - 1)
        start = int(rng.integers(lo, hi + 1))
        direction = int(rng.choice([-1, 1]))
        events.append(FaultEvent(i, types[i], start, dur, direction))
    return events


def build_plan(n: int, events: List[FaultEvent]) -> FaultPlan:
    """Expand events into per-record severity / type arrays."""
    severity = np.zeros(n, dtype=float)
    ftype = np.full(n, NO_FAULT, dtype=object)
    direction = np.zeros(n, dtype=float)
    position = np.zeros(n, dtype=float)
    event_id = np.full(n, -1, dtype=int)
    for ev in events:
        sl = slice(ev.start, ev.start + ev.length)
        k = np.arange(ev.length)
        rise = max(3, int(0.7 * ev.length))
        progress = np.minimum(1.0, (k + 1) / rise)
        severity[sl] = MIN_SEVERITY + (1.0 - MIN_SEVERITY) * progress
        ftype[sl] = ev.fault_type
        direction[sl] = ev.direction
        position[sl] = k
        event_id[sl] = ev.event_id
    return FaultPlan(n, list(events), severity, ftype, direction, position, event_id)


def apply_fault_effects(
    drivers: Dict[str, np.ndarray],
    profile,
    plan: FaultPlan,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Return a copy of ``drivers`` with fault effects applied.

    ``drivers`` holds ``load`` (relative motor load around 1.0), ``speed``
    (RPM), ``vibration`` (mm/s), ``pressure`` (bar) and ``heat_extra`` (extra
    target temperature in deg C).  Motor current and temperature are derived
    from them afterwards, so faults propagate to coupled sensors.
    """
    n = plan.n
    out = {k: np.array(v, dtype=float, copy=True) for k, v in drivers.items()}
    s = plan.severity
    nom = profile.nominal

    over = np.where(plan.fault_type == OVERHEATING, s, 0.0)
    vib = np.where(plan.fault_type == EXCESSIVE_VIBRATION, s, 0.0)
    ovl = np.where(plan.fault_type == MOTOR_OVERLOAD, s, 0.0)
    pres = np.where(plan.fault_type == PRESSURE_ABNORMALITY, s, 0.0)

    # Random streams are always drawn in the same order for reproducibility.
    irregular = np.clip(1.0 + VIBRATION_IRREGULARITY * rng.standard_normal(n), 0.2, None)
    speed_jitter = rng.standard_normal(n)
    phase = float(rng.uniform(0, 2 * np.pi))
    pressure_noise = rng.standard_normal(n)

    # Overheating: gradual temperature rise, slightly higher current.
    out["heat_extra"] += over * profile.temp_rise
    out["load"] *= 1.0 + OVERHEAT_CURRENT_GAIN * over
    out["vibration"] *= 1.0 + 0.10 * over

    # Excessive vibration: growing, irregular vibration and unstable speed.
    out["vibration"] += vib * nom["vibration"] * VIBRATION_GAIN * irregular
    out["speed"] += vib * nom["motor_speed"] * VIBRATION_SPEED_JITTER * speed_jitter
    out["heat_extra"] += vib * 0.10 * profile.temp_rise
    out["load"] *= 1.0 + 0.03 * vib

    # Motor overload: more current, lower speed, more heat over time.
    out["load"] *= 1.0 + OVERLOAD_CURRENT_GAIN * ovl
    out["speed"] *= 1.0 - OVERLOAD_SPEED_DROP * ovl
    out["heat_extra"] += OVERLOAD_HEAT_FRACTION * profile.temp_rise * ovl
    out["vibration"] += ovl * nom["vibration"] * 0.30

    # Pressure abnormality: drift plus oscillation outside the normal band.
    wave = np.sin(2 * np.pi * plan.position / 12.0 + phase)
    out["pressure"] += nom["pressure"] * pres * (
        PRESSURE_DRIFT * plan.direction
        + PRESSURE_WAVE * wave
        + PRESSURE_NOISE * pressure_noise
    )
    out["load"] *= 1.0 + 0.04 * pres * np.abs(plan.direction)
    return out


def label_records(plan: FaultPlan, warning_as_fault: bool = False):
    """Return ``(condition, fault_type, fault_label)`` arrays (see module doc)."""
    sev = plan.severity
    condition = np.full(plan.n, NORMAL, dtype=object)
    condition[(sev > 0) & (sev < WARNING_SEVERITY_THRESHOLD)] = WARNING
    condition[sev >= WARNING_SEVERITY_THRESHOLD] = FAULT
    label = (condition == FAULT).astype(np.int64)
    if warning_as_fault:
        label[condition == WARNING] = 1
    return condition, plan.fault_type.copy(), label
