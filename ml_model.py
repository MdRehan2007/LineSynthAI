"""Predictive-maintenance classifier for LineSynth AI.

A Random Forest predicts ``fault_label`` (0 = normal, 1 = fault) from sensor
readings and simple rolling statistics of those readings.

Leakage controls
----------------
* ``condition``, ``fault_type``, ``fault_label``, ids and timestamps are never
  used as inputs.
* The train/test split is **chronological per machine** (first part of each
  machine's history trains, the last part tests).  If the cut-off would fall
  inside a fault sequence, it is moved to the start of that sequence so that
  no sequence is shared between the two sets.
* A small purge gap (the rolling-window length) is removed from the end of
  the training part so rolling features never straddle the boundary.
* Rolling features only look backwards in time.

IMPORTANT: results obtained on synthetic data say nothing about real-world
predictive-maintenance accuracy.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from generator import available_sensors, dataset_problems

DISCLAIMER = (
    "These results come from synthetic data produced by LineSynth AI. Performance "
    "on synthetic data does not establish real-world predictive-maintenance accuracy."
)
MIN_ROWS = 60
MIN_POSITIVES = 4
DEFAULT_WINDOW = 10


class MLValidationError(Exception):
    """The dataset cannot be used for training (message is user-friendly)."""


@dataclass
class MLResult:
    ok: bool
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None
    split_info: Dict[str, object] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model: object = None


def build_features(df: pd.DataFrame, use_rolling: bool = True,
                   window: int = DEFAULT_WINDOW) -> Tuple[pd.DataFrame, List[str]]:
    """Return the (sorted) frame with feature columns appended and the feature names."""
    sensors = available_sensors(df)
    work = df.sort_values(["machine_id", "timestamp"], kind="stable").reset_index(drop=True)
    for s in sensors:
        work[s] = pd.to_numeric(work[s], errors="coerce")
    features = list(sensors)
    if use_rolling:
        grouped = work.groupby("machine_id")
        for s in sensors:
            work[f"{s}_roll_mean"] = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).mean())
            work[f"{s}_roll_std"] = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0.0)
            features += [f"{s}_roll_mean", f"{s}_roll_std"]
    return work, features


def chronological_group_split(work: pd.DataFrame, test_size: float,
                              purge: int) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, test_mask) over the rows of ``work`` (see module doc)."""
    train = np.zeros(len(work), dtype=bool)
    test = np.zeros(len(work), dtype=bool)
    nonnormal_all = (work["condition"].to_numpy() != "Normal")
    for _, idx in work.groupby("machine_id").indices.items():
        idx = np.asarray(idx)
        m = len(idx)
        if m < 2:
            continue
        nonnormal = nonnormal_all[idx]
        cut = int(round(m * (1.0 - test_size)))
        cut = min(max(cut, 1), m - 1)
        while cut > 0 and nonnormal[cut - 1] and nonnormal[cut]:
            cut -= 1  # never cut through a fault sequence
        train_end = max(0, cut - purge)
        train[idx[:train_end]] = True
        test[idx[cut:]] = True
    return train, test


def _validate(df: Optional[pd.DataFrame], test_size: float) -> None:
    problems = dataset_problems(df)
    if problems:
        raise MLValidationError(" ".join(problems) + " Generate a dataset first.")
    if not 0.1 <= test_size <= 0.5:
        raise MLValidationError("Test size must be between 10 % and 50 %.")
    labels = pd.to_numeric(df["fault_label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()) <= {0, 1}:
        raise MLValidationError("fault_label must contain only 0 and 1.")
    if len(df) < MIN_ROWS:
        raise MLValidationError(
            f"Only {len(df)} records are available; at least {MIN_ROWS} are needed to train "
            "and test a model. Generate a larger dataset."
        )
    if labels.nunique() < 2:
        raise MLValidationError(
            "The dataset contains only one class, so a classifier cannot learn to tell "
            "normal and fault records apart. Generate data with a fault condition or "
            "'Mixed Conditions' and make sure the fault percentage is greater than 0."
        )
    if int((labels == 1).sum()) < MIN_POSITIVES:
        raise MLValidationError(
            f"Only {int((labels == 1).sum())} fault records exist; at least {MIN_POSITIVES} are "
            "needed. Increase the number of records or the fault percentage."
        )


def train_fault_classifier(df: pd.DataFrame, test_size: float = 0.3, n_estimators: int = 150,
                           seed: int = 42, use_rolling: bool = True,
                           window: int = DEFAULT_WINDOW) -> MLResult:
    """Train and evaluate the model.

    Returns an ``MLResult``.  Problems with the *data* (empty, single class,
    too small, ...) give ``ok=False`` with a friendly message.  Unexpected
    errors are not swallowed; the caller is expected to display them.
    """
    try:
        _validate(df, test_size)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                     precision_score, recall_score)

        work, features = build_features(df, use_rolling, window)
        warnings: List[str] = []
        missing = [s for s in ("temperature", "pressure", "vibration", "motor_speed", "motor_current")
                   if s not in work.columns]
        if missing:
            warnings.append("Sensor columns not in the dataset were skipped: " + ", ".join(missing) + ".")

        valid = work[features].notna().all(axis=1).to_numpy()
        if (~valid).any():
            warnings.append(f"{int((~valid).sum())} rows with missing sensor values were excluded.")
        y_all = pd.to_numeric(work["fault_label"]).astype(int).to_numpy()

        purge = window if use_rolling else 1
        train_mask, test_mask = chronological_group_split(work, test_size, purge)
        train_mask &= valid
        test_mask &= valid
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        if train_mask.sum() < 30 or test_mask.sum() < 10:
            raise MLValidationError("Not enough records to build both a training and a test set. "
                                    "Generate more records.")
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise MLValidationError(
                "After the time-ordered split, the training or the test set contains only one class "
                f"(train fault records: {int(y_train.sum())}, test fault records: {int(y_test.sum())}). "
                "Generate more records, use a higher fault percentage, a shorter fault duration, "
                "or more machines."
            )

        model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(seed),
                                       class_weight="balanced", n_jobs=-1)
        model.fit(work.loc[train_mask, features], y_train)
        pred = model.predict(work.loc[test_mask, features])

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "majority_baseline_accuracy": float(max(np.mean(y_test == 0), np.mean(y_test == 1))),
        }
        if pred.sum() == 0:
            warnings.append("The model predicted no faults in the test set, so precision is reported as 0.")

        importance = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                      .sort_values("importance", ascending=False).reset_index(drop=True))
        split_info = {
            "split_method": "Chronological per machine with sequence-safe cut-off and purge gap",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_fault_rows": int(y_train.sum()),
            "test_fault_rows": int(y_test.sum()),
            "purge_rows": int(purge),
            "machines": int(work["machine_id"].nunique()),
        }
        return MLResult(ok=True, message="Model trained successfully.", metrics=metrics,
                        confusion=confusion_matrix(y_test, pred, labels=[0, 1]),
                        feature_importance=importance, split_info=split_info,
                        features=features, warnings=warnings, model=model)
    except MLValidationError as exc:
        return MLResult(ok=False, message=str(exc))


def serialize_model(result: MLResult) -> Optional[bytes]:
    """Return the trained model as joblib bytes, or None if unavailable."""
    if result is None or not result.ok or result.model is None:
        return None
    try:
        import joblib
    except ImportError:
        return None
    buffer = io.BytesIO()
    joblib.dump({"model": result.model, "features": result.features, "metrics": result.metrics,
                 "note": DISCLAIMER}, buffer)
    return buffer.getvalue()
