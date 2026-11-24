"""Automated tests for LineSynth AI core functionality.

Run from the project folder with either:
    python -m unittest discover -s tests -v
    python -m pytest tests -v        (if pytest is installed)

Chart tests are skipped automatically when Plotly is not installed.
"""
import importlib.util
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fault_simulator as fs  # noqa: E402
import generator as gen  # noqa: E402
import ml_model as ml  # noqa: E402

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def make(**kwargs):
    return gen.generate_dataset(gen.GeneratorConfig(**kwargs))


class TestGeneration(unittest.TestCase):
    def test_exactly_100_records(self):
        self.assertEqual(len(make(n_records=100)), 100)

    def test_exactly_10000_records(self):
        self.assertEqual(len(make(n_records=10000)), 10000)

    def test_multiple_machines_keep_exact_count(self):
        df = make(n_records=1003, n_machines=4)
        self.assertEqual(len(df), 1003)
        self.assertEqual(df["machine_id"].nunique(), 4)

    def test_every_machine_type(self):
        for machine in gen.MACHINE_TYPES:
            df = make(machine_type=machine, n_records=200)
            self.assertTrue((df["machine_type"] == machine).all())

    def test_normal_condition_has_no_faults(self):
        df = make(condition="Normal", n_records=500)
        self.assertTrue((df["condition"] == "Normal").all())
        self.assertTrue((df["fault_type"] == "None").all())
        self.assertTrue((df["fault_label"] == 0).all())

    def test_each_fault_type_is_generated(self):
        for fault in fs.FAULT_TYPES:
            df = make(condition=fault, n_records=1000)
            self.assertGreater((df["condition"] == "Fault").sum(), 0, fault)
            self.assertEqual(set(df.loc[df["condition"] != "Normal", "fault_type"]), {fault})
            self.assertTrue((df.loc[df["condition"] == "Normal", "fault_type"] == "None").all())

    def test_mixed_conditions_contains_all_fault_types_and_mostly_normal(self):
        df = make(condition="Mixed Conditions", n_records=5000, fault_pct=10)
        kinds = set(df["fault_type"]) - {"None"}
        self.assertEqual(kinds, set(fs.FAULT_TYPES))
        self.assertGreater((df["condition"] == "Normal").mean(), 0.8)

    def test_required_columns_and_dtypes(self):
        df = make(n_records=300)
        self.assertEqual(list(df.columns), gen.DATASET_COLUMNS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        for s in gen.SENSOR_COLUMNS:
            self.assertTrue(pd.api.types.is_float_dtype(df[s]), s)
        self.assertTrue(pd.api.types.is_integer_dtype(df["fault_label"]))
        self.assertTrue(set(df["condition"]) <= {"Normal", "Warning", "Fault"})
        self.assertTrue(set(df["fault_label"]) <= {0, 1})

    def test_timestamps_are_chronological(self):
        for machines in (1, 3):
            df = make(n_records=600, n_machines=machines)
            self.assertTrue(df["timestamp"].is_monotonic_increasing)
        single = make(n_records=300, sampling_seconds=10)
        self.assertTrue((single["timestamp"].diff().dropna() == pd.Timedelta(seconds=10)).all())

    def test_no_missing_values(self):
        self.assertEqual(int(make(n_records=2000, n_machines=2).isna().sum().sum()), 0)

    def test_reproducible_with_same_seed_and_different_with_other_seed(self):
        a = make(n_records=500, seed=123)
        b = make(n_records=500, seed=123)
        c = make(n_records=500, seed=124)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a[gen.SENSOR_COLUMNS].equals(c[gen.SENSOR_COLUMNS]))

    def test_values_are_physically_plausible(self):
        for fault in ["Normal"] + fs.FAULT_TYPES:
            df = make(condition=fault, n_records=2000)
            self.assertGreater(df["temperature"].min(), 0)
            self.assertGreater(df["pressure"].min(), 0)
            self.assertGreater(df["vibration"].min(), 0)
            self.assertGreater(df["motor_speed"].min(), 0)
            self.assertGreater(df["motor_current"].min(), 0)

    def test_sensor_subset_is_respected(self):
        df = make(n_records=100, sensors=("temperature", "vibration"))
        self.assertIn("temperature", df.columns)
        self.assertNotIn("pressure", df.columns)
        self.assertIn("fault_label", df.columns)


class TestFaultBehaviour(unittest.TestCase):
    def _split(self, fault):
        df = make(condition=fault, n_records=3000, seed=5)
        return df[df["condition"] == "Normal"], df[df["condition"] == "Fault"], df

    def test_overheating_raises_temperature_and_current(self):
        normal, fault, _ = self._split("Overheating")
        self.assertGreater(fault["temperature"].mean(), normal["temperature"].mean() + 5)
        self.assertGreater(fault["motor_current"].mean(), normal["motor_current"].mean())

    def test_excessive_vibration_raises_vibration(self):
        normal, fault, _ = self._split("Excessive Vibration")
        self.assertGreater(fault["vibration"].mean(), 2 * normal["vibration"].mean())

    def test_motor_overload_current_up_speed_down_temperature_up(self):
        normal, fault, _ = self._split("Motor Overload")
        self.assertGreater(fault["motor_current"].mean(), 1.3 * normal["motor_current"].mean())
        self.assertLess(fault["motor_speed"].mean(), normal["motor_speed"].mean())
        self.assertGreater(fault["temperature"].mean(), normal["temperature"].mean() + 3)

    def test_pressure_abnormality_leaves_normal_range(self):
        _, fault, df = self._split("Pressure Abnormality")
        lo, hi = gen.MACHINE_PROFILES[df["machine_type"].iloc[0]].normal_range["pressure"]
        self.assertGreater(((fault["pressure"] < lo) | (fault["pressure"] > hi)).mean(), 0.4)

    def test_faults_are_sequences_not_isolated_points(self):
        df = make(condition="Overheating", n_records=2000, fault_duration=25)
        active = (df["condition"] != "Normal").astype(int).to_numpy()
        runs, current = [], 0
        for v in active:
            if v:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        self.assertTrue(runs)
        self.assertGreaterEqual(min(runs), 5)

    def test_warning_comes_before_fault_and_rule_is_consistent(self):
        df = make(condition="Overheating", n_records=1000)
        self.assertTrue((df.loc[df["condition"] == "Warning", "fault_label"] == 0).all())
        self.assertTrue((df.loc[df["condition"] == "Fault", "fault_label"] == 1).all())
        strict = make(condition="Overheating", n_records=1000, warning_as_fault=True)
        self.assertTrue((strict.loc[strict["condition"] != "Normal", "fault_label"] == 1).all())

    def test_normal_data_mostly_inside_hypothetical_range(self):
        df = make(condition="Normal", n_records=3000)
        profile = gen.MACHINE_PROFILES[df["machine_type"].iloc[0]]
        for s in gen.SENSOR_COLUMNS:
            lo, hi = profile.normal_range[s]
            self.assertGreater(df[s].between(lo, hi).mean(), 0.99, s)

    def test_fault_percentage_is_roughly_respected(self):
        df = make(condition="Mixed Conditions", n_records=5000, fault_pct=20, fault_duration=40)
        share = (df["condition"] != "Normal").mean()
        self.assertAlmostEqual(share, 0.20, delta=0.05)


class TestValidation(unittest.TestCase):
    def test_invalid_settings_raise_config_error(self):
        bad = [
            dict(n_records=5), dict(n_records=1000.5), dict(n_records=10**7),
            dict(sensors=()), dict(sensors=("nonsense",)), dict(seed=-1),
            dict(machine_type="Rocket"), dict(condition="Haunted"),
            dict(fault_pct=0), dict(fault_pct=90), dict(fault_duration=1),
            dict(n_machines=0), dict(n_records=50, n_machines=10),
            dict(sampling_seconds=0),
        ]
        for kwargs in bad:
            with self.assertRaises(gen.ConfigError, msg=str(kwargs)):
                make(**kwargs)

    def test_dataset_problem_detection(self):
        self.assertTrue(gen.dataset_problems(None))
        self.assertTrue(gen.dataset_problems(pd.DataFrame()))
        self.assertTrue(gen.dataset_problems(pd.DataFrame({"a": [1]})))
        self.assertFalse(gen.dataset_problems(make(n_records=100)))
        self.assertEqual(gen.summarize_dataset(None)["total"], 0)


class TestCsvExport(unittest.TestCase):
    def test_csv_roundtrip_with_pandas(self):
        df = make(n_records=500, n_machines=2)
        raw = gen.dataset_to_csv_bytes(df)
        raw.decode("utf-8")
        back = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"])
        self.assertEqual(list(back.columns), list(df.columns))
        self.assertEqual(len(back), len(df))
        for s in gen.SENSOR_COLUMNS:
            np.testing.assert_allclose(back[s], df[s])
        self.assertEqual(back["fault_label"].tolist(), df["fault_label"].tolist())
        self.assertEqual(back["condition"].tolist(), df["condition"].tolist())
        self.assertEqual(back["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                         df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist())

    def test_csv_refuses_empty_data(self):
        with self.assertRaises(ValueError):
            gen.dataset_to_csv_bytes(pd.DataFrame())
        with self.assertRaises(ValueError):
            gen.dataset_to_csv_bytes(None)

    def test_filename_is_descriptive(self):
        name = gen.build_csv_filename({"machine_type": "Milk Processing Machine",
                                      "condition": "Mixed Conditions", "seed": 42}, 1000)
        self.assertTrue(name.startswith("linesynth_milk_processing_machine_mixed_conditions_1000rows_seed42_"))
        self.assertTrue(name.endswith(".csv"))


class TestMachineLearning(unittest.TestCase):
    def test_training_on_mixed_dataset(self):
        df = make(condition="Mixed Conditions", n_records=3000, seed=11)
        result = ml.train_fault_classifier(df, n_estimators=60)
        self.assertTrue(result.ok, result.message)
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertGreaterEqual(result.metrics[key], 0.0)
            self.assertLessEqual(result.metrics[key], 1.0)
        self.assertEqual(result.confusion.shape, (2, 2))
        self.assertEqual(int(result.confusion.sum()), result.split_info["test_rows"])
        self.assertFalse({"condition", "fault_type", "fault_label", "timestamp", "machine_id"} & set(result.features))
        self.assertEqual(len(result.feature_importance), len(result.features))
        self.assertGreater(result.metrics["f1"], 0.5)

    def test_split_is_chronological_and_never_cuts_a_fault_sequence(self):
        df = make(condition="Mixed Conditions", n_records=3000, n_machines=2, seed=3)
        work, _ = ml.build_features(df)
        train, test = ml.chronological_group_split(work, 0.3, purge=10)
        self.assertFalse((train & test).any())
        for _, part in work.groupby("machine_id"):
            t_idx = part.index[train[part.index]]
            s_idx = part.index[test[part.index]]
            self.assertLess(part.loc[t_idx, "timestamp"].max(), part.loc[s_idx, "timestamp"].min())
            boundary = s_idx.min()
            self.assertFalse(part.loc[boundary - 1, "condition"] != "Normal" and part.loc[boundary, "condition"] != "Normal")

    def test_two_machine_dataset_trains(self):
        df = make(condition="Mixed Conditions", n_records=4000, n_machines=2, seed=9)
        self.assertTrue(ml.train_fault_classifier(df, n_estimators=40).ok)

    def test_serialisation(self):
        df = make(n_records=2000, seed=2)
        result = ml.train_fault_classifier(df, n_estimators=20)
        self.assertTrue(result.ok, result.message)
        if importlib.util.find_spec("joblib"):
            self.assertGreater(len(ml.serialize_model(result)), 100)

    def test_single_class_dataset_is_handled(self):
        result = ml.train_fault_classifier(make(condition="Normal", n_records=1000))
        self.assertFalse(result.ok)
        self.assertIn("one class", result.message)

    def test_other_bad_inputs_do_not_crash(self):
        self.assertFalse(ml.train_fault_classifier(None).ok)
        self.assertFalse(ml.train_fault_classifier(pd.DataFrame()).ok)
        self.assertFalse(ml.train_fault_classifier(pd.DataFrame({"x": [1, 2, 3]})).ok)
        self.assertFalse(ml.train_fault_classifier(make(n_records=100).head(30)).ok)
        self.assertFalse(ml.train_fault_classifier(make(n_records=1000), test_size=0.9).ok)

    def test_missing_sensor_columns_are_tolerated(self):
        df = make(condition="Mixed Conditions", n_records=3000, sensors=("temperature", "vibration"), seed=4)
        result = ml.train_fault_classifier(df, n_estimators=30)
        self.assertTrue(result.ok, result.message)
        self.assertTrue(any("not in the dataset" in w for w in result.warnings))


@unittest.skipUnless(HAS_PLOTLY, "plotly is not installed")
class TestCharts(unittest.TestCase):
    def test_all_charts_can_be_created(self):
        import plotly.graph_objects as go
        import visualization as viz

        df = make(condition="Mixed Conditions", n_records=1000, n_machines=2)
        figures = [viz.sensor_time_series(df, s, "MILK-01") for s in gen.SENSOR_COLUMNS]
        figures += [viz.sensor_time_series(df, "temperature", viz.ALL_MACHINES, ["Fault"]),
                    viz.all_sensors_overview(df), viz.condition_distribution_chart(df),
                    viz.fault_type_chart(df), viz.sensor_distribution_chart(df, "vibration"),
                    viz.fault_type_chart(make(condition="Normal", n_records=100))]
        result = ml.train_fault_classifier(df, n_estimators=20)
        self.assertTrue(result.ok, result.message)
        figures += [viz.feature_importance_chart(result.feature_importance),
                    viz.confusion_matrix_chart(result.confusion)]
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)
            fig.to_dict()

    def test_chart_errors_are_readable(self):
        import visualization as viz

        with self.assertRaises(ValueError):
            viz.sensor_time_series(pd.DataFrame(), "temperature")
        with self.assertRaises(ValueError):
            viz.sensor_time_series(make(n_records=100), "nonexistent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
