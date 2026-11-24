"""LineSynth AI - Streamlit application.

Run with:  python -m streamlit run app.py
"""
from __future__ import annotations

import copy
import traceback
from dataclasses import asdict
from datetime import date, datetime, time
from typing import List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="LineSynth AI", page_icon="🏭", layout="wide",
                   initial_sidebar_state="expanded")

try:
    import fault_simulator as fs
    import generator as gen
    import ml_model as ml
    import visualization as viz
except ImportError as exc:  # missing optional dependency such as plotly
    st.error(f"A required package is missing: **{exc.name or exc}**. "
             "Install everything with `python -m pip install -r requirements.txt` and restart the app.")
    st.stop()

APP_TITLE = "LineSynth AI"
APP_SUBTITLE = "Intelligent Synthetic Sensor Data Generation for Food Manufacturing"
PAGES = ["Dashboard", "Sensor Data Generator", "Fault Simulation", "Data Visualization",
         "Dataset Download", "ML Model Training", "About Project"]
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}
CONDITION_LABELS = fs.CONDITIONS

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "cfg_machine_type": gen.MACHINE_TYPES[0],
    "cfg_condition": fs.MIXED,
    "cfg_records": 1000,
    "cfg_sensors": list(gen.SENSOR_COLUMNS),
    "cfg_seed": 42,
    "cfg_machines": 1,
    "cfg_interval": 60,
    "cfg_start_date": date(2025, 1, 1),
    "cfg_fault_pct": 15,
    "cfg_fault_duration": 30,
    "cfg_warning_rule": list(gen.WARNING_RULES)[0],
    "preview_fault": fs.FAULT_TYPES[0],
    "ml_test_size": 30,
    "ml_trees": 150,
    "ml_rolling": True,
    "ml_seed": 42,
    "nav_page": PAGES[0],
}
OTHER_STATE = {"dataset": None, "meta": None, "ml_result": None, "ml_dataset_id": None, "dataset_counter": 0}
WIDGET_KEYS_TO_KEEP = list(DEFAULTS) + ["dash_sensor", "dash_machine", "viz_sensor", "viz_machine", "viz_conditions"]


def init_state() -> None:
    ss = st.session_state
    for key, value in DEFAULTS.items():
        if key not in ss:
            ss[key] = copy.deepcopy(value)
    for key, value in OTHER_STATE.items():
        if key not in ss:
            ss[key] = value
    # Streamlit forgets widget values on pages where the widget is not drawn;
    # re-assigning them keeps every setting alive while the user navigates.
    for key in WIDGET_KEYS_TO_KEEP:
        if key in ss:
            ss[key] = ss[key]


def ensure_choice(key: str, options: List, default=None) -> None:
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]


def ensure_multi(key: str, options: List) -> None:
    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = list(options)
    else:
        st.session_state[key] = [c for c in current if c in options]


def get_dataset() -> Optional[pd.DataFrame]:
    df = st.session_state.get("dataset")
    return df if not gen.dataset_problems(df) else None


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
CSS = """
<style>
.stApp { background: #f4f6fa; }
.block-container { padding-top: 1.6rem; max-width: 1300px; }
[data-testid="stSidebar"] { background: #0b1f3a; }
[data-testid="stSidebar"] * { color: #dbe6f5; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.15rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.55rem 0.8rem; border-radius: 10px; width: 100%; transition: background .15s; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #15325c; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #0f9d9a; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * { color: #ffffff; font-weight: 600; }
.ls-brand { font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: .2px; }
.ls-brand-sub { font-size: .8rem; color: #9db4d3; font-style: italic; margin-bottom: 1.1rem; line-height: 1.35; }
.ls-hero h1 { font-size: 2.1rem; color: #0b1f3a; margin: 0; padding: 0; line-height: 1.1; }
.ls-hero p { color: #4a5d78; font-style: italic; margin: .25rem 0 1.2rem 0; }
.ls-page-title { font-size: 1.35rem; font-weight: 650; color: #0b1f3a; margin-bottom: .15rem; }
.ls-page-blurb { color: #5b6b82; margin-bottom: 1rem; }
.ls-card { background: #ffffff; border-radius: 16px; padding: 1rem 1.2rem;
    border-left: 6px solid #2563eb; box-shadow: 0 1px 3px rgba(11,31,58,.08); }
.ls-card.teal { border-left-color: #0f9d9a; }
.ls-card.red { border-left-color: #e5484d; }
.ls-card.navy { border-left-color: #0b1f3a; }
.ls-card-label { color: #5b6b82; font-size: .85rem; }
.ls-card-value { color: #0b1f3a; font-size: 1.9rem; font-weight: 700; line-height: 1.25; }
.ls-card-sub { color: #7b8aa0; font-size: .78rem; min-height: 1rem; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; padding: .55rem 1.3rem; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: #0f9d9a; border: 1px solid #0f9d9a; color: #ffffff; }
.stButton > button[kind="primary"]:hover { background: #0b7f7d; border-color: #0b7f7d; color: #ffffff; }
.ls-note { background: #eaf1ff; border-radius: 12px; padding: .8rem 1rem; color: #1d3a6b; font-size: .9rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def metric_card(label: str, value, sub: str = "", accent: str = "") -> None:
    st.markdown(
        f'<div class="ls-card {accent}"><div class="ls-card-label">{label}</div>'
        f'<div class="ls-card-value">{value}</div><div class="ls-card-sub">{sub}</div></div>',
        unsafe_allow_html=True)


def page_header(title: str, blurb: str) -> None:
    st.markdown(f'<div class="ls-page-title">{title}</div><div class="ls-page-blurb">{blurb}</div>',
                unsafe_allow_html=True)


def show_chart(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
    except Exception:  # older Streamlit versions use use_container_width
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


def show_table(data: pd.DataFrame, height: Optional[int] = None) -> None:
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def show_error_details(exc: Exception) -> None:
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def require_dataset(action: str = "this page") -> Optional[pd.DataFrame]:
    """Return the dataset, or show a friendly empty state and return None."""
    df = get_dataset()
    if df is None:
        stored = st.session_state.get("dataset")
        if stored is not None:
            for problem in gen.dataset_problems(stored):
                st.error(problem)
        st.info(f"No dataset yet. Open **Sensor Data Generator**, choose your settings and click "
                f"**Generate Synthetic Data**. Then come back to use {action}.")
    return df


# --------------------------------------------------------------------------- #
# Configuration / generation
# --------------------------------------------------------------------------- #
def build_config() -> gen.GeneratorConfig:
    ss = st.session_state
    return gen.GeneratorConfig(
        machine_type=ss["cfg_machine_type"],
        condition=ss["cfg_condition"],
        n_records=ss["cfg_records"],
        sensors=tuple(s for s in gen.SENSOR_COLUMNS if s in ss["cfg_sensors"]),
        seed=ss["cfg_seed"],
        n_machines=ss["cfg_machines"],
        fault_pct=float(ss["cfg_fault_pct"]),
        fault_duration=ss["cfg_fault_duration"],
        sampling_seconds=ss["cfg_interval"],
        start_time=datetime.combine(ss["cfg_start_date"], time(0, 0)),
        warning_as_fault=gen.WARNING_RULES[ss["cfg_warning_rule"]],
    )


def config_changed_since_generation() -> bool:
    meta = st.session_state.get("meta")
    if not meta:
        return False
    try:
        return asdict(build_config()) != meta["config"]
    except Exception:
        return True


def run_generation() -> None:
    ss = st.session_state
    try:
        cfg = build_config()
        cfg.validate()
    except gen.ConfigError as exc:
        for message in exc.errors:
            st.error(message)
        return
    except Exception as exc:
        st.error(f"The settings could not be read: {exc}")
        show_error_details(exc)
        return
    try:
        with st.spinner("Generating synthetic sensor data..."):
            df = gen.generate_dataset(cfg)
    except Exception as exc:  # surface unexpected errors instead of hiding them
        st.error(f"Data generation failed: {exc}")
        show_error_details(exc)
        return
    ss["dataset_counter"] += 1
    ss["dataset"] = df
    ss["meta"] = {"config": asdict(cfg), "dataset_id": ss["dataset_counter"],
                  "generated_at": datetime.now(),
                  "filename": gen.build_csv_filename(asdict(cfg), len(df))}
    ss["ml_result"] = None
    ss["ml_dataset_id"] = None
    for key in ("dash_machine", "viz_machine"):
        ss.pop(key, None)
    st.success(f"Generated {len(df):,} records for {cfg.n_machines} machine(s). "
               "The Dashboard, charts, download and ML pages now use this dataset.")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    page_header("Dashboard", "Overview of the most recently generated dataset.")
    df = get_dataset()
    stats = gen.summarize_dataset(df)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Generated Records", f"{stats['total']:,}", accent="")
    with cols[1]:
        metric_card("Normal Records", f"{stats['normal']:,}", accent="teal")
    with cols[2]:
        metric_card("Fault Records", f"{stats['fault']:,}",
                    f"plus {stats['warning']:,} warning records", accent="red")
    with cols[3]:
        metric_card("Simulated Machines", f"{stats['machines']:,}", accent="navy")
    st.write("")

    if df is None:
        require_dataset("the dashboard")
        return

    sensors = gen.available_sensors(df)
    machines = sorted(df["machine_id"].unique().tolist())
    ensure_choice("dash_sensor", sensors, sensors[0])
    ensure_choice("dash_machine", machines, machines[0])

    st.subheader("Sensor trend")
    c1, c2 = st.columns(2)
    c1.selectbox("Sensor", sensors, key="dash_sensor", format_func=lambda s: gen.SENSOR_LABELS[s])
    c2.selectbox("Machine", machines, key="dash_machine")
    try:
        show_chart(viz.sensor_time_series(df, st.session_state["dash_sensor"],
                                          st.session_state["dash_machine"]), "dash_trend")
    except Exception as exc:
        st.error(f"The trend chart could not be drawn: {exc}")

    left, right = st.columns(2)
    with left:
        show_chart(viz.condition_distribution_chart(df), "dash_condition")
    with right:
        show_chart(viz.fault_type_chart(df), "dash_fault_types")

    st.subheader("Machine condition summary")
    show_table(gen.machine_summary(df))

    st.subheader("Recent records")
    st.caption("The 15 most recent readings of the generated dataset.")
    show_table(df.sort_values("timestamp", kind="stable").tail(15).iloc[::-1])


def page_generator() -> None:
    page_header("Sensor Data Generator",
                "Choose a machine, a condition and the sensors to simulate, then generate a dataset.")
    ss = st.session_state

    c1, c2 = st.columns(2)
    c1.selectbox("Machine type", gen.MACHINE_TYPES, key="cfg_machine_type")
    c2.selectbox("Machine condition", fs.CONDITION_OPTIONS, key="cfg_condition")

    c3, c4, c5 = st.columns(3)
    c3.selectbox("Number of records", gen.RECORD_OPTIONS, key="cfg_records",
                 format_func=lambda n: f"{n:,}")
    c4.number_input("Random seed", min_value=0, max_value=gen.MAX_SEED, step=1, key="cfg_seed",
                    help="The same seed and settings always produce the same dataset.")
    c5.number_input("Number of machines", min_value=1, max_value=gen.MAX_MACHINES, step=1,
                    key="cfg_machines", help="Records are shared across independent machine runs.")

    c6, c7 = st.columns(2)
    c6.number_input("Sampling interval (seconds)", min_value=1, max_value=86400, step=1,
                    key="cfg_interval")
    c7.date_input("Start date", key="cfg_start_date")

    st.multiselect("Sensor parameters", gen.SENSOR_COLUMNS, key="cfg_sensors",
                   format_func=lambda s: gen.SENSOR_LABELS[s],
                   help="Sensors you deselect are left out of the dataset.")

    if ss["cfg_condition"] == fs.NORMAL:
        st.markdown('<div class="ls-note">Normal operation: no fault events are added.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="ls-note">Fault settings: about <b>{ss["cfg_fault_pct"]}%</b> of the records belong to '
            f'fault events of <b>{ss["cfg_fault_duration"]}</b> records each. Change them on the '
            f'<b>Fault Simulation</b> page.</div>', unsafe_allow_html=True)
    st.write("")

    clicked = st.button("Generate Synthetic Data", type="primary", key="btn_generate")
    if clicked:
        run_generation()

    df = get_dataset()
    if df is None:
        st.caption("Your generated data will appear here.")
    else:
        if not clicked and config_changed_since_generation():
            st.warning("The settings above differ from the dataset shown below. "
                       "Click **Generate Synthetic Data** to create a new dataset.")
        stats = gen.summarize_dataset(df)
        k = st.columns(4)
        k[0].metric("Records", f"{stats['total']:,}")
        k[1].metric("Normal", f"{stats['normal']:,}")
        k[2].metric("Warning", f"{stats['warning']:,}")
        k[3].metric("Fault", f"{stats['fault']:,}")
        st.subheader("Generated data")
        show_table(df, height=420)

    with st.expander("Hypothetical operating ranges used by the simulator"):
        st.caption("Illustrative values for synthetic data only. They are not certified "
                   "industrial or food-safety limits.")
        rows = []
        for profile in gen.MACHINE_PROFILES.values():
            row = {"Machine": profile.name}
            for s in gen.SENSOR_COLUMNS:
                lo, hi = profile.normal_range[s]
                row[gen.SENSOR_LABELS[s]] = f"{lo:g} to {hi:g} (nominal {profile.nominal[s]:g})"
            rows.append(row)
        show_table(pd.DataFrame(rows))


def page_fault_simulation() -> None:
    page_header("Fault Simulation",
                "Faults are simulated as developing sequences over time, not as random outliers.")
    ss = st.session_state
    disabled = ss["cfg_condition"] == fs.NORMAL
    if disabled:
        st.info("The selected machine condition is **Normal**, so these settings have no effect. "
                "Choose a fault or Mixed Conditions on the Sensor Data Generator page.")

    c1, c2 = st.columns(2)
    c1.slider("Fault percentage (share of records inside fault events)", 1, 40,
              key="cfg_fault_pct", disabled=disabled)
    c2.slider("Fault duration (records per event)", 5, 200, key="cfg_fault_duration", disabled=disabled)
    st.selectbox("Labelling rule for Warning records", list(gen.WARNING_RULES), key="cfg_warning_rule",
                 help="Warning rows are the early phase of a developing fault.")
    st.caption("Fault percentage is approximate: it is rounded to a whole number of events. "
               "Settings apply the next time you click Generate Synthetic Data.")

    st.subheader("How each condition behaves")
    st.dataframe(pd.DataFrame(
        [{"Condition": "Normal operation",
          "Simulated behaviour": "Stable temperature and pressure, low vibration, steady speed and current."}] +
        [{"Condition": name, "Simulated behaviour": text} for name, text in fs.FAULT_DESCRIPTIONS.items()]),
        hide_index=True)
    st.markdown(
        "**Severity and labels.** Inside a fault event severity grows from 0.2 to 1.0. "
        f"Below {fs.WARNING_SEVERITY_THRESHOLD} the record is labelled **Warning**, from "
        f"{fs.WARNING_SEVERITY_THRESHOLD} upward it is **Fault**. `fault_label` is 1 for Fault records; "
        "Warning records follow the labelling rule selected above (default: 0). "
        "The event ends abruptly, like a maintenance reset.")

    st.subheader("Fault preview")
    st.caption("A short simulated run using the machine type, seed, fault percentage and duration from "
               "your current settings. It does not change your dataset.")
    st.selectbox("Fault to preview", fs.FAULT_TYPES, key="preview_fault")
    try:
        preview_cfg = gen.GeneratorConfig(
            machine_type=ss["cfg_machine_type"], condition=ss["preview_fault"], n_records=400,
            seed=ss["cfg_seed"], fault_pct=float(max(ss["cfg_fault_pct"], 10)),
            fault_duration=min(int(ss["cfg_fault_duration"]), 100), sampling_seconds=ss["cfg_interval"],
            warning_as_fault=gen.WARNING_RULES[ss["cfg_warning_rule"]])
        preview = gen.generate_dataset(preview_cfg)
        show_chart(viz.all_sensors_overview(preview), "fault_preview")
    except Exception as exc:
        st.error(f"The fault preview could not be created: {exc}")
        show_error_details(exc)


def page_visualization() -> None:
    page_header("Data Visualization", "Explore the generated sensor data. Charts support zoom, pan and hover.")
    df = require_dataset("the charts")
    if df is None:
        return
    sensors = gen.available_sensors(df)
    machines = df["machine_id"].unique().tolist()
    machine_options = sorted(machines) if len(machines) == 1 else [viz.ALL_MACHINES] + sorted(machines)
    present = [c for c in CONDITION_LABELS if c in set(df["condition"])]
    ensure_choice("viz_sensor", sensors, sensors[0])
    ensure_choice("viz_machine", machine_options, machine_options[0])
    ensure_multi("viz_conditions", present)

    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.selectbox("Sensor", sensors, key="viz_sensor", format_func=lambda s: gen.SENSOR_LABELS[s])
    c2.selectbox("Machine", machine_options, key="viz_machine")
    c3.multiselect("Show conditions", present, key="viz_conditions")
    sensor = st.session_state["viz_sensor"]
    machine = st.session_state["viz_machine"]
    conditions = st.session_state["viz_conditions"]
    if not conditions:
        st.warning("Select at least one condition to see coloured readings.")

    tab1, tab2, tab3 = st.tabs(["Selected sensor", "All sensors", "Normal vs fault"])
    with tab1:
        try:
            show_chart(viz.sensor_time_series(df, sensor, machine, conditions), "viz_series")
        except Exception as exc:
            st.error(f"The chart could not be drawn: {exc}")
        subset = df if machine == viz.ALL_MACHINES else df[df["machine_id"] == machine]
        subset = subset[subset["condition"].isin(conditions)]
        st.markdown(f"**Individual readings** ({len(subset):,} records match the filters)")
        if subset.empty:
            st.info("No records match the selected filters.")
        else:
            show_table(subset, height=300)
            row_number = st.number_input("Inspect one record (row number in the table above)",
                                         min_value=1, max_value=len(subset), value=1, step=1)
            record = subset.iloc[int(row_number) - 1]
            show_table(pd.DataFrame({"field": record.index.astype(str), "value": record.astype(str).to_numpy()}))
    with tab2:
        try:
            show_chart(viz.all_sensors_overview(df, machine), "viz_all")
        except Exception as exc:
            st.error(f"The overview could not be drawn: {exc}")
    with tab3:
        left, right = st.columns(2)
        with left:
            show_chart(viz.condition_distribution_chart(df), "viz_dist")
        with right:
            show_chart(viz.sensor_distribution_chart(df, sensor), "viz_box")
        show_chart(viz.fault_type_chart(df), "viz_types")
        stats = (df.groupby("condition")[sensor].agg(["count", "mean", "std", "min", "max"])
                 .reindex(CONDITION_LABELS).dropna(how="all").round(3).reset_index())
        st.markdown(f"**{gen.SENSOR_LABELS[sensor]} by condition**")
        show_table(stats)


def page_download() -> None:
    page_header("Dataset Download", "Export the complete generated dataset as a CSV file.")
    df = get_dataset()
    meta = st.session_state.get("meta") or {}
    if df is None:
        st.warning("There is no dataset to download yet. Generate one on the "
                   "**Sensor Data Generator** page first.")
        st.download_button("Download CSV Dataset", data=b"", file_name="linesynth_dataset.csv",
                           mime="text/csv", disabled=True)
        return
    try:
        data = gen.dataset_to_csv_bytes(df)
    except Exception as exc:
        st.error(f"The CSV file could not be prepared: {exc}")
        show_error_details(exc)
        return
    filename = meta.get("filename") or gen.build_csv_filename(None, len(df))
    k = st.columns(3)
    k[0].metric("Rows", f"{len(df):,}")
    k[1].metric("Columns", len(df.columns))
    k[2].metric("File size", f"{len(data) / 1024:,.0f} KB")
    st.caption(f"File name: `{filename}`. Encoding: UTF-8. Timestamps are written as YYYY-MM-DD HH:MM:SS.")
    st.download_button("Download CSV Dataset", data=data, file_name=filename, mime="text/csv",
                       type="primary")
    st.subheader("Preview")
    show_table(df.head(20))


def page_ml() -> None:
    page_header("ML Model Training",
                "Train a Random Forest that predicts fault records from sensor readings.")
    ss = st.session_state
    st.markdown(f'<div class="ls-note">{ml.DISCLAIMER}</div>', unsafe_allow_html=True)
    st.write("")
    df = get_dataset()
    if df is None:
        require_dataset("model training")

    c1, c2, c3 = st.columns(3)
    c1.slider("Test set share (%)", 10, 50, key="ml_test_size", step=5)
    c2.slider("Number of trees", 50, 400, key="ml_trees", step=50)
    c3.number_input("Model seed", min_value=0, max_value=gen.MAX_SEED, step=1, key="ml_seed")
    st.checkbox("Add rolling mean/std features (last 10 readings per machine)", key="ml_rolling")
    st.caption("The split is chronological per machine and never cuts through a fault sequence. "
               "Labels, fault types, ids and timestamps are not used as inputs.")

    if st.button("Train Machine Learning Model", type="primary", key="btn_train"):
        if df is None:
            st.warning("Generate a dataset before training a model.")
        else:
            try:
                with st.spinner("Training the Random Forest..."):
                    result = ml.train_fault_classifier(
                        df, test_size=ss["ml_test_size"] / 100.0, n_estimators=ss["ml_trees"],
                        seed=int(ss["ml_seed"]), use_rolling=ss["ml_rolling"])
                ss["ml_result"] = result
                ss["ml_dataset_id"] = (ss.get("meta") or {}).get("dataset_id")
            except Exception as exc:
                ss["ml_result"] = None
                st.error(f"Model training failed: {exc}")
                show_error_details(exc)

    result = ss.get("ml_result")
    if result is None:
        return
    if not result.ok:
        st.warning(result.message)
        return
    if ss.get("ml_dataset_id") != (ss.get("meta") or {}).get("dataset_id"):
        st.info("These results belong to an earlier dataset. Train again to use the current one.")
    for message in result.warnings:
        st.warning(message)

    m = result.metrics
    k = st.columns(4)
    with k[0]:
        metric_card("Accuracy", f"{m['accuracy']:.1%}", f"always predicting the majority class: {m['majority_baseline_accuracy']:.1%}")
    with k[1]:
        metric_card("Precision", f"{m['precision']:.1%}", "share of predicted faults that were faults", "teal")
    with k[2]:
        metric_card("Recall", f"{m['recall']:.1%}", "share of real faults that were found", "red")
    with k[3]:
        metric_card("F1-score", f"{m['f1']:.1%}", "balance of precision and recall", "navy")
    st.write("")
    left, right = st.columns(2)
    with left:
        show_chart(viz.confusion_matrix_chart(result.confusion), "ml_cm")
    with right:
        show_chart(viz.feature_importance_chart(result.feature_importance), "ml_fi")

    info = result.split_info
    st.markdown(
        f"**Split:** {info['split_method']}. Training rows: {info['train_rows']:,} "
        f"({info['train_fault_rows']:,} fault). Test rows: {info['test_rows']:,} "
        f"({info['test_fault_rows']:,} fault). Purge gap: {info['purge_rows']} rows. "
        f"Machines: {info['machines']}.")
    payload = ml.serialize_model(result)
    if payload is not None:
        st.download_button("Download trained model (.joblib)", data=payload,
                           file_name="linesynth_random_forest.joblib", mime="application/octet-stream")
    st.caption(ml.DISCLAIMER)


def page_about() -> None:
    page_header("About Project", "What LineSynth AI does and how to read its results.")
    st.markdown(f"""
**{APP_TITLE}** generates realistic *synthetic* sensor data for four types of food-manufacturing
machines, simulates fault conditions, lets you explore the data, export it as CSV and train a basic
predictive-maintenance model. It runs completely offline: no API keys, cloud services or hardware.

**How the data is made.** Each machine run starts from slowly varying load, speed, pressure and vibration
signals. Fault events (overheating, excessive vibration, motor overload, pressure abnormality) develop
gradually and affect several sensors at once. Motor current and temperature are derived from the load, with a
thermal lag, so the sensors stay physically related. Measurement noise is added at the end.

**Labels.** `Normal` = no fault active. `Warning` = early phase of a developing fault. `Fault` = fault
has reached at least half of its full severity. `fault_label` is 1 for Fault rows; Warning rows get 0 unless
you choose the other labelling rule on the Fault Simulation page.

**Modules.** `generator.py` (data generation), `fault_simulator.py` (fault events and effects),
`visualization.py` (Plotly charts), `ml_model.py` (Random Forest), `app.py` (this interface).

**Important limits.**
- All operating ranges are hypothetical and are not certified industrial or food-safety limits.
- The simulation is a simplified model. Real machines are more complex.
- A model that scores well on synthetic data has not been shown to work on real plants.

Built with Python, Streamlit, NumPy, Pandas, Plotly and scikit-learn.
""")
    with st.expander("Dataset columns"):
        show_table(pd.DataFrame({
            "Column": gen.DATASET_COLUMNS,
            "Description": ["Sensor reading date and time", "Unique machine identifier", "Selected machine type",
                            "Temperature in °C", "Pressure in bar", "Vibration in mm/s", "Motor speed in RPM",
                            "Motor current in A", "Normal, Warning or Fault", "None or the specific fault",
                            "0 normal, 1 fault"]}))


PAGE_FUNCTIONS = {
    "Dashboard": page_dashboard,
    "Sensor Data Generator": page_generator,
    "Fault Simulation": page_fault_simulation,
    "Data Visualization": page_visualization,
    "Dataset Download": page_download,
    "ML Model Training": page_ml,
    "About Project": page_about,
}


def main() -> None:
    init_state()
    inject_css()
    with st.sidebar:
        st.markdown(f'<div class="ls-brand">{APP_TITLE}</div><div class="ls-brand-sub">{APP_SUBTITLE}</div>',
                    unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        meta = st.session_state.get("meta")
        st.markdown("---")
        if meta:
            st.caption(f"Dataset: {len(get_dataset()) if get_dataset() is not None else 0:,} records "
                       f"({meta['config']['machine_type']})")
        else:
            st.caption("No dataset generated yet.")
    st.markdown(f'<div class="ls-hero"><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>',
                unsafe_allow_html=True)
    try:
        PAGE_FUNCTIONS[st.session_state["nav_page"]]()
    except Exception as exc:  # never hide unexpected errors silently
        st.error(f"Something unexpected went wrong on this page: {exc}")
        show_error_details(exc)


main()
