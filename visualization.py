"""Interactive Plotly charts for LineSynth AI.

All functions take the generated DataFrame (or ML results) and return a
``plotly.graph_objects.Figure``.  They raise ``ValueError`` with a readable
message when required columns are missing.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generator import MACHINE_PROFILES, SENSOR_LABELS, SENSOR_UNITS, available_sensors

CONDITION_ORDER = ["Normal", "Warning", "Fault"]
CONDITION_COLORS = {"Normal": "#0f9d9a", "Warning": "#f5a623", "Fault": "#e5484d"}
SIGNAL_COLOR = "#9fb0c8"
NAVY = "#0b1f3a"
BLUE = "#2563eb"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
ALL_MACHINES = "All machines"


def _require(df: Optional[pd.DataFrame], columns: Sequence[str]) -> None:
    if df is None or len(df) == 0:
        raise ValueError("There is no data to plot.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing columns needed for this chart: " + ", ".join(missing))


def _style(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16, color=NAVY)),
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=20, t=55, b=45),
        font=dict(family=FONT, color="#26364d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hoverlabel=dict(font_size=12),
    )
    return fig


def empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color="#6b7a90"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, "", height)


def _select_machine(df: pd.DataFrame, machine_id: Optional[str]) -> pd.DataFrame:
    if machine_id in (None, ALL_MACHINES):
        return df
    sub = df[df["machine_id"] == machine_id]
    if sub.empty:
        raise ValueError(f"No records found for machine {machine_id}.")
    return sub


def sensor_time_series(df: pd.DataFrame, sensor: str, machine_id: Optional[str] = None,
                       conditions: Optional[Iterable[str]] = None,
                       show_normal_range: bool = True, height: int = 420) -> go.Figure:
    """Sensor value vs time.

    A light line shows the complete signal; markers coloured by condition are
    drawn for the selected ``conditions`` only (Normal / Warning / Fault).
    """
    _require(df, ["timestamp", "machine_id", "condition", sensor])
    d = _select_machine(df, machine_id).sort_values("timestamp", kind="stable")
    shown = list(CONDITION_ORDER if conditions is None else conditions)
    label = SENSOR_LABELS.get(sensor, sensor)
    unit = SENSOR_UNITS.get(sensor, "")

    fig = go.Figure()
    first = True
    for mid, part in d.groupby("machine_id", sort=True):
        fig.add_trace(go.Scattergl(
            x=part["timestamp"], y=part[sensor], mode="lines",
            line=dict(color=SIGNAL_COLOR, width=1), name="Signal" if first else f"Signal {mid}",
            showlegend=first, hoverinfo="skip"))
        first = False

    for cond in CONDITION_ORDER:
        if cond not in shown:
            continue
        sub = d[d["condition"] == cond]
        if sub.empty:
            continue
        fault_type = sub["fault_type"] if "fault_type" in sub.columns else pd.Series("", index=sub.index)
        custom = np.column_stack([sub["machine_id"].astype(str).to_numpy(), fault_type.astype(str).to_numpy()])
        fig.add_trace(go.Scattergl(
            x=sub["timestamp"], y=sub[sensor], mode="markers",
            marker=dict(size=4 if cond == "Normal" else 6, color=CONDITION_COLORS[cond]),
            name=cond, customdata=custom,
            hovertemplate=("<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>" + label + ": %{y:.2f} " + unit +
                           "<br>Machine: %{customdata[0]}<br>Fault type: %{customdata[1]}"
                           "<extra>" + cond + "</extra>")))

    if show_normal_range and "machine_type" in d.columns and d["machine_type"].nunique() == 1:
        profile = MACHINE_PROFILES.get(str(d["machine_type"].iloc[0]))
        if profile and sensor in profile.normal_range:
            lo, hi = profile.normal_range[sensor]
            fig.add_hrect(y0=lo, y1=hi, fillcolor=CONDITION_COLORS["Normal"], opacity=0.08,
                          line_width=0, annotation_text="Hypothetical normal range",
                          annotation_position="top left", annotation_font_size=11)

    fig.update_xaxes(title_text="Time", rangeslider_visible=False)
    fig.update_yaxes(title_text=label)
    return _style(fig, f"{label} vs time", height)


def all_sensors_overview(df: pd.DataFrame, machine_id: Optional[str] = None,
                         sensors: Optional[List[str]] = None) -> go.Figure:
    """Stacked small multiples of every available sensor, faults highlighted."""
    _require(df, ["timestamp", "machine_id", "condition"])
    sensors = [s for s in (sensors or available_sensors(df)) if s in df.columns]
    if not sensors:
        return empty_figure("No sensor columns available.")
    d = _select_machine(df, machine_id).sort_values("timestamp", kind="stable")
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[SENSOR_LABELS.get(s, s) for s in sensors])
    for row, s in enumerate(sensors, start=1):
        for mid, part in d.groupby("machine_id", sort=True):
            fig.add_trace(go.Scattergl(x=part["timestamp"], y=part[s], mode="lines",
                                       line=dict(color=BLUE, width=1), name="Signal",
                                       showlegend=False, hoverinfo="skip"), row=row, col=1)
        for cond in ("Warning", "Fault"):
            sub = d[d["condition"] == cond]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub["timestamp"], y=sub[s], mode="markers",
                marker=dict(size=5, color=CONDITION_COLORS[cond]), name=cond,
                legendgroup=cond, showlegend=(row == 1),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.2f}<extra>" + cond + "</extra>"),
                row=row, col=1)
    fig.update_layout(hovermode="closest")
    return _style(fig, "All sensors (warning and fault records highlighted)", 190 * len(sensors) + 90)


def condition_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of Normal / Warning / Fault record counts."""
    _require(df, ["condition"])
    counts = df["condition"].value_counts().reindex(CONDITION_ORDER).dropna().astype(int)
    if counts.empty:
        return empty_figure("No condition labels found.")
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values), hole=0.55, sort=False,
        marker=dict(colors=[CONDITION_COLORS[c] for c in counts.index]),
        textinfo="percent", hovertemplate="%{label}: %{value:,} records (%{percent})<extra></extra>"))
    return _style(fig, "Normal vs fault distribution", 340)


def fault_type_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of warning / fault records per fault type."""
    _require(df, ["condition", "fault_type"])
    active = df[df["fault_type"] != "None"]
    if active.empty:
        return empty_figure("No fault events in this dataset.", 340)
    table = (active.groupby(["fault_type", "condition"]).size().unstack(fill_value=0)
             .reindex(columns=["Warning", "Fault"], fill_value=0))
    fig = go.Figure()
    for cond in ("Warning", "Fault"):
        fig.add_trace(go.Bar(x=list(table.index), y=table[cond].tolist(), name=cond,
                             marker_color=CONDITION_COLORS[cond],
                             hovertemplate="%{x}<br>%{y:,} " + cond.lower() + " records<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Records")
    return _style(fig, "Records per fault type", 340)


def sensor_distribution_chart(df: pd.DataFrame, sensor: str) -> go.Figure:
    """Box plot comparing a sensor across Normal / Warning / Fault records."""
    _require(df, ["condition", sensor])
    fig = go.Figure()
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub[sensor], name=cond, marker_color=CONDITION_COLORS[cond],
                             boxmean=True))
    fig.update_yaxes(title_text=SENSOR_LABELS.get(sensor, sensor))
    return _style(fig, f"{SENSOR_LABELS.get(sensor, sensor)}: normal vs fault periods", 340)


def feature_importance_chart(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    if importance is None or importance.empty:
        return empty_figure("No feature importance available.")
    top = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
                           marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Importance")
    return _style(fig, "Feature importance", 60 + 26 * len(top) + 60)


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Annotated heat map of a 2x2 confusion matrix (rows = actual)."""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        return empty_figure("Confusion matrix not available.")
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted normal", "Predicted fault"], y=["Actual normal", "Actual fault"],
        text=cm.astype(str), texttemplate="%{text}", textfont=dict(size=20),
        colorscale=[[0, "#eef4ff"], [1, BLUE]], showscale=False,
        hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (test set)", 340)
