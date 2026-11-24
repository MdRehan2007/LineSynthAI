# LineSynth AI

*Intelligent Synthetic Sensor Data Generation for Food Manufacturing*

LineSynth AI is a Streamlit app that generates realistic **synthetic** sensor data for food-production
machines, simulates fault conditions, visualises the readings, exports CSV datasets and trains a
Random Forest for predictive-maintenance research. It runs fully offline: no API keys, cloud services,
databases or physical sensors.

## Features

| Page | What it does |
|---|---|
| Dashboard | Summary cards, sensor trend, fault distribution, machine summary, recent records |
| Sensor Data Generator | Machine type, condition, record count, sensors, seed, number of machines, sampling interval, start date |
| Fault Simulation | Fault percentage and duration, Warning labelling rule, live preview of each fault type |
| Data Visualization | Per-sensor Plotly charts, all-sensor overview, condition filter, normal-vs-fault comparison |
| Dataset Download | UTF-8 CSV of the complete dataset with a descriptive file name |
| ML Model Training | Random Forest, chronological split, metrics, confusion matrix, feature importance, model download |
| About Project | Method, labelling rule and limitations |

## Setup

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>.

Optional: use a virtual environment first (`python -m venv .venv`, then activate it).

## Quick walkthrough

1. Open **Sensor Data Generator**, keep the defaults and click **Generate Synthetic Data**.
2. Look at the **Dashboard** and **Data Visualization** pages.
3. Download the file on **Dataset Download**.
4. Click **Train Machine Learning Model** on **ML Model Training**.

Your settings and the generated dataset stay in the Streamlit session while you switch pages.
Changing a setting never alters an existing dataset; only a new click on *Generate Synthetic Data* does.
Reloading the browser tab starts a new session.

## Dataset columns

| Column | Description |
|---|---|
| timestamp | Reading date and time |
| machine_id | Machine identifier, e.g. `MILK-01` |
| machine_type | Selected machine type |
| temperature | °C |
| pressure | bar |
| vibration | mm/s |
| motor_speed | RPM |
| motor_current | A |
| condition | `Normal`, `Warning` or `Fault` |
| fault_type | `None` or the specific fault |
| fault_label | 0 normal, 1 fault |

Sensors that you deselect in the generator are left out of the file.

## How faults and labels work

Faults are *events*: runs of consecutive records during which severity grows from 0.2 to 1.0, then holds
until the event ends (like a maintenance reset). Events never overlap and are spread over the whole run.

* Severity below 0.5 -> **Warning** (early phase of the fault).
* Severity 0.5 or more -> **Fault**.
* **Labelling rule:** `fault_label` = 1 for Fault rows. Warning rows get 0 by default, or 1 if you choose
  "Warning counts as fault" on the Fault Simulation page. Normal rows are always 0.
* `fault_type` names the developing fault for Warning and Fault rows.

| Condition | Simulated behaviour |
|---|---|
| Overheating | Temperature rises gradually; current and vibration rise slightly |
| Excessive Vibration | Growing, irregular vibration; unstable speed |
| Motor Overload | Higher current, lower speed, temperature climbing over time |
| Pressure Abnormality | Pressure drifts out of its band with oscillating fluctuations |
| Mixed Conditions | Mostly normal data with events of random fault types |

Motor current and temperature are derived from a shared motor load (with a thermal lag), and vibration
grows with speed deviation, so sensors stay physically related.

**Hypothetical ranges.** All nominal values and "normal ranges" (shown in the app and in `generator.py`)
are invented for illustration. They are **not** certified industrial, safety or food-safety limits.

The fault percentage is approximate: it is rounded to a whole number of equally long events. For very
small datasets (a few dozen records) events are at least 5 records long, so the share can be higher.

## Machine learning

* Random Forest classifier (scikit-learn, `class_weight="balanced"`).
* Inputs: the selected sensor columns plus rolling mean/std (last 10 readings per machine, optional).
  Labels, fault types, ids and timestamps are never inputs.
* **No leakage:** chronological split per machine; the cut-off is moved so it never falls inside a fault
  sequence; a purge gap equal to the rolling window is removed from the end of the training part.
* Reported: accuracy (next to the majority-class baseline), precision, recall, F1, confusion matrix and
  feature importance.
* Friendly messages instead of crashes for empty data, missing columns, too few records, single-class
  data, or a split where train or test would contain only one class.

Performance on synthetic data does not establish real-world predictive-maintenance accuracy.

## Project structure

```text
LineSynthAI/
├── app.py              Streamlit interface and navigation
├── generator.py        Synthetic data generation, config validation, CSV export helpers
├── fault_simulator.py  Fault scheduling, effects and labelling
├── visualization.py    Plotly charts
├── ml_model.py         Random Forest training and evaluation
├── requirements.txt
├── README.md
├── tests/test_linesynth.py
└── .streamlit/config.toml
```

## Tests

```bash
python -m unittest discover -s tests -v
# or, if you have pytest:
python -m pytest tests -v
```

The tests cover record counts (100 and 10,000), normal and every fault type, required columns and types,
chronological timestamps, missing values, seed reproducibility, CSV round-trip, ML on a mixed dataset,
single-class handling, input validation and chart creation. The chart tests are skipped automatically
if Plotly is not installed.

## Troubleshooting

* *"A required package is missing"* - run `python -m pip install -r requirements.txt`.
* *Port 8501 already in use* - run `python -m streamlit run app.py --server.port 8502`.
* *ML says the test set has only one class* - generate more records, raise the fault percentage, shorten
  the fault duration or use more than one machine.

## Limitations

The simulator is a simplified model for research and teaching. It does not reproduce the physics of real
machines, and models trained on its output must not be used for real maintenance decisions.
