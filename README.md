# EnergyMind v3 – single-house anomaly detector (XGBoost)

Ready-to-use model for the rest of the EnergyMind platform: it reads **one house** (smart meter, and PV / battery / inverter if any) plus its **transformer**, and returns for every 15-minute step whether there is an anomaly, which one, how sure it is, whether it is a real event or a measurement problem, a suggested action, and **why** (SHAP and LIME).

It was trained on our simulated Tunisian low-voltage network (SimBench LV1.101, 13 houses, 2018, 14 injected anomaly types). The features were audited for leakage: anything that only the simulator could know (the exact appliance sum, S vs P and Q checks) was removed and replaced by the house's own history.

## Quick start

```
cd detector_xgboost
pip install -r requirements.txt
python predict.py --house-file sample_data/house_07.csv --transformer-file sample_data/transformer_house_07.csv --out result.csv
```

The terminal prints, for the first 20 alarm episodes: when, which anomaly, how sure, the SHAP and LIME reasons and the suggested action. Options: `--show N` (episodes printed), `--lime N` (episodes explained by LIME, 0 = faster), `--no-explain`.

`sample_data/` holds 45 days of a house without PV (`house_07`) and a house with PV + battery (`house_09`), with their transformer data and the true injected anomalies (`*_true_labels.csv`, column `split` = train / val / test).

## Use it from Python (platform integration)

```python
from predict import Detector                     # run from inside detector_xgboost/
det = Detector()

# a whole period (at least 28 days of 15-min rows)
out = det.detect(house_df, trafo_df, house_id="house_07", explain=True, lime_episodes=5)

# streaming: the rolling buffer of raw readings (>= 2688 rows = 28 days) -> verdict of the newest step
out = det.detect_latest(house_buffer, trafo_buffer, house_id="house_07")
```

`detect_latest` runs exactly the same feature code as training (checked: streaming and batch give identical features).

## Inputs

| File | Columns |
|---|---|
| house | `timestamp`; `U_1..3`, `I_1..3`, `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `grid_available` (1/0). PV houses also: `P_PV_W`, `battery_soc_percent`, `battery_voltage_v`, `battery_current_a`, `battery_charge_power_w`, `battery_discharge_power_w`, `inverter_ac_output_w`, `inverter_status`, `transfer_switch_status`, `backup_mode_active`, `backup_load_demand_w`, `backup_load_served_w`, `unmet_backup_demand_w` |
| transformer | `timestamp`, `transformer_loading_percent`, `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`, `other_feeders_P_kw` |

One row every 15 minutes, the same timestamps in both, at least 28 days. No appliance-level data is needed.

## Output (one row per 15 min)

| Column | Meaning |
|---|---|
| `anomalies` | `normal`, or up to two types: `a|b` |
| `primary_anomaly`, `confidence` | most likely type and its probability |
| `category` | real grid event / real equipment event / sensor problem / data manipulation |
| `agent_action`, `needs_human` | what the agent may do alone, and what a person must decide |
| `p_<type>` | probability of each of the 14 types |
| `shap_top` | the features that pushed the model towards the detected type, with their values and SHAP contributions, in plain words |
| `lime_top` | LIME's local rules and weights, for the first row of the first N alarm episodes (`--lime N`, default 10) |

Example (a replay of yesterday's data):
```
SHAP: readings identical to 24 h earlier (meter_lag96_match = 1, SHAP +4.91); voltage vs 24 h earlier (U_lag96_absdiff = 0, SHAP +0.94); ...
LIME: meter_lag96_match > 0.00 (+0.35); meter_lag96_match_run > 0.00 (+0.03); ...
```

## Scores (simulated test set, 4 blocks of 14 days never used for training)

| | XGBoost |
|---|---|
| Macro F1 over the 14 types | 0.730 |
| Detection precision / recall | 0.980 / 0.522 |
| Event recall / event precision | 0.693 / 0.827 |
| False alarms | 0.158 per house-day (about one every 6 days per house) |

| Anomaly type | F1 |
|---|---|
| localized grid outage, sustained undervoltage, transformer overload | 1.00 |
| meter frozen, meter dropout, voltage sensor fault, falsified replay | 1.00 |
| battery charge failure | 0.90 |
| battery discharge failure | 0.85 |
| meter noise | 0.83 |
| falsified under-report | 0.63 |
| meter drift, meter current bias, PV partial shading | **not detected** with one house (disabled: validation PR-AUC below 0.5) |

**Read these with their limits.** The data are simulated. Without the appliance sum (which no real meter has), a slow drift or a current bias of a few percent cannot be told apart from normal changes in the house's consumption, and PV shading looks like a cloudy day. The recall of 0.52 mostly comes from these three long-lasting types. Details: [`reports/EnergyMind_model_benchmark.pdf`](reports/EnergyMind_model_benchmark.pdf).

## Explanations: how good are they?

- **SHAP** values are computed by XGBoost itself (TreeSHAP). They are identical to the `shap` package and add up to the model output.
- **LIME** is a second, model-independent opinion. On correctly detected test rows, 2.2 of SHAP's top-5 features are also in LIME's top 5, and both name the same first feature in 50 % of rows (100 % for voltage sensor fault, replay and discharge failure). Agreement is lower when several features carry the same information (the voltage features of an outage).
- Global SHAP per anomaly type and the full comparison: [`reports/explainability.md`](reports/explainability.md).

## Repository

```
detector_xgboost/      the detector: frozen model, feature code, SHAP/LIME, command line + Python API, sample data
  model/model.json         XGBoost weights (native format, no pickle)
  model/model_config.json  feature order, decision thresholds, labels, scores
  model/lime_background.csv.gz  training rows used by LIME
  energymind_features.py   features from raw readings (+ latest_features for streaming)
  explain.py               SHAP and LIME
  predict.py               Detector class and command line
  labels.json              category, agent action and human decision per anomaly type
reports/               benchmark, anomaly-injection method, data sources, leakage audit, explainability
training_code/         the code that produced the model (injection, features, training, audit, reports)
```

## Reports

| Report | Content |
|---|---|
| [`EnergyMind_model_benchmark.pdf`](reports/EnergyMind_model_benchmark.pdf) | six models compared, Phase 1 baselines, leakage audit, SHAP and LIME |
| [`EnergyMind_anomaly_injection_method.pdf`](reports/EnergyMind_anomaly_injection_method.pdf) | how the 14 anomaly types were injected (SimBench power flow), code, validation |
| [`EnergyMind_meters_and_data_sources.pdf`](reports/EnergyMind_meters_and_data_sources.pdf) | where each input comes from in a real deployment |
| [`leakage_audit.md`](reports/leakage_audit.md) | single features that separate an anomaly almost perfectly, expected or suspect |
| [`explainability.md`](reports/explainability.md) | global SHAP per type, LIME vs SHAP agreement |

## Training code

`training_code/` contains the full pipeline (`scripts/02_inject.py` … `14_leakage_audit.py`, `detector/`, `config/`). It needs the full simulated dataset (13 houses, one year), which is not in this repository because of its size. `training_code/scripts/leakage_lib.py` can audit any new model bundle:

```python
from leakage_lib import audit
table = audit(features_df, labels_df, feature_names, label_names)   # flags features with AUC >= 0.99, expected or SUSPECT
```
