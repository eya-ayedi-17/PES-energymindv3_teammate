# EnergyMind v3 – single-house anomaly detector (XGBoost)

Ready-to-use model for the rest of the EnergyMind platform. It reads **one house** (smart meter, and PV / battery / inverter if any), its **transformer**, and **two reference inputs** (a second meter and, for PV houses, a roof irradiance / temperature sensor). For every 15-minute step it returns whether there is an anomaly, which one, how sure it is, whether it is a real event or a measurement problem, a suggested action, and **why** (SHAP and LIME).

It was trained on our simulated Tunisian low-voltage network (SimBench LV1.101, 13 houses, 2018, 14 injected anomaly types). The features were audited for leakage: anything only the simulator could know (the exact appliance sum, S vs P and Q checks) was removed and replaced by the house's own history.

## Why two reference inputs

A meter bias, a slow drift and PV shading are invisible from one house and its transformer alone: a bias multiplies current, power and energy by the same factor (the meter stays consistent with itself) and is smaller than a house's normal day-to-day variation; shading looks like a cloudy day. With only the house and the transformer these three types were not detectable (validation PR-AUC 0.05 to 0.31) and were disabled. The **second meter** gives an independent measurement of the same energy; the **roof sensor** tells whether the sun was there.

| Input | Houses | Meaning |
|---|---|---|
| `check_meter_energy_kwh` | all | energy per 15 min seen by an independent class-1 meter |
| `irradiance_w_m2` | PV houses | irradiance in the plane of the panels (roof reference cell, about ±3 %) |
| `module_temp_c` | PV houses | module temperature (±0.5 °C) |

In our dataset these inputs are **simulated**: computed from the true values the simulator stored, then degraded to the accuracy above. The scores hold if real devices reach this accuracy and are independent of the house meter.

## Quick start

```
cd detector_xgboost
pip install -r requirements.txt
python predict.py --house-file sample_data/house_09.csv --transformer-file sample_data/transformer_house_09.csv --refs-file sample_data/references_house_09.csv --out result.csv
```

The terminal prints, for the first 20 alarm episodes: when, which anomaly, how sure, the SHAP and LIME reasons and the suggested action. Options: `--show N` (episodes printed), `--lime N` (episodes explained by LIME, 0 = faster), `--no-explain`.

`sample_data/` holds 45 days of a house without PV (`house_07`) and a house with PV + battery (`house_09`): house file, transformer file, reference file and the true injected anomalies (`*_true_labels.csv`, column `split` = train / val / test). The first 28 days are the history the features need.

## Use it from Python (platform integration)

```python
from predict import Detector                     # run from inside detector_xgboost/
det = Detector()

# a whole period (at least 28 days of 15-min rows)
out = det.detect(house_df, trafo_df, refs_df, house_id="house_09", explain=True, lime_episodes=5)

# streaming: the rolling buffers of raw readings (>= 2688 rows = 28 days) -> verdict of the newest step
out = det.detect_latest(house_buf, trafo_buf, refs_buf, house_id="house_09")
```

`detect_latest` runs exactly the same feature code as training (checked: streaming and batch give identical features).

## Inputs

| File | Columns |
|---|---|
| house | `timestamp`; `U_1..3`, `I_1..3`, `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `grid_available` (1/0). PV houses also: `P_PV_W`, `battery_soc_percent`, `battery_voltage_v`, `battery_current_a`, `battery_charge_power_w`, `battery_discharge_power_w`, `inverter_ac_output_w`, `inverter_status`, `transfer_switch_status`, `backup_mode_active`, `backup_load_demand_w`, `backup_load_served_w`, `unmet_backup_demand_w` |
| transformer | `timestamp`, `transformer_loading_percent`, `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`, `other_feeders_P_kw` |
| references | `timestamp`, `check_meter_energy_kwh`; PV houses also `irradiance_w_m2`, `module_temp_c` |

One row every 15 minutes, the same timestamps in the three files, at least 28 days. No appliance-level data is needed. The second meter must be a different device from the house meter (a bypass upstream of both would fool both); the roof sensor must sit in the plane of the panels, out of the shadows to detect.

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

Example (shading on a PV house): `solar production vs the theoretical production from the roof sensor (ref_pv_ratio = 0.459, SHAP +1.28)`.

## Scores (simulated test set, 4 blocks of 14 days never used for training)

| | XGBoost |
|---|---|
| Macro F1 over the 14 types | **0.955** |
| Detection precision / recall | 0.983 / 0.992 |
| Exact accuracy (right type, or right "normal") | 0.981 |
| Event recall / event precision | 0.977 / 0.815 |
| False alarms | 0.141 per house-day (about one every 7 days per house) |

| Anomaly type | F1 |
|---|---|
| localized grid outage, sustained undervoltage, transformer overload | 1.00 |
| meter frozen, meter dropout, voltage sensor fault, falsified replay | 1.00 |
| falsified under-report | 0.99 |
| battery charge failure | 0.97 |
| PV partial shading | 0.92 |
| meter current bias | 0.92 |
| meter noise | 0.90 |
| battery discharge failure | 0.85 |
| meter drift | 0.83 |

All 14 types are detected. Without the two reference inputs, XGBoost reaches a macro F1 of 0.730 and cannot detect shading, drift and bias (see section 7 of the benchmark).

**Read these with their limits.** The data and the two reference inputs are simulated. Some types are separable by one signal because they are defined by it (frozen = repeated values, replay = a copy of the day before); the leakage audit lists them. The alarm episodes of meter drift and current bias are fragmented (event precision 0.43 and 0.30 for the validation winner LightGBM), so expect more false episodes of these two types than of the others. Details: [`reports/EnergyMind_model_benchmark.pdf`](reports/EnergyMind_model_benchmark.pdf).

## Explanations: how good are they?

- **SHAP** values are computed by XGBoost itself (TreeSHAP). They are identical to the `shap` package and add up to the model output.
- **LIME** is a second, model-independent opinion. On correctly detected test rows, 2.0 of SHAP's top-5 features are also in LIME's top 5, and both name the same first feature in 56 % of rows (100 % for battery charge failure, replay, under-report, undervoltage, frozen meter and voltage sensor fault). Agreement is lower when several features carry the same information. **Trust SHAP first, use LIME as a second opinion.**
- Global SHAP per anomaly type and the full comparison: [`reports/explainability.md`](reports/explainability.md).

## Extra analyses: relations between the anomaly types

- **Spearman between the true labels**: the anomaly types are almost independent (90 of 91 pairs have |rho| < 0.05; 58 pairs never overlap). The only coupling is localized grid outage ↔ battery discharge failure (+0.14, injected on purpose). It comes from the injection plan, not from physics. Matrix and figure: [`reports/figures/anomaly_spearman.png`](reports/figures/anomaly_spearman.png), [`reports/anomaly_spearman_labels.csv`](reports/anomaly_spearman_labels.csv).
- **Confusion matrix of the XGBoost model** (test): the main confusions are current bias → drift (13.7 % of the bias steps), PV shading → voltage sensor fault (19.5 %) and battery discharge failure → grid outage (12.3 %). [`reports/anomaly_confusion_true_vs_predicted.csv`](reports/anomaly_confusion_true_vs_predicted.csv).
- **What was tried to exploit them** ([`confusion_exploitation.md`](reports/confusion_exploitation.md), [`label_structure_exploitation.md`](reports/label_structure_exploitation.md)): a specialist classifier for bias vs drift gives +0.0006 macro F1 (noise, not integrated); exclusion and coupling rules from the label correlations give −0.0014 and 0.0000 (not used). The one useful idea is a **family** alert, "meter gain error (bias or drift)": the precision for drift goes from 0.72 to 0.96. It is **not yet in the detector output**: a possible next step for the platform.

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
training_code/         the code that produced the model (injection, references, features, training, audit, reports)
```

## Reports

| Report | Content |
|---|---|
| [`EnergyMind_model_benchmark.pdf`](reports/EnergyMind_model_benchmark.pdf) | six models compared, effect of the reference inputs, Phase 1 baselines, leakage audit, SHAP and LIME |
| [`EnergyMind_anomaly_injection_method.pdf`](reports/EnergyMind_anomaly_injection_method.pdf) | how the 14 anomaly types were injected (SimBench power flow), code, validation |
| [`EnergyMind_meters_and_data_sources.pdf`](reports/EnergyMind_meters_and_data_sources.pdf) | where each input comes from in a real deployment, and the two added devices |
| [`leakage_audit.md`](reports/leakage_audit.md) | single features that separate an anomaly almost perfectly, expected or suspect |
| [`explainability.md`](reports/explainability.md) | global SHAP per type, LIME vs SHAP agreement |

## Training code

`training_code/` contains the full pipeline (`scripts/02_inject.py` … `18_recompute_event_metrics.py`, `detector/`, `config/`). It needs the full simulated dataset (13 houses, one year) and the reference inputs (`scripts/17_make_references.py`), which are not in this repository because of their size; they are shared as a separate zip (`energymind_v3_dataset.zip`). `training_code/scripts/leakage_lib.py` can audit any new model bundle:

```python
from leakage_lib import audit
table = audit(features_df, labels_df, feature_names, label_names)   # flags features with AUC >= 0.99, expected or SUSPECT
```
