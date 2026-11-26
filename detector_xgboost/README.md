# EnergyMind - single-house anomaly detector (xgboost)

Detects 14 anomaly types on ONE house from its smart-meter data, the
transformer measurements and two reference inputs (a second meter and, for PV houses, a roof irradiance /
temperature sensor), and explains each detection with SHAP and LIME.
The model weights are frozen in `model/model.json` (xgboost 3.2.0 native format, no pickle).

## Install
```
pip install -r requirements.txt
```

## Try it on the sample data
```
python predict.py --house-file sample_data/house_07.csv --transformer-file sample_data/transformer_house_07.csv --refs-file sample_data/references_house_07.csv --out result_house_07.csv
python predict.py --house-file sample_data/house_09.csv --transformer-file sample_data/transformer_house_09.csv --refs-file sample_data/references_house_09.csv --out result_house_09.csv
```
`house_07` has no PV; `house_09` has PV + battery. Each sample covers 45 days: the first 28 days are the history
the features need, the last 14 days are a test block never used for training. Compare the `anomalies` column with
`sample_data/<house>_true_labels.csv` (true injected anomalies; column `split` says train / val / test).

## Your own house
```
python predict.py --house-file my_house.csv --transformer-file my_trafo.csv --refs-file my_refs.csv --out result.csv --only-anomalies
```
| File | What it needs |
|---|---|
| my_house.csv | `timestamp`; `U_1..3`, `I_1..3`, `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `grid_available` (1/0). PV houses also: `P_PV_W`, `battery_soc_percent`, `battery_voltage_v`, `battery_current_a`, `battery_charge_power_w`, `battery_discharge_power_w`, `inverter_ac_output_w`, `inverter_status`, `transfer_switch_status` (GRID/BATTERY), `backup_mode_active`, `backup_load_demand_w`, `backup_load_served_w`, `unmet_backup_demand_w` |
| my_trafo.csv | `timestamp`, `transformer_loading_percent`, `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`, `other_feeders_P_kw` |
| my_refs.csv | `timestamp`, `check_meter_energy_kwh` (second meter: kWh per 15 min, every house); PV houses also `irradiance_w_m2` (W/m2 in the plane of the panels) and `module_temp_c` (deg C), from a roof reference cell |
| Time | one row every 15 min, same timestamps in the three files, at least 28 days (2688 rows) |

Units: V, A, W, var, VA; power factor 0-1; transformer P in kW, Q in kvar, voltage in per-unit (1.0 = 230.9 V);
`other_feeders_P_kw` = power of the transformer customers outside the monitored group.
Appliance columns (`app_*_W`) are not needed. The second meter must be a different device from the house meter
(ideally its own current transformers, or placed where the customer cannot reach it); the roof sensor must sit
in the plane of the panels and out of the shadows to detect.
Use `--pv-kwp` to give the PV size of a PV house (otherwise estimated from the data).

## Python API and streaming
```python
from predict import Detector
det = Detector()
out = det.detect(house_df, trafo_df, refs_df, house_id="house_07", explain=True, lime_episodes=5)   # a whole period
out = det.detect_latest(house_buf, trafo_buf, refs_buf, house_id="house_07")                       # newest 15-min step
```
`detect_latest` takes the rolling buffer of raw readings (at least 2688 rows) and runs exactly the same
feature code as training.

## Output (one row per 15 min)
`anomalies` ("normal" or up to two: "a|b"), `primary_anomaly`, `confidence`, `category` (real event or sensor /
data problem), `agent_action`, `needs_human`, `p_<type>` for the 14 types, and the explanations:
`shap_top` (features that pushed the model towards the detected type, with their SHAP values) and `lime_top`
(LIME's local rules and weights). The terminal also prints, for the first 20 alarm episodes, when, what,
how sure, the SHAP and LIME reasons and the suggested action (`--show N` to change, `--lime 0` for speed).

## Test scores (simulated test set)
Macro F1 0.955 | detection precision 0.983, recall 0.992,
F1 0.988 | event recall 0.977, event precision 0.815 |
false alarms 0.141 per house-day

| Anomaly | House type | F1 |
|---|---|---|
| localized_grid_outage | A+B | 1.00 |
| sustained_undervoltage | A+B | 1.00 |
| transformer_overload | A+B | 1.00 |
| battery_charge_failure | A | 0.97 |
| battery_discharge_failure | A | 0.85 |
| pv_partial_shading | A | 0.92 |
| meter_frozen | A+B | 1.00 |
| meter_dropout | A+B | 1.00 |
| meter_drift | A+B | 0.83 |
| meter_noise | A+B | 0.90 |
| meter_bias_current | A+B | 0.92 |
| voltage_sensor_fault | A+B | 1.00 |
| falsified_under_report | A+B | 0.99 |
| falsified_replay | A+B | 1.00 |

House type: A+B = all houses, A = PV + battery houses only.

## Limits
- Trained on simulated data (SimBench LV1.101, 2018). Real data will be noisier.
- Features that relied on simulator truth (exact appliance sum, S vs P and Q) were removed after a leakage audit.
- The two reference inputs of the dataset are simulated: computed from true values and degraded to the assumed accuracy
  of the devices (second meter +-1 %, irradiance about +-3 %, temperature +-0.5 C). The scores hold if real devices
  reach this accuracy and are independent of the house meter (a bypass upstream of both meters would fool both).
- Disabled in this mode (cannot be separated on validation): none.

## Files
| File | Content |
|---|---|
| `model/model.json` | frozen model weights |
| `model/model_config.json` | feature order, decision thresholds, labels, versions, test scores |
| `model/lime_background.csv.gz` | sample of training rows used by LIME |
| `sample_data/references_<house>.csv` | the reference inputs of the two sample houses |
| `labels.json` | category, suggested action and human decision for each anomaly type |
| `energymind_features.py` | feature engineering on raw readings (identical to training) |
| `explain.py` | SHAP and LIME explanations |
| `predict.py` | command line and Python API |
