# Leakage audit (single-house features, test split)

Each feature is tested alone against the normal rows for each anomaly type (ROC-AUC). AUC >= 0.99 is flagged: **expected** when the anomaly is defined by that signal (outage -> voltage 0, dropout -> missing data, frozen -> repeated values, replay -> identical to 24 h before, undervoltage -> low transformer voltage, overload -> transformer loading), **SUSPECT** otherwise.

## Features used by the models: 127

- Flagged: 68 (label, feature) pairs; expected: 65; **suspect: 0**.

## Removed features (simulator truth)

These features were used before and are now removed from the single-house models (`features_lib.SIMULATOR_TRUTH_FEATURES`). They compare the meter with the exact appliance sum, which no real house measures, or check S against P and Q, which the injection broke by adding independent noise to each channel.

| Anomaly type | Feature | AUC alone |
|---|---|---|
| falsified_under_report | `P_residual_W` | 1.0000 |
| falsified_under_report | `P_residual_rel` | 1.0000 |
| falsified_under_report | `P_logratio` | 1.0000 |
| falsified_under_report | `P_resid_rel_mean_8` | 0.9993 |
| falsified_under_report | `P_resid_rel_mean_96` | 0.9962 |
| falsified_under_report | `P_resid_rel_std_8` | 0.9939 |
| meter_frozen | `P_resid_rel_std_8` | 0.9986 |
| meter_frozen | `P_resid_step_8` | 0.9946 |
| meter_noise | `S_pq_residual_phase_max` | 1.0000 |
| meter_noise | `IU_S_residual_phase_max` | 1.0000 |
| meter_noise | `P_resid_rel_std_8` | 0.9968 |
| meter_noise | `IU_S_residual_rel` | 0.9962 |

They are replaced by references built from the house's own history (same time of day over the previous 14 days), from its share of the transformer's monitored group (`hist_*` features) and, in the final system, from two reference inputs: a second meter and a roof irradiance / temperature sensor (`ref_*` features).

## Reference features (`ref_*`)

The second meter reads the TRUE grid energy, so the ratio between the house meter and the second meter separates every fault of the house meter by design (bias, drift, under-reporting, replay, frozen). The roof sensor gives the production the panels should deliver, so the ratio measured / theoretical separates shading and charge failures. These pairs are marked `expected`: this is what the devices are for. What the audit checks is that no OTHER feature became a shortcut, and that bias and drift are not separated by one feature alone (they need the combination of the ratio level, its jump and its trend).

| Anomaly type | Reference feature | AUC alone |
|---|---|---|
| battery_charge_failure | `ref_pv_ratio_nc` | 0.9994 |
| battery_charge_failure | `ref_pv_ratio_mean_4` | 0.9984 |
| falsified_under_report | `ref_meter_lr_1h` | 0.9998 |
| falsified_under_report | `ref_meter_lr_24h` | 0.9959 |
| falsified_under_report | `ref_meter_lr_abs_24h` | 0.9948 |
| pv_partial_shading | `ref_pv_ratio_nc` | 0.9969 |
| pv_partial_shading | `ref_pv_ratio_mean_4` | 0.9932 |
| voltage_sensor_fault | `ref_pv_ratio_nc` | 0.9976 |
| voltage_sensor_fault | `ref_pv_ratio_mean_4` | 0.9951 |