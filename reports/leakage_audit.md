# Leakage audit (single-house features, test split)

Each feature is tested alone against the normal rows for each anomaly type (ROC-AUC). AUC >= 0.99 is flagged: **expected** when the anomaly is defined by that signal (outage -> voltage 0, dropout -> missing data, frozen -> repeated values, replay -> identical to 24 h before, undervoltage -> low transformer voltage, overload -> transformer loading), **SUSPECT** otherwise.

## Features used by the models: 111

- Flagged: 59 (label, feature) pairs; expected: 58; **suspect: 0**.

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

They are replaced by references built from the house's own history (same time of day over the previous 14 days) and from its share of the transformer's monitored group (`hist_*` features).