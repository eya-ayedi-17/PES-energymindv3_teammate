"""Step 16 - Export the whole dataset in one clear zip (CSV) to share: injected data, clean data, labels per split,
anomaly catalogue. Copies data/02_injected and data/01_physical_baseline as they are (nothing is regenerated).

Output: export/energymind_v3_dataset/ and export/energymind_v3_dataset.zip
Usage : python 16_export_dataset.py
"""
import json
import shutil
import textwrap

import pandas as pd

from common import D_BASE, D_INJ, HOUSES, LABEL_INFO, LABELS, TYPE_A, V3

D_REF = V3 / "data" / "05_references"

OUT = V3 / "export" / "energymind_v3_dataset"
if OUT.exists():
    for c in OUT.iterdir():
        shutil.rmtree(c) if c.is_dir() else c.unlink()
for d in ("1_data_with_anomalies/houses", "2_labels", "3_clean_data_without_anomalies/houses", "4_reference_inputs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# 1. data with anomalies (model input)
for h in HOUSES:
    pd.read_parquet(D_INJ / "measured" / f"{h}.parquet").to_csv(OUT / "1_data_with_anomalies/houses" / f"{h}.csv", index=False)
pd.read_parquet(D_INJ / "transformer_meter.parquet").to_csv(OUT / "1_data_with_anomalies/transformer.csv", index=False)

# 2. labels: one file per split + the anomaly catalogue + the split calendar
lab = pd.read_parquet(D_INJ / "labels.parquet")
lab["anomalies"] = lab.anomalies.replace("", "normal").fillna("normal")
cols = ["timestamp", "house_id", "split", "block", "anomalies", "n_labels"] + LABELS + ["event_id_1", "event_id_2"]
names = {"train": "labels_train.csv", "val": "labels_validation.csv", "test": "labels_test.csv"}
for sp, fn in names.items():
    lab.loc[lab.split == sp, cols].to_csv(OUT / "2_labels" / fn, index=False)
ev = pd.read_csv(D_INJ / "events.csv")
ev[["event_id", "label", "category", "houses", "n_houses", "start", "end", "duration_h", "split", "block", "params",
    "cause_tunisia"]].sort_values("start").to_csv(OUT / "2_labels" / "anomaly_events.csv", index=False)
cal = (lab[lab.house_id == HOUSES[0]].groupby("block").agg(split=("split", "first"), start=("timestamp", "min"),
                                                           end=("timestamp", "max")).reset_index())
cal.to_csv(OUT / "2_labels" / "split_calendar.csv", index=False)
types = pd.DataFrame([{"anomaly": l, "category": LABEL_INFO[l][2], "house_type": LABEL_INFO[l][3],
                       "events": int((ev.label == l).sum()), "labelled_steps": int(lab[l].sum())} for l in LABELS])
types.to_csv(OUT / "2_labels" / "anomaly_types.csv", index=False)

# 3. clean data (same houses, same year, no anomaly)
for h in HOUSES:
    pd.read_parquet(D_BASE / "measured" / f"{h}.parquet").to_csv(OUT / "3_clean_data_without_anomalies/houses" / f"{h}.csv", index=False)
pd.read_parquet(D_BASE / "transformer_meter.parquet").to_csv(OUT / "3_clean_data_without_anomalies/transformer.csv", index=False)

# 4. reference inputs (second meter, roof sensor): copied as they are
for h in HOUSES:
    shutil.copy(D_REF / f"{h}.csv", OUT / "4_reference_inputs" / f"{h}.csv")
shutil.copy(D_REF / "references_config.json", OUT / "4_reference_inputs" / "references_config.json")

# README
n = {sp: int((lab.split == sp).sum()) for sp in names}
na = {sp: int(((lab.split == sp) & (lab.n_labels > 0)).sum()) for sp in names}
ne = ev.split.value_counts().to_dict()
blocks = {sp: ", ".join(str(b) for b in cal[cal.split == sp].block) for sp in names}
cal_rows = "\n".join(f"| {r.block} | {r.split} | {str(r.start)[:10]} | {str(r.end)[:10]} |" for r in cal.itertuples())
type_rows = "\n".join(f"| {r.anomaly} | {r.category} | {'PV + battery houses' if r.house_type == 'A' else 'all houses'} | "
                      f"{r.events} | {r.labelled_steps:,} |" for r in types.itertuples())
(OUT / "README.md").write_text(textwrap.dedent(f"""\
    # EnergyMind v3 dataset - 13 houses, one transformer, one year, 14 anomaly types

    Simulated Tunisian low-voltage network (SimBench LV1.101 rural grid, one 160 kVA 20/0.4 kV transformer), year 2018,
    one row every 15 minutes (35,040 rows per house). Houses {", ".join(TYPE_A)} have rooftop PV + a LiFePO4 battery;
    the other 9 houses are grid-only. 492 anomaly events of 14 types were injected; the method is described in
    `EnergyMind_anomaly_injection_method.pdf`.

    ## Folders

    | Folder / file | Content |
    |---|---|
    | `1_data_with_anomalies/houses/house_XX.csv` | what each smart meter (and inverter) reports, **with the injected anomalies**: this is the model input |
    | `1_data_with_anomalies/transformer.csv` | transformer measurements (loading %, P, Q, LV voltage, power of the other feeders) |
    | `2_labels/labels_train.csv`, `labels_validation.csv`, `labels_test.csv` | the true anomalies, one row per house and 15-min step |
    | `2_labels/anomaly_events.csv` | the 492 injected events: type, houses, start, end, parameters, Tunisian cause, split |
    | `2_labels/anomaly_types.csv` | the 14 types with their category and counts |
    | `2_labels/split_calendar.csv` | which 14-day block belongs to train, validation or test |
    | `3_clean_data_without_anomalies/` | the same houses and transformer **before** injection (reference, no anomaly) |
    | `4_reference_inputs/house_XX.csv` | the two reference inputs of the final detector: a **second meter** (all houses) and a **roof irradiance / temperature sensor** (PV houses); see below |

    ## Train / validation / test

    The year is cut into 26 blocks of 14 days, assigned **before** the injection; every event lies fully inside one
    block, so no event is shared between two splits.

    | Split | Blocks | House-steps | Anomalous steps | Events |
    |---|---|---|---|---|
    | train | {blocks['train']} | {n['train']:,} | {na['train']:,} | {ne.get('train', 0)} |
    | validation | {blocks['val']} | {n['val']:,} | {na['val']:,} | {ne.get('val', 0)} |
    | test | {blocks['test']} | {n['test']:,} | {na['test']:,} | {ne.get('test', 0)} |

    The house and transformer files are **not** cut by split: they cover the whole year, because the features need the
    previous 28 days of each house (its own usual level). Use the label files to know which rows are train,
    validation or test (column `split`), and compute features on the full year before selecting the rows.

    | Block | Split | From | To |
    |---|---|---|---|
    @@CAL@@

    ## Label files

    | Column | Meaning |
    |---|---|
    | `timestamp`, `house_id` | the 15-min step and the house |
    | `split`, `block` | train / val / test and the 14-day block |
    | `anomalies` | `normal`, or the anomaly types active at that step (`a|b` when two at once) |
    | `n_labels` | number of anomalies at that step (0, 1 or 2) |
    | one column per type (0/1) | the 14 anomaly types |
    | `event_id_1`, `event_id_2` | the event(s) of `anomaly_events.csv` behind the label (-1 = none) |

    A step is labelled only where the anomaly is **visible** in the data (for example a current bias on a PV house only
    while it imports from the grid). The full fault window of each event is in `anomaly_events.csv` (`start`, `end`).

    ## The 14 anomaly types

    | Anomaly | Category | Houses | Events | Labelled steps |
    |---|---|---|---|---|
    @@TYPES@@

    ## Reference inputs (`4_reference_inputs/`)

    | Column | Houses | Unit | Meaning |
    |---|---|---|---|
    | `check_meter_energy_kwh` | all 13 | kWh per 15 min | energy seen by a second, independent meter (class 1, own gain +-1 %, noise 0.3 %). It reads the true grid energy, so it is not fooled by the faults of the house meter |
    | `irradiance_w_m2` | PV houses | W/m2, in the plane of the panels | roof reference cell, about +-3 % |
    | `module_temp_c` | PV houses | deg C | sensor on the back of the module, +-0.5 deg C |

    They were computed from the true values the simulator stored (true grid power, PV available without shading), then degraded to the accuracy of the devices; `references_config.json` lists the noise levels and the seed. They are an assumption about two devices to install, not a measurement. The anomalies and labels were NOT regenerated to build them.

    ## House file columns

    | Columns | Meaning | Unit |
    |---|---|---|
    | `U_1..3`, `I_1..3` | voltage and current per phase | V, A |
    | `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3` | active, reactive, apparent power and power factor per phase | W, var, VA, 0-1 |
    | `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT` | totals of the house (grid side) | W, var, VA, 0-1 |
    | `app_<name>_W` | power of each appliance (simulation only: a real meter does not measure it; the final detector does not use it) | W |
    | `grid_available` | STEG grid present (1) or outage (0) | - |
    | PV houses: `P_PV_W`, `E_PV_kWh`, `battery_*`, `inverter_status`, `inverter_ac_output_w`, `backup_*`, `transfer_switch_status`, `unmet_backup_demand_w` | solar production, battery, hybrid inverter | W, kWh, %, V, A |

    Transformer: `transformer_loading_percent` (%), `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`
    (1.0 = 230.9 V), `other_feeders_P_kw` (customers outside the 13 houses).
    Missing values (empty cells) are part of the anomalies (meter dropout).

    ## Use with the detector

    The single-house XGBoost detector reads one house file, the transformer file and the reference inputs:
    ```
    python predict.py --house-file 1_data_with_anomalies/houses/house_04.csv --transformer-file 1_data_with_anomalies/transformer.csv --refs-file 4_reference_inputs/house_04.csv --out result.csv
    ```
    and its output can be compared with `2_labels/labels_test.csv` (rows of that house, `split = test`).
    """).replace("@@CAL@@", cal_rows).replace("@@TYPES@@", type_rows), encoding="utf-8")
shutil.copy(V3 / "reports" / "EnergyMind_anomaly_injection_method.pdf", OUT / "EnergyMind_anomaly_injection_method.pdf")
z = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
print(json.dumps({"rows": n, "anomalous": na, "events": ne}), "->", z)
