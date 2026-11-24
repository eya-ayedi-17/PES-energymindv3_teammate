"""Step 9 - Export the SINGLE-HOUSE models as stand-alone packages (frozen weights + code) to share.

For each model (xgboost, lightgbm) it writes export/energymind_<model>_single_house/ and a .zip of it:
  model/model.json | model/model.txt   frozen weights in the library's own format (no pickle, no project code)
  model/model_config.json             feature order, labels, thresholds, disabled labels, versions, test scores
  model/lime_background.csv.gz        training rows sample used by LIME
  labels.json                         category, agent action and human decision per anomaly type
  energymind_features.py              feature engineering on raw readings (copied from features_lib.py)
  explain.py                          SHAP and LIME explanations (copied from explain_lib.py)
  predict.py                          command line + Python API (detect, detect_latest)
  requirements.txt, README.md
  sample_data/                        45 days of one grid-only house and one PV house + transformer + true labels
The export checks that the frozen model gives the same probabilities as the trained model.
Usage: python 09_export_packages.py [xgboost] [lightgbm]
"""
import inspect
import json
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features_lib as FL
from common import ACTIONS, CATEGORY_TEXT, D_INJ, LABEL_INFO, LABELS, METER_CH, MODELS, U_BASE, V3

warnings.filterwarnings("ignore")
EXPORT = V3 / "export"
SAMPLE = [("house_07", "2018-10-03", "2018-11-17"), ("house_09", "2018-07-03", "2018-08-17")]   # end in test blocks
HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ stand-alone feature module
def features_module() -> str:
    head = f'''"""EnergyMind - single-house feature engineering on raw 15-min readings (stand-alone copy of features_lib.py).
Input : one house (meter, PV/battery/inverter columns if any) + transformer (timestamp + TRAFO_COLS_SINGLE),
        same timestamps, one row every 15 minutes.
Output: one row per time step with the engineered features (float32).
Streaming: latest_features(buffer_house, buffer_trafo) gives the features of the newest rows; the buffer needs
MIN_HISTORY_STEPS rows. It runs the same code as training (no train/serve skew).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN windows (night, missing data) are expected
U_BASE = {U_BASE!r}
METER_CH = {METER_CH!r}
TYPE_A = []          # a house is PV + battery when it has a P_PV_W column
PV_KWP = {FL.PV_KWP!r}
INV_STATUS = {FL.INV_STATUS!r}
TRAFO_COLS_SINGLE = {FL.TRAFO_COLS_SINGLE!r}
HIST_DAYS = {FL.HIST_DAYS!r}
MIN_HISTORY_STEPS = {FL.MIN_HISTORY_STEPS!r}
RUN_CAP = {FL.RUN_CAP!r}
SIMULATOR_TRUTH_FEATURES = {FL.SIMULATOR_TRUTH_FEATURES!r}

'''
    body = "\n\n".join(inspect.getsource(f) for f in (FL._run_length, FL._roll, FL._same_tod_median, FL.house_features,
                                                       FL.build_single_house_features, FL.latest_features))
    return head + body


PREDICT = r'''"""EnergyMind - anomaly detector for ONE house (@ALGO@ model, frozen weights), with SHAP / LIME explanations.

Command line:
    python predict.py --house-file my_house.csv --transformer-file my_trafo.csv [--out result.csv]
                      [--house-id house_07] [--pv-kwp 4.0] [--only-anomalies] [--explain] [--lime 20]
Python:
    from predict import Detector
    det = Detector()
    out = det.detect(house_df, trafo_df, house_id="house_07", explain=True)       # whole period
    out = det.detect_latest(house_buffer, trafo_buffer, house_id="house_07")       # streaming: newest step only

my_house.csv : timestamp + meter channels (U/I/P/Q/S/PF per phase, P/Q/S/PF_TOT), grid_available, and the
               PV/battery/inverter columns if the house has PV
my_trafo.csv : timestamp, transformer_loading_percent, transformer_P_kw, transformer_Q_kvar, transformer_U_pu,
               other_feeders_P_kw
Both files: one row every 15 minutes, the same timestamps, at least 28 days (MIN_HISTORY_STEPS).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energymind_features import MIN_HISTORY_STEPS, build_single_house_features, latest_features

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "model" / "model_config.json").read_text())
INFO = json.loads((HERE / "labels.json").read_text())
LABELS = CFG["labels"]
PRIORITY = {"battery_discharge_failure", "localized_grid_outage", "transformer_overload",
            "sustained_undervoltage", "falsified_replay"}


def load_model():
@LOADER@


class Detector:
    def __init__(self):
        self.model = load_model()
        self.features = CFG["features"]
        self.th = np.asarray(CFG["thresholds"])

    # full class probabilities (normal + 14 types), used by LIME too
    def predict_proba_full(self, X):
        X = np.asarray(X, np.float32)
@PREDICTOR@
        return P

    def _matrix(self, F):
        return F.reindex(columns=self.features).to_numpy(np.float32)   # missing PV columns -> NaN (grid-only house)

    def detect(self, house: pd.DataFrame, trafo: pd.DataFrame, house_id="house", pv_kwp=None,
               explain=False, lime_episodes=0) -> pd.DataFrame:
        F = build_single_house_features(house, trafo, house_id, pv_kwp=pv_kwp)
        return self._score(F, explain, lime_episodes)

    def detect_latest(self, house_buffer: pd.DataFrame, trafo_buffer: pd.DataFrame, house_id="house", pv_kwp=None,
                      n_last=1, explain=True) -> pd.DataFrame:
        """Streaming: give the last MIN_HISTORY_STEPS (or more) raw readings, get the verdict of the newest n_last steps."""
        F = latest_features(house_buffer, trafo_buffer, house_id, pv_kwp=pv_kwp, n_last=n_last)
        return self._score(F, explain, 0)

    def _score(self, F, explain, lime_episodes):
        X = self._matrix(F)
        P = self.predict_proba_full(X)[:, 1:]
        out = combine(F[["timestamp", "house_id"]].reset_index(drop=True), P, self.th)
        rows = np.where(out.anomalies.to_numpy() != "normal")[0]
        cls = np.array([LABELS.index(p) + 1 if p != "normal" else 0 for p in out.primary_anomaly])
        if explain:
            from explain import shap_contributions, shap_text, top_shap
            out["shap_top"] = ""
            if len(rows):
                C = shap_contributions(self.model, X[rows], CFG["n_trees_used"])
                out.loc[rows, "shap_top"] = [shap_text(top_shap(C[i, cls[r]], X[r], self.features)) for i, r in enumerate(rows)]
        if lime_episodes:
            from explain import LimeExplainer, lime_text
            bg = pd.read_csv(HERE / "model" / "lime_background.csv.gz").reindex(columns=self.features).to_numpy(np.float32)
            lx = LimeExplainer(bg, self.features, ["normal"] + LABELS)
            out["lime_top"] = ""
            starts = [r for r in rows if r == 0 or out.anomalies.iat[r - 1] != out.anomalies.iat[r]][:lime_episodes]
            for r in starts:
                out.loc[r, "lime_top"] = lime_text(lx.explain(X[r], self.predict_proba_full, int(cls[r])))
        return out


def combine(keys: pd.DataFrame, P: np.ndarray, th: np.ndarray) -> pd.DataFrame:
    on_all = P >= th[None, :]
    out = keys.copy()
    labs, prim, conf, cat, act, hum = [], [], [], [], [], []
    for i in range(len(P)):
        on = np.where(on_all[i])[0]
        on = on[np.argsort(-P[i, on])][:2]
        if len(on) == 0:
            labs.append("normal"); prim.append("normal"); conf.append(float(1 - P[i].max())); cat.append("normal")
            act.append("none"); hum.append("")
            continue
        names = [LABELS[j] for j in on]
        labs.append("|".join(names)); prim.append(names[0]); conf.append(float(P[i, on[0]]))
        cat.append(" + ".join(sorted({INFO[n]["category_text"] for n in names})))
        if len(names) == 2 or P[i, on[0]] < 0.7:
            act.append("UNCLEAR CASE (two anomalies or low confidence): flag, log, notify -> human decides. "
                       + " / ".join(f"{n}: {INFO[n]['agent_action']}" for n in names))
        else:
            act.append(("[HIGH] " if names[0] in PRIORITY else "") + INFO[names[0]]["agent_action"])
        hum.append(" / ".join(INFO[n]["human_decision"] for n in names))
    out["anomalies"], out["n_anomalies"] = labs, on_all.sum(1).clip(max=2)
    out["primary_anomaly"], out["confidence"], out["category"] = prim, np.round(conf, 3), cat
    out["agent_action"], out["needs_human"] = act, hum
    for j, l in enumerate(LABELS):
        out[f"p_{l}"] = np.round(P[:, j], 4)
    return out


def read(p):
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house-file", required=True)
    ap.add_argument("--transformer-file", required=True)
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--house-id", default=None, help="name used in the output (default: file name)")
    ap.add_argument("--pv-kwp", type=float, default=None, help="PV size; estimated from the data if not given")
    ap.add_argument("--only-anomalies", action="store_true")
    ap.add_argument("--explain", action="store_true", help="SHAP reasons for every anomalous row (shap_top)")
    ap.add_argument("--lime", type=int, default=0, metavar="N", help="LIME reasons for the first N alarm episodes (lime_top)")
    a = ap.parse_args()
    m, tm = read(a.house_file), read(a.transformer_file)
    hid = a.house_id or Path(a.house_file).stem
    if len(m) < MIN_HISTORY_STEPS:
        print(f"warning: {len(m)} rows < {MIN_HISTORY_STEPS} ({MIN_HISTORY_STEPS // 96} days): early rows have incomplete features")
    res = Detector().detect(m, tm, hid, a.pv_kwp, explain=a.explain, lime_episodes=a.lime)
    if a.only_anomalies:
        res = res[res.anomalies != "normal"]
    res.to_csv(a.out, index=False)
    n_an = int((res.anomalies != "normal").sum())
    print(f"model: {CFG['model']} | house: {hid} | {len(res)} rows written to {a.out}, {n_an} anomalous")
    if CFG.get("disabled_labels"):
        print("not detected in single-house mode:", ", ".join(CFG["disabled_labels"]))
    if n_an:
        print(res[res.anomalies != "normal"].groupby("anomalies").size().rename("steps (15 min)").to_string())


if __name__ == "__main__":
    main()
'''

LOADERS = {
    "xgboost": ('''    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(HERE / "model" / "model.json"))
    return b''',
                '''        import xgboost as xgb
        P = self.model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, CFG["n_trees_used"]))'''),
    "lightgbm": ('''    import lightgbm as lgb
    return lgb.Booster(model_file=str(HERE / "model" / "model.txt"))''',
                 '''        P = self.model.predict(X)'''),
}


def export(algo):
    import importlib.metadata as md
    b = joblib.load(MODELS / "single_house" / f"{algo}.joblib")
    res = json.loads((MODELS / "single_house" / f"{algo}_results.json").read_text())
    mdl = b["model"]
    out = EXPORT / f"energymind_{algo}_single_house"
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "sample_data").mkdir()

    # 1. frozen weights + config + LIME background
    if algo == "xgboost":
        mdl.get_booster().save_model(str(out / "model" / "model.json"))
        n_used = int(mdl.best_iteration) + 1
        weights = "model/model.json"
    else:
        mdl.booster_.save_model(str(out / "model" / "model.txt"), num_iteration=mdl.best_iteration_)
        n_used = int(mdl.best_iteration_)
        weights = "model/model.txt"
    o = res["overall"]
    keys = ["macro_f1", "detect_precision", "detect_recall", "detect_f1", "event_recall_macro", "event_precision_macro",
            "false_alarms_per_house_day"]
    cfg = {"model": f"EnergyMind single-house {algo}", "framework": algo, "framework_version": md.version(algo),
           "weights_file": weights, "n_trees_used": n_used, "n_classes": len(LABELS) + 1,
           "class_order": ["normal"] + LABELS, "labels": LABELS, "features": b["features"],
           "thresholds": [float(x) for x in b["thresholds"]], "disabled_labels": b.get("disabled_labels", []),
           "min_history_steps": FL.MIN_HISTORY_STEPS,
           "inputs": "one house (meter + PV/battery if any) + transformer (" + ", ".join(FL.TRAFO_COLS_SINGLE) + "); "
                     f"15-min steps; at least {FL.MIN_HISTORY_STEPS // 96} days",
           "training": "simulated Tunisian LV network (SimBench LV1.101, 13 houses, 2018, 15 min, 14 anomaly types); "
                       "train/val/test = 14-day blocks; thresholds tuned on validation; leakage-audited features",
           "test_scores": {k: round(float(o[k]), 4) for k in keys}}
    (out / "model" / "model_config.json").write_text(json.dumps(cfg, indent=1))
    pd.read_parquet(MODELS / "single_house" / "lime_background.parquet").round(5).to_csv(
        out / "model" / "lime_background.csv.gz", index=False)
    lab = {l: {"category": LABEL_INFO[l][2], "category_text": CATEGORY_TEXT[LABEL_INFO[l][2]],
               "house_type": LABEL_INFO[l][3], "agent_action": ACTIONS[l][0], "human_decision": ACTIONS[l][1]} for l in LABELS}
    (out / "labels.json").write_text(json.dumps(lab, indent=1))

    # 2. code
    (out / "energymind_features.py").write_text(features_module(), encoding="utf-8")
    shutil.copy(HERE / "explain_lib.py", out / "explain.py")
    ld, pr = LOADERS[algo]
    (out / "predict.py").write_text(PREDICT.replace("@ALGO@", algo).replace("@LOADER@", ld).replace("@PREDICTOR@", pr),
                                    encoding="utf-8")
    req = ["numpy", "pandas", "pyarrow", algo, "lime"]
    (out / "requirements.txt").write_text("\n".join(f"{p}>={md.version(p)}" for p in req if _has(p)) + "\n")

    # 3. sample data: 45 days ending in a test block (the first 28 days are the history the features need)
    tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
    lab_all = pd.read_parquet(D_INJ / "labels.parquet")
    for h, a, z in SAMPLE:
        m = pd.read_parquet(D_INJ / "measured" / f"{h}.parquet")
        sel = (m.timestamp >= a) & (m.timestamp < z)
        m[sel].to_csv(out / "sample_data" / f"{h}.csv", index=False)
        tm[(tm.timestamp >= a) & (tm.timestamp < z)][["timestamp"] + FL.TRAFO_COLS_SINGLE].to_csv(
            out / "sample_data" / f"transformer_{h}.csv", index=False)
        tr = lab_all[(lab_all.house_id == h) & (lab_all.timestamp >= a) & (lab_all.timestamp < z)]
        tr[["timestamp", "anomalies", "split"]].replace({"anomalies": {"": "normal"}}).fillna("normal").to_csv(
            out / "sample_data" / f"{h}_true_labels.csv", index=False)

    # 4. README
    sc = cfg["test_scores"]
    rows = "\n".join(f"| {l} | {LABEL_INFO[l][3]} | {res['per_label'][l]['f1']:.2f} |" if l not in cfg["disabled_labels"]
                     else f"| {l} | {LABEL_INFO[l][3]} | not detected in this mode |" for l in LABELS)
    days = FL.MIN_HISTORY_STEPS // 96
    (out / "README.md").write_text(textwrap.dedent(f"""\
        # EnergyMind - single-house anomaly detector ({algo})

        Detects {len(LABELS) - len(cfg["disabled_labels"])} anomaly types on ONE house from its smart-meter data and the
        transformer measurements, and explains each detection with SHAP and LIME.
        The model weights are frozen in `{weights}` ({algo} {md.version(algo)} native format, no pickle).

        ## Install
        ```
        pip install -r requirements.txt
        ```

        ## Try it on the sample data
        ```
        python predict.py --house-file sample_data/house_07.csv --transformer-file sample_data/transformer_house_07.csv --out result_house_07.csv --explain --lime 10
        python predict.py --house-file sample_data/house_09.csv --transformer-file sample_data/transformer_house_09.csv --out result_house_09.csv --explain
        ```
        `house_07` has no PV; `house_09` has PV + battery. Each sample covers 45 days: the first {days} days are the history
        the features need, the last 14 days are a test block never used for training. Compare the `anomalies` column with
        `sample_data/<house>_true_labels.csv` (true injected anomalies; column `split` says train / val / test).

        ## Your own house
        ```
        python predict.py --house-file my_house.csv --transformer-file my_trafo.csv --out result.csv --only-anomalies --explain
        ```
        | File | What it needs |
        |---|---|
        | my_house.csv | `timestamp`; `U_1..3`, `I_1..3`, `P_1..3`, `Q_1..3`, `S_1..3`, `PF_1..3`; `P_TOT`, `Q_TOT`, `S_TOT`, `PF_TOT`; `grid_available` (1/0). PV houses also: `P_PV_W`, `battery_soc_percent`, `battery_voltage_v`, `battery_current_a`, `battery_charge_power_w`, `battery_discharge_power_w`, `inverter_ac_output_w`, `inverter_status`, `transfer_switch_status` (GRID/BATTERY), `backup_mode_active`, `backup_load_demand_w`, `backup_load_served_w`, `unmet_backup_demand_w` |
        | my_trafo.csv | `timestamp`, `transformer_loading_percent`, `transformer_P_kw`, `transformer_Q_kvar`, `transformer_U_pu`, `other_feeders_P_kw` |
        | Time | one row every 15 min, same timestamps in both files, at least {days} days ({FL.MIN_HISTORY_STEPS} rows) |

        Units: V, A, W, var, VA; power factor 0-1; transformer P in kW, Q in kvar, voltage in per-unit (1.0 = 230.9 V);
        `other_feeders_P_kw` = power of the transformer customers outside the monitored group.
        Appliance columns (`app_*_W`) are not needed: the house's normal level is learnt from its own history.
        Use `--pv-kwp` to give the PV size of a PV house (otherwise estimated from the data).

        ## Python API and streaming
        ```python
        from predict import Detector
        det = Detector()
        out = det.detect(house_df, trafo_df, house_id="house_07", explain=True, lime_episodes=5)   # a whole period
        out = det.detect_latest(house_buffer, trafo_buffer, house_id="house_07")                   # newest 15-min step
        ```
        `detect_latest` takes the rolling buffer of raw readings (at least {FL.MIN_HISTORY_STEPS} rows) and runs exactly the same
        feature code as training.

        ## Output (one row per 15 min)
        `anomalies` ("normal" or up to two: "a|b"), `primary_anomaly`, `confidence`, `category` (real event or sensor /
        data problem), `agent_action`, `needs_human`, `p_<type>` for the 14 types, and with `--explain` / `--lime`:
        `shap_top` (features that pushed the model towards the detected type, with their SHAP values) and `lime_top`
        (LIME's local rules and weights).

        ## Test scores (simulated test set)
        Macro F1 {sc['macro_f1']:.3f} | detection precision {sc['detect_precision']:.3f}, recall {sc['detect_recall']:.3f},
        F1 {sc['detect_f1']:.3f} | event recall {sc['event_recall_macro']:.3f}, event precision {sc['event_precision_macro']:.3f} |
        false alarms {sc['false_alarms_per_house_day']:.3f} per house-day

        | Anomaly | House type | F1 |
        |---|---|---|
        @@ROWS@@

        House type: A+B = all houses, A = PV + battery houses only.

        ## Limits
        - Trained on simulated data (SimBench LV1.101, 2018). Real data will be noisier.
        - Features that relied on simulator truth (exact appliance sum, S vs P and Q) were removed after a leakage audit.
        - Disabled in this mode (cannot be separated on validation): {", ".join(cfg["disabled_labels"]) or "none"}.

        ## Files
        | File | Content |
        |---|---|
        | `{weights}` | frozen model weights |
        | `model/model_config.json` | feature order, decision thresholds, labels, versions, test scores |
        | `model/lime_background.csv.gz` | sample of training rows used by LIME |
        | `labels.json` | category, suggested action and human decision for each anomaly type |
        | `energymind_features.py` | feature engineering on raw readings (identical to training) |
        | `explain.py` | SHAP and LIME explanations |
        | `predict.py` | command line and Python API |
        """).replace("@@ROWS@@", rows), encoding="utf-8")

    # 5. check: frozen model == trained model
    m = pd.read_parquet(D_INJ / "measured" / "house_09.parquet")
    F = FL.build_single_house_features(m, tm, "house_09").iloc[20000:23000]
    X = F.reindex(columns=b["features"]).to_numpy(np.float32)
    P_ref = mdl.predict_proba(X)
    if algo == "xgboost":
        import xgboost as xgb
        bb = xgb.Booster(); bb.load_model(str(out / "model" / "model.json"))
        P_new = bb.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, n_used))
    else:
        import lightgbm as lgb
        P_new = lgb.Booster(model_file=str(out / "model" / "model.txt")).predict(X)
    diff = float(np.abs(P_ref - P_new).max())
    assert diff < 1e-5, f"frozen {algo} model differs from trained model: {diff}"
    z = shutil.make_archive(str(out), "zip", root_dir=EXPORT, base_dir=out.name)
    print(f"{algo}: {out}  (max prob diff frozen vs trained = {diff:.1e})  -> {z}")


def _has(p):
    import importlib.metadata as md
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    for a in sys.argv[1:] or ("xgboost", "lightgbm"):     # native weight formats exist for these two
        export(a)
