"""EnergyMind v3 detector - one combined output per house and time step.

Models (--model):
    single_house (default)         FINAL: best single-house model, chosen on validation (models/single_house_refs).
                                   Input: one house + transformer + the two reference inputs.
    single_house_<algo>            same, algo = xgboost, lightgbm, random_forest, extra_trees, hist_gb, mlp
    single_house_norefs_<algo>     one house + transformer only (no reference inputs): needs no --refs-file; cannot
                                   detect pv_partial_shading, meter_drift, meter_bias_current
    <algo>                         13-house models (compare each house with its neighbours): needs all 13 houses

Library:
    from detector import Detector
    det = Detector("single_house_xgboost")
    out = det.predict({"house_07": house_df}, transformer_df, refs={"house_07": refs_df})

CLI:
    SINGLE-HOUSE mode (main mode):
    python detector.py --house-file my_house.csv --transformer-file my_trafo.csv --refs-file my_refs.csv [--model xgboost]
    Files, one row every 15 minutes, same timestamps, at least 28 days:
      my_house.csv   meter channels (+ PV / battery / inverter columns for a PV house)
      my_trafo.csv   timestamp, transformer_loading_percent, transformer_P_kw, transformer_Q_kvar, transformer_U_pu,
                     other_feeders_P_kw
      my_refs.csv    timestamp, check_meter_energy_kwh (second meter, kWh per 15 min) and, for a PV house,
                     irradiance_w_m2 + module_temp_c (roof sensor)
    SHAP and LIME explanations are printed for the first alarm episodes (--show N, --lime N, --no-explain).

    13-HOUSE mode (compares each house with its neighbours, no reference inputs):
    python detector.py --input <folder> --model xgboost [--out predictions.csv] [--house house_05]
    <folder> holds house_01..house_13 (.parquet or .csv, same columns as data/02_injected/measured)
    and transformer_meter (.parquet or .csv).

Output columns: timestamp, house_id, anomalies ("normal" or "a|b"), n_anomalies, primary_anomaly, confidence,
category (real event vs sensor / data problem), agent_action, needs_human, p_<label> x 14.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from common import ACTIONS, CATEGORY_TEXT, HOUSES, LABEL_INFO, LABELS, MODELS  # noqa: E402
from features_lib import build_features, build_single_house_features  # noqa: E402
from ml_common import Prep  # noqa: E402

# the single models were pickled while Prep lived in __main__ (05_benchmark script) -> make it findable there
import __main__  # noqa: E402
if not hasattr(__main__, "Prep"):
    __main__.Prep = Prep

PRIORITY = {"battery_discharge_failure": "HIGH", "localized_grid_outage": "HIGH", "transformer_overload": "HIGH",
            "sustained_undervoltage": "HIGH", "falsified_replay": "HIGH"}


class Detector:
    def __init__(self, model="best"):
        if model in ("best", "single_house"):          # default = best single-house model (chosen on validation)
            model = json.loads((MODELS / "single_house_refs" / "best_model.json").read_text())["system"]
        elif model == "best_13_houses":
            model = json.loads((MODELS / "best_model.json").read_text())["system"]
        self.name = model
        self.single_house = model.startswith("single_house")
        if self.single_house:                       # one house + transformer (+ reference inputs), no neighbours
            self.use_refs = "norefs" not in model
            self.folder = MODELS / ("single_house_refs" if self.use_refs else "single_house")
            algo = model.replace("single_house_", "").replace("norefs_", "")
            self.s = joblib.load(self.folder / f"{algo}.joblib")
            self.features = self.s["features"]
            self.th = self.s["thresholds"]
        elif model.startswith("per_anomaly"):
            self.pa = joblib.load(MODELS / "per_anomaly" / "per_anomaly_models.joblib")
            self.features = self.pa["features"]
            self.th = self.pa["th_stack"] if model.endswith("stacking") else self.pa["th_BR"]
        else:
            nm = model.replace("single_", "")
            self.s = joblib.load(MODELS / "single" / f"{nm}.joblib")
            self.features = self.s["features"]
            self.th = self.s["thresholds"]

    def proba(self, F: pd.DataFrame) -> np.ndarray:
        X = F.reindex(columns=self.features).to_numpy(np.float32)    # grid-only houses have no PV columns -> NaN
        if self.name.startswith("per_anomaly"):
            P = np.column_stack([self.pa["experts"][l].predict_proba(X)[:, 1] for l in LABELS])
            if self.name.endswith("stacking"):
                M = np.c_[P, F.house_type_A.to_numpy(np.float32)]
                P = np.column_stack([self.pa["metas"][l].predict_proba(M)[:, 1] for l in LABELS])
            return P
        A = self.s["prep"].transform(X)
        return self.s["model"].predict_proba(A)[:, 1:]

    def predict(self, meas: dict, tm: pd.DataFrame, features: pd.DataFrame | None = None,
                explain: bool = False, lime_episodes: int = 0, refs: dict | None = None) -> pd.DataFrame:
        if features is not None:
            F = features
        elif self.single_house:                     # each house on its own: meas may hold ONE house only
            if self.use_refs and not refs:
                raise ValueError("this model needs the reference inputs (second meter, roof sensor): give refs={house: DataFrame} "
                                 "(--refs-file), or use a single_house_norefs_<algo> model")
            F = pd.concat([build_single_house_features(m, tm, h, refs=(refs or {}).get(h) if self.use_refs else None)
                           for h, m in meas.items()], ignore_index=True)
        else:
            F = build_features(meas, tm)
        P = self.proba(F)
        out = self.combine(F[["timestamp", "house_id"]].reset_index(drop=True), P)
        if explain or lime_episodes:
            out = self.explain(F.reset_index(drop=True), out, shap=explain, lime_episodes=lime_episodes)
        return out

    def explain(self, F: pd.DataFrame, out: pd.DataFrame, shap: bool = True, lime_episodes: int = 0, k: int = 5) -> pd.DataFrame:
        """SHAP on every anomalous row (primary anomaly), LIME on the first row of the first `lime_episodes` alarm episodes."""
        from explain_lib import LimeExplainer, lime_text, shap_contributions, shap_text, top_shap
        X = self.s["prep"].transform(F.reindex(columns=self.features).to_numpy(np.float32))
        rows = np.where(out.anomalies.to_numpy() != "normal")[0]
        cls = np.array([LABELS.index(p) + 1 if p != "normal" else 0 for p in out.primary_anomaly])
        out["shap_top"], out["lime_top"] = "", ""
        if shap and len(rows):
            C = shap_contributions(self.s["model"], X[rows])
            out.loc[rows, "shap_top"] = [shap_text(top_shap(C[i, cls[r]], X[r], self.features, k)) for i, r in enumerate(rows)]
        if lime_episodes and len(rows):
            bg_path = self.folder / "lime_background.parquet"
            bg_df = pd.read_parquet(bg_path) if bg_path.exists() else pd.read_csv(self.folder / "lime_background.csv.gz")
            bg = bg_df.reindex(columns=self.features).to_numpy(np.float32)
            lx = LimeExplainer(self.s["prep"].transform(bg), self.features, ["normal"] + LABELS)
            starts = [r for r in rows if r == 0 or out.anomalies.iat[r - 1] != out.anomalies.iat[r]
                      or out.house_id.iat[r - 1] != out.house_id.iat[r]][:lime_episodes]
            for r in starts:
                out.loc[r, "lime_top"] = lime_text(lx.explain(X[r], self.s["model"].predict_proba, int(cls[r]), k))
        return out

    def combine(self, keys: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
        th = np.asarray(self.th)
        Yh = P >= th[None, :]
        # margin over threshold -> confidence in [0, 1]
        out = keys.copy()
        labs, prim, conf, cat, act, hum = [], [], [], [], [], []
        for i in range(len(P)):
            on = np.where(Yh[i])[0]
            on = on[np.argsort(-P[i, on])][:2]
            if len(on) == 0:
                labs.append("normal"); prim.append("normal"); conf.append(float(1 - P[i].max())); cat.append("normal")
                act.append("none"); hum.append("")
                continue
            names = [LABELS[j] for j in on]
            labs.append("|".join(names)); prim.append(names[0]); conf.append(float(P[i, on[0]]))
            cats = sorted({LABEL_INFO[n][2] for n in names})
            cat.append(" + ".join(CATEGORY_TEXT[c] for c in cats))
            if len(names) == 2 or P[i, on[0]] < 0.7:
                act.append("UNCLEAR CASE (two anomalies or low confidence): level-1 only (flag, log, notify) -> human decides. "
                           + " / ".join(f"{n}: {ACTIONS[n][0]}" for n in names))
            else:
                act.append(("[HIGH] " if names[0] in PRIORITY else "") + ACTIONS[names[0]][0])
            hum.append(" / ".join(ACTIONS[n][1] for n in names))
        out["anomalies"], out["n_anomalies"] = labs, Yh.sum(1).clip(max=2)
        out["primary_anomaly"], out["confidence"], out["category"] = prim, np.round(conf, 3), cat
        out["agent_action"], out["needs_human"] = act, hum
        for j, l in enumerate(LABELS):
            out[f"p_{l}"] = np.round(P[:, j], 4)
        return out


def read_folder(folder: Path):
    def rd(stem):
        for ext in (".parquet", ".csv"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])
        raise FileNotFoundError(folder / stem)
    return {h: rd(h) for h in HOUSES}, rd("transformer_meter")


def show_explanations(res, n):
    """one block per alarm episode: when, what, how sure, SHAP and LIME reasons, suggested action."""
    an, hid = res.anomalies.to_numpy(), res.house_id.to_numpy()
    starts = [i for i in range(len(res)) if an[i] != "normal" and (i == 0 or an[i - 1] != an[i] or hid[i - 1] != hid[i])]
    print(f"\n=== {len(starts)} alarm episodes; explanation of the first {min(n, len(starts))} "
          f"(SHAP = exact contribution of each feature, LIME = local second opinion) ===")
    for k, i in enumerate(starts[:n], 1):
        j = i
        while j + 1 < len(res) and an[j + 1] == an[i] and hid[j + 1] == hid[i]:
            j += 1
        r = res.iloc[i]
        print(f"\n[{k}] {r.house_id}  {r.timestamp} -> {res.timestamp.iat[j]}  ({j - i + 1} steps)  {r.anomalies}  "
              f"confidence {r.confidence:.2f}\n    category: {r.category}")
        for tag, col in (("SHAP", "shap_top"), ("LIME", "lime_top")):
            if isinstance(r.get(col), str) and r[col]:
                for part in r[col].split("; "):
                    print(f"    {tag}  {part}")
        print(f"    action: {str(r.agent_action)[:160]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="folder with the 13 houses + transformer_meter (13-house mode)")
    ap.add_argument("--house-file", help="SINGLE-HOUSE mode: one house file (.csv or .parquet)")
    ap.add_argument("--transformer-file", help="SINGLE-HOUSE mode: transformer file (timestamp, transformer_loading_percent, "
                                                "transformer_P_kw, transformer_Q_kvar, transformer_U_pu, other_feeders_P_kw)")
    ap.add_argument("--refs-file", help="SINGLE-HOUSE mode: reference inputs (timestamp, check_meter_energy_kwh and, for a "
                                         "PV house, irradiance_w_m2, module_temp_c)")
    ap.add_argument("--model", default="best")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--house", default=None)
    ap.add_argument("--only-anomalies", action="store_true")
    ap.add_argument("--no-explain", action="store_true", help="no SHAP column (faster)")
    ap.add_argument("--lime", type=int, default=10, metavar="N", help="LIME explanations for the first N alarm episodes (0 = none)")
    ap.add_argument("--show", type=int, default=20, metavar="N", help="print the SHAP / LIME explanation of the first N alarm episodes")
    a = ap.parse_args()

    def rd_file(p):
        p = Path(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])

    if a.house_file:
        if not a.transformer_file:
            ap.error("--house-file needs --transformer-file")
        hid = a.house or Path(a.house_file).stem
        meas = {hid: rd_file(a.house_file)}
        tm = rd_file(a.transformer_file)
        refs = {hid: rd_file(a.refs_file)} if a.refs_file else None
        if not a.model.startswith("single_house"):      # one-house input -> single-house model
            a.model = "single_house" if a.model == "best" else f"single_house_{a.model}"
        a.house = None
    elif a.input:
        meas, tm = read_folder(Path(a.input))
        refs = None
    else:
        ap.error("give --input <folder> (13 houses) or --house-file + --transformer-file (one house)")
    det = Detector(a.model)
    if det.single_house and det.s.get("disabled_labels"):
        print("single-house mode: not detected (validation PR-AUC below 0.5):", ", ".join(det.s["disabled_labels"]))
    if not det.single_house:
        a.no_explain, a.lime = True, 0
    res = det.predict(meas, tm, explain=not a.no_explain, lime_episodes=0 if a.no_explain else a.lime, refs=refs)
    if a.show and (res.anomalies != "normal").any() and "shap_top" in res:
        show_explanations(res, a.show)
    if a.house:
        res = res[res.house_id == a.house]
    if a.only_anomalies:
        res = res[res.anomalies != "normal"]
    res.to_csv(a.out, index=False)
    print(f"model: {det.name} -> {a.out}: {len(res)} rows, {int((res.anomalies != 'normal').sum())} anomalous")
    print(res[res.anomalies != "normal"].groupby(["house_id", "anomalies"]).size().rename("steps").to_string())
