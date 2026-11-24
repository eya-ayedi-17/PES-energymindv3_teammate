"""EnergyMind - anomaly detector for ONE house (xgboost model, frozen weights), with SHAP / LIME explanations.

Command line:
    python predict.py --house-file my_house.csv --transformer-file my_trafo.csv [--out result.csv]
                      [--house-id house_07] [--pv-kwp 4.0] [--only-anomalies] [--lime 10] [--show 20] [--no-explain]
    SHAP (every anomalous row) and LIME (first 10 episodes) are always computed and printed for the first 20
    alarm episodes; they are also written to the shap_top / lime_top columns of the output file.
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
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(HERE / "model" / "model.json"))
    return b


class Detector:
    def __init__(self):
        self.model = load_model()
        self.features = CFG["features"]
        self.th = np.asarray(CFG["thresholds"])

    # full class probabilities (normal + 14 types), used by LIME too
    def predict_proba_full(self, X):
        X = np.asarray(X, np.float32)
        import xgboost as xgb
        P = self.model.predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, CFG["n_trees_used"]))
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
    ap.add_argument("--lime", type=int, default=10, metavar="N",
                    help="LIME reasons for the first N alarm episodes (default 10; 0 = no LIME, faster)")
    ap.add_argument("--show", type=int, default=20, metavar="N",
                    help="print the SHAP / LIME explanation of the first N alarm episodes (default 20; 0 = none)")
    ap.add_argument("--no-explain", action="store_true", help="no SHAP column (faster)")
    a = ap.parse_args()
    m, tm = read(a.house_file), read(a.transformer_file)
    hid = a.house_id or Path(a.house_file).stem
    if len(m) < MIN_HISTORY_STEPS:
        print(f"warning: {len(m)} rows < {MIN_HISTORY_STEPS} ({MIN_HISTORY_STEPS // 96} days): early rows have incomplete features")
    res = Detector().detect(m, tm, hid, a.pv_kwp, explain=not a.no_explain, lime_episodes=a.lime)
    n_an = int((res.anomalies != "normal").sum())
    if a.show and n_an:
        show_explanations(res, a.show)
    if a.only_anomalies:
        res = res[res.anomalies != "normal"]
    res.to_csv(a.out, index=False)
    print(f"\nmodel: {CFG['model']} | house: {hid} | {len(res)} rows written to {a.out}, {n_an} anomalous")
    if CFG.get("disabled_labels"):
        print("not detected in single-house mode:", ", ".join(CFG["disabled_labels"]))
    if n_an:
        print(res[res.anomalies != "normal"].groupby("anomalies").size().rename("steps (15 min)").to_string())


def show_explanations(res, n):
    """print one block per alarm episode (consecutive steps with the same anomalies): when, what, how sure, why."""
    an = res.anomalies.to_numpy()
    starts = [i for i in range(len(res)) if an[i] != "normal" and (i == 0 or an[i - 1] != an[i])]
    print(f"\n=== {len(starts)} alarm episodes; explanation of the first {min(n, len(starts))} "
          f"(SHAP = exact contribution of each feature, LIME = local second opinion) ===")
    for k, i in enumerate(starts[:n], 1):
        j = i
        while j + 1 < len(res) and an[j + 1] == an[i]:
            j += 1
        r = res.iloc[i]
        print(f"\n[{k}] {r.timestamp} -> {res.timestamp.iat[j]}  ({j - i + 1} steps)  {r.anomalies}  "
              f"confidence {r.confidence:.2f}\n    category: {r.category}")
        if isinstance(r.get("shap_top"), str) and r.shap_top:
            for part in r.shap_top.split("; "):
                print(f"    SHAP  {part}")
        if isinstance(r.get("lime_top"), str) and r.lime_top:
            for part in r.lime_top.split("; "):
                print(f"    LIME  {part}")
        print(f"    action: {str(r.agent_action)[:160]}")


if __name__ == "__main__":
    main()
