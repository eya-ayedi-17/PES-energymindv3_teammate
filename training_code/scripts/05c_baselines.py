"""Step 5c - Phase 1 baselines on the v3 single-house data: anomaly or not (detection only, no anomaly type).

The Phase 1 prototype used a statistical baseline, Isolation Forest and One-Class SVM. They are re-implemented here
on the SAME single-house features and evaluated with the SAME metrics as the supervised models, so the gain of the
supervised models is measured, not assumed. (Same model families as Phase 1, not the Phase 1 code itself.)

* statistical : robust z-score of every feature vs the NORMAL training rows (median / MAD); score = mean of the 5
                largest |z|
* isolation_forest : sklearn IsolationForest fitted on normal training rows
* one_class_svm    : sklearn OneClassSVM (RBF) fitted on a sample of normal training rows
The alarm threshold is tuned on validation (best detection F1); the test set is used once.
Output: models/baselines/results.json, reports/baselines.csv
Usage : python 05c_baselines.py
"""
import importlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM

from common import LABELS, MODELS, REPORTS, save_json
from ml_common import Prep, event_metrics

warnings.filterwarnings("ignore")
OUT = MODELS / "baselines"
OUT.mkdir(parents=True, exist_ok=True)
m05b = importlib.import_module("05b_single_house_model")


def tune(score, y):
    best = (-1, None)
    for q in np.linspace(50, 99.8, 120):
        t = np.nanpercentile(score, q)
        f = f1_score(y, score >= t, zero_division=0)
        if f > best[0]:
            best = (f, t)
    return best[1]


def evaluate(name, s_va, s_te, y_va, te_ev, fit_s):
    th = tune(s_va, y_va)
    yh = (s_te >= th).astype(np.int8)
    y = (te_ev[LABELS].to_numpy().sum(1) > 0).astype(int)
    em, _ = event_metrics(te_ev, np.repeat(yh[:, None], len(LABELS), axis=1))   # any alarm counts for any type
    r = {"detect_precision": precision_score(y, yh, zero_division=0), "detect_recall": recall_score(y, yh),
         "detect_f1": f1_score(y, yh), "event_recall_macro": em["event_recall_macro"],
         "event_precision_detection": em["event_precision_detection"],
         "false_alarms_per_house_day": em["false_alarms_per_house_day"], "fit_seconds": fit_s, "threshold": float(th)}
    print(f"{name:18s} detect P {r['detect_precision']:.3f} R {r['detect_recall']:.3f} F1 {r['detect_f1']:.3f} | "
          f"event recall {r['event_recall_macro']:.3f} | FA/day {r['false_alarms_per_house_day']:.3f}")
    return r


def main():
    feats, train, va, te = m05b.build_data()
    norm = train[train.is_anomaly == 0]
    prep = Prep("scale")
    prep.fit(norm[feats].to_numpy(np.float32))
    prep.keep = np.zeros(len(feats), bool)               # no missing-flags: unsupervised models see values only
    Xn = prep.transform(norm[feats].to_numpy(np.float32))
    Xva, Xte = prep.transform(va[feats].to_numpy(np.float32)), prep.transform(te[feats].to_numpy(np.float32))
    y_va = (va[LABELS].to_numpy().sum(1) > 0).astype(int)
    te_ev = te[["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS].reset_index(drop=True)
    res = {}

    t0 = time.time()                                      # statistical: robust z (data already robust-scaled)
    med = np.median(Xn, 0); mad = np.median(np.abs(Xn - med), 0) * 1.4826 + 1e-6
    z = lambda X: np.sort(np.abs((X - med) / mad), axis=1)[:, -5:].mean(1)       # noqa: E731
    res["statistical"] = evaluate("statistical", z(Xva), z(Xte), y_va, te_ev, time.time() - t0)

    rng = np.random.default_rng(0)
    t0 = time.time()
    iso = IsolationForest(n_estimators=300, max_samples=512, random_state=0, n_jobs=4)
    iso.fit(Xn[rng.choice(len(Xn), min(60000, len(Xn)), replace=False)])
    res["isolation_forest"] = evaluate("isolation_forest", -iso.score_samples(Xva), -iso.score_samples(Xte), y_va, te_ev,
                                       time.time() - t0)

    t0 = time.time()
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc.fit(Xn[rng.choice(len(Xn), 6000, replace=False)])
    res["one_class_svm"] = evaluate("one_class_svm", -oc.score_samples(Xva), -oc.score_samples(Xte), y_va, te_ev,
                                    time.time() - t0)

    # supervised single-house models, same detection metrics (from 05b)
    for a in ["xgboost", "lightgbm", "random_forest", "hist_gb", "mlp", "extra_trees"]:
        p = m05b.OUT / f"{a}_results.json"
        if p.exists():
            o = json.loads(p.read_text())["overall"]
            res[f"supervised_{a}"] = {k: o.get(k) for k in ["detect_precision", "detect_recall", "detect_f1", "event_recall_macro",
                                                             "event_precision_detection", "false_alarms_per_house_day", "fit_seconds"]}
    save_json(res, OUT / "results.json")
    pd.DataFrame(res).T.round(4).to_csv(REPORTS / "baselines.csv")
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    main()
