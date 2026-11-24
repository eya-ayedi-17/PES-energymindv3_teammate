"""EnergyMind - leakage audit: find single features that separate an anomaly type almost perfectly.

A feature that alone reaches ROC-AUC >= 0.99 for one anomaly type is either
  * expected by definition (outage -> voltage 0, dropout -> missing data, frozen -> repeated values, ...): fine, or
  * a leak: the simulator gave the model information a real system would not have. It must be removed or rebuilt.
Use it on every new model bundle before trusting its scores (Phase 1 principle: audit the benchmark first).

    from leakage_lib import audit
    table = audit(features_df, labels_df, feature_names, label_names)      # one row per (label, feature) flagged
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# anomaly type -> feature name prefixes that separate it BY DEFINITION of the anomaly (not a leak)
EXPECTED = {
    "localized_grid_outage": ("U_", "grid_available", "meter_", "P_TOT", "I_sum", "S_TOT", "Q_TOT", "PF_", "hist_P_",
                              "house_share", "rest_of_group", "backup_", "transfer_switch", "unmet_", "inverter_"),
    "meter_dropout": ("meter_missing", "meter_n_missing"),
    "meter_frozen": ("meter_repeat", "P_TOT_repeat", "U_repeat"),
    "falsified_replay": ("meter_lag96", "U_lag96", "P_TOT_lag96"),
    "sustained_undervoltage": ("transformer_U", "U_"),
    # an overloaded transformer lowers every voltage; the overload load is UNMETERED (illegal connections), so it
    # appears as transformer power that no meter explains (monitored-group power, energy-balance gap)
    "transformer_overload": ("transformer_", "trafo_", "U_", "group_P", "rest_of_group"),
}
MIN_COVERAGE = 0.5      # a feature defined for fewer positives than this cannot leak much: reported as 'low coverage'


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    r = pd.Series(x).rank().to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    a = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def audit(X: pd.DataFrame, Y: pd.DataFrame, features: list[str], labels: list[str], threshold: float = 0.99,
          min_positives: int = 20) -> pd.DataFrame:
    """X: feature table, Y: 0/1 label table (same rows). Each label is compared with the NORMAL rows only.
    Returns every (label, feature) with AUC >= threshold, marked 'expected' or 'SUSPECT'."""
    normal = Y[labels].to_numpy().sum(1) == 0
    out = []
    for lab in labels:
        pos = Y[lab].to_numpy() == 1
        if pos.sum() < min_positives:
            continue
        sel = pos | normal
        y = pos[sel]
        for f in features:
            x = X.loc[sel, f].to_numpy(float)
            ok = np.isfinite(x)
            if y[ok].sum() < min_positives or (~y[ok]).sum() < min_positives:
                continue
            a = _auc(x[ok], y[ok])
            if a >= threshold:
                cov = float(ok[y].mean())
                verdict = ("expected" if f.startswith(EXPECTED.get(lab, ())) else
                           "low coverage" if cov < MIN_COVERAGE else "SUSPECT")
                out.append({"label": lab, "feature": f, "auc": round(a, 4), "positives_covered": round(cov, 3),
                            "verdict": verdict})
    return pd.DataFrame(out, columns=["label", "feature", "auc", "positives_covered", "verdict"])
