"""Step 19 - Spearman correlation between the 14 anomaly types (and, next to it, the confusion matrix of the model).

1. SPEARMAN between the TRUE labels (all 455,520 house-steps): do two anomaly types tend to be active at the same step?
   Spearman on 0/1 columns equals the phi coefficient. It reflects the DESIGN of the injection (house-wide events overlap
   others, an outage is deliberately paired with a battery discharge failure, at most two anomalies at once, some types
   are never combined), not physics. Rare events give small coefficients even when they overlap often.
2. CONFUSION of the final model (test split): share of the steps of true type A for which the model predicts type B
   (diagonal = recall). Spearman between the model's predicted probabilities, or between a label and a probability, is
   deliberately NOT used: the softmax output squashes all the other probabilities together when the model is sure of one
   type (rho of about 0.95 for almost every pair), and for a rare label the coefficient is capped by its prevalence (about
   0.1 even for a perfect detection), so neither tells which anomalies are confused.
Output: reports/anomaly_spearman_labels.csv, reports/anomaly_confusion_true_vs_predicted.csv,
        reports/anomaly_spearman_top_pairs.csv, reports/figures/anomaly_spearman.png
Usage : python 19_anomaly_correlation.py [model]      (default xgboost)
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import D_INJ, FIGS, LABELS, MODELS, REPORTS
from ml_common import load
from report_style import MPL_FONT

warnings.filterwarnings("ignore")
ALGO = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
lab = pd.read_parquet(D_INJ / "labels.parquet", columns=LABELS)
C1 = lab.corr(method="spearman")
C1.round(4).to_csv(REPORTS / "anomaly_spearman_labels.csv")

Yt = load("test", LABELS)[LABELS].to_numpy(dtype=np.int64)
P = np.load(MODELS / "single_house_refs" / f"{ALGO}_P_test.npy")
th = np.array(json.loads((MODELS / "single_house_refs" / f"{ALGO}_results.json").read_text())["thresholds"])
Yh = (P >= th[None, :]).astype(np.int64)
CONF = pd.DataFrame((Yt.T @ Yh) / np.maximum(Yt.sum(0), 1)[:, None], index=LABELS, columns=LABELS)      # row = true, col = predicted
CONF.round(4).to_csv(REPORTS / "anomaly_confusion_true_vs_predicted.csv")

both = lab.to_numpy(dtype=np.int64).T @ lab.to_numpy(dtype=np.int64)
rows = []
for i, a in enumerate(LABELS):
    for j in range(i + 1, len(LABELS)):
        rows.append({"anomaly_a": a, "anomaly_b": LABELS[j], "spearman_true_labels": C1.iloc[i, j], "steps_together": int(both[i, j])})
T = pd.DataFrame(rows)
T.assign(k=T.spearman_true_labels.abs()).sort_values("k", ascending=False).drop(columns="k").round(4).to_csv(
    REPORTS / "anomaly_spearman_top_pairs.csv", index=False)

SHORT = {"localized_grid_outage": "grid outage", "sustained_undervoltage": "undervoltage", "transformer_overload": "overload",
         "battery_charge_failure": "batt. charge", "battery_discharge_failure": "batt. discharge", "pv_partial_shading": "PV shading",
         "meter_frozen": "frozen", "meter_dropout": "dropout", "meter_drift": "drift", "meter_noise": "noise",
         "meter_bias_current": "current bias", "voltage_sensor_fault": "voltage sensor", "falsified_under_report": "under-report",
         "falsified_replay": "replay"}
plt.rcParams.update({"font.family": MPL_FONT, "font.size": 7.5})
fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), dpi=170)
# left: Spearman between true labels (diverging blue <-> red, grey midpoint)
M = C1.to_numpy().copy(); np.fill_diagonal(M, np.nan)
v = max(0.15, float(np.nanmax(np.abs(M))))
im = axes[0].imshow(M, cmap=LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#c0392b"]), vmin=-v, vmax=v)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02, label="Spearman coefficient")
axes[0].set_title("Spearman between the true labels (455,520 house-steps)", loc="left", fontsize=9.5, color="#1F3864")
for i in range(14):
    for j in range(14):
        if i != j and abs(M[i, j]) >= 0.03:
            axes[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.8)
# right: confusion (sequential, one hue)
im2 = axes[1].imshow(CONF.to_numpy() * 100, cmap=LinearSegmentedColormap.from_list("seq", ["#ffffff", "#1c5cab"]), vmin=0, vmax=100)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02, label="% of the steps of the true type")
axes[1].set_title(f"{ALGO}: share of true type (rows) predicted as type (columns), test", loc="left", fontsize=9.5, color="#1F3864")
for i in range(14):
    for j in range(14):
        x = CONF.iloc[i, j] * 100
        if x >= 2:
            axes[1].text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=6, color="white" if x > 55 else "#0b0b0b")
for ax, yl, xl in ((axes[0], "anomaly type", "anomaly type"), (axes[1], "true type", "predicted type")):
    ax.set_xticks(range(14), [SHORT[l] for l in LABELS], rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(14), [SHORT[l] for l in LABELS], fontsize=7.5)
    ax.set_ylabel(yl); ax.set_xlabel(xl); ax.tick_params(length=0)
fig.suptitle("Left: Spearman between the 14 anomaly types (printed when |rho| >= 0.03; diagonal omitted). Right: confusion matrix of the model "
             "(printed when >= 2 %).", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIGS / "anomaly_spearman.png", facecolor="white", bbox_inches="tight")

off1 = C1.to_numpy()[~np.eye(14, dtype=bool)]
print(f"SPEARMAN of the true labels: {int((np.abs(off1) < 0.05).sum() / 2)} of 91 pairs have |rho| < 0.05 (max {off1.max():.3f}, min {off1.min():.3f}); "
      f"{int((T.steps_together == 0).sum())} pairs never overlap")
print("  strongest:", T.assign(k=T.spearman_true_labels.abs()).sort_values("k", ascending=False).head(3)[["anomaly_a", "anomaly_b", "spearman_true_labels", "steps_together"]].round(3).values.tolist())
print("  overlap most:", T.sort_values("steps_together", ascending=False).head(4)[["anomaly_a", "anomaly_b", "steps_together", "spearman_true_labels"]].round(3).values.tolist())
print("CONFUSION diagonal (recall %):", {SHORT[l]: round(float(CONF.iloc[i, i]) * 100) for i, l in enumerate(LABELS)})
o = CONF.to_numpy().copy(); np.fill_diagonal(o, -1)
idx = np.dstack(np.unravel_index(np.argsort(-o.ravel())[:8], o.shape))[0]
print("TOP CONFUSIONS (true -> predicted, %):", [(SHORT[LABELS[i]], SHORT[LABELS[j]], round(float(o[i, j]) * 100, 1)) for i, j in idx])
