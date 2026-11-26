"""Step 13 - Explainability of the selected single-house model (XGBoost): SHAP and LIME.

1. Check: native TreeSHAP contributions (XGBoost pred_contribs) == shap.TreeExplainer, and they add up to the
   model's output (log-odds) for every row.
2. Global SHAP: for each anomaly type, the features that push the model towards that type on the test rows that
   truly have it (mean SHAP value), plotted as small multiples.
3. LIME vs SHAP: on up to 8 test rows per type (correctly detected), how many of SHAP's top-5 features LIME also
   puts in its top 5 (explanation agreement).
Output: reports/explainability.md, reports/xai_shap_top_features.csv, reports/xai_lime_vs_shap.csv,
        reports/figures/xai_shap_top.png
Usage : python 13_explainability.py [model]      (default: the best single-house model)
"""
import importlib
import json
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, LABELS, MODELS, REPORTS
from explain_lib import LimeExplainer, describe, shap_contributions, top_shap
from report_style import MPL_FONT

warnings.filterwarnings("ignore")
m05b = importlib.import_module("05b_single_house_model")
SH = m05b.OUT                                            # models/single_house_refs (final) unless --no-refs
ALGO = ([x for x in sys.argv[1:] if not x.startswith("--")] or [json.loads((SH / "best_model.json").read_text())["model"]])[0]
b = joblib.load(SH / f"{ALGO}.joblib")
model, feats, th, disabled = b["model"], b["features"], np.asarray(b["thresholds"]), b.get("disabled_labels", [])
_, train, _, te = m05b.build_data()
X = b["prep"].transform(te[feats].to_numpy(np.float32))
Y = te[LABELS].to_numpy()
rng = np.random.default_rng(0)
lines = [f"# Explainability of the single-house {ALGO} model (SHAP and LIME)", ""]

# 1. check ---------------------------------------------------------------------------------------------------
idx = rng.choice(len(X), 300, replace=False)
C = shap_contributions(model, X[idx])
P = model.predict_proba(X[idx])
z = C.sum(-1); pz = np.exp(z - z.max(1, keepdims=True)); pz /= pz.sum(1, keepdims=True)
add_err = float(np.abs(pz - P).max())
lib_err = "not checked (only for XGBoost)"
try:
    import shap
    if ALGO == "xgboost":
        n = int(model.best_iteration) + 1
        sv = np.asarray(shap.TreeExplainer(model.get_booster()[:n]).shap_values(X[idx[:100]]))
        if sv.shape[-1] == C.shape[1]:
            sv = np.moveaxis(sv, -1, 1)
        lib_err = float(np.abs(sv - C[:100, :, :-1]).max())
except Exception as e:                                 # shap not installed or layout change
    lib_err = f"not checked ({type(e).__name__})"
lines += ["## 1. Are the SHAP values right?", "",
          f"- The SHAP contributions plus the bias add up to the model output: max probability difference {add_err:.1e} on 300 test rows.",
          f"- Native TreeSHAP vs the `shap` package (TreeExplainer): max difference "
          f"{lib_err if isinstance(lib_err, str) else f'{lib_err:.1e}'} on 100 rows.", ""]

# 2. global SHAP per anomaly type ----------------------------------------------------------------------------
rows, panels = [], []
for j, lab in enumerate(LABELS):
    if lab in disabled:
        continue
    r = np.where(Y[:, j] == 1)[0]
    if len(r) == 0:
        continue
    r = rng.choice(r, min(1500, len(r)), replace=False)
    c = shap_contributions(model, X[r])[:, j + 1, :-1]
    mean = np.nanmean(c, 0)
    order = np.argsort(-mean)[:8]
    for k, i in enumerate(order):
        rows.append({"anomaly": lab, "rank": k + 1, "feature": feats[i], "mean_shap": round(float(mean[i]), 4),
                     "meaning": describe(feats[i])})
    panels.append((lab, [(feats[i], mean[i]) for i in order[:5]]))
G = pd.DataFrame(rows)
G.to_csv(REPORTS / "xai_shap_top_features.csv", index=False)

plt.rcParams.update({"font.family": MPL_FONT, "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#e4e3df", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
nc = 3; nr = int(np.ceil(len(panels) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(11, 2.0 * nr), dpi=170)
for ax, (lab, items) in zip(axes.ravel(), panels):
    names = [f for f, _ in items][::-1]; vals = [v for _, v in items][::-1]
    ax.barh(range(len(vals)), vals, color="#256abf", height=0.6)
    ax.set_yticks(range(len(vals)), names, fontsize=6.5)
    ax.set_title(lab.replace("_", " "), fontsize=8, loc="left", color="#1F3864")
    ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
for ax in axes.ravel()[len(panels):]:
    ax.axis("off")
fig.supxlabel("mean SHAP value towards the anomaly type (log-odds), test rows that have this anomaly", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "xai_shap_top.png", facecolor="white"); plt.close(fig)

lines += ["## 2. What drives each anomaly type (global SHAP)", "",
          "Mean SHAP value of each feature towards the anomaly type, on the test rows that truly have it. "
          "Top 3 per type (full list: `xai_shap_top_features.csv`).", "",
          "| Anomaly type | Top features (meaning) |", "|---|---|"]
for lab, g in G.groupby("anomaly", sort=False):
    lines.append(f"| {lab} | " + "; ".join(f"`{r.feature}` ({r.meaning})" for r in g.head(3).itertuples()) + " |")
lines += ["", "![SHAP top features](figures/xai_shap_top.png)", ""]

# 3. LIME vs SHAP agreement ----------------------------------------------------------------------------------
bg = b["prep"].transform(train.sample(n=min(6000, len(train)), random_state=0)[feats].to_numpy(np.float32))
lx = LimeExplainer(bg, feats, ["normal"] + LABELS)
Pall = model.predict_proba(X)[:, 1:]
agree = []
for j, lab in enumerate(LABELS):
    if lab in disabled:
        continue
    ok = np.where((Y[:, j] == 1) & (Pall[:, j] >= th[j]))[0]
    for r in rng.choice(ok, min(8, len(ok)), replace=False) if len(ok) else []:
        # compare on the features both methods can see: LIME cannot perturb a missing value, SHAP can credit it
        fin = np.isfinite(X[r])
        c = shap_contributions(model, X[r:r + 1])[0, j + 1].copy()
        c[:-1][~fin] = 0
        s5 = [d["feature"] for d in top_shap(c, X[r], feats, 5)]
        l5 = [d["feature"] for d in lx.explain(X[r], model.predict_proba, j + 1, k=10) if d["weight"] > 0][:5]
        agree.append({"anomaly": lab, "row": int(r), "shap_top5": ", ".join(s5), "lime_top5": ", ".join(l5),
                      "overlap": len(set(s5) & set(l5)), "same_first": bool(s5 and l5 and s5[0] == l5[0])})
A = pd.DataFrame(agree)
A.to_csv(REPORTS / "xai_lime_vs_shap.csv", index=False)
s = A.groupby("anomaly").agg(rows=("row", "size"), mean_overlap=("overlap", "mean"), same_first=("same_first", "mean"))
lines += ["## 3. Does LIME agree with SHAP?", "",
          "On up to 8 correctly detected test rows per type: how many of SHAP's top-5 features (pushing towards the type) "
          "LIME also ranks in its top 5 positive features, and how often both name the same first feature. The comparison "
          "uses the features both methods can see: LIME cannot perturb a missing value (for example the voltages during a "
          "meter dropout), while SHAP can credit it, so missing features are left out of SHAP's list here. "
          "LIME is model-agnostic and approximate (local linear fit on perturbed samples); SHAP is exact for tree models. "
          "Strong agreement means the explanation does not depend on the method; weak agreement is expected when several "
          "features carry the same information (for example the different voltage features of an outage).", "",
          f"Overall: mean overlap **{A.overlap.mean():.1f} / 5**, same first feature in **{100 * A.same_first.mean():.0f} %** of rows.", "",
          "| Anomaly type | Rows | Mean overlap (of 5) | Same first feature |", "|---|---|---|---|"] + \
         [f"| {k} | {int(v.rows)} | {v.mean_overlap:.1f} | {100 * v.same_first:.0f} % |" for k, v in s.iterrows()] + \
         ["", "## How to get explanations for your own data", "",
          "```", "python detector/detector.py --house-file my_house.csv --transformer-file my_trafo.csv --explain --lime 20", "```",
          "`shap_top` (every anomalous row) and `lime_top` (first row of the first 20 alarm episodes) are added to the output."]
(REPORTS / "explainability.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:8]))
print(s.round(2).to_string())
