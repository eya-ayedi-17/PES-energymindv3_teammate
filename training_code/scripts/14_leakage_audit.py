"""Step 14 - Leakage audit of the single-house feature set (and of the removed simulator-truth features).

For every anomaly type, every feature is tested ALONE against the normal rows (ROC-AUC, test split).
AUC >= 0.99 is flagged: 'expected' when the anomaly is defined by that signal, 'SUSPECT' otherwise.
The features removed for leaking simulator truth (features_lib.SIMULATOR_TRUTH_FEATURES) are audited too,
to document why they were removed.
Output: reports/leakage_audit.csv, reports/leakage_audit.md
Usage : python 14_leakage_audit.py
"""
import importlib
import warnings

import pandas as pd

from common import D_INJ, HOUSES, LABELS, REPORTS
from features_lib import SIMULATOR_TRUTH_FEATURES, build_single_house_features, house_features
from leakage_lib import audit

warnings.filterwarnings("ignore")
m05b = importlib.import_module("05b_single_house_model")
feats, _, _, te = m05b.build_data()

# 1. current single-house features (what the models use)
A = audit(te, te, feats, LABELS)
A.insert(0, "feature_set", "single_house (used)")

# 2. the removed features, rebuilt on the same test rows, to show the leak
tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
lab = pd.read_parquet(D_INJ / "labels.parquet")
old = pd.concat([house_features(h, pd.read_parquet(D_INJ / "measured" / f"{h}.parquet")) for h in HOUSES], ignore_index=True)
old = pd.concat([old, lab[LABELS + ["split"]]], axis=1)
old = old[old.split == "test"].reset_index(drop=True)
B = audit(old, old, [f for f in SIMULATOR_TRUTH_FEATURES if f in old.columns], LABELS)
B.insert(0, "feature_set", "removed (simulator truth)")

R = pd.concat([A, B], ignore_index=True)
R.to_csv(REPORTS / "leakage_audit.csv", index=False)

sus = A[A.verdict == "SUSPECT"]
lines = ["# Leakage audit (single-house features, test split)", "",
         "Each feature is tested alone against the normal rows for each anomaly type (ROC-AUC). "
         "AUC >= 0.99 is flagged: **expected** when the anomaly is defined by that signal "
         "(outage -> voltage 0, dropout -> missing data, frozen -> repeated values, replay -> identical to 24 h before, "
         "undervoltage -> low transformer voltage, overload -> transformer loading), **SUSPECT** otherwise.", "",
         f"## Features used by the models: {len(feats)}", "",
         f"- Flagged: {len(A)} (label, feature) pairs; expected: {int((A.verdict == 'expected').sum())}; "
         f"**suspect: {len(sus)}**.", ""]
if len(sus):
    lines += ["| Anomaly type | Feature | AUC |", "|---|---|---|"] + \
             [f"| {r.label} | `{r.feature}` | {r.auc:.4f} |" for r in sus.itertuples()] + [""]
lines += ["## Removed features (simulator truth)", "",
          "These features were used before and are now removed from the single-house models "
          "(`features_lib.SIMULATOR_TRUTH_FEATURES`). They compare the meter with the exact appliance sum, which no real "
          "house measures, or check S against P and Q, which the injection broke by adding independent noise to each channel.", "",
          "| Anomaly type | Feature | AUC alone |", "|---|---|---|"] + \
         [f"| {r.label} | `{r.feature}` | {r.auc:.4f} |" for r in B.sort_values(["label", "auc"], ascending=[True, False]).itertuples()] + \
         ["", "They are replaced by references built from the house's own history (same time of day over the previous "
          "14 days), from its share of the transformer's monitored group (`hist_*` features) and, in the final system, "
          "from two reference inputs: a second meter and a roof irradiance / temperature sensor (`ref_*` features).",
          "", "## Reference features (`ref_*`)", "",
          "The second meter reads the TRUE grid energy, so the ratio between the house meter and the second meter separates "
          "every fault of the house meter by design (bias, drift, under-reporting, replay, frozen). The roof sensor gives the "
          "production the panels should deliver, so the ratio measured / theoretical separates shading and charge failures. "
          "These pairs are marked `expected`: this is what the devices are for. What the audit checks is that no OTHER "
          "feature became a shortcut, and that bias and drift are not separated by one feature alone (they need the "
          "combination of the ratio level, its jump and its trend).", "",
          "| Anomaly type | Reference feature | AUC alone |", "|---|---|---|"] + \
         [f"| {r.label} | `{r.feature}` | {r.auc:.4f} |" for r in A[A.feature.str.startswith("ref_")].sort_values(["label", "auc"], ascending=[True, False]).itertuples()]
(REPORTS / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")
print(R.groupby(["feature_set", "verdict"]).size().to_string())
if len(sus):
    print(sus.to_string(index=False))
