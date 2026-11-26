"""Step 15 - Test data to try the single-house detector by hand, on the TEST blocks only (never used for training).

For each of the 4 test blocks (14 days each) it writes test_data/block_XX_<month>/ with
  house_01.csv ... house_13.csv   raw meter data (same columns as the detector input): 28 days of history + the 14 test days
  transformer.csv                 transformer data for the same period
  references/house_01.csv ...     the two reference inputs for the same period: second meter (all houses), roof sensor
                                  irradiance + temperature (PV houses)
  true_labels.csv                 true anomalies per house and 15-min step (column in_test = 1 for the 14 test days)
  true_events.csv                 the injected events of the block: house, type, start, end, Tunisian cause
and test_data/check_accuracy.py, which compares a result file with the true labels.
The data are copied from data/02_injected (nothing is regenerated).
Usage: python 15_make_test_data.py
"""
import shutil
from pathlib import Path

import pandas as pd

from common import D_INJ, HOUSES, V3
from features_lib import MIN_HISTORY_STEPS, TRAFO_COLS_SINGLE

OUT = V3 / "test_data"
D_REF = V3 / "data" / "05_references"
HIST = pd.Timedelta(minutes=15 * MIN_HISTORY_STEPS)
MONTH = {3: "feb", 9: "may", 16: "aug", 22: "nov"}

lab = pd.read_parquet(D_INJ / "labels.parquet", columns=["timestamp", "house_id", "anomalies", "split", "block"])
tm = pd.read_parquet(D_INJ / "transformer_meter.parquet")
ev = pd.read_csv(D_INJ / "events.csv", parse_dates=["start", "end"])
meas = {h: pd.read_parquet(D_INJ / "measured" / f"{h}.parquet") for h in HOUSES}
if OUT.exists():
    shutil.rmtree(OUT)
summary = []
for b in sorted(lab.loc[lab.split == "test", "block"].unique()):
    t = lab[(lab.split == "test") & (lab.block == b)]
    t0, t1 = t.timestamp.min(), t.timestamp.max()
    a = max(t0 - HIST, tm.timestamp.min())
    d = OUT / f"block_{b:02d}_{MONTH.get(b, 'test')}"
    d.mkdir(parents=True)
    win = lambda df: df[(df.timestamp >= a) & (df.timestamp <= t1)]        # noqa: E731
    for h in HOUSES:
        win(meas[h]).to_csv(d / f"{h}.csv", index=False)
    win(tm)[["timestamp"] + TRAFO_COLS_SINGLE].to_csv(d / "transformer.csv", index=False)
    (d / "references").mkdir()
    for h in HOUSES:
        win(pd.read_csv(D_REF / f"{h}.csv", parse_dates=["timestamp"])).to_csv(d / "references" / f"{h}.csv", index=False)
    L = win(lab).copy()
    L["anomalies"] = L.anomalies.replace("", "normal").fillna("normal")
    L["in_test"] = ((L.timestamp >= t0) & (L.timestamp <= t1)).astype(int)
    L[["timestamp", "house_id", "anomalies", "in_test"]].to_csv(d / "true_labels.csv", index=False)
    E = ev[(ev.split == "test") & (ev.block == b)][["event_id", "label", "category", "houses", "start", "end", "duration_h",
                                                   "params", "cause_tunisia"]].sort_values("start")
    E.to_csv(d / "true_events.csv", index=False)
    summary.append({"folder": d.name, "history_from": a, "test_from": t0, "test_to": t1, "events": len(E)})
    print(d.name, "| test", t0.date(), "->", t1.date(), "|", len(E), "events")

(OUT / "check_accuracy.py").write_text('''"""Compare a detector result with the true labels of one test block.
Usage: python check_accuracy.py <result.csv> <block folder> [house_id]
"""
import sys
import pandas as pd

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
res, folder = sys.argv[1], sys.argv[2]
p = pd.read_csv(res)
t = pd.read_csv(f"{folder}/true_labels.csv")
house = sys.argv[3] if len(sys.argv) > 3 else p.house_id.iloc[0]
t = t[t.house_id == house]
d = p[["timestamp", "anomalies"]].merge(t.rename(columns={"anomalies": "truth"}), on="timestamp")
d = d[d.in_test == 1]
print(f"house {house}: {len(d)} test steps (15 min), {int((d.truth != 'normal').sum())} truly anomalous")
print(f"exact accuracy (right type or right 'normal'): {100 * (d.anomalies == d.truth).mean():.1f} %")
print(f"detection accuracy (anomaly or not)          : {100 * ((d.anomalies != 'normal') == (d.truth != 'normal')).mean():.1f} %")
tp = ((d.anomalies != "normal") & (d.truth != "normal")).sum()
print(f"detection precision {tp / max((d.anomalies != 'normal').sum(), 1):.3f} | recall {tp / max((d.truth != 'normal').sum(), 1):.3f}")
print()
print(pd.crosstab(d.truth, d.anomalies, rownames=["TRUE"], colnames=["PREDICTED"]))
''', encoding="utf-8")
pd.DataFrame(summary).to_csv(OUT / "blocks.csv", index=False)
