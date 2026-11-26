"""Step 18 - Recompute the event-level metrics of every saved model from its saved test probabilities.
(event_metrics counted overlapping events of other types as events of the type; fixed in ml_common.event_metrics.)
No retraining, no new data: reads models/<folder>/<model>_P_test.npy and <model>_results.json, rewrites the event fields.
Usage: python 18_recompute_event_metrics.py
"""
import json
import warnings

import numpy as np

from common import LABELS, MODELS, save_json
from ml_common import decide, event_metrics, load

warnings.filterwarnings("ignore")
te = load("test", ["house_id", "timestamp", "event_id_1", "event_id_2", "n_labels"] + LABELS).reset_index(drop=True)
for folder in ("single", "single_house", "single_house_refs"):
    for rp in sorted((MODELS / folder).glob("*_results.json")):
        name = rp.name.replace("_results.json", "")
        pp = MODELS / folder / f"{name}_P_test.npy"
        if not pp.exists():
            continue
        r = json.loads(rp.read_text())
        em, _ = event_metrics(te, decide(np.load(pp), np.array(r["thresholds"])))
        r["overall"].update({k: v for k, v in em.items() if k != "per_label"})
        r["event_per_label"] = em["per_label"]
        save_json(r, rp)
        print(f"{folder}/{name}: event recall {em['event_recall_macro']:.3f}, precision {em['event_precision_macro']:.3f}, "
              f"events {sum(e['events'] for e in em['per_label'])}")
