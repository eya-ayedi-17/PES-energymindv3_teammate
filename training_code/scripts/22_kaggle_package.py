"""Step 22 - Package the dataset for Kaggle: files at the root of the zip, dataset card (README.md), dataset-metadata.json.

Reads export/energymind_v3_dataset/ (built by 16_export_dataset.py) and writes export/energymind_v3_kaggle.zip.
Upload: Kaggle > Datasets > New Dataset > drop the zip (the metadata file is only needed for the kaggle CLI:
`kaggle datasets create -p <folder>` / `kaggle datasets version -p <folder> -m "message"`).
Usage : python 22_kaggle_package.py
"""
import json
import zipfile

from common import V3

SRC = V3 / "export" / "energymind_v3_dataset"
OUT = V3 / "export" / "energymind_v3_kaggle.zip"
META = {
    "title": "EnergyMind Tunisia v3 - anomalies",
    "subtitle": "13 houses, 1 year, 15 min, 14 injected anomaly types with labels",
    "id": "azertyqsdfghqsdfgh/energymind-tunisia-v3",
    "licenses": [{"name": "ODbL-1.0"}],
    "keywords": ["energy", "smart meter", "anomaly detection", "time series", "tunisia", "photovoltaic", "electricity"],
    "description": (SRC / "README.md").read_text(encoding="utf-8"),
}
assert 6 <= len(META["title"]) <= 50 and 20 <= len(META["subtitle"]) <= 80, "Kaggle title / subtitle length"
(V3 / "export" / "dataset-metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
if OUT.exists():
    OUT.unlink()
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(SRC.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(SRC))
            n += 1
    z.writestr("dataset-metadata.json", json.dumps(META, indent=2))
    n += 1
print(n, "files,", round(OUT.stat().st_size / 1048576, 1), "MB ->", OUT)
