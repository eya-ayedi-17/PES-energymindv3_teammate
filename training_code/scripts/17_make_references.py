"""Step 17 - Reference inputs: a second meter and a simple roof sensor (irradiance + temperature).

Writes data/05_references/house_XX.csv (same 15-min timestamps as the house files). Nothing existing is modified:
the new columns are computed from the TRUE values the simulator already stored (data/02_injected/truth and
data/01_physical_baseline/physical), then degraded to realistic device accuracy.

  check_meter_energy_kwh   all 13 houses  second meter: true grid energy of the 15-min step x its own gain
                                          (class 1: +-1 %) x small noise (0.3 %). It reads the TRUE energy, so it is
                                          not fooled by the faults of the house meter (bias, drift, noise, replay, ...).
  irradiance_w_m2          PV houses      roof reference cell, in the plane of the panels: true irradiance x its own
                                          calibration (+-2 %) x noise (1 %)  => about +-3 %
  module_temp_c            PV houses      sensor on the back of the module: true cell temperature + offset (+-0.5 C)
                                          + noise (0.3 C)

The true irradiance and cell temperature are recovered by inverting the formula of the Phase 1 PV generator
(baseline_dataset/pv.py):  P = poa/1000 * (1 - 0.004 * (T_cell - 25)) * 0.82 * kWp,  T_cell = T_amb + 0.03 * poa,
with the Tunis monthly temperatures (see refs_lib.py). The daily temperature anomaly of Phase 1 is not stored; it moves the
recovered irradiance by less than 1 %.
Usage: python 17_make_references.py
"""
import numpy as np
import pandas as pd

from common import CONFIG, D_BASE, D_INJ, HOUSES, TYPE_A, V3, load_json, save_json
from refs_lib import COEF, F_UNIT, NOISE, PERF, roof_sensor, second_meter

OUT = V3 / "data" / "05_references"
SEED = 17

cfg = load_json(CONFIG / "tunisia_config.json")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
ts = pd.to_datetime(pd.read_parquet(D_INJ / "transformer_meter.parquet").timestamp)
hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60
poa_by_house, check = {}, {}
for h in HOUSES:
    tr = pd.read_parquet(D_INJ / "truth" / f"{h}.parquet")
    df = pd.DataFrame({"timestamp": ts.values, "check_meter_energy_kwh": second_meter(tr.P_grid_true_W.to_numpy(float), rng)})
    if h in TYPE_A:
        ph = pd.read_parquet(D_BASE / "physical" / f"{h}.parquet")          # PV available WITHOUT shading
        kwp = cfg["type_a_systems"][h]["pv_kwp"]
        df["irradiance_w_m2"], df["module_temp_c"], poa = roof_sensor(ph.pv_avail_W.to_numpy(float), kwp, ts, rng)
        poa_by_house[h] = poa
        check[h] = {"kwp": kwp, "poa_max_w_m2": float(poa.max()), "poa_mean_daylight_w_m2": float(poa[poa > 0].mean())}
    df.to_csv(OUT / f"{h}.csv", index=False)

# the weather is shared by the neighbourhood: the recovered irradiance must agree between PV houses
H = list(poa_by_house)
noon = (hour >= 11) & (hour <= 13) & (poa_by_house[H[0]] > 300)
spread = max(float(np.nanmax(np.abs(poa_by_house[a][noon] / poa_by_house[H[0]][noon] - 1))) for a in H[1:])
save_json({"seed": SEED, "noise": NOISE, "pv_constants": {"perf_ratio": PERF, "temp_coeff_per_K": COEF, "f_unit": F_UNIT},
           "check": check, "max_difference_between_pv_houses_at_noon": spread,
           "columns": {"check_meter_energy_kwh": "all houses, kWh per 15 min", "irradiance_w_m2": "PV houses, W/m2 in the panel plane",
                       "module_temp_c": "PV houses, deg C"}}, OUT / "references_config.json")
print("written", OUT, "| PV houses:", H)
print({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in check.items()})
print("irradiance recovered for the 4 PV houses agrees within %.1f %% at noon (they share one weather; the rest is Phase 1's 0.97-1.0 random PV factor)" % (100 * spread))
