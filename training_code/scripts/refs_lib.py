"""EnergyMind - reference inputs from TRUE simulator values: a second meter and a simple roof sensor.
Used by 17_make_references.py (the dataset) and by detector/test_anomaly.py (injected test scenarios).
See 17_make_references.py for the description of the two devices and of the inversion of the PV generator."""
import numpy as np
import pandas as pd

PERF, COEF = 0.82, -0.004                      # panel constants of the Phase 1 PV generator
TAMB_MONTH = (11.5, 12.0, 14.0, 16.5, 20.5, 24.5, 27.5, 28.0, 25.5, 21.0, 16.0, 12.5)    # Tunis climatology
F_UNIT = 0.985                                 # Phase 1 gave each house a random 0.97-1.0 PV factor: its mean
NOISE = {"meter_gain": 0.01, "meter_step": 0.003, "irr_gain": 0.02, "irr_step": 0.01, "irr_offset_w_m2": 1.0,
         "temp_offset_c": 0.5, "temp_step_c": 0.3}


def second_meter(p_true_w, rng):
    """kWh per 15-min step seen by an independent class-1 meter (own gain +-1 %, noise 0.3 %) on the TRUE grid power."""
    p = np.clip(np.asarray(p_true_w, float), 0, None)
    gain = 1 + rng.uniform(-NOISE["meter_gain"], NOISE["meter_gain"])
    return np.round(p * 0.25 / 1000 * gain * (1 + rng.normal(0, NOISE["meter_step"], len(p))), 4)


def roof_sensor(pv_avail_w, kwp, ts, rng):
    """(irradiance W/m2, module temperature C, true irradiance) from the PV available WITHOUT shading.
    The true irradiance is recovered by inverting P = poa/1000 (1 + COEF (T_cell - 25)) PERF kWp, T_cell = T_amb + 0.03 poa."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    T = len(ts)
    month = ts.dt.month.to_numpy() - 1
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60
    day = ((ts.dt.normalize() - ts.dt.normalize().iloc[0]).dt.days).to_numpy()
    tamb = np.array(TAMB_MONTH)[month] + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    kw = np.clip(np.asarray(pv_avail_w, float) / (kwp * 1000 * F_UNIT), 0, None)            # kW per kWp
    poa = kw * 1000 / PERF
    for _ in range(6):                                                                      # poa <-> T_cell fixed point
        tcell = tamb + 0.03 * poa
        poa = kw * 1000 / (PERF * (1 + COEF * (tcell - 25)))
    poa = np.where(kw > 0, poa, 0.0)
    tcell = tamb + 0.03 * poa + rng.normal(0, 2.0, day.max() + 1)[day]                      # real days differ from the climatology
    g = np.clip(rng.normal(1, NOISE["irr_gain"]), 0.95, 1.05)
    irr = poa * g * (1 + rng.normal(0, NOISE["irr_step"], T)) + rng.normal(0, NOISE["irr_offset_w_m2"], T)
    irr = np.round(np.where(poa > 0, np.clip(irr, 0, None), 0.0), 1)
    mod = np.round(tcell + rng.normal(0, NOISE["temp_offset_c"]) + rng.normal(0, NOISE["temp_step_c"], T), 1)
    return irr, mod, poa
