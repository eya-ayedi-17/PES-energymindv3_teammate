"""EnergyMind - explanations of a detection: SHAP (TreeSHAP) and LIME.

* SHAP  : exact TreeSHAP contributions computed by the tree library itself (XGBoost pred_contribs, LightGBM
          pred_contrib). Same values as the `shap` package's TreeExplainer, no extra dependency. For one row and one
          anomaly type they say how much each feature pushed the model's score (log-odds) towards that type.
* LIME  : model-agnostic second opinion (package `lime`). It perturbs the row around its value, fits a local linear
          model and reports the features that drive the probability of the predicted type.
Both explain the predicted anomaly type of a row; agreement between them is a check of explanation quality.

Works with the sklearn wrappers (XGBClassifier, LGBMClassifier) and with the native boosters of the exported package.
This file only needs numpy and pandas (+ xgboost / lightgbm for SHAP, + lime for LIME).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# plain-language meaning of the feature families (prefix -> text)
FEATURE_TEXT = [
    ("ref_meter_lr_abs", "size of the gap between the house meter and the second meter (24 h)"),
    ("ref_meter_lr_24h_vs", "change of the house-meter / second-meter gap vs earlier days"),
    ("ref_meter_lr", "gap between the house meter and the second meter (log ratio of their energies)"),
    ("ref_pv_ratio_vs_prev", "solar production vs theory, compared with the same hours on the previous days"),
    ("ref_pv_ratio", "solar production vs the theoretical production from the roof sensor"),
    ("ref_pv_theory", "theoretical solar production from the roof sensor (irradiance, module temperature)"),
    ("ref_irr", "irradiance measured by the roof sensor"),
    ("ref_module_temp", "module temperature measured by the roof sensor"),
    ("hist_P_logratio", "meter power vs its usual level at this hour (last 14 days)"),
    ("hist_P_step", "sudden step of the meter power vs its usual level"),
    ("hist_P_ref_W", "usual power of this house at this hour"),
    ("hist_share", "house share of the transformer's monitored group vs its usual share"),
    ("hist_day_energy", "last 24 h energy vs the same 24 h one week before"),
    ("house_share_of_group", "house share of the transformer's monitored group"),
    ("rest_of_group", "power of the rest of the monitored group"),
    ("group_P", "power of the monitored group under the transformer"),
    ("meter_missing", "missing meter readings"), ("meter_n_missing", "missing meter readings in the last 6 h"),
    ("meter_repeat", "meter values repeated exactly"), ("P_TOT_repeat", "power value repeated exactly"),
    ("U_repeat", "voltage value repeated exactly"),
    ("meter_lag96", "readings identical to 24 h earlier"), ("P_TOT_lag96", "power vs 24 h earlier"),
    ("U_lag96", "voltage vs 24 h earlier"),
    ("U_rel_trafo", "house voltage vs transformer voltage"), ("U_dev_vs_trafo", "house voltage change vs transformer voltage change"),
    ("U_phase_rel_trafo", "phase voltage vs transformer voltage"), ("U_phase_spread", "spread between the 3 phase voltages"),
    ("U_zero", "no voltage at the meter"), ("U_pu_minus", "house voltage vs its recent level"), ("U_", "house voltage"),
    ("transformer_loading", "transformer loading"), ("trafo_loading", "transformer loading vs its recent level"),
    ("transformer_U", "transformer low-voltage side voltage"), ("transformer_", "transformer measurement"),
    ("other_feeders", "power of the other feeders"),
    ("grid_available", "grid (STEG) available"), ("P_TOT_std", "short-term variability of the power"),
    ("P_TOT_rough", "roughness of the power curve"), ("P_TOT", "meter active power"), ("Q_TOT", "meter reactive power"),
    ("S_TOT", "meter apparent power"), ("PF", "power factor"), ("I_sum", "total current"),
    ("battery_soc", "battery state of charge"), ("soc_", "battery state of charge change"),
    ("battery_charge", "battery charging power"), ("battery_discharge", "battery discharging power"),
    ("battery_I", "battery current vs power consistency"), ("charge_", "battery charging vs available solar power"),
    ("grid_import_soc_high", "importing from the grid although the battery is charged"),
    ("inverter", "inverter output / status"), ("backup", "backup mode"), ("unmet", "backup demand not served"),
    ("transfer_switch", "transfer switch on battery"),
    ("pv_", "solar production vs clear-sky level"), ("P_PV", "solar production"), ("dP_PV", "change of solar production"),
    ("curtail", "solar power not used"), ("importing_flag", "house importing from the grid"),
    ("hour_", "time of day"), ("house_type_A", "house with PV + battery"),
]


def describe(feature: str) -> str:
    for p, t in FEATURE_TEXT:
        if feature.startswith(p):
            return t
    return feature


# ------------------------------------------------------------------ SHAP
def shap_contributions(model, X: np.ndarray, n_trees: int | None = None) -> np.ndarray:
    """TreeSHAP values, shape (rows, classes, features + 1); the last column is the bias.
    model: XGBClassifier / xgboost.Booster / LGBMClassifier / lightgbm.Booster."""
    X = np.asarray(X, np.float32)
    mod = type(model).__module__
    if mod.startswith("xgboost"):
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        if n_trees is None:
            n_trees = int(model.best_iteration) + 1 if hasattr(model, "best_iteration") and model.best_iteration is not None else 0
        c = booster.predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True,
                            iteration_range=(0, n_trees) if n_trees else (0, 0))
        return c if c.ndim == 3 else c[:, None, :]
    if mod.startswith("lightgbm"):
        booster = model.booster_ if hasattr(model, "booster_") else model
        kw = {"num_iteration": model.best_iteration_} if hasattr(model, "best_iteration_") and model.best_iteration_ else {}
        c = booster.predict(X, pred_contrib=True, **kw)
        n_cls = booster.num_model_per_iteration()
        return np.asarray(c).reshape(len(X), n_cls, X.shape[1] + 1)
    raise TypeError(f"SHAP explanations need an XGBoost or LightGBM model, got {type(model).__name__}")


def top_shap(contrib_row: np.ndarray, x_row: np.ndarray, features: list[str], k: int = 5) -> list[dict]:
    """contrib_row: contributions of ONE class for ONE row (features + bias). Largest positive pushes first."""
    c = contrib_row[:-1]
    order = np.argsort(-c)[:k]
    return [{"feature": features[i], "value": None if not np.isfinite(x_row[i]) else float(x_row[i]),
             "shap": float(c[i]), "meaning": describe(features[i])} for i in order if c[i] > 0]


def shap_text(items: list[dict]) -> str:
    parts = []
    for d in items:
        v = "missing" if d["value"] is None else f"{d['value']:.3g}"
        parts.append(f"{d['meaning']} ({d['feature']} = {v}, SHAP +{d['shap']:.2f})")
    return "; ".join(parts)


# ------------------------------------------------------------------ LIME
class LimeExplainer:
    """LIME on the rows' finite features only (LIME cannot perturb missing values: a grid-only house has no PV data).
    background: 2-D array of training feature rows (NaN allowed)."""

    def __init__(self, background: np.ndarray, features: list[str], class_names: list[str], seed: int = 0):
        self.bg = np.asarray(background, float)
        self.features, self.class_names, self.seed = features, class_names, seed
        self._cache = {}

    def _explainer(self, mask: tuple):
        if mask not in self._cache:
            from lime.lime_tabular import LimeTabularExplainer
            cols = np.array(mask)
            bg = self.bg[:, cols]
            bg = bg[np.isfinite(bg).all(1)] if np.isfinite(bg).all(1).sum() >= 200 else np.where(
                np.isfinite(bg), bg, np.nanmedian(bg, axis=0))
            self._cache[mask] = LimeTabularExplainer(bg, feature_names=[self.features[i] for i in cols],
                                                     class_names=self.class_names, discretize_continuous=True,
                                                     random_state=self.seed)
        return self._cache[mask]

    def explain(self, x_row: np.ndarray, predict_proba, cls: int, k: int = 5, num_samples: int = 2000) -> list[dict]:
        """predict_proba(X_full) -> (n, classes). cls = class index in predict_proba's output."""
        x_row = np.asarray(x_row, float)
        mask = tuple(np.where(np.isfinite(x_row))[0].tolist())
        cols = np.array(mask)

        def f(Z):
            full = np.tile(x_row, (len(Z), 1))
            full[:, cols] = Z
            return predict_proba(full.astype(np.float32))
        exp = self._explainer(mask).explain_instance(x_row[cols], f, labels=(cls,), num_features=k, num_samples=num_samples)
        out = []
        for rule, w in exp.as_list(label=cls):
            # the rule text contains the feature name; take the LONGEST matching name (U_rel_trafo vs U_rel_trafo_minus_med96)
            hits = [self.features[i] for i in cols if self.features[i] in rule]
            name = max(hits, key=len) if hits else rule
            out.append({"feature": name, "rule": rule, "weight": float(w), "meaning": describe(name)})
        return out


def lime_text(items: list[dict]) -> str:
    return "; ".join(f"{d['rule']} ({'+' if d['weight'] >= 0 else ''}{d['weight']:.2f})" for d in items)
