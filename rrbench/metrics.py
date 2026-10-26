"""
Goodness-of-fit metrics, observed vs. simulated.

Conventions (identical to Calibration Studio, so numbers match between tools):
- Missing values are removed pair-wise before anything is computed.
- PBIAS = 100 * sum(obs - sim) / sum(obs): positive means the model
  UNDERestimates the observed volume.
- R2 is the squared Pearson correlation (r**2), not NSE.
- Rv = mean(sim) / mean(obs) = 1 - PBIAS/100 (also the beta term of KGE).
- Standard deviations use ddof=0.
"""

from __future__ import annotations

import numpy as np

METRIC_NAMES = ["n", "NSE", "KGE", "RMSE", "MAE", "PBIAS", "r", "R2", "Rv"]

# Calibration objectives, expressed as values to MINIMISE.
OBJECTIVES = ("NSE", "KGE", "RMSE", "MAE", "PBIAS")


def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def evaluate(obs, sim) -> dict:
    """All metrics for one obs/sim pair. Undefined metrics are NaN."""
    o, s = _paired(obs, sim)
    out = dict.fromkeys(METRIC_NAMES, np.nan)
    out["n"] = len(o)
    if len(o) < 2:
        return out

    diff = o - s
    o_mean, s_mean = o.mean(), s.mean()
    o_std, s_std = o.std(), s.std()
    ss_tot = np.sum((o - o_mean) ** 2)

    out["RMSE"] = float(np.sqrt(np.mean(diff**2)))
    out["MAE"] = float(np.mean(np.abs(diff)))
    if ss_tot > 0:
        out["NSE"] = float(1.0 - np.sum(diff**2) / ss_tot)
    if np.sum(o) != 0:
        out["PBIAS"] = float(100.0 * np.sum(diff) / np.sum(o))
    if o_std > 0 and s_std > 0:
        r = float(np.corrcoef(o, s)[0, 1])
        out["r"], out["R2"] = r, r**2
        if o_mean != 0:
            alpha, beta = s_std / o_std, s_mean / o_mean
            out["Rv"] = float(beta)
            out["KGE"] = float(
                1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            )
    return out


def objective_value(obs, sim, name: str = "NSE") -> float:
    """Value to minimise during calibration (lower is better)."""
    name = name.upper()
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown objective {name!r}; choose from {OBJECTIVES}")
    m = evaluate(obs, sim)
    if name in ("NSE", "KGE"):
        value = -m[name]
    elif name == "PBIAS":
        value = abs(m["PBIAS"])
    else:
        value = m[name]
    return float(value) if np.isfinite(value) else 1e6
