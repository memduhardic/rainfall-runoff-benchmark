"""
Input features for the machine-learning models, built from precipitation
and PET only (no observed-flow lags, no temperature, no calendar terms),
so that the ML models see the same information as a rainfall-runoff model
driven by P and PET.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

LAGS = (1, 2, 3, 5, 7, 14, 30)
WINDOWS = (3, 7, 14, 30, 60, 90, 120)
DECAYS = (0.85, 0.95, 0.98)


def _recursive_memory(x: np.ndarray, k: float) -> np.ndarray:
    """y[t] = k * y[t-1] + x[t] - an exponentially decaying memory."""
    return lfilter([1.0], [1.0, -k], x)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df["P"].astype(float)
    pet = df["PET"].astype(float)
    wb = p - pet

    f = pd.DataFrame(index=df.index)
    f["P"] = p
    f["PET"] = pet
    f["WB"] = wb
    f["P_div_PET"] = p / (pet + 0.1)

    for lag in LAGS:
        f[f"P_lag{lag}"] = p.shift(lag)
        f[f"PET_lag{lag}"] = pet.shift(lag)

    for w in WINDOWS:
        f[f"P_sum{w}d"] = p.rolling(w, min_periods=1).sum()
        f[f"PET_sum{w}d"] = pet.rolling(w, min_periods=1).sum()
        f[f"WB_sum{w}d"] = wb.rolling(w, min_periods=1).sum()
        f[f"dry_days{w}d"] = (p < 1.0).astype(float).rolling(w, min_periods=1).sum()

    # Antecedent precipitation index and climatic water-balance memory
    for k in DECAYS:
        f[f"API_k{k}"] = _recursive_memory(p.to_numpy(), k)
        f[f"CWB_k{k}"] = _recursive_memory(wb.to_numpy(), k)

    # Rows near the start (lags not yet available) fall inside the warm-up
    # period and are never used for training or scoring.
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f
