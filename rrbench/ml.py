"""Random Forest and feed-forward ANN (MLP) regressors on P/PET features."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from rrbench.features import build_features


def _train_rows(q: np.ndarray, split) -> np.ndarray:
    rows = split.mask(split.warmup, split.train_end)
    return rows & np.isfinite(q)


def random_forest(df: pd.DataFrame, split, seed=42, n_estimators=500):
    """Returns (simulated flow m3/s, feature importance table)."""
    X = build_features(df)
    q = df["Q"].to_numpy(float)
    rows = _train_rows(q, split)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X.values[rows], q[rows])
    sim = np.maximum(model.predict(X.values), 0.0)

    importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return sim, importance


def ann(df: pd.DataFrame, split, seed=42, hidden_layers=(10,), max_iter=3000):
    """Feed-forward ANN; scalers are fitted on the training rows only."""
    X = build_features(df).values
    q = df["Q"].to_numpy(float)
    rows = _train_rows(q, split)

    x_scaler = StandardScaler().fit(X[rows])
    y_scaler = StandardScaler().fit(q[rows].reshape(-1, 1))

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_scaler.transform(X[rows]), y_scaler.transform(q[rows].reshape(-1, 1)).ravel())

    pred = model.predict(x_scaler.transform(X)).reshape(-1, 1)
    return np.maximum(y_scaler.inverse_transform(pred).ravel(), 0.0)
