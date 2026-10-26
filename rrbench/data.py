"""
Input loading and the common chronological split.

Expected input: one daily CSV with a date column, precipitation (mm/day),
potential evapotranspiration (mm/day), mean air temperature (degC, needed
by HBV only) and observed streamflow (m3/s). Common column names are
recognised automatically, e.g. "Date", "P", "precipitation (mm)", "PET (mm)",
"Temperature (oC)", "Flow (m3/s)".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALIASES = {
    "Date": ["date", "dates", "datesr", "time", "tarih"],
    "P": ["p", "precip", "precipitation", "precipitation (mm)", "prcp", "rain"],
    "PET": ["pet", "e", "pet (mm)", "etp", "potential_evap", "evap"],
    "T": ["t", "temp", "temperature", "temperature (oc)", "temperature (c)", "tmean"],
    "Q": ["q", "qobs", "flow", "flow (m3/s)", "discharge", "observed", "streamflow"],
}
REQUIRED = ("Date", "P", "PET", "Q")


def load_timeseries(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    lookup = {c.lower(): c for c in raw.columns}

    df = pd.DataFrame()
    for name, aliases in ALIASES.items():
        match = next((lookup[a] for a in [name.lower(), *aliases] if a in lookup), None)
        if match is not None:
            df[name] = raw[match]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find column(s) {missing} in {path}. "
            f"Columns present: {list(raw.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    step = df["Date"].diff().median()
    if step != pd.Timedelta(days=1):
        raise ValueError(f"Daily data expected; median time step is {step}.")

    for col in df.columns.drop("Date"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forcings are gap-filled; observed flow is never filled.
    forcings = [c for c in ("P", "PET", "T") if c in df.columns]
    df[forcings] = df[forcings].interpolate(limit_direction="both")
    return df


@dataclass(frozen=True)
class Split:
    """Chronological split shared by every model.

    [0, warmup)            spin-up, never scored
    [warmup, train_end)    train
    [train_end, valid_end) validation
    [valid_end, n)         test - unseen by every model
    """

    n: int
    warmup: int
    train_end: int
    valid_end: int

    def periods(self) -> dict[str, slice]:
        return {
            "train": slice(self.warmup, self.train_end),
            "valid": slice(self.train_end, self.valid_end),
            "test": slice(self.valid_end, self.n),
        }

    def mask(self, start: int, stop: int) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[start:stop] = True
        return m


def make_split(n: int, train_ratio=0.70, valid_ratio=0.15, warmup_days=365) -> Split:
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    if not 0 <= warmup_days < train_end < valid_end < n:
        raise ValueError(
            "Invalid split: need warmup < train_end < valid_end < n "
            f"(got {warmup_days}, {train_end}, {valid_end}, {n})."
        )
    return Split(n, warmup_days, train_end, valid_end)
