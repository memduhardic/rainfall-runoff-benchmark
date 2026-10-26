"""
Rainfall-runoff model benchmark.

Calibrates / trains every selected model on the same chronological split
and scores all of them on one common test period that none of them saw.

    python run.py --input examples/data/L0123001.csv --area 360
    python run.py --models gr4j hbv --objective KGE
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rrbench.calibration import calibrate
from rrbench.data import load_timeseries, make_split
from rrbench.metrics import METRIC_NAMES, OBJECTIVES, evaluate
from rrbench import gr4j, hbv
from rrbench import plotting

CONCEPTUAL = {"gr4j": gr4j, "hbv": hbv}


def mm_day_to_m3s(q_mm, area_km2):
    return q_mm * area_km2 / 86.4
ML = ("rf", "ann", "lstm")
ALL_MODELS = ("gr4j", "hbv", "rf", "ann", "lstm")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="examples/data/L0123001.csv",
                    help="daily CSV with Date, P, PET, (T), Q columns")
    ap.add_argument("--area", type=float, default=360.0, help="catchment area in km2")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    ap.add_argument("--out", default="output", help="output folder")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    ap.add_argument("--warmup", type=int, default=365, help="spin-up days, never scored")
    ap.add_argument("--objective", default="NSE", choices=OBJECTIVES,
                    help="calibration objective for the conceptual models")
    ap.add_argument("--reps", type=int, default=10000, help="max SCE-UA model runs")
    ap.add_argument("--lstm-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def run_conceptual(module, df, split, args):
    missing = [c for c in module.INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"{module.NAME} needs column(s) {missing}")

    # Calibrate on train + validation (after warm-up); test stays unseen.
    end = split.valid_end
    forcing_cal = {c: df[c].to_numpy(float)[:end] for c in module.INPUTS}
    obs_cal = df["Q"].to_numpy(float)[:end]
    score_mask = split.mask(split.warmup, end)[:end]

    def simulate_cal(p):
        return mm_day_to_m3s(module.simulate(p, forcing_cal), args.area)

    params, best = calibrate(simulate_cal, module.PARAMS, obs_cal, score_mask,
                             objective=args.objective, reps=args.reps, seed=args.seed)
    forcing = {c: df[c].to_numpy(float) for c in module.INPUTS}
    sim = mm_day_to_m3s(module.simulate(list(params.values()), forcing), args.area)
    return sim, params


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "parameters").mkdir(exist_ok=True)

    df = load_timeseries(args.input)
    split = make_split(len(df), args.train_ratio, args.valid_ratio, args.warmup)
    d = df["Date"].dt.date
    print(f"\nInput : {args.input}  ({len(df)} days, area {args.area} km²)")
    for name, sl in split.periods().items():
        print(f"  {name:<5}: {d.iloc[sl.start]} -> {d.iloc[sl.stop - 1]}")

    sims, rows = {}, []
    for key in args.models:
        label = key.upper()
        print(f"\n[{label}]")
        t0 = time.time()
        if key in CONCEPTUAL:
            sim, params = run_conceptual(CONCEPTUAL[key], df, split, args)
            (out / "parameters" / f"{label}.json").write_text(json.dumps(params, indent=2))
            print("  parameters: " + ", ".join(f"{k}={v:.3f}" for k, v in params.items()))
        elif key == "rf":
            from rrbench.ml import random_forest
            sim, importance = random_forest(df, split, seed=args.seed)
            importance.to_csv(out / "parameters" / "RF_feature_importance.csv", index=False)
        elif key == "ann":
            from rrbench.ml import ann
            sim = ann(df, split, seed=args.seed)
        else:
            from rrbench.lstm import LSTMConfig, lstm
            sim = lstm(df, split, seed=args.seed, cfg=LSTMConfig(epochs=args.lstm_epochs))
        print(f"  done in {time.time() - t0:.0f} s")

        sims[label] = sim
        obs = df["Q"].to_numpy(float)
        for period, sl in split.periods().items():
            rows.append({"Model": label, "Period": period, **evaluate(obs[sl], sim[sl])})

    metrics = pd.DataFrame(rows, columns=["Model", "Period", *METRIC_NAMES])
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.4f")

    # Same layout Calibration Studio reads directly (Date, Precipitation,
    # Observed + one column per model).
    table = pd.DataFrame({"Date": d, "Precipitation": df["P"], "Observed": df["Q"]})
    for name, sim in sims.items():
        table[name] = np.round(sim, 4)
    table.to_csv(out / "simulations.csv", index=False)

    obs = df["Q"].to_numpy(float)
    test = split.periods()["test"]
    for name, sim in sims.items():
        tm = metrics.query("Model == @name and Period == 'test'").iloc[0]
        plotting.hydrograph(df, sim, name, split, tm, out / "figures" / f"{name}_hydrograph.png")
        plotting.scatter(obs[test], sim[test], name, tm, out / "figures" / f"{name}_scatter_test.png")
    if len(sims) > 1:
        plotting.comparison(df, sims, split, metrics, out / "figures" / "comparison.png")

    summary = metrics[metrics["Period"] == "test"].drop(columns="Period").set_index("Model")
    print("\nTest-period performance")
    print(summary.round(3).to_string())
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
