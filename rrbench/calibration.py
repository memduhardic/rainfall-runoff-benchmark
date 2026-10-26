"""Global calibration of conceptual models with SCE-UA (spotpy)."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import spotpy

from rrbench.metrics import objective_value


class _SpotpySetup:
    def __init__(self, simulate, specs, obs, score_mask, objective):
        self._simulate = simulate
        self._params = [spotpy.parameter.Uniform(n, lo, hi) for n, lo, hi in specs]
        self._obs = obs
        self._mask = score_mask
        self._objective = objective

    def parameters(self):
        return spotpy.parameter.generate(self._params)

    def simulation(self, vector):
        return self._simulate(np.asarray(vector, dtype=float))

    def evaluation(self):
        return self._obs

    def objectivefunction(self, simulation, evaluation):
        sim = np.asarray(simulation, dtype=float)
        if not np.all(np.isfinite(sim)):
            return 1e6
        return objective_value(evaluation[self._mask], sim[self._mask], self._objective)


def calibrate(simulate, specs, obs, score_mask, objective="NSE", reps=10000,
              seed=42, verbose=False):
    """
    Minimise the objective over `score_mask` with SCE-UA.

    simulate   : callable(param_vector) -> simulated series (same length as obs)
    specs      : list of (name, lower, upper)
    score_mask : boolean array, True where the objective is evaluated
    Returns (best_params: dict, best_objective: float).
    """
    setup = _SpotpySetup(simulate, specs, np.asarray(obs, float), score_mask, objective)
    sampler = spotpy.algorithms.sceua(
        setup, dbname="sceua", dbformat="ram", save_sim=False, random_state=seed
    )
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        sampler.sample(reps, ngs=7, kstop=10, peps=0.01, pcento=0.001)

    results = sampler.getdata()
    best = results[np.argmin(results["like1"])]
    params = {name: float(best[f"par{name}"]) for name, _, _ in specs}
    return params, float(best["like1"])
