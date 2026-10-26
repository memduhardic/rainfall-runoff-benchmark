"""
HBV-type daily rainfall-runoff model with snow, soil and two reservoirs
(Bergstrom, 1995). Model structure after the HBV implementation by
Giuliani, Kollat, Herman et al. (https://github.com/mxgiuliani00/hbv;
Herman et al., 2013), re-written in Python so that the water balance closes.
"""

from __future__ import annotations

import numpy as np

from numba import njit

NAME = "HBV"
INPUTS = ("P", "PET", "T")
PARAMS = [  # name, lower, upper
    ("TT", -2.0, 2.0),
    ("CFMAX", 1.0, 5.0),
    ("Beta", 1.0, 5.0),
    ("FC", 100.0, 800.0),
    ("PWP", 0.3, 1.0),
    ("SUMax", 10.0, 100.0),
    ("Kr", 0.01, 0.5),
    ("Ku", 0.01, 0.5),
    ("Kl", 0.001, 0.1),
    ("Kperc", 0.001, 0.2),
]
INIT_SOIL = 0.5   # fraction of FC
INIT_UPPER = 10.0  # mm
INIT_LOWER = 10.0  # mm


@njit(cache=True)
def run(P, T, E, tt, cfmax, beta, fc, pwp, sumax, kr, ku, kl, kperc):
    """Returns (runoff, actual ET, total storage), all in mm."""
    n = len(P)
    q = np.empty(n)
    et_out = np.empty(n)
    storage = np.empty(n)

    snow = 0.0
    sm = INIT_SOIL * fc
    su = INIT_UPPER
    sl = INIT_LOWER
    lp = pwp * fc

    for i in range(n):
        p, t, e = P[i], T[i], E[i]

        # snow routine
        if t < tt:
            snow += p
            peq = 0.0
        else:
            melt = min(cfmax * (t - tt), snow)
            snow -= melt
            peq = p + melt

        # soil routine
        recharge = (sm / fc) ** beta * peq
        sm += peq - recharge
        if sm > fc:
            recharge += sm - fc
            sm = fc
        et = e * min(sm / lp, 1.0) if lp > 0.0 else e
        et = min(et, sm)
        sm -= et

        # upper zone: surface runoff, interflow, percolation
        su += recharge
        qr = kr * max(su - sumax, 0.0)
        qu = ku * su
        perc = kperc * su
        out = qr + qu + perc
        if out > su and out > 0.0:
            scale = su / out
            qr *= scale
            qu *= scale
            perc *= scale
        su -= qr + qu + perc

        # lower zone: baseflow
        sl += perc
        ql = kl * sl
        sl -= ql

        q[i] = qr + qu + ql
        et_out[i] = et
        storage[i] = snow + sm + su + sl
    return q, et_out, storage


def simulate(params, forcing: dict) -> np.ndarray:
    """Runoff in mm/day for a parameter vector in PARAMS order."""
    P = np.ascontiguousarray(forcing["P"], dtype=np.float64)
    T = np.ascontiguousarray(forcing["T"], dtype=np.float64)
    E = np.ascontiguousarray(forcing["PET"], dtype=np.float64)
    q, _, _ = run(P, T, E, *(float(v) for v in params))
    return q
