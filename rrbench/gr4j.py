"""
GR4J daily rainfall-runoff model (Perrin et al., 2003).

Adapted from the Python GR4J by Andrew MacDonald
(https://github.com/amacd31/gr4j), with numba for speed.
Original licence (BSD 3-Clause):

    Copyright (c) 2014, Andrew MacDonald. All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met: (1) redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer;
    (2) redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution;
    (3) the name of the author may not be used to endorse or promote
    products derived from this software without specific prior written
    permission.
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
    IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE
    LIABLE FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import math

import numpy as np

from numba import njit

NAME = "GR4J"
INPUTS = ("P", "PET")
PARAMS = [  # name, lower, upper
    ("X1", 100.0, 1200.0),
    ("X2", -10.0, 5.0),
    ("X3", 20.0, 500.0),
    ("X4", 1.0, 5.0),
]
# Initial stores as a fraction of capacity (the warm-up period removes
# most of their influence).
INIT_PRODUCTION = 0.6
INIT_ROUTING = 0.7


@njit(cache=True)
def _s_curve1(t, x4):
    if t <= 0.0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def _s_curve2(t, x4):
    if t <= 0.0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def run(P, E, x1, x2, x3, x4, s0, r0):
    """Simulated runoff in mm/day. s0, r0: initial stores (mm)."""
    n1 = int(math.ceil(x4))
    n2 = int(math.ceil(2.0 * x4))
    o1 = np.empty(n1)
    o2 = np.empty(n2)
    for k in range(n1):
        o1[k] = _s_curve1(k + 1.0, x4) - _s_curve1(float(k), x4)
    for k in range(n2):
        o2[k] = _s_curve2(k + 1.0, x4) - _s_curve2(float(k), x4)
    uh1 = np.zeros(n1)
    uh2 = np.zeros(n2)

    s, r = s0, r0
    q = np.empty(len(P))
    for i in range(len(P)):
        p, e = P[i], E[i]
        if p > e:
            pn = min((p - e) / x1, 13.0)
            tw = math.tanh(pn)
            ps = x1 * (1.0 - (s / x1) ** 2) * tw / (1.0 + s / x1 * tw)
            es = 0.0
            pr = p - e - ps
        else:
            en = min((e - p) / x1, 13.0)
            tw = math.tanh(en)
            es = s * (2.0 - s / x1) * tw / (1.0 + (1.0 - s / x1) * tw)
            ps = 0.0
            pr = 0.0
        s = s - es + ps

        perc = s - s / (1.0 + (s / (2.25 * x1)) ** 4) ** 0.25
        s -= perc
        pr += perc

        for k in range(n1 - 1):
            uh1[k] = uh1[k + 1] + o1[k] * pr
        uh1[n1 - 1] = o1[n1 - 1] * pr
        for k in range(n2 - 1):
            uh2[k] = uh2[k + 1] + o2[k] * pr
        uh2[n2 - 1] = o2[n2 - 1] * pr

        f = x2 * (r / x3) ** 3.5
        r = max(0.0, r + 0.9 * uh1[0] + f)
        r_new = r / (1.0 + (r / x3) ** 4) ** 0.25
        qr = r - r_new
        r = r_new
        qd = max(0.0, 0.1 * uh2[0] + f)
        q[i] = qr + qd
    return q


def simulate(params, forcing: dict) -> np.ndarray:
    """Runoff in mm/day for a parameter vector [X1, X2, X3, X4]."""
    x1, x2, x3, x4 = (float(v) for v in params)
    P = np.ascontiguousarray(forcing["P"], dtype=np.float64)
    E = np.ascontiguousarray(forcing["PET"], dtype=np.float64)
    return run(P, E, x1, x2, x3, x4, INIT_PRODUCTION * x1, INIT_ROUTING * x3)
