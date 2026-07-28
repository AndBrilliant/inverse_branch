#!/usr/bin/env python3
"""
coincidence_budget_convolved.py -- threshold-convolved coincidence budget.

Companion to coincidence_budget_engineA.py. The observed score
    J_obs = |k_lep - 1| + |k_down - 1| = 3.3e-6 + 1.05e-3 = 1.0533e-3
carries input-propagated uncertainty sigma_J ~= 0.0025 (m_d-dominated), which
exceeds J_obs itself: the threshold is not sharp. Rather than substituting
prior-stability for threshold-stability, we propagate the threshold:

    f_conv = E_null[ Phi( (J_obs - J_null) / sigma_J ) ]

i.e. each null universe contributes the probability that the smeared observed
threshold exceeds its score (J_null >= 0, and t < 0 contributes zero measure
to {t >= J_null}, so no truncation correction is needed).

Null construction matches Engine A exactly: lepton triples log-uniform on
[0.3 MeV, 2 GeV], down triples log-uniform on [2 MeV, 10 GeV]; each null
receives the exact circulant fit in both coordinates and is granted the
post-hoc coordinate choice (k = whichever of direct/inverse lies closer to 1).

Reference run (seed 20260728, N = 2e6, log-uniform):
    raw  f = 9.0e-6  (18 hits; CP95 [5.3, 14.2]e-6 -- consistent with the
                       pooled 6.3e-6 of the 3e7-draw production runs)
    conv f = 3.15e-5 +/- 0.2e-5 (MC)   inflation x5.0 over 6.3e-6
Widened windows [0.1 MeV, 5 GeV] x [1 MeV, 20 GeV] (seed 20260729, N = 2e6):
    conv f = 3.60e-5 +/- 0.2e-5        inflation x4.8 over 7.5e-6
"""
import numpy as np
from scipy.stats import norm, beta as betadist

J_OBS = 3.3e-6 + 1.05e-3
SIG_J = 2.5e-3


def kfit(V):
    """Exact circulant-fit Koide numerator k for amplitude triples V."""
    V = np.asarray(V, float)
    j = np.arange(3)
    C = (V * np.cos(2 * np.pi * j / 3)).sum(-1)
    S = (V * np.sin(2 * np.pi * j / 3)).sum(-1)
    return np.sqrt(2 * (C**2 + S**2)) / V.sum(-1)


def kmin_both_coords(m):
    kd = kfit(np.sqrt(m))
    ki = kfit(1.0 / np.sqrt(m))
    return np.minimum(np.abs(kd - 1), np.abs(ki - 1))


def run(lep_win, dwn_win, N=2_000_000, seed=20260728, batch=100_000):
    rng = np.random.default_rng(seed)
    W = W2 = 0.0
    hits = 0
    for _ in range(N // batch):
        lep = np.exp(rng.uniform(np.log(lep_win[0]), np.log(lep_win[1]), (batch, 3)))
        dwn = np.exp(rng.uniform(np.log(dwn_win[0]), np.log(dwn_win[1]), (batch, 3)))
        J = kmin_both_coords(lep) + kmin_both_coords(dwn)
        w = norm.cdf((J_OBS - J) / SIG_J)
        W += w.sum()
        W2 += (w * w).sum()
        hits += int((J <= J_OBS).sum())
    f_conv = W / N
    se = np.sqrt(max(W2 / N - f_conv**2, 0.0) / N)
    f_raw = hits / N
    if hits:
        lo = betadist.ppf(0.025, hits, N - hits + 1)
        hi = betadist.ppf(0.975, hits + 1, N - hits)
    else:
        lo, hi = 0.0, 3.0 / N
    return f_conv, se, f_raw, hits, lo, hi


if __name__ == "__main__":
    for tag, lw, dw, seed in [
        ("standard", (0.3, 2000.0), (2.0, 10000.0), 20260728),
        ("widened", (0.1, 5000.0), (1.0, 20000.0), 20260729),
    ]:
        fc, se, fr, h, lo, hi = run(lw, dw, seed=seed)
        print(f"[{tag}] N=2e6  raw f = {fr:.2e} ({h} hits, CP95 [{lo:.2e},{hi:.2e}])"
              f"   conv f = {fc:.2e} +/- {se:.1e}")
