# ENGINE A of the coincidence-budget MC (Kimi K3 implementation).
# AUDIT NOTES (2026-07-23): (1) original observed inputs used mtau=1776.86
# and truncated mmu; harmonized runs use full-precision PDG masses via the
# engine-B convention, J_obs = 1.054e-3. (2) At seed 20260723, N=2e7: 120 hits
# (121 with the as-sent inputs). The count "73 of 1e7" quoted in an earlier
# draft is not reproduced by this script and is superseded. (3) The k formula
# sqrt(2(C^2+S^2))/S0 is algebraically identical to the engine-B DFT form.
# two_cone_mc.py — Monte Carlo behind Sec. VI (Statistical significance)
# of "The inverse branch of the compact family cycle" (A. M. Brilliant, 2026).
#
# Question: how often do random universes show the joint self-duality of
# Table I — lepton triple at k=1 (direct coordinate, |k-1| = 9e-6) AND
# down triple at k=1 (inverse coordinate, |k-1| = 1.1e-3)?
#
# Null: lepton triples log-uniform in [0.3 MeV, 2 GeV], down-type triples
# log-uniform in [2 MeV, 10 GeV], sorted. Every null universe is granted
# the post-hoc coordinate choice (its k is whichever of direct/inverse
# lies closer to 1). Score J = |k_lep-1| + |k_down-1|; p = P(J <= J_obs).
# Seeded, self-contained, runs in seconds.

import numpy as np

def circulant_k(V):
    """Exact Brannen-Zenczykowski circulant fit (discrete Fourier inversion).
    V: (...,3) amplitudes in one coordinate. Returns k."""
    j = np.arange(3)
    C = (V * np.cos(2*np.pi*j/3)).sum(-1)
    S = (V * np.sin(2*np.pi*j/3)).sum(-1)
    S0 = V.sum(-1)
    return np.sqrt(2*(C**2 + S**2)) / S0

def best_k(masses):
    """k closest to 1 across the two coordinates (post-hoc fair to the null)."""
    kd = circulant_k(np.sqrt(masses))
    ki = circulant_k(1/np.sqrt(masses))
    return np.where(np.abs(kd-1) < np.abs(ki-1), kd, ki)

if __name__ == "__main__":
    # observed universe (leptons: pole masses; down: common 2 GeV MS-bar)
    k_lep = best_k(np.array([0.510999, 105.658, 1776.86]))
    k_dwn = best_k(np.array([4.70, 93.4, 4966.0]))
    J_obs = abs(k_lep-1) + abs(k_dwn-1)
    print(f"observed: k_lep = {k_lep:.8f}, k_down = {k_dwn:.7f}, J_obs = {J_obs:.6e}")

    rng = np.random.default_rng(20260723)
    N = 20_000_000
    k = 0
    for _ in range(4):  # chunked alternating draws (memory); reproduces the paper's count
        Lep = np.sort(np.exp(rng.uniform(*np.log([0.3, 2000.0]), size=(N//4,3))), axis=1)
        Dwn = np.sort(np.exp(rng.uniform(*np.log([2.0, 10_000.0]), size=(N//4,3))), axis=1)
        J = np.abs(best_k(Lep)-1) + np.abs(best_k(Dwn)-1)
        k += int((J <= J_obs).sum())
    print(f"{N:,} universes: {k} with J <= J_obs  ->  p = {(k+1)/(N+1):.2e}")
