# The inverse branch of the compact family cycle

Reproducibility materials for "The inverse branch of the compact family cycle:
down-type quark masses from waveform-carried partners" (A. M. Brilliant, 2026).

## Structure

```
├── inverse_branch.tex          # manuscript source
├── fig_branches.pdf            # Figure 1
├── scripts/
│   ├── coincidence_budget_engineA.py   # Kimi K3 implementation, seed 20260723
│   ├── coincidence_budget_engineB.py   # independent implementation, seed 20260726
│   ├── coincidence_budget_priors.py    # prior-sensitivity variants
│   ├── fig_branches.py                 # deterministic Figure 1 generator
│   └── ms_race_figure.py              # ms discrimination figure (requires ms_history.csv)
├── lean/
│   ├── InverseBranch.lean              # formal verification of core identities
│   └── lakefile.lean
├── results/
│   └── mc_counts.json                  # archived MC outputs at stated seeds
└── README.md
```

## Reproducibility

Every count printed in the paper is produced by the scripts in `scripts/` at the
stated seeds. To verify:

```bash
python3 scripts/coincidence_budget_engineB.py
# Expected output: 69/10000000 -> f = 6.90e-06
```

Engine A (Kimi K3 implementation) at seed 20260723, N=2×10⁷: 120 hits.
Engine B (this repo) at seed 20260726, N=10⁷: 69 hits.
Pooled: 189/3×10⁷, f = 6.3×10⁻⁶, Clopper–Pearson 95% [5.4, 7.3]×10⁻⁶.

Prior variants (seed 314159, N=10⁷ each):
- Log-uniform (quoted): 6.3×10⁻⁶
- Log-normal (1.5 dec): 5.8×10⁻⁶
- Uniform-in-mass: 0/10⁷ (f < 3.0×10⁻⁷)

## Lean verification

The `lean/` directory contains machine-checked proofs of:
1. The C₃ phase-cancellation identities (Σcos(θₖ)=0, Σcos²(θₖ)=3/2)
2. The cone lemma (Q = 2/3 iff A' = √2)
3. The α relocation ((1+α)⁻² = 2/3 with α = √(3/2)−1)
4. The inverse-branch inheritance (inverse masses satisfy the same cone condition)

All six theorems fully proven, zero sorries, verified under Lean 4.9.0 +
Mathlib v4.9.0. Build: `lake update && lake exe cache get && lake build`
(cache fetch downloads ~2 GB of precompiled Mathlib oleans; build then
completes in seconds).

## Convention

All rates are model-conditional frequencies under stated nulls, never p-values.
