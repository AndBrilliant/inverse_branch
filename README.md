# The inverse branch of the compact family cycle

Reproducibility materials for "The inverse branch of the compact family cycle:
down-type quark masses from waveform-carried partners" (A. M. Brilliant, 2026).

## Structure

```
├── inverse_branch.tex          # manuscript source (submission draft, 2026-07-30)
├── fig_branches.pdf            # Figure 1
├── scripts/
│   ├── coincidence_budget_engineA.py    # Kimi K3 implementation, seed 20260723
│   ├── coincidence_budget_engineB.py    # independent implementation, seed 20260726
│   ├── coincidence_budget_priors.py     # prior-sensitivity variants
│   ├── coincidence_budget_convolved.py  # measurement-convolved budget
│   ├── delta_k_seesaw.py                # leading-order seesaw dk estimate
│   ├── mb_running_uncertainty.py        # m_b running uncertainty
│   ├── fig_branches.py                  # deterministic Figure 1 generator
│   ├── ms_race_figure.py                # ms discrimination figure (requires ms_history.csv)
│   ├── look_elsewhere_anatomy.py        # per-choice look-elsewhere decomposition
│   └── verify_manuscript.py             # prover: independent audit of every printed number
├── lean/
│   ├── InverseBranch.lean               # formal verification of core identities
│   └── lakefile.lean
├── results/
│   ├── mc_counts.json                   # archived MC outputs at stated seeds
│   └── look_elsewhere_anatomy.json      # anatomy outputs (2026-07-30 run)
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

### Independent re-verification (2026-07-30)

All archived engines re-run from a clean checkout at the stated seeds:
- Engine B: 69/10⁷ → f = 6.90×10⁻⁶ ✓
- Engine A: 120/2×10⁷ (121 under the as-sent inputs documented in the script header) ✓
- Prior variants: log-normal 58/10⁷, uniform-in-mass 0/10⁷ ✓
- Convolved budget: 18 raw hits; f_conv = 3.15×10⁻⁵; widened 3.60×10⁻⁵ ✓

## Manuscript audit (prover)

`scripts/verify_manuscript.py` recomputes every number printed in the
manuscript from canonical PDG 2024/2026 and FLAG 2024 inputs (4-loop QCD
running included) and guards the printed strings in the `.tex` source.
Status: 238 PASS / 0 WARN / 0 FAIL, 9 informational NOTEs, exit 0.

```bash
python3 scripts/verify_manuscript.py inverse_branch.tex
```

## Look-elsewhere anatomy

`scripts/look_elsewhere_anatomy.py` decomposes the lepton-anchored register
(paper Table 2) and the freedom ladder (paper Table 3) choice by choice. The
maximal library grant is constructed explicitly: 8 lepton-derived anchors ×
49 simple multipliers ({α^k, k = −2..4} × cofactors {1, 2, 3, 1/2, 1/3, √2,
1/√2}), plus the ninth-anchor augmentation √(2 m_e μ*) required by review
finding #5. All five observed rung forms are generated exactly. Joint
frequencies are exact expectations over the lepton draw (interval unions in
log space), validated against direct Monte Carlo.

Headline numbers (2026-07-30 run; full outputs in
`results/look_elsewhere_anatomy.json`):

- Register anatomy reproduces Table 2 exactly: joint 2.1×10⁻¹⁶ (windows
  variant 7.3×10⁻¹⁵); per-rung attribution 2.8–3.7 dex, delocalized.
- Maximal-grant joints: μ*-only 2.0×10⁻⁸; same-anchor 9.6×10⁻⁸ unaugmented
  → 1.1×10⁻⁷ augmented (union bound over prescriptions, null-favorable).
- Any-anchor reading (anchor may change rung by rung — granted for stress,
  not advocated): 1.2×10⁻⁴ unaugmented, 2.1×10⁻⁴ augmented; direct-MC
  validation 125/10⁶ = 1.25×10⁻⁴.
- Augmentation measured at ×1.1 per rung (×1.8 joint): the paper's granted
  doubling per rung (2⁵ = 32 total) is conservative.
- Freedom ladder verified by direct MC: 84-pair 2.31×10⁻² (paper 2.3×10⁻²);
  12-rational library 6.2×10⁻⁴ (paper 6.8×10⁻⁴, stated 12-list
  reconstruction); corner 0.68 (paper 0.71); 84-single 9.7×10⁻⁴ at the
  observed lepton sharpness 3.3×10⁻⁶ (paper 1.0×10⁻³).
- A geometric-mean relation per se is not rare: 5.3×10⁻³ for any 1%
  geometric-mean relation in a random up-window triple (~1 in 190); the m_u
  form alone 1.0×10⁻³. Rarity lives in the joint, not in any single form.

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
