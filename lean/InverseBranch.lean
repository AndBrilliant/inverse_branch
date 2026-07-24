import Mathlib.Tactic

/-!
# Formal verification: The inverse branch of the compact family cycle

Machine-checked proofs (Lean 4 + Mathlib) of the core algebraic identities in
"The inverse branch of the compact family cycle" (A. M. Brilliant, 2026).

Theorems:
1. `cone_lemma`          — Σz²/(Σz)² = 1/3 + A'²/6 under the sum rules
2. `self_dual_forward`   — A'² = 2 ⟹ Q = 2/3
3. `self_dual_converse`  — Q = 2/3 ⟹ A'² = 2
4. `alpha_relocation`    — (1+α)² = 3/2 ⟹ 1/(1+α)² = 2/3
5. `inverse_branch_cone` — inverse-scaled samples satisfy the same Q
6. `seesaw_fixed_product`— (m_D²/M)·M = m_D²
-/

/-- **Cone lemma.** If the three samples satisfy the sum rules
    S = Σz = 3c₀ and S₂ = Σz² = c₀²(3 + 3A'²/2), then
    S₂/S² = 1/3 + A'²/6. This is Eq. (4) of the paper. -/
theorem cone_lemma (c₀ A' S S2 : ℚ) (hc : c₀ ≠ 0)
    (hS : S = 3 * c₀)
    (hS2 : S2 = c₀ ^ 2 * (3 + 3 * A' ^ 2 / 2)) :
    S2 / S ^ 2 = 1 / 3 + A' ^ 2 / 6 := by
  subst hS hS2
  have hc2 : c₀ ^ 2 ≠ 0 := pow_ne_zero 2 hc
  field_simp
  ring

/-- **Self-duality, forward.** A'² = 2 gives the ratio 2/3 exactly. -/
theorem self_dual_forward (A' : ℚ) (h : A' ^ 2 = 2) :
    1 / 3 + A' ^ 2 / 6 = 2 / 3 := by
  rw [h]; norm_num

/-- **Self-duality, converse.** Ratio 2/3 forces A'² = 2:
    the coefficient √2 in the waveform is not fitted. -/
theorem self_dual_converse (A' : ℚ) (h : 1 / 3 + A' ^ 2 / 6 = 2 / 3) :
    A' ^ 2 = 2 := by linarith

/-- **α relocation.** If (1+α)² = 3/2 then 1/(1+α)² = 2/3:
    α = √(3/2) − 1 is the cone condition in scalar form,
    not an additional constant. -/
theorem alpha_relocation (α : ℚ) (h : (1 + α) ^ 2 = 3 / 2) :
    1 / (1 + α) ^ 2 = 2 / 3 := by
  rw [h]; norm_num

/-- **Inverse-branch inheritance.** Rescaling all samples by 1/M
    (the seesaw inversion m ↦ m_D²/m sends √m ↦ (m_D/√m), i.e.
    v_a = Z_a/M in the inverse coordinate) leaves the participation
    ratio invariant; hence if the direct samples satisfy Q = 2/3,
    the inverse-coordinate images do too. This is the algebraic core
    of Sec. III of the paper. -/
theorem inverse_branch_cone (M Z₀ Z₁ Z₂ : ℚ) (hM : M ≠ 0)
    (hS : Z₀ + Z₁ + Z₂ ≠ 0) :
    ((Z₀ / M) ^ 2 + (Z₁ / M) ^ 2 + (Z₂ / M) ^ 2) /
      (Z₀ / M + Z₁ / M + Z₂ / M) ^ 2
    = (Z₀ ^ 2 + Z₁ ^ 2 + Z₂ ^ 2) / (Z₀ + Z₁ + Z₂) ^ 2 := by
  have hS2 : (Z₀ + Z₁ + Z₂) ^ 2 ≠ 0 := pow_ne_zero 2 hS
  have hM2 : M ^ 2 ≠ 0 := pow_ne_zero 2 hM
  field_simp

/-- **Seesaw fixed product.** The determinant identity: the light mass
    m = m_D²/M and its partner M multiply to m_D², independent of M.
    Every light–partner pair is a geometric-mean pair. -/
theorem seesaw_fixed_product (m_D M : ℚ) (hM : M ≠ 0) :
    m_D ^ 2 / M * M = m_D ^ 2 :=
  div_mul_cancel₀ (m_D ^ 2) hM

