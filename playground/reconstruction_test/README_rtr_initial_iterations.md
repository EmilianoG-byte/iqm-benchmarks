# Wasted initial RTR iterations for State and POVM

## Symptom

In `run_riemannian_optimization`, the POVM and State blocks barely move during the first 3–4 outer iterations.
The Kraus block does not show this. Logs from `1q_mgst_vs_current_state.ipynb` (JIT run, outer iteration 1):

| block | ‖grad f‖ at init | radius accepted | tCG |
|---|---|---|---|
| POVM  | 5.0e8 | 1.6e-6 (3 rejections from 1e-4) | 1 iteration, negative curvature |
| Kraus | 2.1e4 | 1e-1 (0 rejections) | 1–7 iterations, normal |
| State | 9.6e5 | 6.3e-6 (2 rejections from 1e-4) | 1 iteration, negative curvature |

In `2q_mgst_vs_current_state.ipynb`, the first State block rejects all 5 steps. After that, the radius
doubles once per accepted step: 1e-6 → 3e-2 takes 3–4 outer iterations.

## Diagnosis

`get_compressed_perturbed_rep_from_mgst` factorized the ideal pure state and the projective POVM at full rank.
The only regularization was `1e-14·I`, so the columns that raise the rank are ~1e-7. The initial point is
then on the boundary of the PSD cone:

1. **Log-likelihood barrier.** Predicted probabilities are p ≈ 1e-14 where the data has y > 0 because of SPAM errors.
   The gradient N·y/p is huge, and the quadratic model of −log p only holds within ~√p ≈ 1e-6. That is
   exactly the radius that ends up accepted.
2. **Saddle of the factorization (Burer–Monteiro).** For ρ = XX†, the gradient along a zero column is 2G·0 = 0.
   Raising the rank is a second-order effect, so tCG meets negative curvature at its first iteration. It then
   returns the boundary step, and the RTR becomes steepest descent with step Δ. In the 2Q State block, the
   gradient norm *grows* while the radius ramps up, which is the signature of escaping a saddle.

Δ can only double per accepted step, so most of the first outer iterations go into growing the radius.

The Kraus init already avoids this: `perturbed_target_init` starts the non-dominant Kraus operators as small
random matrices.

## Implemented: interior SPAM initialization

`get_initial_gate_set_for_optimization(..., spam_depolarization=ε)` and
`get_compressed_perturbed_rep_from_mgst(..., spam_depolarization=ε)` mix the identity into the targets before factorizing:

    E_i -> (1 - ε) E_i + ε I / num_povm,    ρ -> (1 - ε) ρ + ε I / dim

This preserves Σ E_i = I and Tr ρ = 1. With ε = 1e-2, the smallest eigenvalues are ε/d, and the rank-raising
columns are ~√(ε/d) ≈ 0.05–0.07 instead of 1e-7. The default ε = 0 reproduces the old init exactly.
See the last section of `2q_mgst_vs_current_state.ipynb` for the comparison against the target init.

A related flag, `project_kraus_to_isometry=True`, projects the perturbed Kraus init onto the Stiefel manifold
per gate, using the polar decomposition. It exists because `perturbed_target_init` builds a full-rank channel,
and truncating it to `kraus_rank` < dim² leaves |Σ K†K − I| ≈ 5e-6 for 2Q with rank 4. That is enough for
`are_all_tensors_in_gate_set_isometries` to return False. The flag defaults to False, which keeps the old init.

## Future improvements

Not implemented yet. Each one is independent of the others.

1. **Rejections do not use up the inner budget.** Add `TrustRegionOptions.max_rejections`, so that
   `num_iterations` counts only accepted steps (`run_trust_region_optimization`). A rejected step costs only
   1 HVP and 1 cost evaluation, but with `num_iterations=5` per block it can waste a whole block (2Q State, outer iteration 1).
2. **Function-scaled initial radius.** Δ₀ = clip(c·(f(x₀) − f_floor)/‖g₀‖, Δ_min, max_radius), with c ≈ 0.1, so that
   the linear model removes a fraction c of the reducible cost. `cost_floor` is already computed in
   `run_riemannian_optimization`; pass it through `_optimize_single_operator`. This removes the hand-tuned per-operator
   radii (1e-4 / 1e-3 / 1e-1). For comparison, Manopt and pymanopt use Δ₀ = Δ̄/8, where Δ̄ is the typical distance of
   the manifold. That reflects the manifold's scale only, not the cost's.
   References: Sartenaer, *Automatic determination of an initial trust region in nonlinear programming*, SIAM J. Sci. Comput. 18 (1997);
   Conn, Gould, Toint, *Trust-Region Methods* (2000), §17.2.
3. **Radius update.** The current rule (×0.25 if ρ < 0.25, ×2 if ρ > 0.75 on the boundary) is the standard one
   (Absil, Baker, Gallivan, *Trust-region methods on Riemannian manifolds*, FoCM 2007; Boumal, *An Introduction to
   Optimization on Smooth Manifolds*, Alg. 6.3). Two options in `_update_trust_region_radius_and_xk`:
   - a configurable expansion factor (×2–4 for very successful steps, e.g. ρ > 0.9);
   - shrinking relative to ‖η‖ instead of Δ when tCG stopped inside the region (Conn–Gould–Toint, Ch. 6 and §17.1).
4. **Fisher-preconditioned tCG for State/POVM.** Preconditioned tCG (Absil–Baker–Gallivan 2007; Boumal Ch. 6)
   measures the trust region in the P⁻¹-norm. With P ≈ (Fisher information)⁻¹, the region is approximately the
   Dikin ellipsoid of the self-concordant −log barrier. The radius then has no units (Δ ≈ 1 is about one standard
   deviation), is the same for all operators, and adapts automatically near the PSD boundary. State/POVM have at
   most ~128 real parameters for 2Q, so a dense Fisher matrix plus a Cholesky factorization is cheap.
   This is the principled fix for both the scaling problem and the boundary problem.
   Related: Nocedal, Wright, *Numerical Optimization*, §4.5 (scaled trust regions).
5. **Continuation.** Run a least-squares phase before the log-likelihood, as pyGSTi runs χ² before MLGST. The
   alternative is a log-likelihood regularized below a probability clip (pyGSTi `min_prob_clip`). Both remove the
   barrier during the early iterations.

## Literature on starting inside the PSD cone

- Journée, Bach, Absil, Sepulchre, *Low-rank optimization on the cone of positive semidefinite matrices*, SIAM J. Optim. 20 (2010).
- Boumal, Voroninski, Bandeira, *The non-convex Burer–Monteiro approach works on smooth semidefinite programs*, NeurIPS (2016).
- Řeháček, Hradil, Knill, Lvovsky, *Diluted maximum-likelihood algorithm for quantum tomography*, PRA 75 (2007). Starts at I/d.
- Shang, Zhang, Ng, *Superfast maximum-likelihood reconstruction for quantum tomography*, PRA 95 (2017).
