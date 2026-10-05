# Computational efficiency of the Riemannian trust-region optimization

Notes from the session of 2026-10-01/02. They cover why the optimization *seemed* to get slower after the tCG bug fix, the changes made to the stopping criteria and to the TR loop, and the vectorized + jit-compiled cost function / gradient / Hessian-vector product (HVP) in [`src/mGST/jitted_objective.py`](../../src/mGST/jitted_objective.py). All tables were measured during the session; the scripts in this folder reproduce them.

**TL;DR**

1. The slowdown was **not** caused by the tCG fix, the tangent-space projection or the ratio regularization: on identical data, old and new code cost the same per outer iteration (Table 2). The old runs were faster because the old relative-precision criterion **stopped after 2–3 outer iterations, before convergence** (Table 3).
2. The cost of a run is dominated by eager (op-by-op) automatic differentiation through a Python loop over the gate sequences. Vectorizing the cost over sequences and jit-compiling the Riemannian gradient and HVP makes them **50–100× faster at 1 qubit and 5–20× at 3 qubits**, with compile times **below 2 s even at 3 qubits / 1000 circuits** (Tables 5–6). End to end: **176 s → 10.5 s** for 20 outer iterations at 1q, with an identical trajectory (Table 7).
3. Two new stopping criteria are based on the statistics of the log-likelihood: a relative precision measured against the *reducible* cost `f − f_floor`, and an absolute tolerance (in nats) on the *estimated remaining decrease* `Δf_k q/(1−q)`.

---

## 0. Setup and reproduction

| | |
|---|---|
| Machine | Apple M2 Pro (CPU), JAX 0.4.38, `jax_enable_x64=True` |
| Timing | wall clock, single run, after warm-up unless "first call" is stated. Re-runs vary by roughly ±15 %. |
| Optimization settings (unless stated) | 5 TR iterations per operator, 10 tCG iterations, `tol_grad=1e-8`, log-likelihood, `compute_least_squares=1`, `carry_over_radius=True` |

Datasets (generated with `00_make_data.py`, simulated on `iqmfakeapollo`, 1000 shots, Kraus rank 4, full-rank POVM/state):

| name | gate set | circuits | Kraus tensor | POVM | state | sequence length (mean / max) |
|---|---|---|---|---|---|---|
| 1q | `1QXYI` | 100 | (3, 4, 2, 2) | (2, 2, 2) | (2, 2) | 8.2 / 14 |
| 2q | `2QXYICZ` | 1000 | (6, 4, 4, 4) | (4, 4, 4) | (4, 4) | 8.3 / 14 |
| 3q | `3QXYCZ` | 1000 | (8, 4, 8, 8) | (8, 8, 8) | (8, 8) | 8.2 / 14 |

Scripts (run from this folder; they import `common.py` and read `data/`):

| script | what it measures | tables |
|---|---|---|
| `00_make_data.py {1q,2q,3q}` | generates `data/data_*.pkl` (git-ignored) | – |
| `01_benchmark_tr_calls.py` | time + call counts of a full `run_riemannian_optimization`; old vs new code via `git worktree` (instructions in the docstring); stopping criteria | 2, 3, 8 |
| `02_eager_vs_jit_loop.py` | eager autodiff vs `jax.jit` around the existing loop-based cost (1q only) | 4 |
| `03_scaling_eager_vs_jitted.py` | per-call cost and compile time, eager vs `jitted_objective`, 1q/2q/3q | 5 |
| `04_contraction_strategies.py` | padded scan vs length buckets vs active prefix | 6 |
| `05_end_to_end_use_jit.py` | full optimization with `use_jit=False/True` | 7 |

---

## 1. Diagnosis: "the optimization takes longer since the tCG fix"

### 1.1 What the notebook runs did

Parsed from the outputs of `reconstruction_test/1q_mgst_vs_current_state.ipynb`. All runs used 20 outer iterations and none met a convergence criterion.

**Table 1** – work per notebook run (one tCG inner iteration = one HVP)

| notebook cell | code / schedule | tCG calls | tCG inner iterations | rejected TR steps |
|---|---|---|---|---|
| 23 | oldest run in the notebook (earlier code version), Kraus linearized | 296 | 1035 | 92 |
| 21 | Kraus linearized | 299 | 1061 | 63 |
| 20 | Kraus linearized + Hδ projection | 280 | 1197 | 16 |
| 19 | all linearized | 294 | 1059 | 48 |
| 18 | all linearized + Hδ projection | 295 | 1277 | 26 |

The amount of work changed by at most ~25 %, which cannot explain a "significantly longer" run.

### 1.2 Controlled comparison: old vs new code on identical data

Commit `6d68d64` (before the tCG fix) vs the working tree after `82b38ae`, 1q data, 20 outer iterations, convergence criteria disabled (`01_benchmark_tr_calls.py`).

**Table 2** – full optimization, old vs new code

| schedule | old code | new code | final f (both) |
|---|---|---|---|
| only Kraus linearized | 205.6 s | 198.4 s | 5.18541e4 |
| all operators linearized | 168.6 s | 172.2 s | 5.18540e4 |

**Conclusion:** per outer iteration, the tCG fix, the projection of Hδ and the regularized ratio cost nothing measurable.

The tCG fix does not change the result in exact arithmetic either. The old code computed the boundary step from η_{k+1} = η_k + αδ (already outside the trust region), solving ‖η_{k+1} + τδ‖ = Δ. The relevant root τ is then negative, which explains the frequent `t < 0` warnings, but η_{k+1} + τδ is the *same* boundary point as η_k + τ'δ with τ' > 0 (the new code, as in Steihaug–Toint CG, [Nocedal & Wright, Alg. 7.2]). The fix is about numerical robustness (it removed NaNs), not about the trajectory.

### 1.3 The old relative-precision criterion stopped too early

The old check was `|f_after_kraus − f_after_povm| / (|f| + 1) < relative_precision`, evaluated in the middle of an outer iteration. Running the old code with the criterion enabled:

**Table 3** – old code with the old criterion (1q, Kraus linearized)

| relative_precision | stopped at | time | final f | f after 20 iterations |
|---|---|---|---|---|
| 1e-3 | outer iteration 2 (after Kraus) | 33 s | 5.3396e4 | 5.1854e4 |
| 1e-5 | outer iteration 3 (after Kraus) | 42 s | 5.3130e4 | 5.1854e4 |

The early stops left 1300–1500 nats of log-likelihood on the table (for comparison, the whole statistically irreducible misfit is ~38 nats, Section 3.2). **The old runs were fast because they had not converged.** With the new criterion (Section 3) every run uses all 20 outer iterations, ~3 min eagerly, which is what was being observed.

---

## 2. Changes made

| # | change | where | commit | why | achieved |
|---|---|---|---|---|---|
| 1 | tCG boundary step computed from η_k instead of η_{k+1} | `truncated_conjugate_gradient` | `d43941f` | NaNs on POVM/state; matches the standard Steihaug–Toint step | no NaNs; same boundary point in exact arithmetic (Section 1.2) |
| 2 | Hδ projected onto T_xM inside tCG; removed the projection of δ onto the "tangent space at η" (η is not a point on the manifold) | `truncated_conjugate_gradient` | `82b38ae` | the small normal component of Hδ gets amplified in r once ‖r‖ ≪ ‖αHδ‖ and breaks the CG recurrences | r and δ stay tangent; no cost (Table 2) |
| 3 | regularized quality ratio ρ = (Δf − ε)/(Δm − ε), ε = 10³·eps·max(1, \|f\|) | `compute_quality_quotient` | `82b38ae` | Δf and Δm are differences of ~5e4-sized numbers and lose all precision near convergence: 0/0 or random rejections | robust accept/reject near convergence. Same as `rho_regularization = 1e3` in Manopt/pymanopt [Manopt, pymanopt], following [Conn, Gould, Toint 2000, §17.4.2] |
| 4 | relative precision measured against the reducible cost `f − f_floor`, once per outer iteration | `convergence_criteria_from_relative_precision`, `compute_cost_floor` | `82b38ae` | the old criterion stopped 2–3 iterations in (Table 3) | criterion is relative to what the optimizer can still remove (Section 3.2) |
| 5 | after a rejected TR step, skip recomputing the gradient / re-linearizing (x_k is unchanged) | `run_trust_region_optimization` | `81a14cf` | each rejection wasted one gradient (~0.15 s) or linearization (~0.6 s) at 1q | eager 1q run 198.4 s → 181.4 s, identical trajectory |
| 6 | pass the known f(x_k) into the quality quotient | `compute_quality_quotient(cost_fx=...)` | `81a14cf` | f(x_k) was recomputed each TR iteration | 290 fewer cost evaluations per 1q run (1472 → 1182) |
| 7 | absolute tolerance on the estimated remaining decrease (opt-in) | `convergence_criteria_from_absolute_tolerance`, `run_riemannian_optimization(absolute_tolerance=...)` | `81a14cf` | statistically meaningful, dataset-independent tolerance in nats (Section 3.3) | with 0.1 nats: stops by itself after 38 iterations, 0.87 nats below the 20-iteration value |
| 8 | vectorized, jitted cost / Riemannian gradient / HVP; `use_jit` and `jit_contraction` flags | new `src/mGST/jitted_objective.py`; `run_trust_region_optimization(rgrad_and_euclidean_fn=..., rhvp_and_euclidean_fn=...)`, `_optimize_single_operator(jitted_objective=...)`, `run_riemannian_optimization(use_jit=..., jit_contraction=...)` | **uncommitted** | eager autodiff through a Python loop of jitted kernels was ~99 % of the run time | 1q: 176 s → 10.5 s; 3q: ~6× per outer iteration (Section 4) |
| 9 | Euclidean HVP along the tCG solution accumulated inside tCG, H_e[η] = Σ_j α_j H_e[δ_j] (+ τ H_e[δ] on the boundary), instead of one extra HVP for the quality quotient | `truncated_conjugate_gradient(return_euclidean_hvp=True)`; `return_euclidean` in `riemannian_hessian_vector_fn`, `linearize_riemannian_gradient_and_hvp`, `rhessian_vector_product_from_linear_map` | **uncommitted** | the model m(η) needs H_e[η]; the jitted/linearized paths spent one HVP per TR iteration on it, and the eager path a full eager gradient + HVP (`compute_approx_terms_from_cost_fn`). The HVP is linear in its direction and H_e[δ_j] comes out of the same jvp as the Riemannian H[δ_j], so the sum is free | ρ identical to 10 digits; eager 1q 181.4 s → 153.6 s (`compute_quality_quotient` 20.4 s → 2.2 s); jitted 3q 25.8 s → 23.5 s per 2 outer iterations; jitted 1q 10.5 s → 10.1 s; trajectories unchanged |

---

## 3. Stopping criteria

Notation: f is the negative log-likelihood f = −N Σ_n Σ_i y_{n,i} log p_{n,i}(θ), with N shots, empirical frequencies y and model probabilities p. Δf_k = f_{k−1} − f_k is the decrease over outer iteration k (one POVM → Kraus → state sweep).

### 3.1 Why the old criterion fails for the log-likelihood

`|Δf| / (|f| + 1)` is the classic relative function-change test (e.g. `ftol` in SciPy's L-BFGS-B: (f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ ftol [SciPy]). It assumes that the size of f is the size of what can still be improved. For the log-likelihood that is false: f contains the constant N Σ_n H(y_n) (the empirical entropy), which is ~5.35e4 here, while the part the optimizer can reduce is ~10³ at the start and ~38 at the optimum. Dividing by |f| makes the criterion ~1000× too permissive at the end. Evaluating it after the Kraus step against the value after the POVM step (a partial sweep) made it worse.

### 3.2 Relative precision against the cost floor

The minimum of f over *all* probability assignments is reached at p = y (the "saturated model"):

    f_floor = −N Σ_n Σ_i y_{n,i} log y_{n,i},        f − f_floor = N Σ_n KL(y_n ‖ p_n) ≥ 0.

2(f − f_floor) is the **deviance** of the model [McCullagh & Nelder 1989, Ch. 2]. For a well-specified model at the maximum-likelihood estimate it is asymptotically χ²_k distributed, with k = #independent data − #model parameters (Wilks' theorem [Wilks 1938]). GST uses the same statistic to detect model violation: 2Δlog L compared to χ²_k [Nielsen et al. 2021]. Consequences:

* `f − f_floor` is the part of the cost the optimizer can actually remove, so it is the right denominator: `relative_change = Δf_k / (f_{k−1} − f_floor)`.
* At convergence, `f − f_floor ≈ k/2`. In the 1q notebook run (Table 8) the excess levels off at **≈ 37.5 nats**. For comparison, 100 sequences × (2 − 1) independent outcomes minus a rough count of ~30 gauge-free gate-set parameters gives (100 − 30)/2 ≈ 35 nats. The optimizer reaches the statistical floor.
* Drawback: the denominator tends to k/2, so `relative_precision = 1e-3` means "stop when Δf_k < 1e-3 · k/2 nats" (≈ 0.04 nats here). The meaning of a given value changes with the dataset size and the model, which motivated 3.3.

### 3.3 Absolute tolerance on the estimated remaining decrease (new, opt-in)

**Observation.** In the tail, the decrease per outer iteration shrinks by an almost constant factor (Table 8: q = Δf_k/Δf_{k−1} ≈ 0.71–0.74 for 10 iterations). The alternating (block-coordinate) optimization converges **linearly**, consistent with the linear rates known for block coordinate descent under (local) strong convexity [cf. Beck & Tetruashvili 2013]. Gauge directions are flat directions of f and do not affect Δf.

**Estimate.** If Δf_{k+j} ≈ Δf_k q^j, the decrease still to come is a geometric series:

    f_k − f_∞ ≈ Σ_{j≥1} Δf_k q^j = Δf_k · q / (1 − q).

This is the standard a-posteriori error estimate for linearly convergent iterations: the contraction-mapping bound ‖x_k − x*‖ ≤ q/(1−q) ‖x_k − x_{k−1}‖ [e.g. Kreyszig 1978, §5.1]. It is also the idea behind Aitken's Δ² extrapolation [Aitken 1926], which estimates the limit from the same ratio of successive differences.

**Rule** (`convergence_criteria_from_absolute_tolerance`):

    stop if  max(Δf_k, Δf_k · q/(1−q)) < absolute_tolerance

* The `max` keeps the rule conservative when q < 1/2 (noisy estimates of q early on), because q/(1−q) < 1 there.
* If q ≥ 1 or q is undefined (first iteration, Δf_{k−1} ≤ 0), the remaining decrease is treated as infinite and the rule never fires.

**Why nats, and why ~0.1.** By Wilks' theorem, 2Δf is a likelihood-ratio statistic. Moving one standard deviation along a single parameter direction increases f by 0.5 nats (the usual Δln L = 1/2 rule for likelihood intervals [PDG, Statistics review]). An optimization error of ≲ 0.1 nats is therefore far below the statistical uncertainty of the estimate, whatever the dataset size. The relative criterion has no such interpretation.

**Table 8** – 1q notebook run (cell 18), per outer iteration; floor f_floor = 53481.03

| iteration | f | f − f_floor | Δf | q | Δf·q/(1−q) |
|---|---|---|---|---|---|
| 5 | 53559.45 | 78.42 | 34.21 | 0.18 | 7.75 |
| 8 | 53531.34 | 50.31 | 5.61 | 0.66 | 10.78 |
| 10 | 53524.96 | 43.93 | 2.64 | 0.71 | 6.34 |
| 12 | 53521.80 | 40.77 | 1.31 | 0.71 | 3.18 |
| 14 | 53520.21 | 39.18 | 0.66 | 0.71 | 1.61 |
| 16 | 53519.38 | 38.35 | 0.35 | 0.73 | 0.94 |
| 18 | 53518.93 | 37.90 | 0.19 | 0.73 | 0.52 |
| 20 | 53518.69 | 37.66 | 0.10 | 0.71 | 0.25 |

On this trace the rule stops at iteration 16 for a tolerance of 1.0 nat and at iteration 19 for 0.5, and does not stop within 20 iterations for 0.1. A live run (`01_benchmark_tr_calls.py --outer 40 --absolute-tolerance 0.1`) stopped by itself after **38** outer iterations, at f = 5.185319e4 vs 5.185406e4 after 20 (0.87 nats lower). That took 318 s eagerly, which is affordable with `use_jit=True`.

**For least squares** the cost is not in nats. A natural choice is a fraction of the shot-noise floor `estimate_noise_floor_threshold(..., multiplier=1)`. Not done yet.

---

## 4. The jitted objective: what was done and why it works

### 4.1 Where the time went

`cost_function_jax_mps` loops over the sequences in Python and calls a small jitted kernel per sequence (`log_likelihood_mps_single_gate_sequence_jit`). The arrays are tiny (2×2 to 8×8), so the floating-point work is negligible and the run time is **overhead**:

* every cost evaluation dispatches one XLA call per sequence (100–1000 per evaluation);
* outside `jit`, `jax.grad` / `jax.jvp` / `jax.linearize` run eagerly. JAX traces the Python loop with autodiff tracers and processes every per-sequence call (and its JVP/transpose) separately, with Python bookkeeping for each of them;
* the trust-region code needs ~1 gradient per TR iteration and ~4 HVPs per tCG solve, i.e. ~300 gradients and ~1000 HVPs per 20-iteration 1q run.

**Table 4** – Kraus operator at 1q, per call (`02_eager_vs_jit_loop.py`)

| | eager (current path) | `jax.jit` around the loop-based cost | vectorized + jit (Section 4.3) |
|---|---|---|---|
| cost | 5.2 ms | – | 0.3–0.5 ms |
| Riemannian gradient | 178 ms | 2.6 ms, **compile 7.7 s** | 1.6 ms, compile 0.12 s |
| `linearize` (once per TR iteration) | 601 ms | – | not needed |
| apply linear map (once per tCG iteration) | 34 ms | – | – |
| HVP, non-linearized | 237 ms | 9.2 ms, **compile 46 s** | 4.5 ms, compile 0.25 s |

### 4.2 Why jitting the existing code compiles slowly

`jax.jit` traces the Python function once with abstract values and hands the resulting program to XLA. **Python control flow is executed during tracing, so a Python `for` loop is unrolled** [JAX docs, "Control flow and logical operators with JIT"]. The loop over sequences and the inner loop over gates turn the cost into a program with one contraction per gate of every sequence (Σ_n ℓ_n ≈ 8.2 × #circuits einsums). The gradient roughly triples it and the HVP (forward-over-reverse) roughly doubles that again. XLA compile time grows at least linearly with program size: 7.7 s / 46 s at 1q with 100 circuits, and an order of magnitude more at 1000 circuits. This matches the earlier experience that "jitting the gradient and HVP was too expensive for more qubits". The cause was the **number of circuits** that got unrolled, not the number of qubits.

### 4.3 Vectorization: one batched contraction for all sequences

The model probability of POVM element i for sequence n with gates g_{n,1}, …, g_{n,ℓ_n} (applied in that order) is

    p_{n,i} = Tr[ E_i  Φ_{g_{n,ℓ_n}} ∘ … ∘ Φ_{g_{n,1}}(ρ) ],   Φ_g(ρ) = Σ_r K_{g,r} ρ K_{g,r}^†,   E_i = A_i^† A_i,   ρ = B B^†.

Instead of contracting sequence by sequence, all sequences are contracted at once. The state is a batch ρ_b (shape `(num_seq, d, d)`), and at position t every sequence gathers its own Kraus operators `K[gates[:, t]]` (shape `(num_seq, r, d, d)`) and applies them in one batched einsum:

```python
jnp.einsum("bijk,bkl,biml->bjm", K[gates[:, t]], rho, K[gates[:, t]].conj())
```

The final probabilities are one more einsum, `"ijk,bkl,ijl->bi"`. The program now contains one einsum per *position*, not per gate of every sequence, and each einsum does useful batched work. Sequences of different lengths are handled in one of two ways (`jit_contraction`):

**`"padded"`** – pad every sequence to the maximal length L with the index of an identity channel (K_0 = I, the other Kraus operators 0, appended to the Kraus tensor) and loop over positions with `jax.lax.scan`. `scan` traces its body **once** and compiles it as a loop [JAX docs, `jax.lax.scan`], so the program size and compile time are independent of the number of circuits, of L and of the number of qubits (which only changes shapes). Cost: the identities are multiplied too, and only Σℓ / (num_seq · L) ≈ 58–59 % of the work is real gates.

**`"prefix"` (default)** – the identities cannot be skipped at run time per sequence: inside `vmap`/batched code a branch becomes a `select` that evaluates both sides. But **the sequences are data, known before tracing**, so the skipping can be decided at trace time. Sort the sequences by decreasing length. At position t, exactly the first n_t = #{n : ℓ_n > t} sequences still have a gate. n_t is a plain Python integer, so `rho.at[:n_t].set(apply(K[gates[:n_t, t]], rho[:n_t]))` has static shapes and compiles fine. The loop over t is unrolled (L bodies, L = 14 here), so compile time grows with the maximal length but not with the number of circuits, and the work is exactly Σℓ gate applications: no identities. In general, **a data-dependent decision that is fixed for the whole optimization can be moved from run time to trace time.**

A third option, length buckets (padding within groups of similar length, one `scan` per bucket), sits in between (Table 6) and is implemented only in `04_contraction_strategies.py`.

### 4.4 Gradient and HVP inside `jit`

* **Gradient.** `riemannian_and_euclidean_gradient_fn` (Euclidean gradient via `jax.grad`, conjugated to df/dx* following the Wirtinger-calculus convention [Hunger 2007], then projected onto the tangent space of the Stiefel manifold) is reused unchanged, but called *inside* `jax.jit`. Forward pass, reverse pass and projection become a single XLA program.
* **HVP.** Forward-over-reverse: `jax.jvp` of the gradient function [Pearlmutter 1994; JAX autodiff cookbook, "Hessian-vector products"]. One `jvp` gives both the gradient at x and its directional derivative D grad f(x)[z]. The Riemannian HVP is then assembled with the Riemannian connection of the Stiefel manifold (`riemannian_connection`, Eq. 5.4 of arXiv:2009.10159 as cited in `riemannian.py`; see also [Absil, Mahony, Sepulchre 2008, §5.5] and [Boumal 2023, Ch. 7]).
* **Euclidean HVP for free.** The `jvp` of `(rgrad, egrad)` also returns the Euclidean HVP D egrad(x)[z], which is what the second-order model m(η) in the quality quotient needs (`compute_model_approximation`, after [Manton 2002, Eqs. 31–33]). `rhvp_fn` returns it. Because the HVP is linear in its direction, tCG accumulates H_e[η] = Σ_j α_j H_e[δ_j] next to η = Σ_j α_j δ_j (change 9), so the quality quotient needs no HVP of its own.
* **Why `linearize` is no longer needed.** `jax.linearize` pays the forward pass once and returns a linear map whose applications are cheaper. That amortization mattered eagerly (601 ms once + 34 ms per application vs 237 ms per HVP). Inside `jit` a full `jvp` (primal + tangent) costs 4.5 ms, so recomputing the primal each time is cheaper than any eager linear map. The linear map returned by `linearize` is also a Python closure over residuals, which cannot be returned from a jitted function as easily.

### 4.5 Making sure compilation happens once

`jax.jit` caches compiled programs by the *function object* and the shapes/dtypes of its arguments [JAX docs, "Just-in-time compilation", caching]. Two design decisions follow:

1. **The optimized and the fixed operators are arguments**, never closed-over arrays: `rgrad_fn(kraus, povm, state)`, `rhvp_fn(kraus, povm, state, z)`. The old pattern, `cost_fn_x = lambda x: cost(kraus=x, povm=povm_psd, ...)`, creates a new function each TR call. Jitting such a lambda would recompile every time, and its closed-over operators would become constants baked into the program.
2. **The data are compile-time constants on purpose.** The gate indices, the sort order, n_t and the probabilities never change during a run, so baking them in is free and is what makes the prefix trick possible.

`JittedObjective` holds the jitted cost and lazily builds `(rgrad_fn, rhvp_fn)` per `(operator_type, metric)`, so each is compiled once per run. `bind(operator_type, kraus, povm, state, metric)` returns ordinary Python closures with the two other operators fixed. They are cheap and call the *same* jitted functions, so they don't trigger recompilation. They plug into `run_trust_region_optimization` through the new arguments `rgrad_and_euclidean_fn` / `rhvp_and_euclidean_fn`.

### 4.6 Correctness safeguards

* The vectorized cost matches `cost_function_jax_mps` to ~1e-16 (relative) and the jitted gradient / HVP match the eager ones to ~1e-15 (Tables 5, 6).
* End-to-end trajectories are identical: the final f agrees to all printed digits (5.18540613e4 for eager, prefix and padded at 1q; Table 7).
* `JittedObjective.validate` compares the jitted cost with the user's `cost_function` at the initial point and raises a `ValueError` if they differ. The jitted objective re-implements only `cost_function_jax_mps` (log-likelihood or least squares), and this prevents silently optimizing a different objective, e.g. a regularized cost. Tested with a cost scaled by 1.01.

### 4.7 Results

**Table 5** – per call, eager vs jitted with the **padded** contraction (`03_scaling_eager_vs_jitted.py --contraction padded`); jitted compile time in brackets

| | operator | eager rgrad | eager `linearize` + apply | eager HVP | jitted rgrad | jitted HVP |
|---|---|---|---|---|---|---|
| 1q, 100 seq | Kraus | 164 ms | 575 + 32 ms | 186 ms | 1.6 ms (0.11 s) | 4.6 ms (0.17 s) |
| | POVM | 97 ms | 161 + 11 ms | 72 ms | 0.5 ms (0.06 s) | 0.5 ms (0.09 s) |
| 2q, 1000 seq | Kraus | 1826 ms | 5934 + 370 ms | 2470 ms | 34 ms (0.26 s) | 91 ms (0.51 s) |
| | POVM | 633 ms | 1488 + 120 ms | 842 ms | 11 ms (0.12 s) | 11 ms (0.19 s) |
| 3q, 1000 seq | Kraus | 1662 ms | 6162 + 338 ms | 2680 ms | 130 ms (0.41 s) | 306 ms (0.86 s) |
| | POVM | 505 ms | 1468 + 120 ms | 733 ms | 43 ms (0.14 s) | 47 ms (0.22 s) |

Relative differences between the eager and jitted results were ≤ 9e-15 in all cases.

**Table 6** – contraction strategies, jitted, Kraus operator (`04_contraction_strategies.py`); compile time in brackets

| | padded scan | 4 length buckets | active prefix |
|---|---|---|---|
| 1q rgrad | 3.0 ms (0.24 s) | 1.6 ms (0.28 s) | 2.9 ms (0.39 s) |
| 1q HVP | 7.3 ms (0.34 s) | 3.6 ms (0.52 s) | 5.4 ms (0.94 s) |
| 3q rgrad | 121 ms (0.43 s) | 90 ms (0.61 s) | **70 ms** (0.87 s) |
| 3q HVP | 292 ms (0.84 s) | 217 ms (1.27 s) | **161 ms** (1.92 s) |
| useful work | 58–59 % | in between | 100 % |

At 1q all variants are a few ms (overhead-dominated) and the ordering fluctuates between runs. At 3q the work is FLOP-dominated and the prefix contraction is 1.7–1.8× faster than padding, as expected from 59 % useful work.

**Table 7** – end to end through `run_riemannian_optimization` (`05_end_to_end_use_jit.py`), times include compilation

| run | `use_jit=False` | `use_jit=True, "prefix"` | `use_jit=True, "padded"` | final f |
|---|---|---|---|---|
| 1q, 20 outer iterations | 176.0 s | **10.5 s** | 11.5 s | 5.18540613e4 (all three) |
| 3q, 2 outer iterations | 162.0 s | **25.8 s** | 35.8 s | 1.59192021e6 (all three) |

### 4.8 Recipe for future workflows

1. **Measure first.** Count calls and time them (`01_benchmark_tr_calls.py`) before assuming an algorithmic change made things slower. Here the cause was the iteration count, not the cost per iteration.
2. **Small arrays + Python loops = overhead-bound.** If the tensors are tiny, eager JAX spends its time on dispatch and tracing bookkeeping, not on arithmetic.
3. **Do not `jit` a Python loop over data items.** It gets unrolled and compile time grows with the data. Express the loop over data as a **batch dimension** (batched einsum or `vmap`) and the loop over time/depth as `lax.scan` (constant compile time) or as a short unrolled loop over a *small* static range.
4. **Move data-dependent structure to trace time.** Anything fixed for the whole run (sequence lengths, sort orders, masks, active sets) can be precomputed in NumPy and used as static shapes and slices. That is how identity padding was eliminated.
5. **Jit the whole derivative**, not just the cost: `jit(grad(f))` and `jit(jvp(grad(f)))` are single programs; `grad(jit(f))` evaluated eagerly is not.
6. **Pass changing arrays as arguments, bake constant data in.** Build the jitted functions once per run and bind fixed arguments with cheap Python closures *outside* `jit`.
7. **HVPs: forward-over-reverse** (`jvp` of `grad`). Inside `jit` it is cheap enough that `linearize`-style amortization isn't needed. Return any by-products you need, like the Euclidean HVP for the TR model.
8. **Validate** the fast path against the reference implementation (values and full trajectories) and keep a flag to switch back (`use_jit`).

---

## 5. Still to do, and what to expect

| item | why | expected effect |
|---|---|---|
| Commit `jitted_objective.py`, the `use_jit` wiring and the accumulated Euclidean HVP (changes 8, 9) | uncommitted | – |
| Unit tests for `jitted_objective` (NLL and LS, Kraus rank 1 and > 1, both contractions, vs `cost_function_jax_mps`) and for `convergence_criteria_from_absolute_tolerance` | no tests cover `trust_region.py` / `jitted_objective.py` | regression safety for the paper results |
| Jit the whole tCG loop (`lax.while_loop`, `lax.cond` for negative curvature / boundary) | each tCG iteration still runs ~20 tiny eager ops and host syncs (`if curvature < 0`) between HVPs | likely the dominant remaining cost at 1q (10.5 s per run, not profiled); smaller gain at 3q where HVP FLOPs dominate |
| Profile the 3q jitted run (25.8 s for 2 outer iterations) | breakdown not measured yet | decide between tCG jitting and contraction work |
| Reuse a `JittedObjective` across calls of `run_riemannian_optimization` (optional argument) | every call recompiles (~1–3 s at 3q per operator) | removes repeated compilation in notebook sweeps |
| Jitted top-level cost evaluations (initial cost, `compute_least_squares` values) | still use the loop-based `cost_function` | minor: a few calls per outer iteration |
| Long sequences: bound the prefix unroll (chunks of positions with `scan`, or buckets) | prefix compile time grows with the maximal length L | keeps compile time bounded if L ≫ 14 |
| Memory check at 4–5 qubits | reverse mode stores ρ for every position: num_seq × L × d² complex values (3q: ~14 MB, 5q: ~230 MB for 1000 × 14) | probably fine; use `jax.checkpoint` if not |
| GPU | the batched einsums are GPU-friendly | untested |
| Default `absolute_tolerance` (e.g. 0.1 nats) and an LS analogue (fraction of the shot-noise floor) | currently opt-in, LS has no natural unit | dataset-independent stopping |
| `isinstance(compute_least_squares, int)` is also true for `True` | `compute_least_squares=True` computes LS after every operator | cosmetic |
| GDS path (`GradientDescentOptions`) is not jitted | only TR uses the jitted objective | – |
| Remove the superseded prototypes in `playground/time_complexity/` (`jit_prototype.py`, `end_to_end_jit.py`, `scaling.py`) | replaced by the module and the scripts here | – |

---

## References

* [Absil, Baker, Gallivan 2007] P.-A. Absil, C. G. Baker, K. A. Gallivan, *Trust-region methods on Riemannian manifolds*, Found. Comput. Math. 7, 303 (2007).
* [Absil, Mahony, Sepulchre 2008] P.-A. Absil, R. Mahony, R. Sepulchre, *Optimization Algorithms on Matrix Manifolds*, Princeton University Press (2008).
* [Aitken 1926] A. C. Aitken, *On Bernoulli's numerical solution of algebraic equations*, Proc. R. Soc. Edinburgh 46, 289 (1926).
* [Beck & Tetruashvili 2013] A. Beck, L. Tetruashvili, *On the convergence of block coordinate descent type methods*, SIAM J. Optim. 23, 2037 (2013).
* [Boumal 2023] N. Boumal, *An Introduction to Optimization on Smooth Manifolds*, Cambridge University Press (2023), https://www.nicolasboumal.net/book (tCG: Alg. 6.4; stopping tolerances: §6.4).
* [Brieger et al. 2023] R. Brieger, I. Roth, M. Kliesch, *Compressive gate set tomography*, PRX Quantum 4, 010325 (2023) (mGST).
* [Conn, Gould, Toint 2000] A. R. Conn, N. I. M. Gould, P. L. Toint, *Trust-Region Methods*, SIAM (2000), §17.4.2 (regularized ratio).
* [Hunger 2007] R. Hunger, *An introduction to complex differentials and complex differentiability*, Technical report, TU München (2007) (cited in `riemannian_hessian_vector_fn`).
* [JAX docs] https://docs.jax.dev: "Just-in-time compilation" (tracing, caching), "Control flow and logical operators with JIT" (Python loops are unrolled), `jax.lax.scan`, `jax.linearize`, "The Autodiff Cookbook" (Hessian-vector products, forward-over-reverse).
* [Kreyszig 1978] E. Kreyszig, *Introductory Functional Analysis with Applications*, Wiley (1978), §5.1 (Banach fixed-point theorem, a-posteriori error estimate).
* [Manopt, pymanopt] `trustregions` with `rho_regularization = 1e3`: https://www.manopt.org, https://pymanopt.org/docs/stable/_modules/pymanopt/optimizers/trust_regions.html (also linked in `run_trust_region_optimization`).
* [Manton 2002] J. H. Manton, *Optimization algorithms exploiting unitary constraints*, IEEE Trans. Signal Process. 50, 635 (2002).
* [McCullagh & Nelder 1989] P. McCullagh, J. A. Nelder, *Generalized Linear Models*, 2nd ed., Chapman & Hall (1989) (deviance).
* [Nielsen et al. 2021] E. Nielsen et al., *Gate Set Tomography*, Quantum 5, 557 (2021) (2Δlog L vs χ²_k model-violation test).
* [Nocedal & Wright] J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006), Alg. 7.2 (CG–Steihaug).
* [PDG] Particle Data Group, *Review of Particle Physics*, "Statistics" review (likelihood intervals, Δln L = 1/2).
* [Pearlmutter 1994] B. A. Pearlmutter, *Fast exact multiplication by the Hessian*, Neural Comput. 6, 147 (1994).
* [SciPy] `scipy.optimize.minimize(method="L-BFGS-B")`, option `ftol`.
* [Steihaug 1983] T. Steihaug, *The conjugate gradient method and trust regions in large scale optimization*, SIAM J. Numer. Anal. 20, 626 (1983); [Toint 1981] P. L. Toint, *Towards an efficient sparsity exploiting Newton method for minimization*, in *Sparse Matrices and Their Uses*, Academic Press (1981).
* [Wilks 1938] S. S. Wilks, *The large-sample distribution of the likelihood ratio for testing composite hypotheses*, Ann. Math. Statist. 9, 60 (1938).
