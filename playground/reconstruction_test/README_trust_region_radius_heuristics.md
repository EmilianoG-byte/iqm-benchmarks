# Trust region radius heuristics

How `radius_init` (Δ₀) and `max_radius` (Δ̄) are chosen automatically in `run_trust_region_optimization`,
where the heuristics come from, and what they changed in practice.

Companion to `README_rtr_initial_iterations.md`, which deals with the *initialization* side of the same problem.

## TL;DR

Hand-tuned radii (`radius_init=1e-1` for Kraus, `1e-3` for POVM/State) wasted most of the budget ramping Δ up.
Resolving Δ₀ and Δ̄ from the geometry of the manifold or from the initial gradient norm cut the runtime roughly
in half and — for the first time in this notebook — made the optimization **stop on a convergence criterion**
instead of running into the outer-iteration cap.

---

## 1. `typical_dist`: the length scale of the manifold

Every point of the Stiefel manifold has the same Frobenius norm:

$$X \in \mathrm{St}(n,p) \implies \|X\|_F^2 = \operatorname{tr}(X^H X) = \operatorname{tr}(I_p) = p .$$

So the manifold lives on a sphere of radius $\sqrt p$ in $\mathbb{R}^{n\times p}$, and $\|X - Y\|_F \le 2\sqrt p$.
For a product of $k$ independent factors the squared norms add, giving $\sqrt{kp}$.

Pymanopt encodes exactly this ([`pymanopt/manifolds/stiefel.py`](https://pymanopt.org/docs/stable/_modules/pymanopt/manifolds/stiefel.html#Stiefel)):

```python
@property
def typical_dist(self):
    return np.sqrt(self._p * self._k)
```

Note what it is **not**: the intrinsic dimension of $\mathrm{St}(n,p)^k$ is $k\left(np - \tfrac{p(p+1)}{2}\right)$,
and pymanopt only falls back to `sqrt(manifold.dim)` for manifolds that do not implement `typical_dist`.
Stiefel overrides it precisely because the manifold is bounded: `sqrt(dim)` would wildly overestimate how far
one can travel.

### Mapping onto the mGST blocks

`get_isometry_dimensions_from_tensor` now returns `(n, p, k)`, with `k = tensor.size // (n * p)` — the same
leading dimension `tensor_to_isometry` reshapes to. Per block:

| block | tensor shape | n | p | k | `typical_dist` |
|---|---|---|---|---|---|
| Kraus | `(num_gates, rank, d, d)` | `rank·d` | `d` | `num_gates` | $\sqrt{G\,d}$ |
| POVM  | `(num_povm, rank, d)` | `num_povm·rank` | `d` | 1 | $\sqrt d$ |
| State | `(d, rank)` | `d·rank` | 1 | 1 | 1 |

The Kraus tensor is a **product** of `num_gates` Stiefel factors (the constraint $\sum_k K_{g,k}^H K_{g,k} = I$
holds per gate), which is why `k` matters. The POVM is a single stacked isometry ($\sum_i E_i = I$ couples all
effects), and the State is the $p=1$ case — a unit vector, consistent with $\operatorname{tr}\rho = 1$.

### Canonical metric correction

Pymanopt's Stiefel uses the plain Euclidean metric (`inner_product` is a bare `tensordot`), so its
`typical_dist` is Euclidean. Our tCG can also run under the canonical metric, and Δ is measured in whichever
metric is selected (`compute_step_size`, `norm_solution >= radius`), so the scale has to match.

For $z = xA + x_\perp B$ with $A$ skew-Hermitian:

$$\|z\|_e^2 = \|A\|_F^2 + \|B\|_F^2, \qquad \|z\|_c^2 = \tfrac12\|A\|_F^2 + \|B\|_F^2 ,$$

because $I_n - \tfrac12 xx^H$ removes half of the component in $x$'s own column space — those rotational
directions are double-counted by the Euclidean metric, since the constraint $x^Hx = I_p$ is symmetric.
Hence $\|z\|_e/\sqrt2 \le \|z\|_c \le \|z\|_e$, with equality to the lower bound for purely rotational
directions. Equivalently, applied formally to $x$ itself: $\operatorname{tr}(x^Hx) - \tfrac12\operatorname{tr}(x^Hxx^Hx) = p - p/2$.

$$\text{typical\_dist} = \sqrt{k\,p}\ (\text{euclidean}), \qquad \sqrt{k\,p/2}\ (\text{canonical}).$$

## 2. Where Δ₀ = Δ̄/8 comes from

From [`pymanopt/optimizers/trust_regions.py`](https://pymanopt.org/docs/latest/_modules/pymanopt/optimizers/trust_regions.html):

```python
if Delta_bar is None:
    try:
        Delta_bar = manifold.typical_dist
    except NotImplementedError:
        Delta_bar = np.sqrt(manifold.dim)
if Delta0 is None:
    Delta0 = Delta_bar / 8
```

The `/8` is **a code convention, not a theorem**. The lineage is GenRTR (Absil–Baker–Gallivan, 2004–2007) →
Manopt → Manopt.jl → pymanopt. The RTR convergence theory holds for *any* Δ₀ ∈ (0, Δ̄], so Δ₀ only affects
efficiency. The same file notes that the radius doubles on successful steps and quarters on failures, so a
too-small Δ₀ costs only O(log) iterations — which is exactly the cost we were paying.

Pymanopt even warns about this at runtime: after 5 consecutive `TR+` or `TR-` it prints
*"Consider increasing/decreasing options.Delta_bar by an order of magnitude"*.

## 3. The gradient-based alternative

Classical trust-region practice sets Δ₀ from the first-order information instead:
Δ₀ = ‖g₀‖, or 0.1‖g₀‖, or by probing curvature along −g₀. Surveyed in
**Conn, Gould & Toint, _Trust-Region Methods_, SIAM (2000), §17.2**.

Function-aware, which is the point: `typical_dist` knows nothing about the cost. The failure mode is the
mirror image — where the log-likelihood is steep (model probabilities near zero, see
`README_rtr_initial_iterations.md`) ‖g₀‖ explodes and Δ₀ with it.

Hence the three modes implemented in `resolve_trust_region_radii`:

| `radius_heuristic` | Δ₀ | Δ̄ |
|---|---|---|
| `"typical_distance"` | `typical/8` | `typical` |
| `"gradient"` | `0.1·‖r∇f(x₀)‖` | `typical` |
| `"auto"` (default) | `min(0.1·‖r∇f(x₀)‖, typical/8)` | `typical` |

Δ̄ always comes from the geometry: the gradient carries no information about how far one can travel.
Δ₀ is then clamped into `[1e-8·Δ̄, Δ̄]`, which guards the saddle case where ‖g₀‖ ≈ 0 (zero POVM/State columns)
and an unclamped gradient rule would return Δ₀ = 0.

The same book (§17.4.2) is already the source of the `rho_reg` regularization in `compute_quality_quotient`,
and pymanopt cites it in the same place.

## 4. Measured effect (2 qubits, rank 4, 1000 sequences × 1000 shots)

Gate set `2QXYICZ` (6 gates, d = 4), interior SPAM init with ε = 1e-2, all from
`2q_mgst_vs_current_state.ipynb`.

### Resolved radii

| block | `typical_distance` Δ₀ | `gradient` Δ₀ | Δ̄ |
|---|---|---|---|
| POVM  | 2.50e-01 | 3.12e-02 | 2.00 ( = $\sqrt4$ ) |
| Kraus | 6.12e-01 | 7.65e-02 | 4.90 ( = $\sqrt{24}$ ) |
| State | 1.25e-01 | 2.50e-01 | 1.00 |

### Outcome

| run | Δ₀ source | outer iters | convergence reason | wall time |
|---|---|---|---|---|
| baseline, target SPAM | hand-tuned 1e-1 / 1e-3 | **20/20** | *none met* | 1 min 16.8 s |
| baseline, interior SPAM | hand-tuned 1e-1 / 1e-3 | **20/20** | *none met* | 1 min 15.8 s |
| manual pymanopt cell | `typical/8` | 10/20 | rel. change 9.30e-04 < 1e-3 🎯 | 37 s |
| `radius_heuristic="typical_distance"` | `typical/8` | 10/20 | rel. change 9.30e-04 < 1e-3 🎯 | 38.5 s |
| `radius_heuristic="gradient"` | `0.1·‖g₀‖` | 11/20 | rel. change 8.04e-04 < 1e-3 🎯 | 43.5 s |

Three things worth recording:

1. **The convergence criterion actually fires.** Both baselines exhausted all 20 outer iterations with
   `"No convergence criteria met yet ⏳"`. Every heuristic run stopped early on
   `convergence_criteria_from_relative_precision`. The optimizer was never badly conditioned — it was simply
   spending its budget growing Δ instead of descending.

2. **The built-in `"typical_distance"` reproduces the hand-rolled cell bit-for-bit.** Carried radii
   `{kraus: 0.15309310892394862, povm: 0.25, state: 0.125}` versus the manual cell's
   `{kraus: Array(0.15309311), povm: Array(0.25), state: Array(0.125)}` — same numbers, now plain Python floats
   rather than traced JAX scalars.

3. **POVM and State never needed to move their radius.** Under `typical_distance` they end at exactly their
   Δ₀ (0.25 and 0.125). Compare the interior-SPAM baseline, which after 20 outer iterations had only crawled to
   `{povm: 0.128, state: 0.008}` from 1e-3 — the State radius is **16× smaller** than where the heuristic
   simply starts.

The two heuristics are close enough in outcome that the choice is not critical here; `"auto"` takes the smaller
of the two, which in this instance means the gradient value for POVM/Kraus and the geometric value for State.

### Interaction with `carry_over_radius`

The heuristics only fire on the first outer iteration per block: `_optimize_single_operator` passes the carried
radius as `radius_init_override`, which is non-`None` from the second iteration on. Δ̄ is not carried and is
re-resolved each call — deterministically, since it depends only on `(p, k, metric)`. Visible in the logs: the
Kraus block prints `Δ0 = 6.12e-01` on the first outer iteration and `Δ0 = 3.06e-01` (the carried, halved value)
on the second.

## 5. What was *not* done

- **Sartenaer's automatic initial radius** (*SIAM J. Sci. Comput.* **18**(6), 1997), which probes curvature
  along −g₀ at a few trial radii. It needs 2–3 extra cost evaluations per block — cheap with the jitted MPS
  contraction, and it would remove `gradient_fraction` as a knob. Worth revisiting if the current heuristics
  ever misfire.
- Carrying Δ̄ across outer iterations.
- Changing the ρ-based expansion/contraction rules in `_update_trust_region_radius_and_xk`.

## Sources

- Absil, Baker & Gallivan, *Trust-region methods on Riemannian manifolds*, **Foundations of Computational
  Mathematics 7(3), 303–330 (2007)**, [doi:10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
  — RTR and the Δ̄/Δ₀/ρ′ parameters; global convergence for any Δ₀ ∈ (0, Δ̄].
- Conn, Gould & Toint, *Trust-Region Methods*, SIAM (2000) — §17.2 initial-radius heuristics (the `0.1·‖g₀‖`
  family), §17.4.2 the regularized ρ ratio.
- Boumal, *An Introduction to Optimization on Smooth Manifolds*, CUP (2023), Ch. 6
  — [nicolasboumal.net/book](https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf).
- Pymanopt `Stiefel.typical_dist`:
  <https://pymanopt.org/docs/stable/_modules/pymanopt/manifolds/stiefel.html#Stiefel>
- Pymanopt `TrustRegions.run` (`Delta_bar`, `Delta0 = Delta_bar / 8`, the TR± warnings):
  <https://pymanopt.org/docs/latest/_modules/pymanopt/optimizers/trust_regions.html>
- Manopt.jl trust regions (same defaults, Julia side):
  <https://manoptjl.org/v0.2/solvers/trust_regions.html>
- Sartenaer, *Automatic determination of an initial trust region in nonlinear programming*,
  **SIAM J. Sci. Comput. 18(6), 1788–1803 (1997)**.

## Code

- `resolve_trust_region_radii`, `compute_typical_distance`, `RADIUS_HEURISTICS`, `RADIUS_FLOOR_FRACTION`
  — `src/mGST/trust_region.py`
- `TrustRegionOptions.radius_init` / `.max_radius` (now `None` by default) and `.radius_heuristic`
  — `src/mGST/typing.py`
- `get_isometry_dimensions_from_tensor` returning `(n, p, k)` — `src/mGST/utility_functions_comparisons.py`
- `tests/unit/test_trust_region_radius.py`
