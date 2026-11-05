"""Parametric bootstrap of the maximum-likelihood estimate (Nielsen et al. 2021, Secs. 6.3.2 and 6.3.5).

In the bootstrap world the estimate theta_hat plays the role of the truth:
  fit:     draw B new datasets from theta_hat (same design, same number of shots) and rerun exactly the estimation
           procedure of script 07 on each (same initial point, same optimizer settings). Replicates are saved to
           data/bootstrap/ and existing ones are skipped, so the step can be resumed or extended.
  analyse: gauge-fix every replicate with the same Pi as script 13 (Hilbert-space unitary closest to the target), then
           for the test probabilities, Pauli transfer matrix elements and infidelities report
           - the bootstrap standard error s (standard deviation over replicates) and bias (mean over replicates - estimate),
           - s compared with the actual spread of the estimator over the 40 independent datasets of script 07,
           - s compared with the Fisher-information width of script 13 (consistent gauge gradient, plug-in),
           - whether the truth lies in the standard, percentile and basic intervals for this dataset,
           - the coverage obtained when s from this dataset is used as the width for all 40 estimates (a proxy that
             assumes s varies little between datasets).

Usage:
    python 14_parametric_bootstrap.py fit <estimate_index> <num_replicates>          # about 22 s per replicate
    python 14_parametric_bootstrap.py analyse <estimate_index> [<estimate_index> ...]
"""
import glob
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst, get_initial_gate_set_for_optimization
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices, compute_probability_matrices, run_optimization_workflow
from mGST.typing import TrustRegionOptions, OperatorSchedule
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "bootstrap"
num_gates, kraus_rank, dim, m, N = 6, 4, 4, 1000, 400
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
truth = tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))
indices_dict = generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)
load = lambda f: tuple(jnp.asarray(np.load(f)[k]) for k in ("kraus", "povm", "state"))
estimate_file = lambda j: DATA / "crb_estimates" / f"default_{j}.npz"


def fit(j, num_replicates):
    """Resample from estimate j and refit with the settings of script 07 ('default': absolute tolerance 1, 20 outer iterations)."""
    OUT.mkdir(parents=True, exist_ok=True)
    K, A, B = load(estimate_file(j))
    target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
    init = get_initial_gate_set_for_optimization(target_superops=target_superops, kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim,
                                                 spam_depolarization=1e-2, project_kraus_to_isometry=True)
    tr_options = TrustRegionOptions(num_iterations=5, verbose=False, verbose_cg=False, num_iterations_cg=10, tol_grad=1e-8, radius_heuristic="typical_distance")
    schedule = OperatorSchedule.default(20, tr_options)
    options = {"schedule": {op: schedule for op in ["povm", "kraus", "state"]}, "num_iterations_outer": 20, "relative_precision": 1e-3,
               "global_gradient_norm_tol": 1e-10, "noise_threshold": None, "compute_least_squares": 1, "optimization_verbose": False, "absolute_tolerance": 1.0}
    for b in range(num_replicates):
        path = OUT / f"est{j}_b{b}.npz"
        if path.exists():
            continue
        t0 = time.time()
        probability_matrices = compute_probability_matrices(K, A, B, gate_indices=indices_dict["gate_indices"], num_shots=m, seed=100000 + 1000 * j + b)
        result = run_optimization_workflow(num_shots=m, init_operators=init, operators_psd_true={"kraus": K, "povm": A, "state": B},
                                           target_superops=target_superops, optimization_options=options, use_log_likelihood=True,
                                           indices_dict=indices_dict, probability_matrices=probability_matrices, gauge_optimize=False)
        op = result["optimized_operators"]
        np.savez(path, kraus=np.asarray(op["kraus"]), povm=np.asarray(op["povm"]), state=np.asarray(op["state"]),
                 final_cost=float(result["cost_fn_history"]["main"][-1]), reason=str(result["convergence_reason"]))
        print(f"estimate {j}, replicate {b}: {time.time() - t0:.0f}s, {result['convergence_reason']}", flush=True)


# ---------- analysis: gauge fixing and quantities, as in script 13 ----------
def pad(indices_list):
    gates = np.full((len(indices_list), max(len(s) for s in indices_list)), -1, dtype=np.int32)
    for row, s in enumerate(indices_list):
        gates[row, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
    return jnp.asarray(gates)

train = pad(indices_dict["gate_indices"])
test = pad([list(c) for c in np.random.default_rng(2024).integers(0, num_gates, size=(200, 10))])
shapes = [x.shape for x in truth]
to_vec = lambda K, A, B: ga._operators_to_real_vector(K, A, B)
to_ops = lambda theta: ga._real_vector_to_operators(theta, shapes)
pauli1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
paulis = jnp.asarray([np.kron(a, b) for a in pauli1 for b in pauli1])
ptm = lambda K: jnp.real(jnp.einsum("iab,grbc,jcd,grad->gij", paulis, K, paulis, K.conj())) / dim
R_target = ptm(jnp.asarray(target_operators["kraus"]))
h0 = jnp.zeros(15)

def act(h, theta):
    K, A, B = to_ops(theta)
    U = expm(1j * jnp.einsum("j,jab->ab", h, paulis[1:]) / 2)
    return to_vec(jnp.einsum("ab,grbc,dc->grad", U, K, U.conj()), A @ U.conj().T, U @ B)

gauge_cost = lambda h, theta: jnp.sum((ptm(to_ops(act(h, theta))[0]) - R_target) ** 2)
cost_and_grad = jax.jit(jax.value_and_grad(gauge_cost))

def gauge_fix(theta):
    for _ in range(2):
        fun = lambda h: tuple(np.asarray(v, dtype=float) for v in cost_and_grad(jnp.asarray(h), theta))
        theta = act(jnp.asarray(minimize(fun, np.zeros(15), jac=True, method="BFGS", options=dict(gtol=1e-12, maxiter=2000)).x), theta)
    return theta

def gauge_dependent(theta):
    R = ptm(to_ops(theta)[0])
    return jnp.concatenate([R.reshape(-1), 1 - jnp.einsum("gij,gij->g", R_target, R) / dim ** 2])

def all_quantities(theta):
    """Test probabilities (800), Pauli transfer matrix elements (1536) and infidelities (6) of a gauge-fixed gate set."""
    return np.concatenate([np.asarray(ga._circuit_probabilities(*to_ops(theta), test)).reshape(-1), np.asarray(gauge_dependent(theta))])

def fisher_sigma(theta):
    """Plug-in Fisher widths sqrt(d^T F^+ d), with the gradient of f o Pi for the gauge-dependent quantities (script 13)."""
    K, A, B = to_ops(theta)
    T = np.asarray(ga.tangent_space_basis(K, A, B))
    JT = np.asarray(ga._tangent_space_jacobian(K, A, B, T, train))
    p = np.asarray(ga._circuit_probabilities(K, A, B, train)).reshape(-1)
    rank = ga._numerical_rank(jnp.asarray(np.linalg.svd(JT, compute_uv=False)), JT.shape, None)
    _, s, vt = np.linalg.svd(JT / np.sqrt(p)[:, None], full_matrices=False)
    lam, V = m * s[:rank] ** 2, vt[:rank].T
    jac_theta = np.asarray(jax.jacfwd(gauge_dependent)(theta))
    jac_h = np.asarray(jax.jacfwd(lambda h: gauge_dependent(act(h, theta)))(h0))
    H_hh = np.asarray(jax.hessian(gauge_cost, argnums=0)(h0, theta))
    H_htheta = np.asarray(jax.jacfwd(jax.grad(gauge_cost, argnums=0), argnums=1)(h0, theta))
    D = np.vstack([np.asarray(ga._tangent_space_jacobian(K, A, B, T, test)), (jac_theta - jac_h @ np.linalg.solve(H_hh, H_htheta)) @ T])
    return np.sqrt(((D @ V) ** 2 / lam).sum(axis=1))


def compare_with_perform_gauge(point):
    """Largest difference between the superoperators gauge-fixed by gauge_fix and by mGST's perform_gauge
    (pyGSTi gaugeopt_to_target, unitary gauge group, weight 1 per gate and 0.1 on SPAM)."""
    from mGST.simulate import perform_gauge
    from mGST.utility_functions_comparisons import get_mgst_tensors_from_psd_representation
    target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
    mine = np.asarray(get_mgst_tensors_from_psd_representation(*to_ops(gauge_fix(to_vec(*point))))["kraus"])
    pygsti_gauged = np.asarray(perform_gauge(dict(zip(("kraus", "povm", "state"), point)), target_superops, num_gates)[0])
    return np.abs(mine - pygsti_gauged).max()


def analyse(estimates):
    print("max |superoperator(gauge_fix) - superoperator(perform_gauge)|: truth "
          + ", ".join(f"{compare_with_perform_gauge(p):.1e}" for p in [truth] + [load(estimate_file(j)) for j in estimates])
          + " (estimates " + ", ".join(map(str, estimates)) + ")")
    f_truth = all_quantities(gauge_fix(to_vec(*truth)))
    files = sorted(glob.glob(f"{DATA}/crb_estimates/default_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
    F_est = np.array([all_quantities(gauge_fix(to_vec(*load(f)))) for f in files])       # 40 independent datasets
    actual_sd = F_est.std(axis=0, ddof=1)
    n_test, n_ptm = 800, num_gates * 16 * 16
    groups = {"test probabilities": np.arange(n_test),
              "Pauli transfer matrix elements": n_test + np.flatnonzero(actual_sd[n_test:n_test + n_ptm] > 1e-10),
              "infidelities": np.arange(n_test + n_ptm, n_test + n_ptm + num_gates)}
    for j in estimates:
        reps = sorted(OUT.glob(f"est{j}_b*.npz"))
        F_boot = np.array([all_quantities(gauge_fix(to_vec(*load(f)))) for f in reps])
        f_hat = F_est[j]
        s_boot, bias = F_boot.std(axis=0, ddof=1), F_boot.mean(axis=0) - f_hat
        sigma_F = fisher_sigma(gauge_fix(to_vec(*load(estimate_file(j)))))
        lo, hi = np.percentile(F_boot, [2.5, 97.5], axis=0)
        intervals = {"standard  f ± 1.96 s": (f_hat - 1.96 * s_boot, f_hat + 1.96 * s_boot),
                     "percentile": (lo, hi), "basic  [2f - q97.5, 2f - q2.5]": (2 * f_hat - hi, 2 * f_hat - lo)}
        print(f"\n=== estimate {j}: {len(reps)} bootstrap replicates "
              f"(relative precision of each s: ±{1 / np.sqrt(2 * (len(reps) - 1)):.0%}) ===")
        for name, idx in groups.items():
            print(f"{name} ({len(idx)}):")
            print(f"  median s_boot / actual sd (40 datasets) {np.median(s_boot[idx] / actual_sd[idx]):.2f}   "
                  f"median s_boot / Fisher width {np.median(s_boot[idx] / sigma_F[idx]):.2f}   "
                  f"median |bias| / s_boot {np.median(np.abs(bias[idx]) / s_boot[idx]):.2f}")
            inside = {k: np.mean((v[0][idx] <= f_truth[idx]) & (f_truth[idx] <= v[1][idx])) for k, v in intervals.items()}
            print("  truth inside the 95% interval for this dataset: " + ", ".join(f"{k.split()[0]} {100 * v:.0f}%" for k, v in inside.items()))
            proxy = np.mean(np.abs(F_est[:, idx] - f_truth[idx]) <= 1.96 * s_boot[idx])
            print(f"  coverage proxy (width 1.96 s_boot of this dataset applied to all 40 estimates): {100 * proxy:.1f}%")
        k = len(groups["Pauli transfer matrix elements"])
        B_needed = lambda k: 1 + norm.ppf(1 - 0.05 / (2 * k)) ** 2 / (2 * 0.25 ** 2)   # s/sd ~ N(1, 1/(2(B-1))), Bonferroni over k
        print(f"replicates needed for every s to be within ±25% of its limit with probability 95%: "
              f"one quantity B ≈ {B_needed(1):.0f}, all {k} matrix elements B ≈ {B_needed(k):.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(int(sys.argv[2]), int(sys.argv[3]))
    else:
        analyse([int(a) for a in sys.argv[2:]])
