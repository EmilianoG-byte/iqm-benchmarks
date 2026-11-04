"""Do Fisher-information confidence intervals (pyGSTi style) contain the truth? Empirical coverage on the 40 estimates.

For a scalar f the interval is f(estimate) +- z sqrt(d^T F^+ d), as in Nielsen et al. 2021, Sec. 6.3.6, eq. (78), with
z = 1 (nominal 68.3%) and z = 1.96 (nominal 95%). Coverage = fraction of intervals that contain the true value.
F and d are evaluated at each estimate ("plug-in", as with real data) and, for comparison, at the truth ("oracle").

1) Gauge-invariant: the 200 x 4 held-out test probabilities (as in 08 and 09), also restricted to the first 135
   Fisher directions.
2) Gauge-dependent: Pauli transfer matrix elements and entanglement infidelities of the 6 gates, after fixing the gauge
   with Pi = the Hilbert-space unitary U that brings the gate Pauli transfer matrices closest to the target (Frobenius
   norm). For these quantities U is the only gauge freedom of our model: Kraus mixing and the purifications leave
   superoperators, effects and the state unchanged. Two gradients:
   a) "perpendicular": d = T^T grad f; F^+ drops its gauge part. This is pyGSTi's 'std' projection (non-gauge space
      perpendicular to the gauge space), here in the Euclidean metric of (K, A, B).
   b) "consistent": d = T^T grad(f o Pi), with Pi differentiated by the implicit function theorem. This is the
      gradient of the quantity that is actually reported (f of the gauge-fixed estimate).
Needs the estimates of 07_crb_estimates_runner.py (default set). Runtime: about 4 minutes.
"""
import glob
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
num_gates, kraus_rank, dim, m, N = 6, 4, 4, 1000, 400
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
truth = tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))

def pad(indices_list):
    gates = np.full((len(indices_list), max(len(s) for s in indices_list)), -1, dtype=np.int32)
    for row, s in enumerate(indices_list):
        gates[row, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
    return jnp.asarray(gates)

train = pad(generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"])
test = pad([list(c) for c in np.random.default_rng(2024).integers(0, num_gates, size=(200, 10))])
load = lambda f: tuple(jnp.asarray(np.load(f)[k]) for k in ("kraus", "povm", "state"))
files = sorted(glob.glob(f"{DATA}/crb_estimates/default_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))

shapes = [x.shape for x in truth]
to_vec = lambda K, A, B: ga._operators_to_real_vector(K, A, B)
to_ops = lambda theta: ga._real_vector_to_operators(theta, shapes)

pauli1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
paulis = jnp.asarray([np.kron(a, b) for a in pauli1 for b in pauli1])          # 16 two-qubit Paulis

def ptm(K):
    """Pauli transfer matrices R_g[i, j] = tr(P_i G_g(P_j)) / d, shape (num_gates, 16, 16)."""
    return jnp.real(jnp.einsum("iab,grbc,jcd,grad->gij", paulis, K, paulis, K.conj())) / dim

R_target = ptm(jnp.asarray(target_operators["kraus"]))

def act(h, theta):
    """Hilbert-space gauge U = exp(i sum_j h_j P_j / 2) (15 traceless generators): K -> U K U^H, A -> A U^H, B -> U B."""
    K, A, B = to_ops(theta)
    U = expm(1j * jnp.einsum("j,jab->ab", h, paulis[1:]) / 2)
    return to_vec(jnp.einsum("ab,grbc,dc->grad", U, K, U.conj()), A @ U.conj().T, U @ B)

def gauge_cost(h, theta):
    return jnp.sum((ptm(to_ops(act(h, theta))[0]) - R_target) ** 2)

cost_and_grad = jax.jit(jax.value_and_grad(gauge_cost))
h0 = jnp.zeros(15)

def gauge_fix(theta):
    """Pi(theta): minimize the gauge cost over U (BFGS, twice, the second time from the result of the first)."""
    for _ in range(2):
        fun = lambda h: tuple(np.asarray(v, dtype=float) for v in cost_and_grad(jnp.asarray(h), theta))
        res = minimize(fun, np.zeros(15), jac=True, method="BFGS", options=dict(gtol=1e-12, maxiter=2000))
        theta = act(jnp.asarray(res.x), theta)
    return theta, float(res.fun), float(np.abs(res.jac).max())

def quantities(theta):
    """Gauge-dependent quantities: all Pauli transfer matrix elements and the entanglement infidelity of each gate."""
    R = ptm(to_ops(theta)[0])
    infidelity = 1 - jnp.einsum("gij,gij->g", R_target, R) / dim ** 2
    return jnp.concatenate([R.reshape(-1), infidelity])

def analyse(theta):
    """Fisher eigen-decomposition at theta and the squared interval widths d^T F^+ d for all quantities."""
    K, A, B = to_ops(theta)
    T = np.asarray(ga.tangent_space_basis(K, A, B))
    JT = np.asarray(ga._tangent_space_jacobian(K, A, B, T, train))
    p = np.asarray(ga._circuit_probabilities(K, A, B, train)).reshape(-1)
    rank = ga._numerical_rank(jnp.asarray(np.linalg.svd(JT, compute_uv=False)), JT.shape, None)
    _, s, vt = np.linalg.svd(JT / np.sqrt(p)[:, None], full_matrices=False)
    lam, V = m * s[:rank] ** 2, vt[:rank].T
    var = lambda D: ((D @ V) ** 2 / lam).sum(axis=1)                            # rows of D are tangent gradients d

    D_test = np.asarray(ga._tangent_space_jacobian(K, A, B, T, test))
    jac_theta = np.asarray(jax.jacfwd(quantities)(theta))                         # gradient of f
    jac_h = np.asarray(jax.jacfwd(lambda h: quantities(act(h, theta)))(h0))      # derivative along the gauge U
    H_hh = np.asarray(jax.hessian(gauge_cost, argnums=0)(h0, theta))
    H_htheta = np.asarray(jax.jacfwd(jax.grad(gauge_cost, argnums=0), argnums=1)(h0, theta))
    grad_fPi = jac_theta - jac_h @ np.linalg.solve(H_hh, H_htheta)               # implicit function theorem
    gauge_dirs = np.asarray(jax.jacfwd(lambda h: act(h, theta))(h0))             # 928 x 15 Hilbert-space gauge directions
    leak = np.abs(grad_fPi @ gauge_dirs).max() / np.abs(jac_theta @ gauge_dirs).max()
    return dict(rank=rank, lam=lam,
                p_test=np.asarray(ga._circuit_probabilities(K, A, B, test)).reshape(-1),
                var_test=var(D_test), var_test_135=((D_test @ V[:, :135]) ** 2 / lam[:135]).sum(axis=1),
                f=np.asarray(quantities(theta)), var_perp=var(jac_theta @ T), var_cons=var(grad_fPi @ T), leak=leak)

theta_truth, cost_truth, _ = gauge_fix(to_vec(*truth))
oracle = analyse(theta_truth)
print(f"truth: gauge cost after fixing {cost_truth:.3e}; |grad(f o Pi) . gauge| / |grad f . gauge| = {oracle['leak']:.1e} (0 = f o Pi is gauge invariant)")

results, costs, jacs = [], [], []
for f in files:
    theta, cost, jac = gauge_fix(to_vec(*load(f)))
    results.append(analyse(theta))
    costs.append(cost)
    jacs.append(jac)
print(f"{len(files)} estimates: gauge cost after fixing {np.min(costs):.3e} .. {np.max(costs):.3e}, max final |gradient| {max(jacs):.1e}, "
      f"ranks {min(r['rank'] for r in results)}..{max(r['rank'] for r in results)}")

def coverage(name, err, sig):
    err, sig = np.asarray(err), np.asarray(sig)
    cov = [100 * np.mean(np.abs(err) <= z * sig) for z in (1.0, 1.96)]
    ratio = np.sqrt((err ** 2).mean(axis=0)) / np.median(sig, axis=0) if err.ndim == 2 else np.nan
    print(f"  {name:<44} {cov[0]:6.1f}% {cov[1]:7.1f}%   median 95% half-width {np.median(1.96 * sig):.2e}   median rmse/width-std {np.nanmedian(ratio):.2f}")

print(f"\n{'':<46} nominal 68.3%  nominal 95%")
print("gauge-invariant: 800 test probabilities")
err_test = np.array([r["p_test"] for r in results]) - oracle["p_test"]
coverage("full F, oracle (at the truth)", err_test, np.broadcast_to(np.sqrt(oracle["var_test"]), err_test.shape))
coverage("full F, plug-in (at each estimate)", err_test, np.array([np.sqrt(r["var_test"]) for r in results]))
coverage("first 135 directions, oracle", err_test, np.broadcast_to(np.sqrt(oracle["var_test_135"]), err_test.shape))

n_ptm = num_gates * 16 * 16
err_f = np.array([r["f"] for r in results]) - oracle["f"]
nontrivial = np.sqrt(oracle["var_cons"]) > 1e-10                                # drops the constant first row (trace preservation)
for label, sl in [(f"gauge-dependent: Pauli transfer matrix elements ({nontrivial[:n_ptm].sum()} non-constant)", slice(0, n_ptm)),
                  ("gauge-dependent: entanglement infidelities (6 gates)", slice(n_ptm, None))]:
    keep = nontrivial[sl]
    e = err_f[:, sl][:, keep]
    print(label)
    coverage("perpendicular projection, plug-in", e, np.array([np.sqrt(r["var_perp"][sl][keep]) for r in results]))
    coverage("consistent with the gauge fixing, plug-in", e, np.array([np.sqrt(r["var_cons"][sl][keep]) for r in results]))
    coverage("consistent with the gauge fixing, oracle", e, np.broadcast_to(np.sqrt(oracle["var_cons"][sl][keep]), e.shape))
print(f"\ntrue infidelities: {np.round(oracle['f'][n_ptm:], 5)}")
print(f"mean estimated infidelities: {np.round(np.mean([r['f'][n_ptm:] for r in results], axis=0), 5)}")
print(f"95% half-widths (consistent, oracle): {np.round(1.96 * np.sqrt(oracle['var_cons'][n_ptm:]), 5)}")
