"""Q1: how does the Fisher spectrum depend on the evaluation point (truth, maximum-likelihood estimate, prior guesses)?
Q2: does the Cramer-Rao bound (tangent-space Fisher information, pseudo-inverse) predict the spread of the actual estimates?
Also saves the spectra for the figure (data/crb_spectra.npz). Run 09_crb_efficiency.py afterwards to add the remaining keys.
Needs the estimates of 07_crb_estimates_runner.py in data/crb_estimates.
"""
import glob
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst, get_initial_gate_set_for_optimization
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
num_gates, kraus_rank, dim, m, N = 6, 4, 4, 1000, 400
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")

def noisy_point(strength, seed):
    noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=strength, seed=seed)
    superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
    return tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))

def pad(indices_list):
    gates = np.full((len(indices_list), max(len(s) for s in indices_list)), -1, dtype=np.int32)
    for row, s in enumerate(indices_list):
        gates[row, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
    return jnp.asarray(gates)

truth = noisy_point(5e-2, 42)
train = pad(generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"])
rng = np.random.default_rng(2024)
test = pad([list(c) for c in rng.integers(0, num_gates, size=(200, 10))])  # held-out length-10 circuits, as in the MVE

def analyse(point, gates=train):
    """Fisher information on the tangent space at `point`, its spectrum, and the CRB covariance of the test probabilities."""
    K, A, B = point
    T = ga.tangent_space_basis(K, A, B)
    JT = ga._tangent_space_jacobian(K, A, B, T, gates)
    p = ga._circuit_probabilities(K, A, B, gates).reshape(-1)
    sv = np.linalg.svd(np.asarray(JT), compute_uv=False)
    rank = ga._numerical_rank(jnp.asarray(sv), JT.shape, None)
    _, s, vt = np.linalg.svd(np.asarray(JT / jnp.sqrt(p)[:, None]), full_matrices=False)
    lam, V = m * s[:rank] ** 2, vt[:rank].T          # nonzero Fisher eigenvalues and eigenvectors (tangent coordinates)
    J_test = np.asarray(ga._tangent_space_jacobian(K, A, B, T, test))   # gradients of the test probabilities, same basis
    G = J_test @ V                                   # components along the Fisher eigenvectors
    crb_var = (G ** 2 / lam).sum(axis=1)             # diag of J_test F^+ J_test^T
    p_test = np.asarray(ga._circuit_probabilities(K, A, B, test)).reshape(-1)
    return dict(rank=rank, lam=lam, crb_std=np.sqrt(crb_var), p_test=p_test, G=G)

def summary(name, r):
    std = 1 / np.sqrt(r["lam"])
    n = lambda eps: int((std < eps).sum())
    crb_mve = np.mean(0.5 * np.sqrt(2 / np.pi) * r["crb_std"].reshape(-1, dim).sum(axis=1))
    print(f"{name:<34} rank {r['rank']:>3} | 1/sqrt(λ) at k=1: {std[0]:.1e}, k=100: {std[min(99, len(std)-1)]:.1e}, k=last: {std[-1]:.1e}"
          f" | n(1e-2) {n(1e-2):>3}, n(3e-3) {n(3e-3):>3} | CRB-predicted MVE(L=10) {crb_mve:.2e}")

print("== Q1: Fisher spectrum at different points, same design (N=400, m=1000) ==")
res_truth = analyse(truth)
summary("truth", res_truth)

U = expm(1j * (lambda h: (h + h.conj().T) / 2)(jnp.asarray(np.random.default_rng(3).normal(size=(dim, dim)) + 1j * np.random.default_rng(4).normal(size=(dim, dim)))))
K, A, B = truth
gauged = (jnp.einsum("ab,grbc,dc->grad", U, K, U.conj()), A @ U.conj().T, U @ B)
res_gauged = analyse(gauged)
print(f"{'truth, random unitary gauge':<34} max|Δp| = {np.abs(res_gauged['p_test'] - res_truth['p_test']).max():.1e}, "
      f"max relative change of λ_k = {np.abs(res_gauged['lam'] / res_truth['lam'] - 1).max():.1e}")

files = sorted(glob.glob(f"{DATA}/crb_estimates/default_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
files_converged = sorted(glob.glob(f"{DATA}/crb_estimates/converged_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
load = lambda f: tuple(jnp.asarray(np.load(f)[k]) for k in ("kraus", "povm", "state"))
est_results = [analyse(load(f)) for f in files]
for i in range(3):
    summary(f"MLE estimate #{i}", est_results[i])
ranks = [r["rank"] for r in est_results]
print(f"{'':<34} ranks at the {len(ranks)} estimates: min {min(ranks)}, median {int(np.median(ranks))}, max {max(ranks)}")
k_max = min(ranks)
lam_ratio = np.array([r["lam"][:k_max] / res_truth["lam"][:k_max] for r in est_results])
print(f"{'':<34} λ_k(estimate)/λ_k(truth): k<=100: {np.percentile(lam_ratio[:, :100], [5, 50, 95]).round(2)}, "
      f"k=101..400: {np.percentile(lam_ratio[:, 100:400], [5, 50, 95]).round(2)}, k=401..{k_max}: {np.percentile(lam_ratio[:, 400:], [5, 50, 95]).round(3)}  (5/50/95%)")

def choi_spectrum(K):
    """Normalized Choi eigenvalues of every gate (= squared singular values of the matrix of vec(K_r)), decreasing."""
    s = np.linalg.svd(np.asarray(K).reshape(K.shape[0], K.shape[1], -1), compute_uv=False) ** 2
    return s / s.sum(axis=1, keepdims=True)
print("Choi eigenvalues 2..4 of gate 0, truth:    ", choi_spectrum(truth[0])[0, 1:])
for i in range(3):
    print(f"Choi eigenvalues 2..4 of gate 0, estimate {i}:", choi_spectrum(load(files[i])[0])[0, 1:])
smallest = np.array([choi_spectrum(load(f)[0])[:, -1] for f in files])
print(f"smallest Choi eigenvalue per gate, truth: {choi_spectrum(truth[0])[:, -1].min():.1e} (min over gates); estimates: median over gates and estimates {np.median(smallest):.1e}, min {smallest.min():.1e}")

def perturb(point, size, seed):
    """Random perturbation of the given size, mapped back onto the Stiefel manifolds by the polar decomposition."""
    K, A, B = point
    key = np.random.default_rng(seed)
    noise = lambda x: size * jnp.asarray(key.normal(size=x.shape) + 1j * key.normal(size=x.shape)) / np.sqrt(2 * x.size)
    def to_isometry(X):  # polar factor = closest isometry
        u, _, vh = jnp.linalg.svd(X, full_matrices=False)
        return u @ vh
    Kp = jnp.stack([to_isometry((K[g] + noise(K[g])).reshape(-1, dim)).reshape(K.shape[1:]) for g in range(num_gates)])
    Ap = to_isometry((A + noise(A)).reshape(-1, dim)).reshape(A.shape)
    Bp = B + noise(B)
    return Kp, Ap, Bp / jnp.linalg.norm(Bp)
for size in [1e-3, 1e-2]:
    r = analyse(perturb(load(files[0]), size, 0))
    summary(f"estimate #0 + random perturbation {size:.0e}", r)
summary("prior guess: target+other noise 5e-2", res_prior := analyse(noisy_point(5e-2, 7)))
summary("prior guess: target+other noise 1e-2", res_prior_weak := analyse(noisy_point(1e-2, 7)))
target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
init = get_initial_gate_set_for_optimization(target_superops=target_superops, kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim, spam_depolarization=1e-2, project_kraus_to_isometry=True)
init = tuple(jnp.asarray(init[k]) for k in ("kraus", "povm", "state"))
print("   (init point: norm of non-dominant Kraus operators per gate:", np.round(np.asarray(jnp.sqrt(jnp.sum(jnp.abs(init[0][:, 1:]) ** 2, axis=(1, 2, 3)))), 4), ")")
summary("target gates + depolarized SPAM", res_init := analyse(init))

print("\n== Q2: CRB vs actual spread of the MLE (test probabilities of 200 length-10 circuits) ==")
for tag, fs, results in [("default (abs. tol 1)", files, est_results), ("converged (abs. tol 1e-4, rel. 1e-8)", files_converged, None)]:
    if not fs:
        continue
    P = np.array([r["p_test"] for r in results]) if results else np.array([analyse(load(f))["p_test"] for f in fs])
    emp_std = P.std(axis=0, ddof=1)
    bias = P.mean(axis=0) - res_truth["p_test"]
    ratio = emp_std / res_truth["crb_std"]
    mve = np.mean(0.5 * np.abs(P - res_truth["p_test"]).reshape(len(fs), -1, dim).sum(axis=2), axis=1)
    print(f"{tag}: {len(fs)} estimates | empirical std / CRB std (truth): 5% {np.percentile(ratio, 5):.2f}, median {np.median(ratio):.2f}, 95% {np.percentile(ratio, 95):.2f}"
          f" | median |bias|/CRB std {np.median(np.abs(bias) / res_truth['crb_std']):.2f} (noise level of this ratio ~ {1/np.sqrt(len(fs)):.2f})"
          f" | actual MVE(L=10) {mve.mean():.2e} ± {mve.std(ddof=1):.1e}")
    if results:
        plug = np.array([r["crb_std"] for r in results])
        print(f"   CRB std computed at the estimate / at the truth: 5% {np.percentile(plug / res_truth['crb_std'], 5):.2f}, median {np.median(plug / res_truth['crb_std']):.2f}, 95% {np.percentile(plug / res_truth['crb_std'], 95):.2f}")
contrib = (res_truth["G"] ** 2 / res_truth["lam"])  # per-direction contribution to the CRB variance of each test probability
share = contrib.sum(axis=0) / contrib.sum()
print(f"share of the total CRB variance of the test probabilities from eigen-directions k=1..100: {share[:100].sum():.2f}, 101..400: {share[100:400].sum():.2f}, 401..624: {share[400:].sum():.2f}")

spectra_N = {}
for n_circ in [50, 150, 250, 400, 800]:
    gates = pad(generate_sequence_indices(num_gates=num_gates, num_circuits=n_circ, seq_len_list=[1, 8, 14], seed=7)["gate_indices"])
    spectra_N[n_circ] = analyse(truth, gates)["lam"]
np.savez(DATA / "crb_spectra.npz", **{f"N{k}": v for k, v in spectra_N.items()}, truth=res_truth["lam"], estimate=est_results[0]["lam"],
         prior=res_prior["lam"], init=res_init["lam"])
