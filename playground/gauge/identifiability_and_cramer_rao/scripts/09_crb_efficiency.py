"""Per-direction test of the Cramer-Rao bound for the maximum-likelihood estimate, gauge-invariantly.

Usage: python 09_crb_efficiency.py [<tag>]   (tag = default (default) or converged, the file prefix in data/crb_estimates)

Whitened Jacobian W = diag(p)^(-1/2) J_T = U S V^T at the truth, F = m W^T W, so lambda_k = m s_k^2.
For an efficient estimator, e = sqrt(m) U^T diag(p)^(-1/2) (p_train(estimate) - p_train(truth)) has independent standard
normal components: E[e_k^2] = 1 means the k-th direction attains the bound, < 1 means the estimate moves less than an
unbiased estimator must (bias), > 1 means it is less efficient than the bound.
The printed "Var(e_k)" is the mean of e_k^2 over the estimates (no mean subtracted), which is >= the variance.
With tag=default this also adds "unitary" and "var_e" to data/crb_spectra.npz (run 08_crb_analysis.py first);
otherwise it writes data/crb_spectra_<tag>.npz.
"""
import glob
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst, get_initial_gate_set_for_optimization
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
tag = sys.argv[1] if len(sys.argv) > 1 else "default"
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
files = sorted(glob.glob(f"{DATA}/crb_estimates/{tag}_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))

K, A, B = truth
T = ga.tangent_space_basis(K, A, B)
p = np.asarray(ga._circuit_probabilities(K, A, B, train)).reshape(-1)
W = np.asarray(ga._tangent_space_jacobian(K, A, B, T, train)) / np.sqrt(p)[:, None]
U, s, Vt = np.linalg.svd(W, full_matrices=False)
rank = 624
U, s, V, lam = U[:, :rank], s[:rank], Vt[:rank].T, m * s[:rank] ** 2
std_k = 1 / np.sqrt(lam)

P_train = np.array([np.asarray(ga._circuit_probabilities(*load(f), train)).reshape(-1) for f in files])
e = np.sqrt(m) * ((P_train - p) / np.sqrt(p)) @ U          # (num_estimates, rank)
var_e = (e ** 2).mean(axis=0)                               # truth is known, so no mean subtracted
print(f"{len(files)} MLE estimates ({tag}), N={N}, m={m}. Var(e_k): 1 = attains the CRB, <1 = moves less than any unbiased estimator must")
bins = [(1, 50), (51, 100), (101, 135), (136, 200), (201, 300), (301, 400), (401, 500), (501, 624)]
print(f"{'k range':>10} {'CRB std 1/sqrt(λ_k)':>22} {'median Var(e_k)':>16}")
for lo, hi in bins:
    sl = slice(lo - 1, hi)
    print(f"{lo:>4}-{hi:<5} {std_k[lo-1]:>9.1e} .. {std_k[hi-1]:<9.1e} {np.median(var_e[sl]):>14.2f}")
print(f"(sampling uncertainty of a single Var(e_k) with {len(files)} estimates: ±{np.sqrt(2/len(files)):.2f})")

# test probabilities: empirical spread vs CRB restricted to the first k_cut directions, and vs a Van Trees (Bayesian) bound
J_test = np.asarray(ga._tangent_space_jacobian(K, A, B, T, test))
G = J_test @ V
p_test = np.asarray(ga._circuit_probabilities(K, A, B, test)).reshape(-1)
P_test = np.array([np.asarray(ga._circuit_probabilities(*load(f), test)).reshape(-1) for f in files])
rmse = np.sqrt(((P_test - p_test) ** 2).mean(axis=0))
mve_actual = np.mean(0.5 * np.abs(P_test - p_test).reshape(len(files), -1, dim).sum(axis=2))
mve_pred = lambda var: np.mean(0.5 * np.sqrt(2 / np.pi) * np.sqrt(var).reshape(-1, dim).sum(axis=1))
print(f"\ntest probabilities (200 length-10 circuits): actual MVE {mve_actual:.2e}")
for k_cut in [100, 135, 200, 300, 400, 624]:
    var = (G[:, :k_cut] ** 2 / lam[:k_cut]).sum(axis=1)
    print(f"  CRB using directions 1..{k_cut:<3}: median rmse/CRB std {np.median(rmse / np.sqrt(var)):.2f}, predicted MVE {mve_pred(var):.2e}")
for tau in [1.0, 0.1, 0.05, 0.01]:
    var = (G ** 2 / (lam + 1 / tau ** 2)).sum(axis=1)
    print(f"  Van Trees, prior std τ={tau:<5}: median rmse/bound {np.median(rmse / np.sqrt(var)):.2f}, predicted MVE {mve_pred(var):.2e}")

target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
init = get_initial_gate_set_for_optimization(target_superops=target_superops, kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim, spam_depolarization=1e-2, project_kraus_to_isometry=True)
Ki = np.asarray(init["kraus"])
sv = np.linalg.svd(Ki.reshape(num_gates, kraus_rank, -1), compute_uv=False) ** 2
print("\ninit point (perturbed target): normalized Choi eigenvalues of gate 0:", (sv[0] / sv[0].sum()).round(4))
smallest = np.array([(lambda c: c[:, -1] / c.sum(axis=1))(np.linalg.svd(np.asarray(load(f)[0]).reshape(num_gates, kraus_rank, -1), compute_uv=False) ** 2) for f in files])
print(f"smallest Choi eigenvalue per gate over the estimates: median {np.median(smallest):.1e}, fraction below 1e-6: {np.mean(smallest < 1e-6):.2f}")

if tag == "default":
    K_unitary = jnp.zeros_like(K).at[:, 0].set(jnp.asarray(target_operators["kraus"])[:, 0])
    out = ga.probability_jacobian_spectrum(K_unitary, A, B, [list(map(int, np.asarray(r)[np.asarray(r) >= 0][::-1])) for r in train], num_shots=m)
    print("exactly unitary gates + true SPAM: rank", out["rank"])
    d = dict(np.load(DATA / "crb_spectra.npz"))
    d["unitary"] = np.asarray(out["fisher_eigenvalues"])[: out["rank"]]
    d["var_e"] = var_e
    np.savez(DATA / "crb_spectra.npz", **d)
else:
    np.savez(DATA / f"crb_spectra_{tag}.npz", var_e=var_e)
