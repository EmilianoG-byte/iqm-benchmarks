"""Validate gauge_analysis.probability_jacobian_spectrum and compute the numbers for the conditioning / Fisher discussion."""
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm
from mGST import gauge_analysis as ga

num_gates, kraus_rank, dim, m = 6, 4, 4, 1000
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = (jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*noisy_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))
design = lambda N, seed: generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=seed)["gate_indices"]

print("== correctness ==")
seqs = design(40, 0)
gates = np.full((len(seqs), 14), -1, dtype=np.int32)
for r, s in enumerate(seqs):
    gates[r, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
p_new = ga._circuit_probabilities(K, A, B, jnp.asarray(gates))
p_lib = jnp.stack([jnp.real(contract_mps_all_povm(K, A, B, s)) for s in seqs])
print("max |p - p_library|:", float(jnp.abs(p_new - p_lib).max()))
T = ga.tangent_space_basis(K, A, B)
shapes = [K.shape, A.shape, B.shape]
theta = ga._operators_to_real_vector(K, A, B)
Dg = jax.jacfwd(lambda t: ga._isometry_constraints(t, shapes))(theta)
print("Dg", Dg.shape, "rank", np.linalg.matrix_rank(np.asarray(Dg)), "| T", T.shape, "| T^T T = I:", bool(jnp.allclose(T.T @ T, jnp.eye(T.shape[1]), atol=1e-12)),
      "| max|Dg T|:", f"{float(jnp.abs(Dg @ T).max()):.1e}", "| max_num_parameters:", ga.max_num_parameters(2, num_gates, kraus_rank))
# J_T column 0 against a finite difference along T[:, 0] followed by the exact retraction-free check (first order)
res = ga.probability_jacobian_spectrum(K, A, B, seqs)
for N in [100, 400]:
    t0 = time.time(); r = ga.probability_jacobian_spectrum(K, A, B, design(N, 1)); t1 = time.time()
    rb = ga.probability_jacobian_spectrum(K, A, B, design(N, 1), batch_size=64)
    print(f"N={N}: rank {r['rank']} (bound {min(r['num_independent_data'], 624)}), {t1-t0:.1f}s; batch_size=64 same singular values: {bool(jnp.allclose(r['singular_values'], rb['singular_values'], rtol=1e-8, atol=1e-14))}")
K_unitary = jnp.zeros_like(K).at[:, 0].set(jnp.asarray(target_operators["kraus"])[:, 0])
print("N=400 at exactly unitary gates: rank", ga.probability_jacobian_spectrum(K_unitary, A, B, design(400, 1))["rank"])

print("\n== default tolerance (max(shape)*eps) vs 1e-10, 30 designs ==")
for N in [205, 208, 210]:
    ranks_default, ranks_old, smin = [], [], []
    for r in range(30):
        out = ga.probability_jacobian_spectrum(K, A, B, design(N, 1000 * N + r))
        sv = out["singular_values"]
        ranks_default.append(out["rank"]); ranks_old.append(int((sv > 1e-10 * sv[0]).sum()))
        bound = min(out["num_independent_data"], 624)
        smin.append(float(sv[bound - 1] / sv[0]))
    print(f"N={N}: bound {bound}; deficient with default tol: {sum(x < bound for x in ranks_default)}/30, with 1e-10: {sum(x < bound for x in ranks_old)}/30;"
          f" σ_bound/σ_max min {min(smin):.1e} median {np.median(smin):.1e}; next σ (true zero) {float(sv[bound]/sv[0]) if bound < len(sv) else 0:.1e}")

print("\n== Fisher information, m = 1000 shots ==")
incoherent = jnp.sqrt(jnp.sum(jnp.abs(K[:, 1:]) ** 2, axis=(1, 2, 3)))
print("norm of the non-dominant Kraus operators per gate (scale of the incoherent parameters):", np.round(np.asarray(incoherent), 4))
print(f"{'N':>4} {'1/sqrt(λ_1)':>12} {'1/sqrt(λ_93)':>13} {'1/sqrt(λ_94)':>13} {'1/sqrt(λ_min)':>14}   # directions with CRB std < ε:  ε=1e-1   3e-2   1e-2   3e-3")
for N in [50, 100, 150, 200, 250, 400, 800]:
    out = ga.probability_jacobian_spectrum(K, A, B, design(N, 7), num_shots=m)
    lam = np.asarray(out["fisher_eigenvalues"])[: out["rank"]]
    std = 1 / np.sqrt(lam)
    counts = [int((std < eps).sum()) for eps in [1e-1, 3e-2, 1e-2, 3e-3]]
    print(f"{N:>4} {std[0]:>12.1e} {std[92]:>13.1e} {std[93]:>13.1e} {std[-1]:>14.1e}   rank {out['rank']:>3}: " + "  ".join(f"{c:>5}" for c in counts))

print("\n== conditioning of J_T vs Fisher (does the 1/p weighting matter?) N=400 ==")
out = ga.probability_jacobian_spectrum(K, A, B, design(400, 7), num_shots=m)
sv, lam = np.asarray(out["singular_values"][:624]), np.asarray(out["fisher_eigenvalues"][:624])
print(f"σ_min/σ_max = {sv[-1]/sv[0]:.1e};  sqrt(λ_min/λ_max) = {np.sqrt(lam[-1]/lam[0]):.1e}; σ_93/σ_94 = {sv[92]/sv[93]:.1f}, sqrt(λ_93/λ_94) = {np.sqrt(lam[92]/lam[93]):.1f}")

print("\n== design-to-design spread of the CRB std of the weakest direction, m = 1000, 20 designs ==")
for N in [150, 250, 400]:
    worst = []
    for r in range(20):
        out = ga.probability_jacobian_spectrum(K, A, B, design(N, 5000 + 100 * N + r), num_shots=m)
        worst.append(1 / np.sqrt(float(out["fisher_eigenvalues"][out["rank"] - 1])))
    q = np.percentile(worst, [0, 25, 50, 75, 100])
    print(f"N={N}: min {q[0]:.1e}  25% {q[1]:.1e}  median {q[2]:.1e}  75% {q[3]:.1e}  max {q[4]:.1e}")
