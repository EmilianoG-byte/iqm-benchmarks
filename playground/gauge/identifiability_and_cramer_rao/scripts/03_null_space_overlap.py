"""Do the unidentified directions at small N overlap with the 93 stiff (coherent) directions?

Stiff subspace S = top-93 right singular vectors of the tangent-space Jacobian at N=400.
For each N, compute the principal angles between S and null(J_N) (the directions invisible to the data).
cos = 1 -> some stiff direction is completely unseen; cos ~ 0 -> all stiff directions are seen.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

num_gates, kraus_rank, dim, n_stiff = 6, 4, 4, 93

target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = (jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*noisy_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))

shapes = [K.shape, A.shape, B.shape]
sizes = [int(np.prod(s)) for s in shapes]

def unpack(theta):
    out, i = [], 0
    for s, n in zip(shapes, sizes):
        out.append((theta[i:i + n] + 1j * theta[i + n:i + 2 * n]).reshape(s))
        i += 2 * n
    return out

theta0 = jnp.concatenate([jnp.concatenate([x.reshape(-1).real, x.reshape(-1).imag]) for x in (K, A, B)])

def constraints(theta):
    k, a, b = unpack(theta)
    herm = lambda m: jnp.concatenate([m.real.reshape(-1), m.imag.reshape(-1)])
    c_gates = [herm(jnp.einsum("rij,rik->jk", k[g].conj(), k[g]) - jnp.eye(dim)) for g in range(num_gates)]
    c_povm = herm(jnp.einsum("prj,prk->jk", a.conj(), a) - jnp.eye(dim))
    c_state = jnp.atleast_1d(jnp.real(jnp.trace(b @ b.conj().T)) - 1.0)
    return jnp.concatenate(c_gates + [c_povm, c_state])

_, s, vt = np.linalg.svd(np.asarray(jax.jacfwd(constraints)(theta0)))
T = vt[int((s > 1e-10 * s[0]).sum()):].T

rng = np.random.default_rng(0)
N_max = 400
seqs = generate_sequence_indices(num_gates=num_gates, num_circuits=N_max, seq_len_list=[1, 8, 14])["gate_indices"]
seqs = [seqs[i] for i in rng.permutation(N_max)]

def probs(theta, circuits):
    k, a, b = unpack(theta)
    return jnp.concatenate([jnp.real(contract_mps_all_povm(k, a, b, c)) for c in circuits])

JT_full = np.asarray(jax.jacfwd(lambda t: probs(t, seqs))(theta0)) @ T  # (4*N_max, 815)

_, sv_full, vt_full = np.linalg.svd(JT_full)
V_stiff = vt_full[:n_stiff].T  # (815, 93)
V_soft = vt_full[n_stiff:624].T  # (815, 531)
print(f"stiff/soft boundary at N=400: σ_93/σ_max = {sv_full[n_stiff-1]/sv_full[0]:.2e}, σ_94/σ_max = {sv_full[n_stiff]/sv_full[0]:.2e}")

print(f"\n{'N':>4} {'rank':>5} {'null dim':>9} {'non-gauge null':>15} {'max cos(null, stiff)':>21} {'# cos>0.1':>10} {'# cos>0.5':>10}")
for N in [50, 100, 150, 200]:
    JT = JT_full[: dim * N]
    _, sv, vt = np.linalg.svd(JT)
    rank = int((sv > 1e-9 * sv[0]).sum())
    null = vt[rank:].T  # includes the 191 gauge directions
    cosines = np.linalg.svd(null.T @ V_stiff, compute_uv=False)
    print(f"{N:>4} {rank:>5} {null.shape[1]:>9} {624-rank:>15} {cosines.max():>21.3f} {(cosines>0.1).sum():>10} {(cosines>0.5).sum():>10}")

# How well is each stiff direction constrained at N, relative to N=400 (after removing what the null space can absorb)?
print("\nσ_min of J_N restricted to the stiff subspace, normalised by sqrt(N/400)·σ_min at N=400:")
ref = np.linalg.svd(JT_full @ V_stiff, compute_uv=False).min()
for N in [50, 100, 150, 200]:
    smin = np.linalg.svd(JT_full[: dim * N] @ V_stiff, compute_uv=False).min()
    print(f"  N={N}: {smin / (np.sqrt(N / N_max) * ref):.2f}")
