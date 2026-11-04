"""How much does rank(J_T) depend on the particular random sequence design?

For each N, draw many independent designs with `generate_sequence_indices` and record
rank(J_T), the deficit w.r.t. min(3 n_circuits, 624), and the conditioning.
Also: rank at a point with exactly unitary gates (rank also depends on *where* it is evaluated).
"""
import random
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

num_gates, kraus_rank, dim, n_ident = 6, 4, 4, 624
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

def pack(*xs):
    return jnp.concatenate([jnp.concatenate([x.reshape(-1).real, x.reshape(-1).imag]) for x in xs])

def constraints(theta):
    k, a, b = unpack(theta)
    herm = lambda m: jnp.concatenate([m.real.reshape(-1), m.imag.reshape(-1)])
    c_gates = [herm(jnp.einsum("rij,rik->jk", k[g].conj(), k[g]) - jnp.eye(dim)) for g in range(num_gates)]
    c_povm = herm(jnp.einsum("prj,prk->jk", a.conj(), a) - jnp.eye(dim))
    c_state = jnp.atleast_1d(jnp.real(jnp.trace(b @ b.conj().T)) - 1.0)
    return jnp.concatenate(c_gates + [c_povm, c_state])

def tangent_basis(theta):
    _, s, vt = np.linalg.svd(np.asarray(jax.jacfwd(constraints)(theta)))
    return vt[int((s > 1e-10 * s[0]).sum()):].T

identity_kraus = jnp.zeros((1, kraus_rank, dim, dim), dtype=complex).at[0, 0].set(jnp.eye(dim))

def circuit_probs(k, a, b, idx_padded):
    """Same contraction as contract_mps_all_povm; padding (-1) acts as the identity channel."""
    k_ext = jnp.concatenate([k, identity_kraus], axis=0)
    def step(rho, idx):
        kk = k_ext[jnp.where(idx < 0, num_gates, idx)]
        return jnp.einsum("ijk,kl,iml->jm", kk, rho, kk.conj()), None
    rho, _ = lax.scan(step, b @ b.conj().T, idx_padded[::-1])
    return jnp.real(jnp.einsum("ijk,kl,ijl->i", a, rho, a.conj()))

@jax.jit
def jacobian(theta, idx_array):
    f = lambda t: jax.vmap(circuit_probs, in_axes=(None, None, None, 0))(*unpack(t), idx_array).reshape(-1)
    return jax.jacfwd(f)(theta)

theta0 = pack(K, A, B)
T0 = tangent_basis(theta0)

# sanity check against the library contraction
design = generate_sequence_indices(num_gates=num_gates, num_circuits=20, seq_len_list=[1, 8, 14])
p_lib = jnp.concatenate([jnp.real(contract_mps_all_povm(K, A, B, c)) for c in design["gate_indices"]])
p_new = jax.vmap(circuit_probs, in_axes=(None, None, None, 0))(K, A, B, jnp.asarray(design["gate_indices_with_negs"])).reshape(-1)
print("max |p_scan - p_library| =", float(jnp.abs(p_lib - p_new).max()))

def analyse(theta, T, idx_array):
    JT = np.asarray(jacobian(theta, jnp.asarray(idx_array))) @ T
    sv = np.linalg.svd(JT, compute_uv=False)
    rank = int((sv > 1e-10 * sv[0]).sum())
    return rank, sv[rank - 1] / sv[0], sv

R = 30
print(f"\n{R} independent designs per N (true noisy gate set)")
print(f"{'N':>4} {'n_circ':>6} {'bound':>6} {'rank min/med/max':>17} {'#deficient':>10} {'σ_min/σ_max  min / median / max':>34}")
for N in [50, 100, 150, 200, 205, 208, 210, 215, 250, 300, 400]:
    ranks, smins, ncs = [], [], []
    for r in range(R):
        random.seed(1000 * N + r)
        idx = generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14])["gate_indices_with_negs"]
        rank, smin, _ = analyse(theta0, T0, idx)
        ranks.append(rank); smins.append(smin); ncs.append(len(idx))
    bound = min(3 * ncs[0], n_ident)
    deficient = sum(rk < min(3 * nc, n_ident) for rk, nc in zip(ranks, ncs))
    print(f"{N:>4} {ncs[0]:>6} {bound:>6} {min(ranks):>5}/{int(np.median(ranks)):>4}/{max(ranks):>4} {deficient:>10}"
          f"   {min(smins):.1e} / {np.median(smins):.1e} / {max(smins):.1e}")

# rank also depends on the point: exactly unitary gates (K_2..K_4 = 0), same noisy SPAM
K_unitary = jnp.zeros_like(K).at[:, 0].set(jnp.asarray(target_operators["kraus"])[:, 0])
theta_u = pack(K_unitary, A, B)
assert float(jnp.abs(constraints(theta_u)).max()) < 1e-10
T_u = tangent_basis(theta_u)
random.seed(0)
idx = generate_sequence_indices(num_gates=num_gates, num_circuits=400, seq_len_list=[1, 8, 14])["gate_indices_with_negs"]
for label, th, T in [("true noisy gates", theta0, T0), ("exactly unitary gates", theta_u, T_u)]:
    rank, _, sv = analyse(th, T, idx)
    print(f"\nN=400, {label}: rank = {rank}; σ/σ_max of last kept = {sv[rank-1]/sv[0]:.1e}; next = {sv[rank]/sv[0] if rank < len(sv) else 0:.1e}")
