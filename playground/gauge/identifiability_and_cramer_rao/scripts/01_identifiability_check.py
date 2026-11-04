"""Identifiability check for the 2Q MVE-vs-sequences study.

1. Choi spectrum of the true noisy gates (how close to unitary?).
2. Rank of the probability Jacobian restricted to the model tangent space vs number of sequences.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

noise_strength = float(sys.argv[1]) if len(sys.argv) > 1 else 5e-2
num_qubits, num_gates, kraus_rank = 2, 6, 4
dim = 2**num_qubits

target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=noise_strength, seed=42)
noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = get_compressed_rep_from_mgst_output(*noisy_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim)
K, A, B = jnp.asarray(K), jnp.asarray(A), jnp.asarray(B)
print(f"noise_strength={noise_strength}  shapes K{K.shape} A{A.shape} B{B.shape}")

# ---------------- 1. Choi spectra ----------------
print("\n== Noise channels identical across gates? ==")
nk = np.asarray(noise["kraus"])
print("max |noise_kraus[g] - noise_kraus[0]| over gates:", max(np.abs(nk[g] - nk[0]).max() for g in range(num_gates)))

print("\n== Normalised Choi eigenvalues of each true gate (sum = 1) ==")
for g in range(num_gates):
    choi = sum(np.outer(k.reshape(-1), k.reshape(-1).conj()) for k in np.asarray(K[g])) / dim
    ev = np.sort(np.linalg.eigvalsh(choi))[::-1]
    print(f"gate {g}: " + "  ".join(f"{e:.2e}" for e in ev[:kraus_rank]) + f"   | entanglement infidelity 1-λ1 = {1-ev[0]:.2e}")
rho = np.asarray(B @ B.conj().T)
print("state eigenvalues:", np.round(np.sort(np.linalg.eigvalsh(rho))[::-1], 5))

# ---------------- 2. Jacobian rank ----------------
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

theta0 = pack(K, A, B)

def constraints(theta):
    k, a, b = unpack(theta)
    herm = lambda m: jnp.concatenate([m.real.reshape(-1), m.imag.reshape(-1)])
    c_gates = [herm(jnp.einsum("rij,rik->jk", k[g].conj(), k[g]) - jnp.eye(dim)) for g in range(num_gates)]
    c_povm = herm(jnp.einsum("prj,prk->jk", a.conj(), a) - jnp.eye(dim))
    c_state = jnp.atleast_1d(jnp.real(jnp.trace(b @ b.conj().T)) - 1.0)
    return jnp.concatenate(c_gates + [c_povm, c_state])

assert np.abs(constraints(theta0)).max() < 1e-8, np.abs(constraints(theta0)).max()
Dg = np.asarray(jax.jacfwd(constraints)(theta0))
_, s, vt = np.linalg.svd(Dg)
r = int((s > 1e-10 * s[0]).sum())
T = vt[r:].T  # orthonormal basis of the tangent space
print(f"\nambient real params: {theta0.size}, independent constraints: {r}, tangent dim: {T.shape[1]} (max_num_parameters = 815)")

rng = np.random.default_rng(0)
N_max = 400
seqs = generate_sequence_indices(num_gates=num_gates, num_circuits=N_max, seq_len_list=[1, 8, 14])["gate_indices"]
seqs = [seqs[i] for i in rng.permutation(N_max)]  # mix short/long so that prefixes are representative

def probs(theta, circuits):
    k, a, b = unpack(theta)
    return jnp.concatenate([jnp.real(contract_mps_all_povm(k, a, b, c)) for c in circuits])

J_full = np.asarray(jax.jacfwd(lambda t: probs(t, seqs))(theta0))  # (4*N_max, 928)
JT_full = J_full @ T

print("\n== rank of dp/dtheta on the tangent space (gauge directions are automatically in the kernel) ==")
print(f"{'N':>5} {'3N':>6} {'rank':>6} {'σ_min(nonzero)/σ_max':>22}")
for N in [50, 100, 150, 180, 200, 205, 208, 210, 220, 250, 300, 400]:
    JT = JT_full[: dim * N]
    sv = np.linalg.svd(JT, compute_uv=False)
    rank = int((sv > 1e-9 * sv[0]).sum())
    print(f"{N:>5} {3*N:>6} {rank:>6} {sv[rank-1]/sv[0]:>22.2e}")

# Spectrum at saturation: how many directions are "soft" relative to the strongest?
sv = np.linalg.svd(JT_full, compute_uv=False)
rank = int((sv > 1e-9 * sv[0]).sum())
sv = sv[:rank] / sv[0]
print(f"\nN={N_max}: identifiable directions = {rank}")
for thr in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
    print(f"  # singular values with σ/σ_max < {thr:.0e}: {(sv < thr).sum()}")
print("  σ/σ_max for indices 85..100:", " ".join(f"{i}:{sv[i]:.1e}" for i in range(85, 101)))
ratios = sv[:-1] / sv[1:]
print("  largest consecutive gaps (index, ratio):", [(int(i) + 1, round(float(ratios[i]), 1)) for i in np.argsort(ratios)[::-1][:3]])
