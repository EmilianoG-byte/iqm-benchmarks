"""Check the 191 gauge directions at the true gate set (supports DERIVATIONS.md §A.6).

Builds the infinitesimal gauge generators (linear in theta) for
    Kraus mixing      K_gr -> sum_s V_rs K_gs        6 x 16
    POVM purification A_i  -> W_i A_i                4 x 16
    state purification B   -> B Y                    16
    Hilbert space      K -> U K U^H, A_i -> A_i U^H, B -> U B   16 (incl. the global phase, counted twice on purpose)
as columns of a 928 x 192 matrix G and reports
    1) rank G (expected 191: the global phase of U equals a combination of purification phases),
    2) max |Dh G| (gauge directions are tangent),
    3) ||J G|| / (||J|| ||G||) (gauge directions do not change any probability).
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

num_gates, kraus_rank, dim, N = 6, 4, 4, 60
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = (np.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))

shapes = [K.shape, A.shape, B.shape]
sizes = [int(np.prod(s)) for s in shapes]

def unpack(theta):
    out, i = [], 0
    for s, n in zip(shapes, sizes):
        out.append((theta[i:i + n] + 1j * theta[i + n:i + 2 * n]).reshape(s))
        i += 2 * n
    return out

def pack(*xs):
    return np.concatenate([np.concatenate([np.asarray(x).reshape(-1).real, np.asarray(x).reshape(-1).imag]) for x in xs])

theta0 = pack(K, A, B)

def constraints(theta):
    k, a, b = unpack(theta)
    herm = lambda m: jnp.concatenate([m.real.reshape(-1), m.imag.reshape(-1)])
    c_gates = [herm(jnp.einsum("rij,rik->jk", k[g].conj(), k[g]) - jnp.eye(dim)) for g in range(num_gates)]
    c_povm = herm(jnp.einsum("prj,prk->jk", a.conj(), a) - jnp.eye(dim))
    c_state = jnp.atleast_1d(jnp.real(jnp.trace(b @ b.conj().T)) - 1.0)
    return jnp.concatenate(c_gates + [c_povm, c_state])

seqs = generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"]

def probs(theta):
    k, a, b = unpack(theta)
    return jnp.concatenate([jnp.real(contract_mps_all_povm(k, a, b, c)) for c in seqs])

def hermitian_basis(n):
    basis = []
    for j in range(n):
        for l in range(j, n):
            E = np.zeros((n, n), complex)
            E[j, l] = E[l, j] = 1
            basis.append(E)
            if l > j:
                E = np.zeros((n, n), complex)
                E[j, l], E[l, j] = 1j, -1j
                basis.append(E)
    return basis

H4 = hermitian_basis(dim)
zK, zA, zB = np.zeros_like(K), np.zeros_like(A), np.zeros_like(B)
cols = []
for g in range(num_gates):                                   # Kraus mixing of gate g: dK_gr = i sum_s H_rs K_gs
    for H in H4:
        dK = zK.copy()
        dK[g] = 1j * np.einsum("rs,sij->rij", H, K[g])
        cols.append(pack(dK, zA, zB))
for i in range(A.shape[0]):                                  # POVM purification: dA_i = i H A_i
    for H in H4:
        dA = zA.copy()
        dA[i] = 1j * H @ A[i]
        cols.append(pack(zK, dA, zB))
for H in H4:                                                 # state purification: dB = B i H
    cols.append(pack(zK, zA, 1j * B @ H))
for H in H4:                                                 # Hilbert space gauge: dK = i[H, K], dA = -i A H, dB = i H B
    cols.append(pack(1j * (np.einsum("ij,grjk->grik", H, K) - np.einsum("grij,jk->grik", K, H)), -1j * A @ H, 1j * H @ B))
G = np.array(cols).T                                         # 928 x 192

s = np.linalg.svd(G, compute_uv=False)
print(f"G: {G.shape}, rank {(s > 1e-10 * s[0]).sum()}  (singular values #191 / #192 relative to max: {s[190] / s[0]:.1e} / {s[191] / s[0]:.1e})")
Dh = np.asarray(jax.jacfwd(constraints)(jnp.asarray(theta0)))
print(f"max |Dh G| = {np.abs(Dh @ G).max():.1e}   (gauge directions are tangent)")
J = np.asarray(jax.jacfwd(probs)(jnp.asarray(theta0)))
print(f"||J G|| / (||J|| ||G||) = {np.linalg.norm(J @ G, 2) / (np.linalg.norm(J, 2) * np.linalg.norm(G, 2)):.1e}   (gauge directions leave all {J.shape[0]} probabilities unchanged)")
