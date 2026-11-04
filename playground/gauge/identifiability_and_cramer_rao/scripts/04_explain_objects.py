"""Concrete numbers for: Dg (shape, rank), T vs T T^T, J vs J_T, and Hessian-vs-Jacobian precision."""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

num_gates, kraus_rank, dim = 6, 4, 4
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

print("== Q2: Dg ==")
print("theta0 shape:", theta0.shape, "| g(theta0) shape:", constraints(theta0).shape, "| max|g(theta0)|:", float(jnp.abs(constraints(theta0)).max()))
Dg = np.asarray(jax.jacfwd(constraints)(theta0))
print("Dg shape:", Dg.shape)
block = Dg[:32]  # rows coming from gate 0's 4x4 Hermitian matrix
print("gate-0 block: identically-zero rows (Im of diagonal):", int((np.abs(block).max(axis=1) < 1e-14).sum()),
      "| rank of the 32 rows:", np.linalg.matrix_rank(block))
U, s, Vt = np.linalg.svd(Dg)  # full_matrices=True -> Vt is 928 x 928
print("numpy shapes: U", U.shape, " s", s.shape, " Vt", Vt.shape)
print("s/s_max around the cut:", " ".join(f"#{i+1}:{s[i]/s[0]:.1e}" for i in [0, 111, 112, 113, 114, 224]))
r = int((s > 1e-10 * s[0]).sum())
T = Vt[r:].T
print(f"rank r = {r}; T = Vt[r:].T has shape {T.shape}  (= 928 - {r})")
print("max |Dg @ T| (columns of T are in ker Dg):", float(np.abs(Dg @ T).max()))

print("\n== Q3: T vs P = T T^T ==")
P = T @ T.T
print("T^T T = I_815 ?", np.allclose(T.T @ T, np.eye(T.shape[1]), atol=1e-12), "| P shape", P.shape,
      "| P^2 = P ?", np.allclose(P @ P, P, atol=1e-12), "| P = P^T ?", np.allclose(P, P.T, atol=1e-12), "| trace P =", round(float(np.trace(P)), 6))
rng = np.random.default_rng(0)
t = T @ rng.normal(size=T.shape[1]); t /= np.linalg.norm(t)
z = rng.normal(size=theta0.size); z /= np.linalg.norm(z)
print("constraint violation |g(θ0 + ε v)| for a tangent v vs a random v:")
for eps in [1e-2, 1e-3, 1e-4]:
    print(f"  ε={eps:.0e}:  tangent {float(jnp.linalg.norm(constraints(theta0 + eps * t))):.1e}   random {float(jnp.linalg.norm(constraints(theta0 + eps * z))):.1e}")
# tangent space is only a real subspace: multiply a tangent vector by i
tk, ta, tb = unpack(jnp.asarray(t))
it = jnp.concatenate([jnp.concatenate([(1j * x).reshape(-1).real, (1j * x).reshape(-1).imag]) for x in (tk, ta, tb)])
print("|Dg v| for tangent v:", f"{np.linalg.norm(Dg @ t):.1e}", "| |Dg (i·v)|:", f"{np.linalg.norm(Dg @ np.asarray(it)):.1e}  -> i·v is NOT tangent")

print("\n== Q3: J vs J_T ==")
N_max = 400
seqs = generate_sequence_indices(num_gates=num_gates, num_circuits=N_max, seq_len_list=[1, 8, 14])["gate_indices"]
seqs = [seqs[i] for i in rng.permutation(N_max)]

def probs(theta, circuits):
    k, a, b = unpack(theta)
    return jnp.concatenate([jnp.real(contract_mps_all_povm(k, a, b, c)) for c in circuits])

J = np.asarray(jax.jacfwd(lambda th: probs(th, seqs))(theta0))
JT = J @ T
print("J shape", J.shape, "| T shape", T.shape, "| J_T shape", JT.shape)
_, jvp_col = jax.jvp(lambda th: probs(th, seqs), (theta0,), (jnp.asarray(T[:, 0]),))
print("max |J_T[:,0] - JVP along T[:,0]|:", float(np.abs(JT[:, 0] - np.asarray(jvp_col)).max()))
print("rank J (all 928 directions):", np.linalg.matrix_rank(J), "| rank J_T:", np.linalg.matrix_rank(JT))

print("\n== Q5: precision, exact data (Hess = J_T^T J_T) ==")
for N in [210, 400]:
    JTn = JT[: dim * N]
    sv = np.linalg.svd(JTn, compute_uv=False)
    ev = np.sort(np.linalg.eigvalsh(JTn.T @ JTn))[::-1]
    gauge_floor = np.abs(ev[624:]).max() / ev[0]  # the 191 eigenvalues that are exactly zero in exact arithmetic
    print(f"N={N}: σ_624/σ_max = {sv[623]/sv[0]:.1e} -> exact λ_624/λ_max = {(sv[623]/sv[0])**2:.1e}; "
          f"computed λ_624/λ_max = {ev[623]/ev[0]:.1e}; rounding floor of the 191 zero eigenvalues = {gauge_floor:.1e}")
    print(f"        σ_623/σ_max = {sv[622]/sv[0]:.1e} -> λ_623/λ_max = {(sv[622]/sv[0])**2:.1e}")
