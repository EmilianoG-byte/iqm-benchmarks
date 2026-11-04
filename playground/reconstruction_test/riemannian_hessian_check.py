"""Connect the tangent-space Jacobian J_T to the Riemannian Hessian of the GST cost.

1. T T^T equals the Stiefel orthogonal projector P_X(Z) = Z - X sym(X^H Z) (embedded metric), blockwise.
2. Riemannian Hessian via autodiff, in the tangent basis T, using the Lagrangian form
       Hess f(x) = T^T [ ∇²f̄(x) - Σ_i λ_i ∇²g_i(x) ] T,   λ = argmin ||∇f̄(x) - Dg(x)^T λ||
   for the least-squares cost f̄(θ) = ½||p(θ) - y||².
   - exact data y = p(θ*): compare with Gauss-Newton J_T^T J_T
   - sampled data (1000 shots): size of the residual-dependent terms
3. How JAX treats complex inputs.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST.low_level_jit import contract_mps_all_povm

num_gates, kraus_rank, dim, N, num_shots = 6, 4, 4, 60, 1000

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

theta0 = pack(K, A, B)

def constraints(theta):
    k, a, b = unpack(theta)
    herm = lambda m: jnp.concatenate([m.real.reshape(-1), m.imag.reshape(-1)])
    c_gates = [herm(jnp.einsum("rij,rik->jk", k[g].conj(), k[g]) - jnp.eye(dim)) for g in range(num_gates)]
    c_povm = herm(jnp.einsum("prj,prk->jk", a.conj(), a) - jnp.eye(dim))
    c_state = jnp.atleast_1d(jnp.real(jnp.trace(b @ b.conj().T)) - 1.0)
    return jnp.concatenate(c_gates + [c_povm, c_state])

Dg = np.asarray(jax.jacfwd(constraints)(theta0))
_, s, vt = np.linalg.svd(Dg)
T = vt[int((s > 1e-10 * s[0]).sum()):].T

# ---------- 1. T T^T is the Stiefel projector ----------
def stiefel_proj(X, Z):
    XhZ = X.conj().T @ Z
    return Z - X @ (XhZ + XhZ.conj().T) / 2

rng = np.random.default_rng(1)
z = rng.normal(size=theta0.size)
zk, za, zb = unpack(jnp.asarray(z))
proj_blockwise = pack(
    jnp.stack([stiefel_proj(K[g].reshape(-1, dim), zk[g].reshape(-1, dim)).reshape(K[g].shape) for g in range(num_gates)]),
    stiefel_proj(A.reshape(-1, dim), za.reshape(-1, dim)).reshape(A.shape),
    stiefel_proj(B.reshape(-1, 1), zb.reshape(-1, 1)).reshape(B.shape),
)
print("1) max |T T^T z - P_X(z)| (Stiefel projector, blockwise):", float(jnp.abs(T @ (T.T @ z) - proj_blockwise).max()))

# ---------- 2. Riemannian Hessian vs Gauss-Newton ----------
seqs = generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14])["gate_indices"]

def probs(theta):
    k, a, b = unpack(theta)
    return jnp.concatenate([jnp.real(contract_mps_all_povm(k, a, b, c)) for c in seqs])

J = np.asarray(jax.jacfwd(probs)(theta0))
JT = J @ T
GN = JT.T @ JT
p_exact = np.asarray(probs(theta0))
counts = np.concatenate([rng.multinomial(num_shots, np.clip(p_exact[i:i + dim], 0, None) / np.clip(p_exact[i:i + dim], 0, None).sum()) for i in range(0, p_exact.size, dim)])
p_sampled = counts / num_shots

def riemannian_hessian_in_basis(y):
    cost = lambda t: 0.5 * jnp.sum((probs(t) - y) ** 2)
    egrad = np.asarray(jax.grad(cost)(theta0))
    lam = np.linalg.lstsq(Dg.T, egrad, rcond=None)[0]  # normal component of the Euclidean gradient
    lagrangian = lambda t: cost(t) - jnp.dot(lam, constraints(t))
    H_amb = np.asarray(jax.hessian(lagrangian)(theta0))  # 928 x 928, ordinary Euclidean Hessian
    H = T.T @ H_amb @ T
    return 0.5 * (H + H.T), np.linalg.norm(egrad), np.linalg.norm(T.T @ egrad)

def spectrum_summary(M, name):
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    nonzero = ev[np.abs(ev) > 1e-9 * ev[0]]
    print(f"   {name}: rank {len(nonzero)}, λ_max {ev[0]:.3e}, smallest |λ| kept {np.abs(nonzero).min():.3e}, most negative {ev.min():.3e}")

print(f"\n2) N={N} circuits, 3N={3*N}")
print(f"   rank(J_T) = {np.linalg.matrix_rank(JT, tol=1e-9*np.linalg.norm(JT, 2))}")
spectrum_summary(GN, "Gauss-Newton J_T^T J_T")
for label, y in [("exact data", p_exact), ("1000-shot data", p_sampled)]:
    H, g_norm, rg_norm = riemannian_hessian_in_basis(y)
    print(f" - {label}: |Euclidean grad| = {g_norm:.2e}, |Riemannian grad| = {rg_norm:.2e}")
    print(f"   rel. difference ||Hess - J_T^T J_T|| / ||J_T^T J_T|| = {np.linalg.norm(H - GN, 2) / np.linalg.norm(GN, 2):.2e}")
    spectrum_summary(H, "Riemannian Hessian")

# ---------- 3. complex inputs in JAX ----------
print("\n3) JAX with complex inputs")
f = lambda z: jnp.real(z * jnp.conj(z))  # |z|^2, real-valued, not holomorphic
try:
    jax.jacfwd(f)(jnp.array(1.0 + 1.0j))
except Exception as e:
    print("   jacfwd on complex input ->", type(e).__name__, ":", str(e).splitlines()[0][:160])
print("   jax.grad(|z|^2) at z=1+1j ->", jax.grad(f)(jnp.array(1.0 + 1.0j)), "(true gradient in R^2 is (2,2))")
