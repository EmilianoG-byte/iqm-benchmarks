"""Shared helpers for scripts 15 and 16: model setup, Pauli transfer matrices, gauge fixing, gauge-consistent gradients
and the Fisher eigen-decomposition. Same conventions as scripts 09 and 13 (gauge fixing Pi = Hilbert-space unitary that
brings the gate Pauli transfer matrices closest to the target in Frobenius norm)."""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
num_gates, kraus_rank, dim = 6, 4, 4
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")


def gate_set(noise_strength=5e-2, seed=42):
    """(K, A, B): target gates followed by random noise channels. The defaults give the truth of scripts 07-14;
    (5e-2, 7) is the prior guess of script 08."""
    noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=noise_strength, seed=seed)
    superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
    return tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))


truth = gate_set()
shapes = [x.shape for x in truth]
to_vec = lambda K, A, B: ga._operators_to_real_vector(K, A, B)
to_ops = lambda theta: ga._real_vector_to_operators(theta, shapes)
load = lambda f: tuple(jnp.asarray(np.load(f)[k]) for k in ("kraus", "povm", "state"))


def pad(indices_list):
    """Padded int array (num_circuits, max_len), -1 = identity, sequences reversed to application order."""
    gates = np.full((len(indices_list), max(len(s) for s in indices_list)), -1, dtype=np.int32)
    for row, s in enumerate(indices_list):
        gates[row, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
    return jnp.asarray(gates)


pauli1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
paulis = jnp.asarray([np.kron(a, b) for a in pauli1 for b in pauli1])          # 16 two-qubit Paulis


def ptm(K):
    """Pauli transfer matrices R_g[i, j] = tr(P_i G_g(P_j)) / d, shape (num_gates, 16, 16)."""
    return jnp.real(jnp.einsum("iab,grbc,jcd,grad->gij", paulis, K, paulis, K.conj())) / dim


R_target = ptm(jnp.asarray(target_operators["kraus"]))
ptm_vector = lambda theta: ptm(to_ops(theta)[0]).reshape(-1)                   # 6 * 16 * 16 = 1536 elements


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


def ptm_gradient(theta):
    """Rows: gradients of the gauge-fixed elements f o Pi in the 928 real coordinates, at a gauge-fixed theta.
    Implicit function theorem: grad(f o Pi) = grad f - (df/dh) H_hh^-1 H_h,theta."""
    jac_theta = np.asarray(jax.jacfwd(ptm_vector)(theta))
    jac_h = np.asarray(jax.jacfwd(lambda h: ptm_vector(act(h, theta)))(h0))
    H_hh = np.asarray(jax.hessian(gauge_cost, argnums=0)(h0, theta))
    H_htheta = np.asarray(jax.jacfwd(jax.grad(gauge_cost, argnums=0), argnums=1)(h0, theta))
    return jac_theta - jac_h @ np.linalg.solve(H_hh, H_htheta)


def fisher(theta, gates, m):
    """Fisher information F = m W^T W on the tangent space, W = diag(p)^-1/2 J_T = U S V^T.
    Returns T, the nonzero eigenvalues lam (decreasing), their eigenvectors V, the matching U, p, and an orthonormal
    basis `kernel` of ker F (gauge directions plus, below the identifiability threshold, unidentifiable ones)."""
    K, A, B = to_ops(theta)
    T = np.asarray(ga.tangent_space_basis(K, A, B))
    JT = np.asarray(ga._tangent_space_jacobian(K, A, B, T, gates))
    p = np.asarray(ga._circuit_probabilities(K, A, B, gates)).reshape(-1)
    rank = ga._numerical_rank(jnp.asarray(np.linalg.svd(JT, compute_uv=False)), JT.shape, None)
    u, s, vt = np.linalg.svd(JT / np.sqrt(p)[:, None], full_matrices=True)        # full vt: complete kernel also if 4N < 815
    return dict(T=T, lam=m * s[:rank] ** 2, V=vt[:rank].T, U=u[:, :rank], p=p, kernel=vt[rank:].T, rank=rank)
