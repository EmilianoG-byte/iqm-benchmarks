"""Experiment design for the precision of gauge-fixed Pauli transfer matrix elements, without fitting.

Generic version of `ptm_tools.py` (any number of qubits, gates and ranks; the gate set and target are arguments) plus
the design scan of script 16 and the rate-resolution check of PTM_PRECISION.md Section 3.4. Playground version of a
future `src/mGST` module. Conventions as in PTM_PRECISION.md:
- gauge fixing Pi = Hilbert-space unitary that brings the gate Pauli transfer matrices closest to the target ones;
- d_f = T^T grad(f o Pi) (implicit function theorem), sigma_LAN^2 = sum_k (d_f . v_k)^2 / lambda_k (eq. 4);
- van Trees sigma_vT^2(tau) = sum_k (d_f . v_k)^2 / (lambda_k + 1 / tau^2) + tau^2 |kernel part of d_f|^2;
- weak share = fraction of sigma_LAN^2 from directions with 1 / sqrt(lambda_k) > eps.
LAN = local asymptotic normality. All Fisher quantities are computed at m = 1 shot and rescaled (lambda_k is
proportional to m).
"""
from functools import lru_cache
from pathlib import Path
from itertools import product
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from jax.scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

from mGST import gauge_analysis as ga
from mGST.simulate import generate_sequence_indices


# ---------------------------------------------------------------- operators, Pauli transfer matrices, gauge fixing

def to_vec(kraus, povm, state):
    return ga._operators_to_real_vector(kraus, povm, state)


def to_ops(theta, shapes):
    return ga._real_vector_to_operators(theta, shapes)


@lru_cache
def pauli_basis(num_qubits):
    """The 4^n n-qubit Pauli matrices (unnormalized), P_0 = identity. Shape (4^n, 2^n, 2^n). A NumPy array, so that the
    cache never holds a JAX tracer."""
    pauli1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
    paulis = []
    for factors in product(pauli1, repeat=num_qubits):
        P = np.eye(1)
        for f in factors:
            P = np.kron(P, f)
        paulis.append(P)
    return np.array(paulis)


def _num_qubits(dim):
    return int(round(np.log2(dim)))


def pauli_transfer_matrices(kraus):
    """R_g[i, j] = tr(P_i G_g(P_j)) / d for Kraus operators of shape (num_gates, kraus_rank, d, d)."""
    dim = kraus.shape[-1]
    paulis = pauli_basis(_num_qubits(dim))
    return jnp.real(jnp.einsum("iab,grbc,jcd,grad->gij", paulis, kraus, paulis, kraus.conj())) / dim


def _act(h, theta, shapes):
    """Hilbert-space gauge U = exp(i sum_a h_a P_a / 2): K -> U K U^H, A -> A U^H, B -> U B."""
    kraus, povm, state = to_ops(theta, shapes)
    paulis = pauli_basis(_num_qubits(state.shape[0]))
    U = expm(1j * jnp.einsum("a,abc->bc", h, paulis[1:]) / 2)
    return to_vec(jnp.einsum("ab,grbc,dc->grad", U, kraus, U.conj()), povm @ U.conj().T, U @ state)


def _gauge_cost(h, theta, shapes, target_ptms):
    return jnp.sum((pauli_transfer_matrices(to_ops(_act(h, theta, shapes), shapes)[0]) - target_ptms) ** 2)


_gauge_cost_and_grad = jax.jit(jax.value_and_grad(_gauge_cost), static_argnums=2)


def _shapes(kraus, povm, state):
    return tuple(tuple(x.shape) for x in (kraus, povm, state))


def gauge_fix_to_target(kraus, povm, state, target_ptms):
    """Pi: apply the Hilbert-space unitary that minimizes sum_g |R_g - R_g^target|_F^2 (BFGS, twice).

    Returns:
        (kraus, povm, state) after the gauge transformation, and a dict with the final cost and gradient norm.
    """
    shapes = _shapes(kraus, povm, state)
    theta, h0 = to_vec(kraus, povm, state), np.zeros(state.shape[0] ** 2 - 1)
    target_ptms = jnp.asarray(target_ptms)
    for _ in range(2):
        fun = lambda h: tuple(np.asarray(v, dtype=float) for v in _gauge_cost_and_grad(jnp.asarray(h), theta, shapes, target_ptms))
        res = minimize(fun, h0, jac=True, method="BFGS", options=dict(gtol=1e-12, maxiter=2000))
        theta = _act(jnp.asarray(res.x), theta, shapes)
    return tuple(to_ops(theta, shapes)), dict(cost=float(res.fun), grad=float(np.abs(res.jac).max()))


def gauge_fixed_ptm_gradient(kraus, povm, state, target_ptms):
    """Rows: gradients of the gauge-fixed elements f o Pi in the real coordinates of `to_vec`, at a gauge-fixed point.

    Implicit function theorem (PTM_PRECISION eq. 2): grad(f o Pi) = grad f - (df/dh) H_hh^-1 H_h,theta.
    Shape (num_gates * 4^n * 4^n, num_real_parameters); element (g, i, j) is row (g * 4^n + i) * 4^n + j.
    """
    shapes = _shapes(kraus, povm, state)
    theta, target_ptms = to_vec(kraus, povm, state), jnp.asarray(target_ptms)
    h0 = jnp.zeros(state.shape[0] ** 2 - 1)
    stationarity = float(jnp.abs(jax.grad(_gauge_cost)(h0, theta, shapes, target_ptms)).max())
    if stationarity > 1e-6:
        raise ValueError(f"The point is not gauge-fixed (gauge gradient {stationarity:.1e}); call gauge_fix_to_target first.")
    ptm_vector = lambda t: pauli_transfer_matrices(to_ops(t, shapes)[0]).reshape(-1)
    jac_theta = np.asarray(jax.jacfwd(ptm_vector)(theta))
    jac_h = np.asarray(jax.jacfwd(lambda h: ptm_vector(_act(h, theta, shapes)))(h0))
    H_hh = np.asarray(jax.hessian(_gauge_cost, argnums=0)(h0, theta, shapes, target_ptms))
    H_htheta = np.asarray(jax.jacfwd(jax.grad(_gauge_cost, argnums=0), argnums=1)(h0, theta, shapes, target_ptms))
    return jac_theta - jac_h @ np.linalg.solve(H_hh, H_htheta)


# ---------------------------------------------------------------- Fisher information

def pad_sequences(indices_list):
    """Padded int array (num_circuits, max_len), -1 = identity, sequences reversed to application order
    (as in `probability_jacobian_spectrum`)."""
    gates = np.full((len(indices_list), max(1, max(len(s) for s in indices_list))), -1, dtype=np.int32)
    for row, s in enumerate(indices_list):
        gates[row, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
    return jnp.asarray(gates)


def fisher_decomposition(kraus, povm, state, indices_list, num_shots=1, basis=None, batch_size=None):
    """Fisher information F = m W^T W on the tangent space, W = diag(p)^-1/2 J_T = U S V^T.

    Returns a dict with the tangent basis T, the nonzero eigenvalues lam (decreasing), their eigenvectors V, the
    matching U, p, an orthonormal basis `kernel` of ker F (gauge directions plus, below the identifiability threshold,
    unidentifiable ones) and the rank. The rank is taken from the singular values of J_T, as in
    `probability_jacobian_spectrum`.
    """
    T = np.asarray(ga.tangent_space_basis(kraus, povm, state)) if basis is None else np.asarray(basis)
    gates = pad_sequences(indices_list)
    batch_size = gates.shape[0] if batch_size is None else batch_size
    JT = np.concatenate([np.asarray(ga._tangent_space_jacobian(kraus, povm, state, jnp.asarray(T), gates[s:s + batch_size]))
                         for s in range(0, gates.shape[0], batch_size)])
    p = np.asarray(ga._circuit_probabilities(kraus, povm, state, gates)).reshape(-1)
    rank = ga._numerical_rank(jnp.asarray(np.linalg.svd(JT, compute_uv=False)), JT.shape, None)
    # complete V (and kernel) also when there are fewer data than tangent directions
    u, s, vt = np.linalg.svd(JT / np.sqrt(p)[:, None], full_matrices=JT.shape[0] < JT.shape[1])
    return dict(T=T, lam=num_shots * s[:rank] ** 2, V=vt[:rank].T, U=u[:, :rank], p=p, kernel=vt[rank:].T, rank=rank)


# ---------------------------------------------------------------- per-element widths

def element_widths(D, fd, m_grid, taus, eps=0.01, kernel_tol=1e-12):
    """Three numbers per element (PTM_PRECISION 3.7) for gradients D (rows d_f, tangent coordinates).

    fd must come from `fisher_decomposition(..., num_shots=1)`.

    Returns a dict with
    - "estimable": d_f has no component in ker F outside the gauge (relative squared norm below kernel_tol);
    - "sigma_lan1": LAN standard deviation at m = 1 (infinite if not estimable); at m shots divide by sqrt(m);
    - "sigma_vt": van Trees standard deviation, shape (len(taus), num_elements, len(m_grid));
    - "weak_share": share of sigma_LAN^2 from directions with 1 / sqrt(m lambda_k) > eps, shape (num_elements, len(m_grid)).
    """
    lam1, m_grid = fd["lam"], np.asarray(m_grid, dtype=float)
    G2 = (D @ fd["V"]) ** 2
    ker_abs = np.linalg.norm(D @ fd["kernel"], axis=1) ** 2
    estimable = ker_abs / np.linalg.norm(D, axis=1) ** 2 < kernel_tol
    C = G2 / lam1
    sigma_lan1 = np.where(estimable, np.sqrt(C.sum(axis=1)), np.inf)
    m_lam = np.outer(lam1, m_grid)
    sigma_vt = np.stack([np.sqrt(G2 @ (1 / (m_lam + tau ** -2)) + tau ** 2 * ker_abs[:, None]) for tau in taus])
    weak_share = (C @ (m_lam < eps ** -2)) / C.sum(axis=1)[:, None]
    return dict(estimable=estimable, sigma_lan1=sigma_lan1, sigma_vt=sigma_vt, weak_share=weak_share)


# ---------------------------------------------------------------- incoherent rates (Choi eigenvalues)

def choi_eigenvalues(kraus):
    """Nonzero eigenvalues of the normalized Choi matrices, decreasing. Shape (num_gates, kraus_rank).

    They are the eigenvalues of the Kraus Gram matrix tr(K_r^dagger K_s) / d, which is invariant under the Hilbert-space
    gauge and changes only by a unitary similarity under Kraus mixing, so they are gauge-invariant.
    """
    gram = jnp.einsum("grab,gsab->grs", kraus.conj(), kraus) / kraus.shape[-1]
    return jnp.linalg.eigvalsh(gram)[:, ::-1]


def small_choi_gradient(kraus, povm, state):
    """Values (num_gates * (kraus_rank - 1),) and gradients (same rows, real coordinates of `to_vec`) of the small
    Choi eigenvalues mu_2..mu_r of every gate: the incoherent rates of PTM_PRECISION 3.4."""
    shapes = _shapes(kraus, povm, state)
    small = lambda t: choi_eigenvalues(to_ops(t, shapes)[0])[:, 1:].reshape(-1)
    mu = np.asarray(choi_eigenvalues(kraus))
    gaps = -np.diff(mu, axis=1) / mu[:, :1]
    if gaps.min() < 1e-6:
        raise ValueError(f"Nearly degenerate Choi eigenvalues (relative gap {gaps.min():.1e}); their gradient is ill-defined.")
    theta = to_vec(kraus, povm, state)
    return np.asarray(small(theta)), np.asarray(jax.jacfwd(small)(theta))


def lan_sigma1(D, fd):
    """LAN standard deviation at m = 1 of the quantities with tangent gradients D (rows); infinite if not estimable."""
    ker2 = np.linalg.norm(D @ fd["kernel"], axis=1) ** 2 / np.linalg.norm(D, axis=1) ** 2
    return np.where(ker2 < 1e-12, np.sqrt((((D @ fd["V"]) ** 2) / fd["lam"]).sum(axis=1)), np.inf)


def prior_width(kraus, fd_ref, num_shots_ref=1000, eps=0.01):
    """A priori van Trees width tau_0 = sqrt(|small Kraus part|^2 / (rank - n_eff(eps))) (PTM_PRECISION 3.6):
    the squared norm of the incoherent part, spread evenly over the directions not resolved to eps at a reference
    design. fd_ref from `fisher_decomposition(..., num_shots=1)`."""
    mu = np.asarray(choi_eigenvalues(kraus))
    small2 = kraus.shape[-1] * mu[:, 1:].sum()                       # sum_g s_g, s_g = d (1 - mu_1) for a trace-preserving gate
    n_eff = int((num_shots_ref * fd_ref["lam"] > eps ** -2).sum())
    return float(np.sqrt(small2 / (fd_ref["rank"] - n_eff)))


# ---------------------------------------------------------------- realistic noise: gate set of a fake backend

def _kraus_list(channel):
    """Kraus operators (list of matrices) of a qiskit channel or qiskit-aer QuantumError."""
    from qiskit.quantum_info import Kraus, SuperOp
    if hasattr(channel, "to_quantumchannel"):
        channel = channel.to_quantumchannel()
    return [np.asarray(k) for k in Kraus(SuperOp(channel)).data]


def _then(first, second):
    """Kraus operators of the channel `second` applied after `first`."""
    return [b @ a for a in first for b in second]


def _kron(left, right):
    return [np.kron(a, b) for a in left for b in right]


def fake_backend_gate_set(backend, pair, target_kraus, idle_ns=32.0, spectator_idle=True, excited_population=0.01,
                          return_full=False):
    """Gate set of the 2QXYICZ gates on the physical qubits `pair` of a qiskit-iqm fake backend, in the format of the
    other evaluation points: (kraus (6, 4, 4, 4), povm, state), compressed to Kraus rank 4 (and SPAM rank 4).

    Noise, as `IQMFakeBackend._create_noise_model` builds it from `backend.error_profile`, applied after each gate:
    - prx (the one-qubit gates Rx, Ry): thermal relaxation over the prx duration, then one-qubit depolarizing;
    - cz: thermal relaxation of both qubits over the cz duration, then two-qubit depolarizing;
    - readout: the per-qubit confusion matrix, as a POVM.
    Two additions that the Aer noise model of the fake backend does not have (switch off for a comparison with Aer):
    - spectator_idle: thermal relaxation of the idle qubit during a one-qubit gate, and of both qubits during the Idle
      gate (a delay of idle_ns). Without it the Idle gate is exactly unitary, a singular point of the model;
    - excited_population: thermal population of |1> in the prepared state of each qubit. Without it the state has
      rank 1, also a singular point for a model of state rank 4.
    The ideal gates are the mGST targets (`target_kraus`, shape (6, 1, 4, 4), qubit pair[0] = left tensor factor), so
    the qubit ordering is the one of mGST. With return_full=True also returns the uncompressed gate set (Kraus rank
    up to 16) and a table of per-gate numbers.
    """
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    from mGST.utility_functions_comparisons import get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst

    ep = backend.error_profile
    names = [backend.index_to_qubit_name(q) for q in pair]
    t_prx, t_cz = ep.single_qubit_gate_durations["prx"], ep.two_qubit_gate_durations["cz"]
    relax = lambda q, t: _kraus_list(thermal_relaxation_error(ep.t1s[q], ep.t2s[q], t))
    ident = [np.eye(2)]
    rates_cz = ep.two_qubit_gate_depolarizing_error_parameters["cz"]
    p_cz = rates_cz.get(tuple(names), rates_cz.get(tuple(names[::-1])))
    if p_cz is None:
        raise ValueError(f"{names} is not a cz pair of the backend")

    def one_qubit(position):
        q, other = names[position], names[1 - position]
        noisy = _then(relax(q, t_prx), _kraus_list(depolarizing_error(ep.single_qubit_gate_depolarizing_error_parameters["prx"][q], 1)))
        spectator = relax(other, t_prx) if spectator_idle else ident
        return _kron(noisy, spectator) if position == 0 else _kron(spectator, noisy)

    idle = _kron(relax(names[0], idle_ns), relax(names[1], idle_ns)) if spectator_idle else [np.eye(4)]
    cz = _then(_kron(relax(names[0], t_cz), relax(names[1], t_cz)), _kraus_list(depolarizing_error(p_cz, 2)))
    # 2QXYICZ order: Idle, Rx(q0), Rx(q1), Ry(q0), Ry(q1), CZ; noise channel after the ideal gate
    noise = [idle, one_qubit(0), one_qubit(1), one_qubit(0), one_qubit(1), cz]
    target = np.asarray(target_kraus)[:, 0]
    full = [[k @ target[g] for k in noise[g]] for g in range(6)]
    rank = max(len(f) for f in full)
    kraus_full = np.zeros((6, rank, 4, 4), dtype=complex)
    for g, f in enumerate(full):
        kraus_full[g, :len(f)] = f

    effects = []                                     # outcome index 2 r0 + r1, qubit pair[0] = left factor
    for q in names:
        e0, e1 = ep.readout_errors[q]["0"], ep.readout_errors[q]["1"]        # P(read 1 | 0), P(read 0 | 1)
        effects.append([np.diag([1 - e0, e1]), np.diag([e0, 1 - e1])])
    povm = np.array([np.kron(effects[0][r0], effects[1][r1]) for r0 in range(2) for r1 in range(2)])
    rho1 = np.diag([1 - excited_population, excited_population])
    state = np.kron(rho1, rho1)

    X = kraus_tensor_to_mgst(jnp.asarray(kraus_full))
    E, rho = jnp.asarray(povm.reshape(4, 16)), jnp.asarray(state.reshape(16))
    kraus4, povm4, state4 = get_compressed_rep_from_mgst_output(X, E, rho, kraus_rank=4, state_rank=4, povm_rank=4)
    # Truncating the Kraus rank breaks trace preservation (sum_r K_r^dagger K_r != I by the discarded Choi weight).
    # Project back onto the model (a Stiefel manifold per gate): the closest isometry to the stacked Kraus operators,
    # i.e. the polar factor U V^dagger of their singular value decomposition.
    stacked = np.asarray(kraus4).reshape(6, -1, 4)
    u, _, vh = np.linalg.svd(stacked, full_matrices=False)
    kraus4 = (u @ vh).reshape(np.asarray(kraus4).shape)
    compressed = (jnp.asarray(kraus4), jnp.asarray(povm4), jnp.asarray(state4))
    if not return_full:
        return compressed
    full_set = tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(X, E, rho, kraus_rank=16, state_rank=4, povm_rank=4))   # Choi rank <= d^2 = 16
    mu = np.asarray(choi_eigenvalues(full_set[0]))
    table = pd.DataFrame({"1 - mu_1": 1 - mu[:, 0], "Choi weight beyond rank 4": mu[:, 4:].sum(axis=1),
                          "mu_2..mu_6": [np.round(m[1:6], 6) for m in mu]})
    return compressed, full_set, table


# ---------------------------------------------------------------- near-gauge directions and depth check

def pauli_representation(theta, shapes):
    """Gate set as real vectors in the Pauli basis: PTMs R_g (flattened), state r_p = tr(P_p rho) / 2 and effects
    e_ip = tr(P_p E_i) / 2, so that p_i = e_i . R r for one gate (E_i = A_i^dagger A_i, the mGST psd convention)."""
    kraus, povm, state = to_ops(theta, shapes)
    P = jnp.asarray(pauli_basis(_num_qubits(kraus.shape[-1])))
    norm = np.sqrt(kraus.shape[-1])
    E = jnp.einsum("irb,irc->ibc", povm.conj(), povm)
    r = jnp.real(jnp.einsum("pab,ba->p", P, state @ state.conj().T)) / norm
    e = jnp.real(jnp.einsum("pab,iba->ip", P, E)) / norm
    return pauli_transfer_matrices(kraus), r, e


def gl_gauge_residuals(point, fd, ks):
    """Relative residual of Fisher eigendirections ks after removing the best-fitting GL gauge direction.

    The probabilities are invariant under the full gauge group of invertible, trace-preserving frame changes M:
    R_g -> M R_g M^-1, r -> M r, e_i -> M^-T e_i. The Kraus model only has its unitary subgroup as an exact symmetry,
    but a tangent direction whose effect on (R, r, e) is, to first order, G R_g - R_g G, G r, -G^T e_i for some
    generator G (first row zero) changes no probability to first order: a near-gauge direction. Residual ~ 0 means
    the direction is (nearly) such a GL gauge direction; strong directions give residuals ~ 1."""
    kraus, povm, state = point["operators"]
    shapes = _shapes(kraus, povm, state)
    theta = to_vec(kraus, povm, state)
    R, r, e = (np.asarray(x) for x in pauli_representation(theta, shapes))
    n = R.shape[-1]
    cols = []
    for i in range(1, n):
        for j in range(n):
            G = np.zeros((n, n)); G[i, j] = 1
            cols.append(np.concatenate([(G @ R - R @ G).reshape(-1), G @ r, (-e @ G).reshape(-1)]))
    L = np.array(cols).T
    flat = lambda th: jnp.concatenate([x.reshape(-1) for x in pauli_representation(th, shapes)])
    out = []
    for k in ks:
        d = np.asarray(jax.jvp(flat, (theta,), (jnp.asarray(point["T"] @ fd["V"][:, k]),))[1])
        coef, *_ = np.linalg.lstsq(L, d, rcond=None)
        out.append(np.linalg.norm(d - L @ coef) / np.linalg.norm(d))
    return np.array(out)


def output_spread(p, lengths):
    """Per sequence length l: mean total variation distance between the output distributions of the circuits of
    length l and their mean, s_l. p has shape (num_circuits, num_outcomes). When s_l is comparable to the shot-noise
    total variation distance (`tvd_noise`), the circuits of that length can no longer be told apart."""
    ls = np.unique(lengths)
    return ls, np.array([0.5 * np.abs(p[lengths == l] - p[lengths == l].mean(axis=0)).sum(axis=1).mean() for l in ls])


def tvd_noise(p, m):
    """Expected total variation distance between the frequencies of m shots and p, for large m:
    0.5 sum_i E|f_i - p_i| = 0.5 sum_i sqrt(2 p_i (1 - p_i) / (pi m)) (mean absolute deviation of a Gaussian)."""
    return 0.5 * np.sqrt(2 * p * (1 - p) / (np.pi * m)).sum(axis=-1)


def depth_check(point, N, L_list, seed, m_grid, tau, n_weak=40, gl_tol=1e-2, verbose=True):
    """Per L_max (one design seed, one N): the widths of the PTM elements and rates, both with all Fisher directions
    and without the near-GL-gauge ones (residual < gl_tol among the n_weak weakest), plus the outcome probabilities.

    Dropping the near-gauge directions approximates the widths of GL-gauge-fixed quantities: their components along
    those directions would be removed by a gauge fixing over the full gauge group (the implicit-function correction of
    PTM_PRECISION eq. 2 is not applied)."""
    kraus, povm, state = point["operators"]
    rows = []
    for L in L_list:
        t0 = time.time()
        indices = generate_sequence_indices(num_gates=kraus.shape[0], num_circuits=N, seq_len_list=seq_len_list(L), seed=seed)["gate_indices"]
        fd = fisher_decomposition(kraus, povm, state, indices, num_shots=1, basis=point["T"])
        t_fisher = time.time() - t0
        ks = np.arange(fd["rank"] - n_weak, fd["rank"])
        resid = gl_gauge_residuals(point, fd, ks)
        near = ks[resid < gl_tol]
        keep = np.setdiff1d(np.arange(fd["rank"]), near)
        fd_ng = dict(fd, V=fd["V"][:, keep], lam=fd["lam"][keep])
        row = dict(L=L, N=N, seed=seed, rank=fd["rank"], t_fisher=t_fisher, n_near_gauge=len(near), gl_resid=resid, lam1=fd["lam"],
                   lengths=np.array([len(s) for s in indices]), p=fd["p"].reshape(len(indices), -1))
        for tag, f in [("", fd), ("_ng", fd_ng)]:
            w = element_widths(point["D"], f, m_grid, [tau])
            row["estimable" + tag] = w["estimable"].mean()
            row["sigma_lan1" + tag] = w["sigma_lan1"]
            row["max_vt" + tag] = w["sigma_vt"][0].max(axis=0)
            row["q95_vt" + tag] = np.quantile(w["sigma_vt"][0], 0.95, axis=0)
            row["rate_sigma1" + tag] = lan_sigma1(point["D_rate"], f)
        rows.append(row)
        if verbose:
            print(f"L_max {L:2d} N {N}: rank {fd['rank']}, near-gauge directions {len(near)}, max sigma_LAN(m=1000) "
                  f"{row['sigma_lan1'].max() / np.sqrt(1000):.2e} (without near-gauge {row['sigma_lan1_ng'].max() / np.sqrt(1000):.2e}), "
                  f"Fisher {t_fisher:.0f}s, total {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- design scan

def seq_len_list(L_max):
    """[1, L_max // 2 + 1, L_max]; gives [1, 8, 14] for L_max = 14, as in scripts 07-16."""
    return [1, max(3, L_max // 2 + 1), L_max]


def required_shots(z_sigma, m_grid, delta):
    """Smallest m with z sigma(m) <= delta, by log-log interpolation on m_grid (z_sigma decreasing in m).
    Returns m_grid[0] if the target is met on the whole grid and inf if it is never met."""
    ok = z_sigma <= delta
    if not ok.any():
        return np.inf
    i = int(np.argmax(ok))
    if i == 0:
        return float(m_grid[0])
    x0, x1 = np.log(m_grid[i - 1]), np.log(m_grid[i])
    y0, y1 = np.log(z_sigma[i - 1]), np.log(z_sigma[i])
    return float(np.exp(x0 + (np.log(delta) - y0) * (x1 - x0) / (y1 - y0)))


def prepare_point(gate_set, target_ptms):
    """Everything about the evaluation point that does not depend on the design: the gauge-fixed operators, the tangent
    basis T, the gradients D of the non-constant gauge-fixed Pauli transfer matrix elements and of the small Choi
    eigenvalues (both in tangent coordinates)."""
    (kraus, povm, state), info = gauge_fix_to_target(*gate_set, target_ptms)
    T = np.asarray(ga.tangent_space_basis(kraus, povm, state))
    D_all = gauge_fixed_ptm_gradient(kraus, povm, state, target_ptms) @ T
    keep = np.linalg.norm(D_all, axis=1) > 1e-10                   # the constant first rows vanish on the tangent space
    mu_small, rate_grad = small_choi_gradient(kraus, povm, state)
    return dict(operators=(kraus, povm, state), gauge_info=info, T=T, D=D_all[keep], keep=keep,
                mu_small=mu_small, D_rate=rate_grad @ T)


def design_scan(point, N_list, L_list, seeds, m_grid, taus, eps=0.01, batch_size=None, verbose=True):
    """Fisher-based design scan (test T1 of PTM_PRECISION 5) at an evaluation point from `prepare_point`.

    Per design (L_max, N, design seed): the Fisher information at m = 1, and from it the per-element widths. Stored per
    design: the LAN width of every element at m = 1, the largest and median van Trees width and the median and 75th
    percentile of the weak share on m_grid, the LAN width at m = 1 of the small Choi eigenvalues, and the eigenvalues.
    """
    kraus, povm, state = point["operators"]
    rows = []
    for L in L_list:
        for N in N_list:
            for seed in seeds:
                t0 = time.time()
                indices = generate_sequence_indices(num_gates=kraus.shape[0], num_circuits=N, seq_len_list=seq_len_list(L), seed=seed)["gate_indices"]
                fd = fisher_decomposition(kraus, povm, state, indices, num_shots=1, basis=point["T"], batch_size=batch_size)
                w = element_widths(point["D"], fd, m_grid, taus, eps)
                rows.append(dict(L=L, N=N, seed=seed, n_circ=len(indices), rank=fd["rank"], estimable=w["estimable"].mean(),
                                 sigma_lan1=w["sigma_lan1"], max_vt=w["sigma_vt"].max(axis=1), med_vt=np.median(w["sigma_vt"], axis=1),
                                 weak_med=np.median(w["weak_share"], axis=0), weak_q75=np.quantile(w["weak_share"], 0.75, axis=0),
                                 rate_sigma1=lan_sigma1(point["D_rate"], fd), lam1=fd["lam"]))
                if verbose:
                    r = rows[-1]
                    print(f"L_max {L:2d} N {N:4d} seed {seed}: rank {r['rank']}, estimable {100 * r['estimable']:5.1f}%, "
                          f"max sigma_LAN(m=1000) {r['sigma_lan1'].max() / np.sqrt(1000):.2e}, "
                          f"max sigma(mu)/mu (m=1000) {(r['rate_sigma1'] / np.sqrt(1000) / point['mu_small']).max():.2f}, {time.time() - t0:.0f}s", flush=True)
    return rows


def save_scan(path, rows, point, m_grid, taus, eps, tau0, label):
    np.savez(path, rows=np.array(rows, dtype=object), m_grid=m_grid, taus=np.asarray(taus), eps=eps, tau0=tau0, label=label,
             mu_small=point["mu_small"], n_elements=int(point["keep"].sum()))


def load_scan(path):
    data = np.load(path, allow_pickle=True)
    meta = {k: data[k] for k in data.files if k != "rows"}
    meta["label"], meta["tau0"], meta["eps"] = str(meta["label"]), float(meta["tau0"]), float(meta["eps"])
    return list(data["rows"]), meta


def bonferroni_z(n_elements, level=0.95):
    """z such that all n elements are within z sigma with probability >= level (union bound)."""
    return float(norm.ppf(1 - (1 - level) / (2 * n_elements)))


# ---------------------------------------------------------------- summaries

def design_grid(rows, meta, quantity, z=1.96, tau_index=0, reduce=np.median):
    """Array (len(L_list), len(N_list), len(m_grid)) of a quantity on the (L_max, N, m) grid, reduced over design seeds.

    quantity: "lan" (z max_f sigma_LAN), "vt" (z max_f sigma_vT at taus[tau_index]), "weak" (median weak share),
    "rate" (max over the small Choi eigenvalues of sigma(mu) / mu).
    """
    m = np.asarray(meta["m_grid"], dtype=float)
    L_list, N_list = sorted({r["L"] for r in rows}), sorted({r["N"] for r in rows})
    fn = {"lan": lambda r: z * r["sigma_lan1"].max() / np.sqrt(m),
          "vt": lambda r: z * r["max_vt"][tau_index],
          "weak": lambda r: r["weak_med"],
          "rate": lambda r: (r["rate_sigma1"] / meta["mu_small"]).max() / np.sqrt(m)}[quantity]
    out = np.array([[reduce([fn(r) for r in rows if r["L"] == L and r["N"] == N], axis=0) for N in N_list] for L in L_list])
    return out, L_list, N_list


def scan_table(rows, meta, delta, z=1.96, rate_threshold=0.5):
    """One row per (L_max, N), medians over design seeds: required shots per circuit for a 95% half-width delta of the
    worst element (LAN; van Trees for each tau), total shots N m, the fraction of estimable elements, the seed spread
    of the LAN width, and at the LAN-required m: the median weak share and the worst rate resolution sigma(mu) / mu."""
    m_grid, taus, mu = np.asarray(meta["m_grid"], dtype=float), meta["taus"], meta["mu_small"]
    out = []
    for L in sorted({r["L"] for r in rows}):
        for N in sorted({r["N"] for r in rows}):
            rs = [r for r in rows if r["L"] == L and r["N"] == N]
            s1 = np.array([r["sigma_lan1"].max() for r in rs])
            m_lan = float(np.median((z * s1 / delta) ** 2))
            rec = {"L_max": L, "N": N, "m LAN": m_lan}
            for t, tau in enumerate(taus):
                rec[f"m vT({tau:.3g})"] = float(np.median([required_shots(z * r["max_vt"][t], m_grid, delta) for r in rs]))
            rec["N m LAN (1e6)"] = N * m_lan / 1e6
            for t, tau in enumerate(taus):
                rec[f"N m vT({tau:.3g}) (1e6)"] = N * rec[f"m vT({tau:.3g})"] / 1e6
            rec["estimable %"] = 100 * float(np.median([r["estimable"] for r in rs]))
            rec["seed spread LAN"] = float(s1.max() / s1.min()) if np.isfinite(s1).all() else np.nan
            i = min(np.searchsorted(m_grid, m_lan), len(m_grid) - 1) if np.isfinite(m_lan) else len(m_grid) - 1
            rec["weak share at m LAN"] = float(np.median([r["weak_med"][i] for r in rs]))
            rec["max sigma(mu)/mu at m LAN"] = float(np.median([(r["rate_sigma1"] / mu).max() / np.sqrt(m_lan) for r in rs]))
            rec["rates resolved"] = rec["max sigma(mu)/mu at m LAN"] <= rate_threshold
            out.append(rec)
    return pd.DataFrame(out)


def scaling_slopes(rows, meta, m_ref=1000, L_fit=None, N_fit=None, tau_index=0):
    """Slopes of log(max_f sigma) at m_ref versus log L_max (per N) and versus log N (per L_max), for LAN and van Trees
    (test T2). Returns two DataFrames."""
    lan, L_list, N_list = design_grid(rows, meta, "lan", z=1)
    vt, _, _ = design_grid(rows, meta, "vt", z=1, tau_index=tau_index)
    i = np.searchsorted(np.asarray(meta["m_grid"], dtype=float), m_ref)
    L_fit = L_list if L_fit is None else L_fit
    N_fit = [N for N in N_list if N >= 400] if N_fit is None else N_fit
    li, ni = [L_list.index(L) for L in L_fit], [N_list.index(N) for N in N_fit]
    slope = lambda x, y: np.polyfit(np.log(x), np.log(y), 1)[0]
    vs_L = pd.DataFrame([{"N": N_list[n], "LAN": slope(L_fit, lan[li, n, i]), "van Trees": slope(L_fit, vt[li, n, i])} for n in range(len(N_list))])
    vs_N = pd.DataFrame([{"L_max": L_list[l], "LAN": slope(N_fit, lan[l, ni, i]), "van Trees": slope(N_fit, vt[l, ni, i])} for l in range(len(L_list))])
    return vs_L, vs_N


def candidate_designs(rows, meta, delta, z=1.96, rate_threshold=0.5):
    """Decision rule for the regime where the incoherent part exceeds delta (PTM_PRECISION 3.5): per (L_max, N), the
    design needs m_design = max(m_LAN, m_rate), with m_LAN from z max_f sigma_LAN <= delta and m_rate from
    max sigma(mu) / mu <= rate_threshold (rates resolved, so that LAN can hold). Medians over design seeds.

    Returns (per_design, cheapest): all (L_max, N) with m_LAN, m_rate, m_design, the binding constraint and N m_design,
    and the cheapest N per L_max in total shots."""
    out = []
    for L in sorted({r["L"] for r in rows}):
        for N in sorted({r["N"] for r in rows}):
            rs = [r for r in rows if r["L"] == L and r["N"] == N]
            m_lan = np.array([(z * r["sigma_lan1"].max() / delta) ** 2 for r in rs])
            m_rate = np.array([((r["rate_sigma1"] / meta["mu_small"]).max() / rate_threshold) ** 2 for r in rs])
            m_design = np.maximum(m_lan, m_rate)
            out.append({"L_max": L, "N": N, "m LAN": np.median(m_lan), "m rate": np.median(m_rate),
                        "m design": np.median(m_design), "m design min": m_design.min(), "m design max": m_design.max(),
                        "binding": "LAN" if np.median(m_lan) >= np.median(m_rate) else "rates",
                        "N m design (1e6)": N * np.median(m_design) / 1e6})
    per_design = pd.DataFrame(out)
    cheapest = per_design.loc[per_design.groupby("L_max")["N m design (1e6)"].idxmin()].reset_index(drop=True)
    return per_design, cheapest


# ---------------------------------------------------------------- regime-aware decision, cost, incremental scans

def regime(mu_small, num_gates, delta):
    """"A" if the bias cap 3 max_g (1 - mu_1,g) is below delta / 2, else "B" (VALIDITY 7.1 step 4; the margin 2 is a
    judgment call). mu_small: the small Choi eigenvalues as stored by `save_scan`, (num_gates * (rank - 1),); for a
    trace-preserving gate 1 - mu_1 is their sum. Regime A: the unresolved incoherent part cannot spoil delta, design
    to the van Trees width. Regime B: the rates must be resolved, design to max(LAN, rate)."""
    cap = 3 * np.asarray(mu_small).reshape(num_gates, -1).sum(axis=1).max()
    return ("A" if cap < delta / 2 else "B"), float(cap)


def design_requirements(rows, meta, delta, z=1.96, regime_label=None, num_gates=6, rate_threshold=0.5, tau_index=0, m_max=None):
    """Per (L_max, N), medians over design seeds: required shots per circuit for a 95% half-width delta of the worst
    element under LAN, van Trees (taus[tau_index]) and rate resolution, and m_design by the regime:
    A: m_vT (LAN is the conservative end); B: max(m_LAN, m_rate). Also the binding constraint, the total shots, the
    seed spread of m_design and whether m_design <= m_max."""
    m_grid, mu = np.asarray(meta["m_grid"], dtype=float), np.asarray(meta["mu_small"])
    label = regime_label or regime(mu, num_gates, delta)[0]
    out = []
    for L in sorted({r["L"] for r in rows}):
        for N in sorted({r["N"] for r in rows}):
            rs = [r for r in rows if r["L"] == L and r["N"] == N]
            if not rs:
                continue
            m_lan = np.array([(z * r["sigma_lan1"].max() / delta) ** 2 for r in rs])
            m_vt = np.array([required_shots(z * r["max_vt"][tau_index], m_grid, delta) for r in rs])
            m_rate = np.array([((r["rate_sigma1"] / mu).max() / rate_threshold) ** 2 for r in rs])
            m_design = m_vt if label == "A" else np.maximum(m_lan, m_rate)
            rec = {"L_max": L, "N": N, "seeds": len(rs), "regime": label, "m LAN": np.median(m_lan), "m vT": np.median(m_vt),
                   "m rate": np.median(m_rate), "m design": np.median(m_design),
                   "binding": "vT" if label == "A" else ("LAN" if np.median(m_lan) >= np.median(m_rate) else "rates"),
                   "seed spread": float(m_design.max() / m_design.min()) if len(rs) > 1 else np.nan,
                   "N m (1e6)": N * np.median(m_design) / 1e6, "estimable %": 100 * float(np.median([r["estimable"] for r in rs]))}
            rec["feasible"] = True if m_max is None else bool(rec["m design"] <= m_max)
            out.append(rec)
    return pd.DataFrame(out)


def cost_table(req, r_list=(0.0, 1e2, 1e3, 1e4)):
    """Cheapest feasible (L_max, N) for the cost N (r + m_design) per circuit-cost ratio r (the cost of one circuit in
    units of one shot; r = 0 is total shots). req from `design_requirements`."""
    feasible = req[req["feasible"]]
    out = []
    for r in r_list:
        cost = feasible["N"] * (r + feasible["m design"])
        i = cost.idxmin()
        out.append({"circuit cost r (shots)": r, "L_max": feasible.at[i, "L_max"], "N": feasible.at[i, "N"],
                    "m design": feasible.at[i, "m design"], "cost (1e6 shot-equivalents)": cost[i] / 1e6})
    return pd.DataFrame(out)


def depth_returns(req):
    """Local exponents d log m_design / d log L_max between neighbouring L_max, per N. -1: doubling L_max halves the
    shots; -0.3: it saves about 20%; > 0: more depth hurts."""
    out = {}
    for N, t in req.groupby("N"):
        t = t.sort_values("L_max")
        L, m = t["L_max"].to_numpy(float), t["m design"].to_numpy(float)
        out[N] = pd.Series(np.diff(np.log(m)) / np.diff(np.log(L)), index=[f"{int(a)}->{int(b)}" for a, b in zip(L[:-1], L[1:])])
    return pd.DataFrame(out).T.rename_axis("N")


def extend_scan(cache_out, point, N_list, L_list, seeds, m_grid, taus, eps, tau0, label, cache_in=None, batch_size=None):
    """Design scan over N_list x L_list x seeds that reuses the rows of cache_out (if it exists) or cache_in, computes
    only the missing designs with `design_scan`, and saves everything to cache_out. The reused cache must have the
    same m_grid and taus."""
    rows = []
    for path in [cache_out, cache_in]:
        if path is not None and Path(path).exists():
            old, meta = load_scan(path)
            if not (np.allclose(meta["m_grid"], m_grid) and np.allclose(meta["taus"], taus)):
                raise ValueError(f"{path}: different m_grid or taus, cannot reuse")
            rows = [r for r in old if r["N"] in N_list and r["L"] in L_list and r["seed"] in seeds]
            break
    done = {(r["L"], r["N"], r["seed"]) for r in rows}
    for L in L_list:
        for N in N_list:
            for seed in seeds:
                if (L, N, seed) not in done:
                    rows += design_scan(point, [N], [L], [seed], m_grid, taus, eps, batch_size=batch_size)
                    save_scan(cache_out, rows, point, m_grid, taus, eps, tau0, label)    # save after every design
    save_scan(cache_out, rows, point, m_grid, taus, eps, tau0, label)
    return rows


# ---------------------------------------------------------------- plots

TEAL_NAVY = LinearSegmentedColormap.from_list("teal_navy", ["#E6F7F5", "#9FE3DC", "#2EC4B6", "#294C60", "#14232D"])
INK, MUTED = "#1F2933", "0.45"
BUDGETS = (1e5, 3e5, 1e6, 3e6, 1e7)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(colors="0.35", labelsize=8)


def _sci(x):
    e = int(np.floor(np.log10(x)))
    c = x / 10 ** e
    return rf"$10^{{{e}}}$" if np.isclose(c, 1) else rf"${c:.0f}\cdot10^{{{e}}}$"


def _axes_log_N_m(ax, N_list, ylim, pad=0.0):
    ax.set_xticks(np.log10(N_list))
    ax.set_xticklabels([str(N) for N in N_list])
    ax.set_xlim(np.log10(N_list[0]) - pad, np.log10(N_list[-1]) + pad)
    ax.set_ylim(*ylim)
    ticks = np.arange(np.ceil(ylim[0]), np.floor(ylim[1]) + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"$10^{{{int(t)}}}$" for t in ticks])
    _style(ax)


def _budget_lines(ax, N_list, ylim, budgets=BUDGETS, label=True):
    """Lines of constant total shots N m (slope -1 in log-log), labelled where they leave the panel on the right."""
    x = np.log10([N_list[0], N_list[-1]])
    for b in budgets:
        y = np.log10(b) - x
        ax.plot(x, y, color="0.72", lw=0.7, zorder=3)
        if label and ylim[0] < y[1] < ylim[1] - 0.1:
            ax.text(x[1] - 0.01, y[1] + 0.04, rf"$Nm = ${_sci(b)}", fontsize=6.5, color=MUTED, ha="right", va="bottom", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def plot_width_map(rows, meta, quantity="lan", z=1.96, tau_index=0, deltas=(2e-2, 1e-2, 5e-3, 2e-3), ylim=(2, 6),
                   extra_line=None, title=None):
    """One panel per L_max: filled contours of z max_f sigma over (N, m), contour lines at the target half-widths deltas
    (everything above a line reaches that delta for every element), and lines of constant total shots.

    quantity: "lan" or "vt" (van Trees at taus[tau_index]). extra_line: (tau_index, delta) to add the van Trees
    contour of another tau as a dotted line."""
    grid, L_list, N_list = design_grid(rows, meta, quantity, z=z, tau_index=tau_index)
    m = np.log10(np.asarray(meta["m_grid"], dtype=float))
    bounds = np.log10([1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 1.0])
    norm_ = BoundaryNorm(bounds, TEAL_NAVY.N)
    fig, axes = plt.subplots(1, len(L_list), figsize=(3.2 * len(L_list), 3.4), dpi=150, sharey=True, constrained_layout=True)
    X, Y = np.meshgrid(np.log10(N_list), m)
    for ax, L, G in zip(np.atleast_1d(axes), L_list, grid):
        Zg = np.log10(np.where(np.isfinite(G), G, 1.0)).T
        cf = ax.contourf(X, Y, Zg, levels=bounds, cmap=TEAL_NAVY, norm=norm_, extend="both")
        cs = ax.contour(X, Y, Zg, levels=np.log10(sorted(deltas)), colors=INK, linewidths=1.1, linestyles="solid", negative_linestyles="solid", zorder=5)
        ax.clabel(cs, fmt={lv: rf"$\delta$={10 ** lv:.0e}" for lv in cs.levels}, fontsize=6.5, inline_spacing=2)
        if extra_line is not None:
            G2, _, _ = design_grid(rows, meta, "vt", z=z, tau_index=extra_line[0])
            ax.contour(X, Y, np.log10(G2[L_list.index(L)]).T, levels=[np.log10(extra_line[1])], colors=INK, linewidths=1.0, linestyles=":", negative_linestyles=":", zorder=5)
        _budget_lines(ax, N_list, ylim)
        _axes_log_N_m(ax, N_list, ylim)
        ax.set_title(rf"$L_{{\max}}$ = {L}", fontsize=9, color=INK)
        ax.set_xlabel("Number of circuits $N$", fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("Shots per circuit $m$", fontsize=8.5)
    cbar = fig.colorbar(cf, ax=axes, pad=0.01, aspect=30, ticks=bounds[1:-1])
    cbar.set_ticklabels([f"{10 ** b:.0e}" for b in bounds[1:-1]])
    cbar.set_label(rf"${z:.2f}\,\max_f\sigma_f$ (95% half-width, worst element)", fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, colors="0.35")
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


def plot_validity_map(rows, meta, z=1.96, delta=1e-2, tau_index=0, ylim=(2, 6), rate_lines=(0.5, 1.0), title=None):
    """Two rows of panels per L_max over (N, m). Top: median weak share over elements (metric 3). Bottom: max over the
    small Choi eigenvalues of sigma(mu) / mu, with lines at rate_lines (LAN needs sigma(mu) << 2 mu, PTM_PRECISION 3.4).
    Solid and dashed ink lines: where the LAN and the van Trees width reach delta."""
    weak, L_list, N_list = design_grid(rows, meta, "weak")
    rate, _, _ = design_grid(rows, meta, "rate")
    lan, _, _ = design_grid(rows, meta, "lan", z=z)
    vt, _, _ = design_grid(rows, meta, "vt", z=z, tau_index=tau_index)
    m = np.log10(np.asarray(meta["m_grid"], dtype=float))
    X, Y = np.meshgrid(np.log10(N_list), m)
    fig, axes = plt.subplots(2, len(L_list), figsize=(3.2 * len(L_list), 6.4), dpi=150, sharex=True, sharey=True, constrained_layout=True)
    weak_bounds = np.linspace(0, 1, 6)
    rate_bounds = np.log10([1e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100])
    for j, L in enumerate(L_list):
        top, bottom = axes[0, j], axes[1, j]
        cf_w = top.contourf(X, Y, weak[j].T, levels=weak_bounds, cmap=TEAL_NAVY)
        cf_r = bottom.contourf(X, Y, np.log10(rate[j]).T, levels=rate_bounds, cmap=TEAL_NAVY, norm=BoundaryNorm(rate_bounds, TEAL_NAVY.N), extend="both")
        cs = bottom.contour(X, Y, np.log10(rate[j]).T, levels=np.log10(sorted(rate_lines)), colors="#C2410C", linewidths=1.2, linestyles="solid", negative_linestyles="solid", zorder=5)
        bottom.clabel(cs, fmt={lv: rf"$\sigma/\mu$={10 ** lv:.1f}" for lv in cs.levels}, fontsize=6.5)
        halo = [pe.withStroke(linewidth=3, foreground="white")]           # readable on the dark bands
        for ax in (top, bottom):
            ax.contour(X, Y, np.log10(lan[j]).T, levels=[np.log10(delta)], colors=INK, linewidths=1.1, linestyles="solid", negative_linestyles="solid", zorder=5).set_path_effects(halo)
            ax.contour(X, Y, np.log10(vt[j]).T, levels=[np.log10(delta)], colors=INK, linewidths=1.1, linestyles="--", negative_linestyles="--", zorder=5).set_path_effects(halo)
            _budget_lines(ax, N_list, ylim)
            _axes_log_N_m(ax, N_list, ylim)
        top.set_title(rf"$L_{{\max}}$ = {L}", fontsize=9, color=INK)
        bottom.set_xlabel("Number of circuits $N$", fontsize=8.5)
    axes[0, 0].set_ylabel("Shots per circuit $m$", fontsize=8.5)
    axes[1, 0].set_ylabel("Shots per circuit $m$", fontsize=8.5)
    for cf, ax_row, label, ticks, fmt in [(cf_w, axes[0], "median weak share", weak_bounds, "{:.1f}"),
                                          (cf_r, axes[1], r"$\max\,\sigma(\mu)/\mu$ (small Choi eigenvalues)", rate_bounds[1:-1], None)]:
        cbar = fig.colorbar(cf, ax=ax_row, pad=0.01, aspect=20, ticks=ticks)
        cbar.set_ticklabels([fmt.format(t) for t in ticks] if fmt else [f"{10 ** t:g}" for t in ticks])
        cbar.set_label(label, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, colors="0.35")
    handles = [plt.Line2D([], [], color=INK, lw=1.1, label=rf"LAN reaches $\delta$={delta:.0e}"),
               plt.Line2D([], [], color=INK, lw=1.1, ls="--", label=rf"van Trees ($\tau$={meta['taus'][tau_index]:.3f}) reaches $\delta$"),
               plt.Line2D([], [], color="#C2410C", lw=1.2, label=r"rate resolution $\sigma(\mu)/\mu$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


def plot_decision(rows, meta, delta=1e-2, z=1.96, tau_index=0, rate_threshold=0.5, ylim=(2, 6), title=None):
    """One panel per L_max: required shots per circuit versus N for a 95% half-width delta of the worst element.
    LAN (solid), van Trees at taus[tau_index] (dashed), rate resolution sigma(mu)/mu <= rate_threshold (dotted), and the
    design requirement max(LAN, rates) as a wide band; error bars: min-max over design seeds. Gray lines: constant total
    shots N m, as in the maps. The ring marks the cheapest N in total shots per L_max, the filled ring the cheapest
    overall."""
    per_design, cheapest = candidate_designs(rows, meta, delta, z, rate_threshold)
    m_grid = np.asarray(meta["m_grid"], dtype=float)
    L_list, N_list = sorted({r["L"] for r in rows}), sorted({r["N"] for r in rows})
    best = cheapest.loc[cheapest["N m design (1e6)"].idxmin()]
    fig, axes = plt.subplots(1, len(L_list), figsize=(3.2 * len(L_list), 3.4), dpi=150, sharey=True, constrained_layout=True)
    x = np.log10(N_list)
    for ax, L in zip(np.atleast_1d(axes), L_list):
        d = per_design[per_design["L_max"] == L]
        m_vt = [np.median([required_shots(z * r["max_vt"][tau_index], m_grid, delta) for r in rows if r["L"] == L and r["N"] == N]) for N in N_list]
        ax.plot(x, np.log10(d["m design"]), color="#9FE3DC", lw=7, alpha=0.7, solid_capstyle="round", zorder=1, label=r"design: max(LAN, rates)")
        ax.vlines(x, np.log10(d["m design min"]), np.log10(d["m design max"]), color="#294C60", lw=1.0, zorder=2)
        ax.plot(x, np.log10(d["m LAN"]), "o-", color="#294C60", lw=1.4, ms=4, zorder=4, label="LAN width")
        ax.plot(x, np.log10(m_vt), "s--", color="#2EC4B6", lw=1.4, ms=3.5, zorder=4, label=rf"van Trees, $\tau$={meta['taus'][tau_index]:.3f}")
        ax.plot(x, np.log10(d["m rate"]), "^:", color="#C2410C", lw=1.2, ms=3.5, zorder=4, label=rf"rates resolved, $\sigma(\mu)/\mu\leq${rate_threshold}")
        c = cheapest[cheapest["L_max"] == L].iloc[0]
        filled = c["L_max"] == best["L_max"]
        ax.plot(np.log10(c["N"]), np.log10(c["m design"]), "o", ms=12, mfc=INK if filled else "none", mec=INK, mew=1.5, alpha=0.35 if filled else 1, zorder=5)
        ax.annotate(f"N={c['N']}, m≈{c['m design']:.0f}\n{c['N m design (1e6)']:.2g}M shots", (np.log10(c["N"]), np.log10(c["m design"])),
                    xytext=(-10, 12), textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", lw=0.5))
        _budget_lines(ax, N_list, ylim, label=False)
        _axes_log_N_m(ax, N_list, ylim, pad=0.04)
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_title(rf"$L_{{\max}}$ = {L}", fontsize=9, color=INK)
        ax.set_xlabel("Number of circuits $N$", fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel(rf"Shots per circuit $m$ for $\delta$={delta:.0e}", fontsize=8.5)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig, per_design, cheapest
