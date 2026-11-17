"""Matrix-free Fisher-information analysis, for gate sets that are too large for the dense route of `ptm_design.py`
(four qubits: 9 gates of Kraus rank 4 have 18 047 tangent directions, and 10 000 circuits give 160 000 probabilities,
so the whitened Jacobian W alone would need 23 GB). Implements VALIDITY.md Sections 4.5, 4.6, 7 and 8.

Nothing here builds the Jacobian, a tangent basis or an eigen-decomposition. Everything is expressed with products
of the Fisher information with a vector,

    F_1 v = P J^T diag(1/p) J P v      (Fisher information of one shot of every circuit; F = m F_1 for m shots),

where J is the Jacobian of the circuit probabilities in the real coordinates of `gauge_analysis._operators_to_real_vector`,
applied with `jax.jvp` (J v) and `jax.vjp` (J^T w), and P is the orthogonal projector onto the tangent space of the
model, applied factor by factor with `mGST.riemannian.project_onto_tangent_space`. In an orthonormal tangent basis T
this is the matrix T^T (P J^T diag(1/p) J P) T = J_T^T diag(1/p) J_T of README Section 4.5, so for a tangent gradient d

    d^T F^+ d   (README 4.6)   and   d^T (m F_1 + I / tau^2)^-1 d   (van Trees, DERIVATIONS E.3)

are the same numbers as in `ptm_design.py`. Vectors are "ambient": real vectors of length 2 * (number of complex
entries of kraus, povm, state). Tangent vectors are ambient vectors with P v = v.

What is computed per quantity f (a real function of (kraus, povm, state)):
    1. its tangent gradient d_f, with the gauge-fixing correction of PTM_PRECISION eq. (2) if f is gauge dependent;
    2. one Lanczos run on F_1 started at d_f (`mGST.stochastic_lanczos_quadrature.lanczos_from_vector`). Its Gauss
       quadrature gives d_f^T g(F_1) d_f for every function g at once, hence the van Trees width for every number of
       shots m and prior width tau, and a lower bound on the LAN width (LAN = local asymptotic normality);
    3. optionally one conjugate-gradient solve (m F_1 + P / tau^2) v = d_f, which gives the direction v_f needed for
       the nonlinearity ratio kappa (VALIDITY 4.5 b) and for the optimizer certificate (VALIDITY 4.6).

Brackets. For g(x) = 1 / (m x + 1 / tau^2) the Gauss rule is a lower bound and the Gauss-Radau rule with a prescribed
node at 0 an upper bound on d^T g(F_1) d (Golub & Meurant, "Matrices, Moments and Quadrature with Applications",
Princeton University Press 2010, Ch. 6; cited from memory, verified numerically against the dense result in script 20).
The van Trees width is therefore returned as an interval. For the LAN width (tau -> infinity) only the lower bound
exists, and it converges slowly because the spectrum of F_1 spans many decades (README 5.3).

Cost: one product F_1 v is one forward-mode and one reverse-mode pass through all circuits. Measured on a laptop CPU at
four qubits (9 gates, Kraus rank 4): about 1.8 s per 1000 circuits with contraction="scan" (the existing
`gauge_analysis._circuit_probabilities`) and about 0.6 s with contraction="grouped" (after about 50 s of compilation).
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from mGST import gauge_analysis as ga
from mGST.riemannian import project_onto_tangent_space, retraction_polar_decomposition
from mGST.utility_functions_comparisons import tensors_to_isometries, isometry_to_tensor
from mGST.stochastic_lanczos_quadrature import lanczos_from_vector, nodes_and_weights_from_lanczos

import ptm_design as pdz

OPERATOR_TYPES = ("kraus", "povm", "state")
CONTRACTIONS = ("scan", "grouped")


# ---------------------------------------------------------------- counting

def num_identifiable_parameters(shapes):
    """Number of gauge-invariant parameters n_id of the model (VALIDITY Section 8; 624 for the two-qubit study).

    Each term is the dimension of a Stiefel factor minus its Kraus-mixing or purification gauge, and the last term is
    the Hilbert-space gauge. (`gauge_analysis.min_num_parameters` returns one less, see README Section 2.2.)
    """
    (num_gates, kraus_rank, dim, _), (num_povm, povm_rank, _), (_, state_rank) = shapes
    gates = num_gates * (2 * dim ** 2 * kraus_rank - dim ** 2 - kraus_rank ** 2)
    state = 2 * dim * state_rank - state_rank ** 2 - 1
    povm = 2 * num_povm * dim * povm_rank - dim ** 2 - num_povm * povm_rank ** 2
    return gates + state + povm - (dim ** 2 - 1)


def num_tangent_directions(shapes):
    """Dimension of the tangent space of the model (815 for the two-qubit study)."""
    (num_gates, kraus_rank, dim, _), (num_povm, povm_rank, _), (_, state_rank) = shapes
    return num_gates * (2 * dim ** 2 * kraus_rank - dim ** 2) + (2 * num_povm * dim * povm_rank - dim ** 2) + 2 * dim * state_rank - 1


# ---------------------------------------------------------------- tangent space, factor by factor

def project_tangent(operators, vector):
    """Orthogonal projection of an ambient vector onto the tangent space at `operators`.

    Per Stiefel factor X the projector is Z -> Z - X sym(X^dagger Z) (README 4.2), from `mGST.riemannian`.
    """
    shapes = pdz._shapes(*operators)
    parts = []
    for x, z, operator_type in zip(operators, pdz.to_ops(vector, shapes), OPERATOR_TYPES):
        x_iso, z_iso = tensors_to_isometries(x, z, operator_type=operator_type)
        parts.append(isometry_to_tensor(project_onto_tangent_space(x_iso, z_iso), x.shape))
    return pdz.to_vec(*parts)


def _normal_acceleration(operators, vector):
    """Second-order term of the polar retraction: R_X(t V) = X + t V - (t^2 / 2) X V^dagger V + O(t^3) per factor
    (expansion of (X + t V)(I + t^2 V^dagger V)^-1/2, DERIVATIONS C.2). Returns the ambient vector -X V^dagger V."""
    shapes = pdz._shapes(*operators)
    parts = []
    for x, z, operator_type in zip(operators, pdz.to_ops(vector, shapes), OPERATOR_TYPES):
        x_iso, z_iso = tensors_to_isometries(x, z, operator_type=operator_type)
        parts.append(isometry_to_tensor(-x_iso @ (jnp.swapaxes(z_iso.conj(), -1, -2) @ z_iso), x.shape))
    return pdz.to_vec(*parts)


def retract(operators, vector, step):
    """Polar retraction of the tangent vector `step * vector`: the gate set R_theta(step * vector).

    Uses `mGST.riemannian.retraction_polar_decomposition`, which moves along -z, hence the sign."""
    shapes = pdz._shapes(*operators)
    new = []
    for x, z, operator_type in zip(operators, pdz.to_ops(vector, shapes), OPERATOR_TYPES):
        x_iso, z_iso = tensors_to_isometries(x, z, operator_type=operator_type)
        polar = lambda xi, zi: retraction_polar_decomposition(x=xi, z=-zi, step_size=step)
        new.append(isometry_to_tensor(jax.vmap(polar)(x_iso, z_iso) if x_iso.ndim == 3 else polar(x_iso, z_iso), x.shape))
    return tuple(new)


# ---------------------------------------------------------------- circuit probabilities

def _grouped_probabilities(kraus, povm, state, gates):
    """Same probabilities as `gauge_analysis._circuit_probabilities`, about 3 times faster at four qubits.

    `gates` is a NumPy array (num_circuits, max_length) in application order with -1 padding, known at trace time.
    The loops over positions and gates are unrolled: each gate is applied to all circuits that contain it at that
    position with two large matrix products, and padding is never multiplied. Compilation time grows with
    max_length * num_gates.
    """
    rho_init = state @ state.conj().T
    rho = jnp.broadcast_to(rho_init, (gates.shape[0],) + rho_init.shape)
    for position in range(gates.shape[1]):
        for gate in range(kraus.shape[0]):
            rows = np.nonzero(gates[:, position] == gate)[0]
            if rows.size == 0:
                continue
            half = jnp.einsum("ijk,nkl->nijl", kraus[gate], rho[rows])
            rho = rho.at[rows].set(jnp.einsum("nijl,iml->njm", half, kraus[gate].conj()))
    return jnp.real(jnp.einsum("ijk,bkl,ijl->bi", povm, rho, povm.conj()))


@dataclass
class _Batch:
    """Jitted derivatives of the probabilities of one batch of circuits. `function` maps theta to (n, num_outcomes)."""
    function: Callable
    _jitted: dict = field(default_factory=dict, repr=False)

    def _get(self, name, build):
        if name not in self._jitted:
            self._jitted[name] = jax.jit(build())
        return self._jitted[name]

    def probabilities(self, theta):
        return self._get("p", lambda: self.function)(theta)

    def jvp(self, theta, v):
        """J v, shape (n, num_outcomes)."""
        return self._get("jvp", lambda: lambda t, u: jax.jvp(self.function, (t,), (u,))[1])(theta, v)

    def vjp(self, theta, cotangent):
        """J^T w, an ambient vector."""
        return self._get("vjp", lambda: lambda t, w: jax.vjp(self.function, t)[1](w)[0])(theta, cotangent)

    def product(self, theta, v, rare):
        """J^T C J v with C = diag(1/p), or the Fisher weighting of the data in which the outcomes flagged in `rare`
        are merged into one bin per circuit: their common weight is (sum of J v over the bin) / (sum of p over the bin)
        (VALIDITY 4.5 c). With `rare` all False this is J^T diag(1/p) J v."""
        def build():
            def product(t, u, mask):
                p, jv = jax.jvp(self.function, (t,), (u,))
                bin_jv = jnp.sum(jnp.where(mask, jv, 0.0), axis=1, keepdims=True)
                bin_p = jnp.sum(jnp.where(mask, p, 0.0), axis=1, keepdims=True)
                cotangent = jnp.where(mask, bin_jv / jnp.where(bin_p > 0, bin_p, 1.0), jv / p)
                return jax.vjp(self.function, t)[1](cotangent)[0]
            return product
        return self._get("product", build)(theta, v, rare)

    def curve_derivatives(self, theta, v, acceleration):
        """First and second derivative at t = 0 of the probabilities along theta + t v + (t^2 / 2) acceleration:
        J v and v^T (second derivative of p) v + J acceleration."""
        def build():
            def derivatives(t, u, a):
                first = lambda s: jax.jvp(self.function, (s,), (u,))[1]
                jv, second = jax.jvp(first, (t,), (u,))
                return jv, second + jax.jvp(self.function, (t,), (a,))[1]
            return derivatives
        return self._get("curve", build)(theta, v, acceleration)


# ---------------------------------------------------------------- the Fisher information as an operator

@dataclass
class FisherProblem:
    """Evaluation point and design. Build it with `fisher_problem`."""
    operators: tuple
    shapes: tuple
    theta: jnp.ndarray
    batches: list
    probabilities: np.ndarray                 # (num_circuits, num_outcomes) at the evaluation point
    batch_sizes: list = field(repr=False, default=None)
    _project: Callable = field(repr=False, default=None)

    @property
    def num_circuits(self):
        return self.probabilities.shape[0]

    def project(self, v):
        return self._project(v)

    def _split(self, array):
        sizes = np.cumsum([0] + list(self.batch_sizes))
        return [array[sizes[i]:sizes[i + 1]] for i in range(len(self.batches))]

    def jacobian_vector(self, v):
        """J P v: change of all probabilities per unit step along the tangent part of v. Shape (num_circuits, num_outcomes)."""
        v = self.project(v)
        return jnp.concatenate([batch.jvp(self.theta, v) for batch in self.batches])

    def jacobian_transpose(self, cotangent):
        """P J^T w for w of shape (num_circuits, num_outcomes)."""
        parts = self._split(jnp.asarray(cotangent))
        return self.project(sum(batch.vjp(self.theta, part) for batch, part in zip(self.batches, parts)))

    def matvec(self, v, rare=None):
        """F_1 v = P J^T diag(1/p) J P v: the Fisher information of one shot of every circuit, applied to an ambient
        vector. With `rare` (boolean, shape of `probabilities`) the flagged outcomes of each circuit are merged into
        one bin first (`rare_outcomes`)."""
        v = self.project(v)
        masks = self._split(jnp.zeros(self.probabilities.shape, dtype=bool) if rare is None else jnp.asarray(rare))
        return self.project(sum(batch.product(self.theta, v, mask) for batch, mask in zip(self.batches, masks)))

    def curve_derivatives(self, v):
        """First and second derivative of all probabilities along the polar-retraction curve t -> R_theta(t P v)."""
        v = self.project(v)
        acceleration = _normal_acceleration(self.operators, v)
        parts = [batch.curve_derivatives(self.theta, v, acceleration) for batch in self.batches]
        return jnp.concatenate([p[0] for p in parts]), jnp.concatenate([p[1] for p in parts])


def fisher_problem(kraus, povm, state, indices_list, batch_size=None, contraction="scan"):
    """Set up the Fisher operator at the gate set (kraus, povm, state) for the design `indices_list`.

    Args:
        kraus, povm, state: evaluation point. Shapes (num_gates, kraus_rank, dim, dim), (num_povm, povm_rank, dim),
            (dim, state_rank). It must lie on the model manifold.
        indices_list: gate sequences without padding, e.g. generate_sequence_indices(...)["gate_indices"].
        batch_size: number of circuits per forward / reverse pass, to bound memory. None: all circuits at once.
            The reverse pass stores about 80 kB per circuit and gate position at four qubits with Kraus rank 4.
        contraction: "scan" uses `gauge_analysis._circuit_probabilities` (compiles in seconds). "grouped" uses
            `_grouped_probabilities` (about 3 times faster per product, about a minute of compilation per batch).
    """
    if contraction not in CONTRACTIONS:
        raise ValueError(f"Invalid contraction {contraction}. Valid options are: {CONTRACTIONS}")
    operators = tuple(jnp.asarray(x) for x in (kraus, povm, state))
    shapes = pdz._shapes(*operators)
    theta = pdz.to_vec(*operators)
    violation = float(jnp.abs(ga._isometry_constraints(theta, list(shapes))).max())
    if violation > 1e-8:
        raise ValueError(f"The operators are not on the model manifold (max constraint violation {violation:.1e}).")
    gates = np.asarray(pdz.pad_sequences(indices_list))
    batch_size = gates.shape[0] if batch_size is None else batch_size
    batches = []
    for start in range(0, gates.shape[0], batch_size):
        chunk = gates[start:start + batch_size]
        if contraction == "scan":
            function = lambda t, chunk=jnp.asarray(chunk): ga._circuit_probabilities(*pdz.to_ops(t, shapes), chunk)
        else:
            function = lambda t, chunk=chunk: _grouped_probabilities(*pdz.to_ops(t, shapes), chunk)
        batches.append(_Batch(function))
    batch_probabilities = [np.asarray(batch.probabilities(theta)) for batch in batches]
    probabilities = np.concatenate(batch_probabilities)
    if probabilities.min() < 1e-12:
        raise ValueError("The Fisher information is not defined for outcome probabilities equal to zero.")
    return FisherProblem(operators=operators, shapes=shapes, theta=theta, batches=batches, probabilities=probabilities,
                         batch_sizes=[len(p) for p in batch_probabilities], _project=jax.jit(lambda v: project_tangent(operators, v)))


# ---------------------------------------------------------------- gauge fixing and gradients of quantities

def _gauge_cost(h, theta, shapes, target_kraus):
    """sum_g |R_g - R_g^target|_F^2 after the Hilbert-space gauge transformation U(h) of `ptm_design._act`, without
    building Pauli transfer matrices R.

    The Pauli transfer matrix is the matrix of the channel in an orthonormal operator basis, so the Frobenius distance
    equals that of the superoperators S = sum_r K_r (x) conj(K_r), and <S, S'> = sum_rs |tr(K_r^dagger K'_s)|^2.
    """
    kraus = pdz.to_ops(pdz._act(h, theta, shapes), shapes)[0]
    overlap = lambda a, b: jnp.sum(jnp.abs(jnp.einsum("grab,gsab->grs", a.conj(), b)) ** 2)
    return overlap(kraus, kraus) - 2 * overlap(kraus, target_kraus) + overlap(target_kraus, target_kraus)


_gauge_cost_and_grad = jax.jit(jax.value_and_grad(_gauge_cost), static_argnums=2)


def gauge_fix_to_target(kraus, povm, state, target_kraus, newton_steps=5):
    """Pi: apply the Hilbert-space unitary that brings the gate Pauli transfer matrices closest to the target ones
    (BFGS, twice, as `ptm_design.gauge_fix_to_target`, followed by Newton steps; same cost, see `_gauge_cost`).

    Args:
        target_kraus: target gates as Kraus operators, shape (num_gates, target_rank, dim, dim) (target_rank = 1 for
            unitary targets).

    Returns:
        (kraus, povm, state) after the gauge transformation, and a dict with the final cost and gradient norm.
    """
    shapes = pdz._shapes(kraus, povm, state)
    theta, h0 = pdz.to_vec(kraus, povm, state), np.zeros(state.shape[0] ** 2 - 1)
    target_kraus = jnp.asarray(target_kraus)
    for _ in range(2):
        fun = lambda h: tuple(np.asarray(x, dtype=float) for x in _gauge_cost_and_grad(jnp.asarray(h), theta, shapes, target_kraus))
        res = minimize(fun, h0, jac=True, method="BFGS", options=dict(gtol=1e-12, maxiter=2000))
        theta = pdz._act(jnp.asarray(res.x), theta, shapes)
    # Newton polish: BFGS stalls at a gradient of about 1e-5 for the 255 gauge parameters of four qubits.
    for _ in range(newton_steps):
        cost, gradient = _gauge_cost_and_grad(jnp.asarray(h0), theta, shapes, target_kraus)
        if float(jnp.abs(gradient).max()) < 1e-10:
            break
        hessian = np.asarray(jax.hessian(_gauge_cost, argnums=0)(jnp.asarray(h0), theta, shapes, target_kraus))
        theta = pdz._act(jnp.asarray(-np.linalg.solve(hessian, np.asarray(gradient))), theta, shapes)
    cost, gradient = _gauge_cost_and_grad(jnp.asarray(h0), theta, shapes, target_kraus)
    return tuple(pdz.to_ops(theta, shapes)), dict(cost=float(cost), grad=float(jnp.abs(gradient).max()))


def gauge_context(problem, target_kraus):
    """Second derivatives of the gauge cost at a gauge-fixed evaluation point, for `quantity_gradient`.

    Returns a dict with the target, the Hessian H_hh of the gauge cost with respect to the gauge parameters h
    ((dim^2 - 1) x (dim^2 - 1)) and its smallest eigenvalue. The gauge fixing is differentiable only if H_hh is
    invertible (assumption G1 of VALIDITY Section 3).
    """
    target_kraus = jnp.asarray(target_kraus)
    h0 = jnp.zeros(problem.shapes[2][0] ** 2 - 1)
    stationarity = float(jnp.abs(jax.grad(_gauge_cost)(h0, problem.theta, problem.shapes, target_kraus)).max())
    if stationarity > 1e-6:
        raise ValueError(f"The point is not gauge-fixed (gauge gradient {stationarity:.1e}); call gauge_fix_to_target first.")
    H_hh = np.asarray(jax.hessian(_gauge_cost, argnums=0)(h0, problem.theta, problem.shapes, target_kraus))
    return dict(target_kraus=target_kraus, H_hh=H_hh, min_eigenvalue=float(np.linalg.eigvalsh(H_hh).min()), h0=h0)


def quantity_gradient(problem, quantity, gauge=None):
    """Tangent gradient d_f (ambient vector with P d = d) of a real quantity f(kraus, povm, state).

    Args:
        quantity: function (kraus, povm, state) -> real scalar, written with jax.numpy.
        gauge: None for a gauge-invariant quantity. For a gauge-dependent one (a Pauli transfer matrix element, a
            fidelity to the target), the dict from `gauge_context`: the gradient is then that of f o Pi,
            grad f - (df/dh) H_hh^-1 H_h,theta (implicit function theorem, PTM_PRECISION eq. 2), with the last
            factor applied as one reverse-mode gradient instead of building H_h,theta.
    """
    theta, shapes = problem.theta, problem.shapes
    scalar = lambda t: quantity(*pdz.to_ops(t, shapes))
    gradient = jax.grad(scalar)(theta)
    if gauge is not None:
        df_dh = jax.grad(lambda h: scalar(pdz._act(h, theta, shapes)))(gauge["h0"])
        w = jnp.asarray(np.linalg.solve(gauge["H_hh"], np.asarray(df_dh)))
        gauge_gradient = lambda t: jax.grad(_gauge_cost, argnums=0)(gauge["h0"], t, shapes, gauge["target_kraus"])
        gradient = gradient - jax.grad(lambda t: jnp.vdot(w, gauge_gradient(t)))(theta)
    return problem.project(gradient)


# ---------------------------------------------------------------- quantities

def pauli_transfer_matrices(kraus):
    """R_g[i, j] = tr(P_i G_g(P_j)) / d, as `ptm_design.pauli_transfer_matrices`, through the Heisenberg-picture
    operators sum_r K_r^dagger P_i K_r so that no intermediate exceeds num_gates * rank * 4^n * d^2 entries."""
    dim = kraus.shape[-1]
    paulis = jnp.asarray(pdz.pauli_basis(pdz._num_qubits(dim)))
    evolved = jnp.einsum("grab,griad->gibd", kraus.conj(), jnp.einsum("iac,grcd->griad", paulis, kraus))
    return jnp.real(jnp.einsum("gibd,jdb->gij", evolved, paulis)) / dim


def ptm_element(gate, i, j):
    """Pauli transfer matrix element R_gate[i, j]. Gauge dependent: use with `gauge_context`."""
    def quantity(kraus, povm, state):
        dim = kraus.shape[-1]
        paulis = pdz.pauli_basis(pdz._num_qubits(dim))
        return jnp.real(jnp.einsum("ab,rbc,cd,rad->", paulis[i], kraus[gate], paulis[j], kraus[gate].conj())) / dim
    return quantity


def ptm_combination(coefficients):
    """sum_gij C[g, i, j] R_g[i, j] for real coefficients C of shape (num_gates, 4^n, 4^n). Gauge dependent.

    With independent standard normal C on the non-constant elements (rows i >= 1), the variance of this quantity
    averaged over C is the sum of the variances of the elements, so a few random C estimate the root-mean-square
    width over all elements (`random_ptm_coefficients`).
    """
    coefficients = jnp.asarray(coefficients)
    return lambda kraus, povm, state: jnp.sum(coefficients * pauli_transfer_matrices(kraus))


def random_ptm_coefficients(shapes, seed, gate=None):
    """Standard normal coefficients on the non-constant Pauli transfer matrix elements (rows i >= 1) of one gate, or
    of all gates if gate is None. Returns the coefficients and the number of elements they cover."""
    num_gates, _, dim, _ = shapes[0]
    C = np.random.default_rng(seed).normal(size=(num_gates, dim ** 2, dim ** 2))
    C[:, 0, :] = 0.0
    if gate is not None:
        C[np.arange(num_gates) != gate] = 0.0
    return C, int(np.count_nonzero(C))


def entanglement_infidelity(gate, target_kraus):
    """1 - sum_rs |tr(T_s^dagger K_r)|^2 / d^2: entanglement infidelity of `gate` to its target. Gauge dependent."""
    target = jnp.asarray(target_kraus)[gate]
    def quantity(kraus, povm, state):
        return 1 - jnp.sum(jnp.abs(jnp.einsum("sab,rab->sr", target.conj(), kraus[gate])) ** 2) / kraus.shape[-1] ** 2
    return quantity


def choi_eigenvalue(gate, k):
    """k-th largest eigenvalue of the normalized Choi matrix of `gate` (k = 0: coherent part; k >= 1: incoherent
    rates). Gauge invariant. Not differentiable where eigenvalues cross."""
    return lambda kraus, povm, state: pdz.choi_eigenvalues(kraus)[gate, k]


def state_eigenvalue(k):
    """k-th largest eigenvalue of the state rho = B B^dagger. Gauge invariant."""
    return lambda kraus, povm, state: jnp.linalg.eigvalsh(state.conj().T @ state)[::-1][k]


def effect_eigenvalue(outcome, k):
    """k-th largest eigenvalue of the POVM effect E_outcome = A^dagger A. Gauge invariant."""
    return lambda kraus, povm, state: jnp.linalg.eigvalsh(povm[outcome] @ povm[outcome].conj().T)[::-1][k]


def circuit_probability(sequence, outcome):
    """Predicted probability of `outcome` for one gate sequence (as in `indices_list`). Gauge invariant."""
    gates = pdz.pad_sequences([list(sequence)])
    return lambda kraus, povm, state: ga._circuit_probabilities(kraus, povm, state, gates)[0, outcome]


# ---------------------------------------------------------------- widths from one Lanczos run

def lanczos_quadrature(problem, d, num_steps, rare=None, breakdown_rtol=1e-10):
    """Lanczos recurrence for F_1 started at the tangent gradient d (full reorthogonalization).

    Returns a dict with the recurrence coefficients, |d|^2 and the number of steps kept. The run is truncated where
    the Krylov space becomes invariant (beta below breakdown_rtol * max |alpha|); the quadrature is then exact.
    """
    d = problem.project(d)
    norm2 = float(d @ d)
    alphas, betas = lanczos_from_vector(lambda v: problem.matvec(v, rare), d / np.sqrt(norm2), order_m=num_steps, reorth=True)
    alphas, betas = np.asarray(alphas, dtype=float), np.asarray(betas, dtype=float)
    small = np.nonzero(~(betas > breakdown_rtol * np.abs(alphas).max()))[0]          # also catches NaN
    exact = small.size > 0
    if exact:
        alphas, betas = alphas[:small[0] + 1], betas[:small[0]]
    return dict(alphas=alphas, betas=betas, norm2=norm2, exact=exact, num_steps=len(alphas))


def _gauss(quadrature):
    nodes, weights = nodes_and_weights_from_lanczos(jnp.asarray(quadrature["alphas"]), jnp.asarray(quadrature["betas"]))
    return np.asarray(nodes), np.asarray(weights)


def _gauss_radau_at_zero(quadrature):
    """Gauss-Radau rule with a prescribed node at 0 (a lower bound of the spectrum of F_1): the last diagonal entry of
    the Lanczos matrix is replaced by delta_k, where T_k delta = beta_k^2 e_k (Golub & Meurant 2010, Ch. 6)."""
    alphas, betas = quadrature["alphas"], quadrature["betas"]
    k = len(alphas) - 1
    T = np.diag(alphas[:k]) + np.diag(betas[:k - 1], 1) + np.diag(betas[:k - 1], -1)
    rhs = np.zeros(k)
    rhs[-1] = betas[k - 1] ** 2
    last = np.linalg.solve(T, rhs)[-1]
    nodes, weights = nodes_and_weights_from_lanczos(jnp.asarray(np.append(alphas[:k], last)), jnp.asarray(betas))
    return np.asarray(nodes), np.asarray(weights)


def van_trees_sigma(quadrature, m, tau):
    """Van Trees width sqrt(d^T (m F_1 + I / tau^2)^-1 d) (PTM_PRECISION 3.7) as an interval.

    Args:
        m: number of shots per circuit (scalar or array).
        tau: prior standard deviation per tangent direction.

    Returns:
        (lower, upper), each shaped like m: Gauss (lower) and Gauss-Radau (upper) quadrature. They coincide when the
        Lanczos run ended exactly. The true value lies in between, up to rounding.
    """
    m = np.asarray(m, dtype=float)
    value = lambda nodes, weights: np.sqrt(quadrature["norm2"] * (weights / (m[..., None] * np.clip(nodes, 0, None) + tau ** -2)).sum(axis=-1))
    lower = value(*_gauss(quadrature))
    if quadrature["exact"] or quadrature["num_steps"] < 2:
        return lower, (lower if quadrature["exact"] else np.full_like(lower, np.inf))
    return lower, value(*_gauss_radau_at_zero(quadrature))


def lan_sigma(quadrature, m, rtol=1e-10):
    """LAN width sqrt(d^T F^+ d) = sqrt(d^T F_1^+ d / m) from the Gauss rule: a lower bound that increases with the
    number of Lanczos steps and is exact when the run ended exactly.

    Returns:
        (sigma, kernel_share): kernel_share is the weight of d on Ritz values below rtol * (largest Ritz value). A
        value that is not small means that d has a component the design does not constrain at all: the quantity is not
        estimable and the LAN width is infinite (PTM_PRECISION 2.4).
    """
    nodes, weights = _gauss(quadrature)
    keep = nodes > rtol * nodes.max()
    return np.sqrt(quadrature["norm2"] * (weights[keep] / nodes[keep]).sum() / np.asarray(m, dtype=float)), float(weights[~keep].sum())


# ---------------------------------------------------------------- direction of a quantity, nonlinearity, optimizer certificate

def solve_direction(problem, d, m, tau, tol=1e-8, maxiter=1000, rare=None):
    """v = (m F_1 + P / tau^2)^-1 d by conjugate gradients (`jax.scipy.sparse.linalg.cg`).

    This is F^+ d_f of VALIDITY 4.5 (b) regularized by the prior width tau: the mean displacement of the gate set per
    unit change of f. d . v is the squared van Trees width. The condition number is at most 1 + m tau^2 lambda_max(F_1).

    Returns:
        (v, relative_residual).
    """
    d = problem.project(d)
    operator = lambda v: m * problem.matvec(v, rare) + problem.project(v) / tau ** 2
    v, _ = cg(operator, d, tol=tol, maxiter=maxiter)
    return v, float(jnp.linalg.norm(operator(v) - d) / jnp.linalg.norm(d))


def nonlinearity_ratio(problem, v, m):
    """Nonlinearity ratio kappa(v) of VALIDITY 4.5 (b): second-order over first-order change of the whitened
    predictions after one Cramer-Rao standard deviation along the tangent direction v,

        kappa = |diag(p)^-1/2 c''(0)| / (2 sqrt(m) v^T F_1 v),   c(t) = p(R_theta(t v)),

    with R the polar retraction. c''(0) is exact (two nested `jax.jvp`, plus the second-order term of the retraction).
    kappa scales like 1 / sqrt(m). Pass: kappa <= about 0.1 (kappa = 1 / (4 u) for a rate with resolution u).

    Returns a dict with kappa and the step t_1 = 1 / sqrt(m v^T F_1 v).
    """
    first, second = problem.curve_derivatives(v)
    sqrt_p = np.sqrt(problem.probabilities)
    information = float(np.sum((np.asarray(first) / sqrt_p) ** 2))                        # v^T F_1 v
    return dict(kappa=float(np.linalg.norm(np.asarray(second) / sqrt_p)) / (2 * np.sqrt(m) * information),
                step=1 / np.sqrt(m * information))


def nonlinearity_ratio_finite_step(problem, v, m):
    """The same ratio without second derivatives: |w(t_1) - t_1 w'(0)| with w(t) the whitened change of the
    predictions along the polar retraction (`retract`) and t_1 the one-standard-deviation step. It contains all orders
    beyond the first, so it agrees with `nonlinearity_ratio` only while kappa is small."""
    v = problem.project(v)
    first = np.asarray(problem.jacobian_vector(v))
    sqrt_p = np.sqrt(problem.probabilities)
    step = 1 / np.sqrt(m * np.sum((first / sqrt_p) ** 2))
    moved = pdz.to_vec(*retract(problem.operators, v, step))
    p_moved = np.concatenate([np.asarray(batch.probabilities(moved)) for batch in problem.batches])
    return float(np.linalg.norm(np.sqrt(m) * (p_moved - problem.probabilities - step * first) / sqrt_p))


def log_likelihood_gradient(problem, frequencies, m):
    """Riemannian gradient (tangent ambient vector) of the log-likelihood sum n log p at the evaluation point, for
    observed frequencies of shape (num_circuits, num_outcomes). Zero at an exact maximum-likelihood estimate."""
    return problem.jacobian_transpose(m * np.asarray(frequencies) / problem.probabilities)


def optimizer_shift(v, gradient):
    """Certificate of VALIDITY 4.6: first-order distance v . g of a quantity from its value at the exact maximum of
    the likelihood, for v from `solve_direction` and g from `log_likelihood_gradient`. Compare with the width."""
    return float(v @ gradient)


# ---------------------------------------------------------------- data-level checks

def rare_outcomes(problem, m, threshold=5.0):
    """Boolean mask of the outcomes with expected count m p below `threshold` (assumption R1 of VALIDITY Section 3).
    Pass it as `rare` to `lanczos_quadrature` for the merged-outcome width of VALIDITY 4.5 (c)."""
    return m * problem.probabilities < threshold


def expected_count_summary(problem, m, threshold=5.0):
    """Shares of outcomes with small expected counts at m shots per circuit."""
    counts = m * problem.probabilities
    return dict(min_probability=float(problem.probabilities.min()), share_below_threshold=float((counts < threshold).mean()),
                share_below_one=float((counts < 1).mean()), probability_mass_below_threshold=float(problem.probabilities[counts < threshold].sum() / problem.num_circuits))


def deviance(problem, frequencies, m):
    """Deviance 2 [l_sat - l(theta)] = 2 m sum_c KL(f_c || p_c(theta)) at the evaluation point (VALIDITY 4.7), with
    the Wilks reference k = (num_outcomes - 1) * num_circuits - n_id and N_sigma = (deviance - k) / sqrt(2 k).

    N_sigma is given for reference only: the chi-squared reference is not valid when rates are unresolved or outcomes
    are rare (VALIDITY Section 5). Calibrate it with simulated refits.
    """
    f, p = np.asarray(frequencies), problem.probabilities
    value = 2 * m * float(np.sum(np.where(f > 0, f * np.log(np.where(f > 0, f, 1.0) / p), 0.0)))
    k = (p.shape[1] - 1) * p.shape[0] - num_identifiable_parameters(problem.shapes)
    return dict(deviance=value, wilks_degrees_of_freedom=k, n_sigma=(value - k) / np.sqrt(2 * k))
