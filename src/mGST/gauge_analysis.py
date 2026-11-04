"""Module containing the tools to study then Gauge present in our model"""

import numpy as np
import jax
import jax.numpy as jnp
from mGST.typing import Matrix, Tensor
from mGST.linear_algebra import generate_anti_hermitian_basis, is_anti_hermitian, is_square_matrix, random_anti_hermitian_matrix

def velocity_at_x(x:Matrix, A: Matrix)->Matrix:
    "Compute the tangent velocity vector at x"
    _basic_checks_for_isometry(A)
    n, p = x.shape
    dim_A = A.shape[0]
    if dim_A > n:
        raise ValueError(f"A must have dimensions compatible with x. Got {A.shape}, expected at most ({n}, {n})")
    dim_id = int(n / dim_A)
    identity = jnp.eye(dim_id)
    # here we are assuming that the isometry n dimension is ordered as (kraus_rank, dim_id).
    return jnp.kron(A, identity) @ x 

def _basic_checks_for_isometry(A: Matrix):
    "Check that A is a square anti-hermitian matrix"
    if not is_square_matrix(A):
        raise ValueError("A must be a square matrix")
    if not is_anti_hermitian(A):
        raise ValueError("A must be anti-hermitian")

def velocity_at_x_mps(x:Matrix, A: Matrix)->Matrix:
    """Compute the tangent velocity vector at x, when the Gauge transformation is applied along the bond dimension (MPS style)
    """
    _basic_checks_for_isometry(A)
    n, p = x.shape
    dim_A = A.shape[0]
    if dim_A != p:
        raise ValueError(f"A must have dimensions compatible with x. Got {A.shape}, expected ({p}, {p})")
    
    # we split back into kraus tensor dimensions
    kraus_rank = int(n / p)
    identity = jnp.eye(kraus_rank)
    # here we are assuming that the isometry n dimension is ordered as (kraus_rank, dim_id).
    return jnp.kron(identity, A.conj().T) @ x @ A

def random_vertical_velocity(x:Matrix, seed:int, dim_A: int)->Matrix:
    "Generate a random vertical velocity vector at x"
    A = random_anti_hermitian_matrix(dim_A, seed)
    return velocity_at_x(x=x, A=A)


VALID_GAUGES = ["kraus", "mps"]

def generate_vertical_fundamental_vectors(x: Matrix, dim_A: int, gauge_type:str)->Matrix:
    "Generate the vertical fundamental vectors at x"
    if gauge_type not in VALID_GAUGES:
        raise ValueError(f"Invalid gauge_type. Expected one of {VALID_GAUGES}, got {gauge_type}")
    basis = generate_anti_hermitian_basis(dim_A)
    if gauge_type == "kraus":
        velocity_fn = velocity_at_x
    elif gauge_type == "mps":
        velocity_fn = velocity_at_x_mps
    return jnp.array([velocity_fn(x=x, A=A) for A in basis])

def total_num_parameters(num_qubits:int, num_gates:int, kraus_rank:int, state_rank:int, povm_rank:int, verbose: bool = False) -> int:
    """Compute the total number of parameters in the model given the ranks of the state, povm and gates
    
    Args:
        num_qubits: number of qubits in the system
        num_gates: number of gates in the model
        kraus_rank: rank of the gates
        state_rank: rank of the state
        povm_rank: rank of the povm
    Returns:
        total number of parameters in the model
    """
    
    dim = 2 ** num_qubits
    num_povm = dim
    if not 1<= state_rank <= dim:
        raise ValueError(f"state_rank must be between 1 and {dim}")
    if not 1<= povm_rank <= dim:
        raise ValueError(f"povm_rank must be between 1 and {dim}")
    if not 1<= kraus_rank <= dim**2:
        raise ValueError(f"kraus_rank must be between 1 and {dim**2}")

    dim_povm = dim_stiefel_tangent_space(num_povm*povm_rank, dim)
    dim_gates = num_gates * dim_stiefel_tangent_space(dim*kraus_rank, dim)
    dim_state = dim_stiefel_tangent_space(dim*state_rank, 1)

    if verbose:
        print(f"dim_povm: {dim_povm}, dim_gates: {dim_gates}, dim_state: {dim_state}")

    return dim_povm + dim_gates + dim_state
    
def dim_stiefel_tangent_space(n:int, p:int) -> int:
    """Compute the dimension of the tangent space of the Stiefel manifold at a point of size (n, p)"""
    return 2*n*p - p**2  
    
def max_num_parameters(num_qubits:int, num_gates:int, kraus_rank:int, state_rank:int = None, povm_rank:int = None, verbose: bool = False) -> int:
    """Compute the maximum number of parameters in the model given the ranks of the gates, assuming full rank for the state and povm"""
    dim = 2 ** num_qubits
    if state_rank is None:
        state_rank = dim
    if povm_rank is None:
        povm_rank = dim
        
    kraus_rank = _set_kraus_rank(kraus_rank, dim)    
    return total_num_parameters(num_qubits, num_gates, kraus_rank, state_rank, povm_rank, verbose=verbose)

def min_num_parameters(num_qubits:int, num_gates:int, kraus_rank:int|str, state_rank:int = None, povm_rank:int = None, verbose: bool = False) -> int:
    """Compute the minimum number of parameters after gauge fixing.

    This counts only the gauge-invariant degrees of freedom by subtracting the
    gauge group dimension (d^2 + rank^2 per object) from each component's tangent
    space dimension.

    Args:
        num_qubits: Number of qubits in GST
        num_gates: Number of gates in the gate set.
        kraus_rank: Kraus rank r_K of each gate.
        state_rank: Rank r_B of the initial state. If None, it is assumed to be full rank (d).
        povm_rank: Rank r_A of each POVM effect. If None, it is assumed to be full rank (d).

    Returns:
        Minimum number of gauge-invariant parameters.
    """
    dim = 2 ** num_qubits
    num_povm = dim

    if state_rank is None:
        state_rank = dim
    if povm_rank is None:
        povm_rank = dim

    kraus_rank = _set_kraus_rank(kraus_rank, dim)
    gauge_povm = num_povm * unitary_group_dimension(povm_rank)
    gauge_gates = num_gates * unitary_group_dimension(kraus_rank)
    gauge_state = unitary_group_dimension(state_rank)
    mps_gauge = unitary_group_dimension(dim)
    
    num_parameters = total_num_parameters(num_qubits, num_gates, kraus_rank, state_rank, povm_rank)
    
    if verbose:
        print(f"Gauge dimensions: POVM={gauge_povm}, Gates={gauge_gates}, State={gauge_state}")
    return num_parameters - (gauge_povm + gauge_gates + gauge_state + mps_gauge)

def _set_kraus_rank(kraus_rank:int|str, dim:int) -> int:
    """Compute the Kraus rank for a given dimension.

    Args:
        kraus_rank: Kraus rank r_K of each gate. Can be an integer or the string "max".
        dim: Dimension of the Hilbert space (2^num_qubits)
    Returns:
        Maximum Kraus rank (d^2) for the given number of qubits.
    """
    if isinstance(kraus_rank, str):
        if kraus_rank == "max":
            kraus_rank = dim**2
        else:
            raise ValueError(f"Invalid kraus_rank string: {kraus_rank}. Expected 'max' or an integer.")
    else:
        if not 1 <= kraus_rank <= dim**2:
            raise ValueError(f"kraus_rank must be between 1 and {dim**2}, got {kraus_rank}.")
    
    return kraus_rank

def unitary_group_dimension(dimension:int) -> int:
    """Compute the dimension of the unitary group U(dimension).

    Args:
        dimension: Dimension of the unitary group.

    Returns:
        Dimension of the unitary group U(dimension).
    """
    return dimension**2

def _operators_to_real_vector(*operators: Tensor) -> jnp.ndarray:
    """Stack the real and imaginary parts of all operators into one real vector."""
    return jnp.concatenate([jnp.concatenate([op.real.reshape(-1), op.imag.reshape(-1)]) for op in operators])

def _real_vector_to_operators(theta: jnp.ndarray, shapes: list[tuple[int, ...]]) -> list[Tensor]:
    """Inverse of `_operators_to_real_vector`."""
    operators, start = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        operators.append((theta[start:start + size] + 1j * theta[start + size:start + 2 * size]).reshape(shape))
        start += 2 * size
    return operators

def _isometry_constraints(theta: jnp.ndarray, shapes: list[tuple[int, ...]]) -> jnp.ndarray:
    """Defining function g of the model (product of Stiefel manifolds): g(theta) = 0 exactly on the model.

    Real and imaginary parts of sum_i K_i^dagger K_i - I (one per gate) and of sum_ij A_ij^dagger A_ij - I (POVM),
    and Tr(B B^dagger) - 1 (state). The rows are redundant (Hermitian matrices), only the kernel of Dg matters.
    """
    kraus_tensor, povm_psd, state_psd = _real_vector_to_operators(theta, shapes)
    dim = state_psd.shape[0]
    gates = jnp.einsum("grij,grik->gjk", kraus_tensor.conj(), kraus_tensor) - jnp.eye(dim)
    povm = jnp.einsum("prj,prk->jk", povm_psd.conj(), povm_psd) - jnp.eye(dim)
    state = jnp.real(jnp.trace(state_psd.conj().T @ state_psd)) - 1
    hermitian = jnp.concatenate([gates.reshape(-1), povm.reshape(-1)])
    return jnp.concatenate([hermitian.real, hermitian.imag, state[None]])

def _numerical_rank(singular_values: jnp.ndarray, shape: tuple[int, int], rtol: float | None) -> int:
    """Number of singular values above rtol * sigma_max. Default rtol as in numpy.linalg.matrix_rank."""
    if rtol is None:
        rtol = max(shape) * jnp.finfo(singular_values.dtype).eps
    return int(jnp.sum(singular_values > rtol * singular_values[0]))

def tangent_space_basis(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, rtol: float | None = None) -> Matrix:
    """Orthonormal basis T of the tangent space of the model at (kraus_tensor, povm_psd, state_psd).

    The tangent space is ker Dg, with g the defining function `_isometry_constraints`, and T spans the null space of Dg
    (right singular vectors with zero singular value). T T^T is the orthogonal projector onto the tangent space.
    Coordinates are those of `_operators_to_real_vector`, with the Euclidean inner product Re Tr(X^dagger Y).

    Args:
        kraus_tensor: Kraus operators. Shape (num_gates, kraus_rank, dim, dim).
        povm_psd: PSD roots of the POVM elements. Shape (num_povm, povm_rank, dim).
        state_psd: PSD root of the state. Shape (dim, state_rank).
        rtol: Relative tolerance for the rank of Dg. See `_numerical_rank`.

    Returns:
        Matrix of shape (num_real_parameters, max_num_parameters) with orthonormal columns.
    """
    shapes = [kraus_tensor.shape, povm_psd.shape, state_psd.shape]
    theta = _operators_to_real_vector(kraus_tensor, povm_psd, state_psd)
    constraints = lambda t: _isometry_constraints(t, shapes)
    violation = float(jnp.abs(constraints(theta)).max())
    if violation > 1e-8:
        raise ValueError(f"The operators are not on the model manifold (max constraint violation {violation:.1e}).")
    constraint_jacobian = jax.jacfwd(constraints)(theta)
    _, singular_values, vt = jnp.linalg.svd(constraint_jacobian)  # full_matrices=True, so vt is square
    rank = _numerical_rank(singular_values, constraint_jacobian.shape, rtol)
    return vt[rank:].T

def _circuit_probabilities(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, gates: jnp.ndarray) -> Tensor:
    """Outcome probabilities of all circuits. Shape (num_circuits, num_povm).

    Same contraction as the "padded" strategy of `jitted_objective.make_vectorized_cost`: column t of gates holds
    the t-th gate applied to the state, and the padding -1 is the identity channel.
    """
    _, kraus_rank, dim, _ = kraus_tensor.shape
    identity_gate = jnp.zeros((1, kraus_rank, dim, dim), kraus_tensor.dtype).at[0, 0].set(jnp.eye(dim))
    kraus_extended = jnp.concatenate([kraus_tensor, identity_gate], axis=0)  # index -1 is the identity

    def apply_position(rho, gates_t):
        kraus = kraus_extended[gates_t]
        return jnp.einsum("bijk,bkl,biml->bjm", kraus, rho, kraus.conj()), None

    rho_init = state_psd @ state_psd.conj().T
    rho, _ = jax.lax.scan(apply_position, jnp.broadcast_to(rho_init, (gates.shape[0],) + rho_init.shape), gates.T)
    return jnp.real(jnp.einsum("ijk,bkl,ijl->bi", povm_psd, rho, povm_psd.conj()))

@jax.jit
def _tangent_space_jacobian(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, basis: Matrix, gates: jnp.ndarray) -> Matrix:
    """J_T = J T: derivatives of the circuit probabilities along the columns of basis.

    Returns:
        Matrix of shape (num_circuits * num_povm, basis.shape[1]). Rows are ordered circuit by circuit.
    """
    shapes = [kraus_tensor.shape, povm_psd.shape, state_psd.shape]

    def probabilities(coordinates):
        d_kraus, d_povm, d_state = _real_vector_to_operators(basis @ coordinates, shapes)
        return _circuit_probabilities(kraus_tensor + d_kraus, povm_psd + d_povm, state_psd + d_state, gates).reshape(-1)

    return jax.jacfwd(probabilities)(jnp.zeros(basis.shape[1]))

def probability_jacobian_spectrum(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, indices_list: list,
                                  num_shots: int | None = None, rtol: float | None = None, batch_size: int | None = None) -> dict:
    """Singular values and rank of the Jacobian of the circuit probabilities on the tangent space of the model.

    The rank is the number of gauge-invariant directions that the circuits constrain at this point (local
    identifiability in the noiseless limit). Gauge directions are in the kernel automatically. The rank is at most
    min((num_povm - 1) * num_circuits, max_num_parameters - gauge dimension). It depends on the point: e.g. at
    exactly unitary gates the incoherent directions are not seen to first order.

    If num_shots is given, also returns the eigenvalues of the Fisher information of the outcome counts,
    F = num_shots * J_T^T diag(1/p) J_T. By the Cramer-Rao bound, an unbiased estimator has standard deviation at least
    1/sqrt(lambda_k) along the k-th eigenvector of F (in the units of `tangent_space_basis`).

    Args:
        kraus_tensor: Kraus operators. Shape (num_gates, kraus_rank, dim, dim).
        povm_psd: PSD roots of the POVM elements. Shape (num_povm, povm_rank, dim).
        state_psd: PSD root of the state. Shape (dim, state_rank).
        indices_list: Gate sequences without padding, e.g. indices_dict["gate_indices"].
        num_shots: Number of shots per circuit. If None, the Fisher eigenvalues are not computed.
        rtol: Singular values below rtol * sigma_max count as zero. Default max(J_T.shape) * machine epsilon.
        batch_size: Number of circuits per Jacobian evaluation, to reduce memory. If None, all circuits at once.
            Regardless of the batch size, the Jacobian will be computed for all circuits.

    Returns:
        Dictionary with
        - "rank": numerical rank of J_T.
        - "singular_values": singular values of J_T, in decreasing order.
        - "num_independent_data": (num_povm - 1) * num_circuits.
        - "dim_tangent_space": dimension of the tangent space (`max_num_parameters`).
        - "fisher_eigenvalues": eigenvalues of F, in decreasing order (only if num_shots is given).
    """
    basis = tangent_space_basis(kraus_tensor, povm_psd, state_psd)
    num_circuits, num_povm = len(indices_list), povm_psd.shape[0]
    max_length = max(1, max(len(indices) for indices in indices_list))
    gates = np.full((num_circuits, max_length), -1, dtype=np.int32)
    for row, indices in enumerate(indices_list):
        gates[row, :len(indices)] = np.asarray(indices, dtype=np.int32)[::-1]  # application order

    batch_size = num_circuits if batch_size is None else batch_size
    jacobian = jnp.concatenate([
        _tangent_space_jacobian(kraus_tensor, povm_psd, state_psd, basis, jnp.asarray(gates[start:start + batch_size]))
        for start in range(0, num_circuits, batch_size)
    ])
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    result = {
        "rank": _numerical_rank(singular_values, jacobian.shape, rtol),
        "singular_values": singular_values,
        "num_independent_data": (num_povm - 1) * num_circuits,
        "dim_tangent_space": basis.shape[1],
    }

    if num_shots is not None:
        probabilities = _circuit_probabilities(kraus_tensor, povm_psd, state_psd, jnp.asarray(gates)).reshape(-1)
        if float(probabilities.min()) < 1e-12:
            raise ValueError("The Fisher information is not defined for outcome probabilities equal to zero.")
        # eigenvalues of F from the singular values of diag(p)^(-1/2) J_T, without squaring the condition number
        weighted_singular_values = jnp.linalg.svd(jacobian / jnp.sqrt(probabilities)[:, None], compute_uv=False)
        result["fisher_eigenvalues"] = num_shots * weighted_singular_values**2
    return result