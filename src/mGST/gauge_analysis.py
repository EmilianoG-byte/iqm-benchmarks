"""Module containing the tools to study then Gauge present in our model"""

import jax.numpy as jnp
from mGST.typing import Matrix
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