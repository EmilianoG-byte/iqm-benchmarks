"""
All functions compiled with numba, such as tensor contractions, derivatives...
"""

import os

from numba import njit, prange
import numpy as np
import warnings


def kill_files(folder):
    """Delete all files in the specified folder.

    This function iterates over all files in the given folder and attempts to delete them.
    If a file cannot be deleted, an error message is printed.

    Parameters
    ----------
    folder : str
        The path to the folder containing the files to be deleted.
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except OSError as e:
            print(f"Failed on filepath: {file_path}. Error: {e}")


def kill_numba_cache():
    """Delete all __pycache__ folders in the project directory tree.

    This function iterates through the project directory tree, looking for __pycache__ folders.
    When found, it attempts to delete all files within the __pycache__ folders using the
    `kill_files` function. If the files cannot be deleted, an error message is printed.
    """
    root_folder = os.path.realpath(__file__ + "/../../")
    for root, dirnames, _ in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except OSError as e:
                    print(f"Failed on {root}. Error: {e}")


@njit(cache=True)
def local_basis(x, b, length):
    """Convert a base-10 integer to an integer in a specified base with a fixed number of digits.

    This function takes an integer `x` in base-10 and converts it to base `b`.
    The result is returned as an array of length `length` with leading zeros.

    Parameters
    ----------
    x : int
        The input number in base-10 to be converted.
    b : int
        The target base to convert the input number to.
    length : int
        The number of output digits in the target base representation.

    Returns
    -------
    numpy.ndarray
        A numpy array of integers representing the base-`b` digits of the converted number,
        with leading zeros if necessary. The length of the array is `length`.
    """
    r = np.zeros(length).astype(np.int32)
    k = 1
    while x > 0:
        r[-k] = x % b
        x //= b
        k += 1
    return r


@njit(cache=True)
def contract(X, j_vec):
    """Contract a sequence of matrices in the given order.

    This function computes the product of a sequence of matrices specified by
    the indices in `j_vec`. The result is the contracted product of the matrices
    in the given order.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
    j_vec : numpy.ndarray
        A 1D array of indices specifying the order in which to contract the matrices in X.

    Returns
    -------
    numpy.ndarray
        The contracted product of the matrices specified by the indices in `j_vec`.
    """
    j_vec = j_vec[j_vec >= 0]
    res = np.eye(X[0].shape[0])
    res = res.astype(np.complex128)
    for j in j_vec:
        res = res.dot(X[j])
    return res


@njit(cache=True, fastmath=True)  # , parallel=True)
def objf(X, E, rho, J, y, mle=False):
    """Calculate the objective function value for matrices, POVM elements, and target values.

    This function computes the objective function value based on input matrices X, POVM elements E,
    density matrix rho, and target values y.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the objective function will be evaluated.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    mle : bool
        If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns
    -------
    float
        The objective function value for the given set of matrices, POVM elements,
        and target values, normalized by m and n_povm.
    """
    m = len(J)
    n_povm = y.shape[0]
    objf_: float = 0
    for i in prange(m):  # pylint: disable=not-an-iterable
        j = J[i][J[i] >= 0]
        state = rho
        for ind in j[::-1]:
            state = X[ind] @ state
        for o in range(n_povm):
            if mle:
                objf_ -= np.log(abs(E[o].conj() @ state)) * y[o, i]
            else:
                objf_ += abs(E[o].conj() @ state - y[o, i]) ** 2 / m / n_povm
    return objf_

def cost_function_numba(K, E, rho, J, y):
    num_gates = K.shape[0]
    dim = K.shape[2]
    # einsum is not supported by numba
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((num_gates, dim**2, dim**2))
    
    return objf(X, E, rho, J, y)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def contract_mps_general_povm(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int])->jnp.ndarray:
    """Compute the inner product contraction <povm|kraus|state> for a single matrix in povm_psd

    NOTE: Actually, due to the ellipssis this should work in general for all POVM's as well. The only difference is that the output will be a vector of dimension num_povm instead of a scalar.

    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
        
    Returns:
        jnp.ndarray: tensor of dimension: (num_povm), corresponding to the contracted inner products <povm_i| K_{j_n} ... K_{j_1} |state>
        for i in range(num_povm).
    """
    # Initialize right tensor as the state
    right_tensor = state_psd @ state_psd.conj().T  # dim_up_in, dim_down_in
    # right_tensor = state_psd
    optimal_path = [(0, 1), (0, 1)]
    # Iterate through the Kraus tensors in reverse order
    for idx in reversed(gates_indices):
        k = kraus_tensor[idx]  # kraus_rank, dim_up_out, dim_up_in
        # (kraus_rank, dim_up_out, dim_up_in) x (dim_up_in, dim_down_in) x (kraus_rank, dim_down_out, dim_down_in) -> dim_up_out, dim_down_in
        right_tensor = jnp.einsum("ijk,kl,iml->jm", k, right_tensor, k.conj(), optimize=optimal_path)
        
    # (num_povm, rank_povm, dim_up_in) x (dim_up_in, dim_down_in) x (num_povm, rank_povm, dim_down_in) -> num_povm
    return jnp.einsum("...jk, kl, ...jl ->...", povm_psd, right_tensor, povm_psd.conj(), optimize=optimal_path)

def contract_mps_abstract_povm_coherent(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int])->jnp.ndarray:
    """Compute the inner product contraction <povm|kraus|state> for all matrices in povm_psd using the optimal contraction path for a coherent quantum channel (i.e. kraus rank = 1).
    
    WARNING: the following contraction only makes sense assuming that the kraus rank is 1, i.e. the quantum channel is coherent/unitary.
    
    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank = 1, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
    """

    # Erase leg with dimension 1
    kraus_tensor = jnp.squeeze(kraus_tensor) # num_gates, dim_out, dim_in

    right_tensor = state_psd  # dim_in, rank_state
    for idx in reversed(gates_indices):
        k = kraus_tensor[idx]  # dim_out, dim_in
        right_tensor = k @ right_tensor  # dim_out, rank_state
        
    contracted_tensor = povm_psd @ right_tensor # num_povm, rank_povm, rank_state
    return jnp.einsum("...jk, ...jk -> ...", contracted_tensor, contracted_tensor.conj()) # num_povm

def contract_mps_all_povm(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int])->jnp.ndarray:
    """Compute the inner product contraction <povm|kraus|state> for all matrices in povm_psd

    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
        
    Returns:
        jnp.ndarray: tensor of dimension: (num_povm), corresponding to the contracted inner products <povm_i| K_{j_n} ... K_{j_1} |state>
        for i in range(num_povm).
    """
    # Initialize right tensor as the state
    right_tensor = state_psd @ state_psd.conj().T  # dim_up_in, dim_down_in
    # right_tensor = state_psd
    optimal_path = [(0, 1), (0, 1)]
    # Iterate through the Kraus tensors in reverse order
    for idx in reversed(gates_indices):
        k = kraus_tensor[idx]  # kraus_rank, dim_up_out, dim_up_in
        # (kraus_rank, dim_up_out, dim_up_in) x (dim_up_in, dim_down_in) x (kraus_rank, dim_down_out, dim_down_in) -> dim_up_out, dim_down_in
        right_tensor = jnp.einsum("ijk,kl,iml->jm", k, right_tensor, k.conj(), optimize=optimal_path)
        
    # (num_povm, rank_povm, dim_up_in) x (dim_up_in, dim_down_in) x (num_povm, rank_povm, dim_down_in) -> num_povm
    return jnp.einsum("ijk, kl, ijl -> i", povm_psd, right_tensor, povm_psd.conj(), optimize=optimal_path)
    # return jnp.sum(povm_psd.conj() * right_tensor.T, axis=(1, 2))
    
def contract_mps_all_povm_coherent(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int])->jnp.ndarray:
    """Compute the inner product contraction <povm|kraus|state> for all matrices in povm_psd using the optimal contraction path for a coherent quantum channel (i.e. kraus rank = 1).
    
    WARNING: the following contraction only makes sense assuming that the kraus rank is 1, i.e. the quantum channel is coherent/unitary.
    
    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank = 1, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
    """

    # Erase leg with dimension 1
    kraus_tensor = jnp.squeeze(kraus_tensor) # num_gates, dim_out, dim_in

    right_tensor = state_psd  # dim_in, rank_state
    for idx in reversed(gates_indices):
        k = kraus_tensor[idx]  # dim_out, dim_in
        right_tensor = k @ right_tensor  # dim_out, rank_state
        
    contracted_tensor = povm_psd @ right_tensor # num_povm, rank_povm, rank_state
    return jnp.einsum("ijk, ijk -> i", contracted_tensor, contracted_tensor.conj()) # num_povm

def least_squares_mps_single_gate_sequence_coherent(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int], prob_vector:jnp.ndarray):
    """Compute the full least squares for a single set of gate indices and a single probability vector, assuming coherent quantum channels (i.e. kraus rank = 1).
    
    The equation for this looks like: 
    C = sum_{i=1}^{num_povm} | <povm_i| K_{j_n} ... K_{j_1} |state> - p_i |^2

    WARNING: the following contraction only makes sense assuming that the kraus rank is 1, i.e. the quantum channel is coherent/unitary.

    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank = 1, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
        prob_vector: tensor of dimension: (num_povm)
    Returns:
        scalar corresponding to the least squares cost function value for the given gate sequence and probability vector.
    """
    inner_prod_vector = contract_mps_all_povm_coherent(kraus_tensor, povm_psd, state_psd, gates_indices) # num_povm
    return mean_square_error_vectors(inner_prod_vector, prob_vector) # num_povm -> scalar

least_squares_single_gate_sequence_coherent_jit = jax.jit(least_squares_mps_single_gate_sequence_coherent)

def least_squares_mps_single_gate_sequence(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int], prob_vector:jnp.ndarray):
    """Compute the least squares least squares for a single set of gate indices and a single probability vector.
    
    The equation for this looks like: 
    C = sum_{i=1}^{num_povm} | <povm_i| K_{j_n} ... K_{j_1} |state> - p_i |^2

    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
        prob_vector: tensor of dimension: (num_povm)
        
    Returns:
       scalar corresponding to the least squares for the given gate sequence and probability vector.
    """
    inner_prod_vector = contract_mps_all_povm(kraus_tensor, povm_psd, state_psd, gates_indices) # num_povm
    return mean_square_error_vectors(inner_prod_vector, prob_vector) # num_povm -> scalar

least_squares_mps_single_gate_sequence_jit = jax.jit(least_squares_mps_single_gate_sequence)

def log_likelihood_mps_single_gate_sequence(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, gates_indices:list[int], prob_vector:jnp.ndarray):
    """Compute the log-likelihood for a single set of gate indices and a single probability vector.
    
    The equation for this looks like: 
    L = sum_{i=1}^{num_povm} log(<povm_i| K_{j_n} ... K_{j_1} |state>) * p_i

    Args:
        kraus_tensor: tensor of dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        gates_indices: list of indices that dictate which kraus_tensor[idx] will be chosen for each contraction loop.
        prob_vector: tensor of dimension: (num_povm)
        
    Returns:
        jnp.ndarray: scalar corresponding to the log-likelihood for the given gate sequence and probability vector.
    """
    inner_prod_vector = contract_mps_all_povm(kraus_tensor, povm_psd, state_psd, gates_indices) # num_povm
    return log_likelihood_vectors(inner_prod_vector, prob_vector) # num_povm -> scalar

log_likelihood_mps_single_gate_sequence_jit = jax.jit(log_likelihood_mps_single_gate_sequence)

def log_likelihood_vectors(estimate:jnp.ndarray, target:jnp.ndarray)->float:
    """Compute the log-likelihood between the estimate and target vectors.
    
    Args:
        estimate: vector of the estimate operator
        target: vector of the target operator
    Returns:
        The log-likelihood between the two vectors.
    """
    # we take the absolute value for numerical stabillity.
    cost_vector = jnp.log(jnp.abs(estimate)) * target # num_povm
    return jnp.sum(cost_vector) # num_povm -> scalar

def mean_square_error_vectors(estimate:jnp.ndarray, target:jnp.ndarray)->float:
    """Compute the mean square error between the estimate and target vectors.
    
    Args:
        estimate: vector of the estimate operator
        target: vector of the target operator
    Returns:
        The mean square error between the two vectors.
    """
    cost_vector = jnp.abs(estimate - target)**2 # num_povm
    return jnp.sum(cost_vector) # num_povm -> scalar

def frobenius_distance(A:jnp.ndarray, B:jnp.ndarray):
    """Compute the Frobenius norm between two matrices A and B."""
    return jnp.linalg.norm(A - B)**2

def frobenius_distance_expanded(estimate:jnp.ndarray, target:jnp.ndarray)->float:
    """Expressions "equivalent" to the ``frobenius_distance`` function, used for better stability in the optimization.
    
    Important: when using this function, we DO NOT need to add the ``**2`` (squaring the value).

    Notes:
        * I have added the jnp.trace(target.conj().T @ target) even though it's a constant value, to make the function positive.
        * The ``jnp.real`` is needed because ``jax.grad`` needs a real-valued function.

    Args:
        estimate: matrix of the estimate operator
        target: matrix of the target operator

    Returns:
       Frobenius distance between the two operators.
    """
    return jnp.real(jnp.trace(target.conj().T @ target) + jnp.trace(estimate.conj().T @ estimate) - 2 * jnp.trace(estimate.conj().T @ target).real)

def state_infidelity(operator_sqrt_est:jnp.ndarray, operator_sqrt_target:jnp.ndarray, pure:bool=True)->float:
    """Compute the infidelity between the estimate and target states.
    
    Args:
        operator_sqrt_est: cholesky factor of the estimation matrix of dimensions (dim, rank_state_est)
        operator_sqrt_target: cholesky factor of the target matrix of dimensions (dim, rank_state_target)
        pure: If True, the target state is assumed to be a pure state (rank 1).
    Returns:
        The infidelity between the two operators.
    """
    return 1 - state_fidelity(operator_sqrt_est, operator_sqrt_target, pure=pure)

def state_fidelity(operator_sqrt_est:jnp.ndarray, operator_sqrt_target:jnp.ndarray, pure:bool=False)->float:
    """Compute the fidelity between the estimate and target states.

    This implementation uses:
    f(rho, sigma) = ||sqrt(rho) * sqrt(sigma)||_tr^2

    Args:
        operator_sqrt_est: cholesky factor of the estimation matrix of dimensions (dim, rank_state_est)
        operator_sqrt_target: cholesky factor of the target matrix of dimensions (dim, rank_state_target)
        pure: If True, the target state is assumed to be a pure state (rank 1).
            A more efficient implementation is used in this case.

    Returns:
        The fidelity between the two operators.
    """
    if pure:
        operator_sqrt_target = operator_sqrt_target[:, 0:1]  # rank 1
        return _state_fidelity_pure(operator_sqrt_est, operator_sqrt_target)
    return jnp.linalg.norm(operator_sqrt_est.conj().T @ operator_sqrt_target, ord="nuc") ** 2

def _state_fidelity_pure(operator_sqrt_est:jnp.ndarray, operator_sqrt_target:jnp.ndarray)->float:
    """Compute the fidelity between the estimate and target state, assuming the target is a pure state (rank 1).
    
    Args:
        operator_sqrt_est: cholesky factor of the estimation matrix of dimensions (dim, rank_state)
        operator_sqrt_target: cholesky factor of the target matrix of dimensions (dim, 1)
    Returns:
        The fidelity between the two operators.
    """
    new_state = operator_sqrt_est.conj().T @ operator_sqrt_target
    return jnp.vdot(new_state, new_state).real    
    


def state_fidelity_from_density_matrices(rho_est:jnp.ndarray, rho_target:jnp.ndarray)->float:
    from mGST.utility_functions_comparisons import factorize_psd_truncated
    """Compute the fidelity between the estimate and target states.

    This implementation uses:
    f(rho, sigma) = trace(sqrt(rho) * sigma * sqrt(rho))**2
    """
    # return jnp.trace(factorize_psd_truncated(rho_est @ rho_target, unique_srt=True))**2
    rho_est_sqrt = factorize_psd_truncated(rho_est, unique_srt=True)
    return jnp.trace(factorize_psd_truncated(rho_est_sqrt.conj().T @ rho_target @ rho_est_sqrt, unique_srt=True))**2
    
    
def process_fidelity(kraus_tensor_est:jnp.ndarray, kraus_tensor_target:jnp.ndarray, unitary:bool=True)->float:
    """Compute the process fidelity between the estimate and target Kraus operators.
    
    With the new implementation using ellipsis, we can use the same function for individual kraus and tensor of 4 dimensions.
    
    Note: In both cases we sum over the num_gates dimension!
    
    This assumes the first dimension (ommited in the ellipsis) is the number of gates.
    
    Notes:
        * The default behaviour is to assume one of the channels is unitary and return the entanglement fidelity.
            Namely: F_e(target, estimate) = tr(S(target)^dagger * S(estimate))
        * if unitary is False, we use the more general expression for the process fidelity.
            Namely: F_p(target, estimate) = F(Choi(target), Choi(estimate)), with F(rho, sigma) the state fidelity. See ``state_fidelity``.

    Args:
        kraus_tensor_est: kraus tensor of the estimate of dimensions (..., kraus_rank, dim_out, dim_in)
        kraus_tensor_target: kraus tensor of the target of dimensions (..., kraus_rank, dim_out, dim_in)
        unitary: If True, at least one of the quantum channels is assumed to be unitary. 
            Then, the fidelity can be computed as the Hilbert Schmidt inner product of the superoperators. This is equivalent to calculating the state fidelity for pure states.
            If False, we instead return the state fidelity between the choi states of the two arrays channels. 
        
    Returns:
        The process fidelity between the two quantum channels.
    """
    dim = kraus_tensor_est.shape[-1]
    
    if unitary:
        # tr(|psi><psi|^dagger * rho) = <psi| rho |psi> 
        return (hilbert_schmidt_inner_product(kraus_tensor_est, kraus_tensor_target).real) / dim**2
    
    # This is equivalent to using the state_fidelity on the individual choi matrices. Namely:
    # choi_sqrt_est_test = kraus_tensor_est.reshape(rank_kraus, dim**2).T
    # choi_sqrt_target_test = kraus_tensor_target.reshape(rank_kraus, dim**2).T
    # state_fidelity(operator_sqrt_est=choi_sqrt_est_test, operator_sqrt_target=choi_sqrt_target_test) / dim**2
    
    choi_product = jnp.einsum("...roi, ...qoi -> ...rq", kraus_tensor_est.conj(), kraus_tensor_target)
    return jnp.sum(jnp.linalg.norm(choi_product, ord="nuc", axis=(-2, -1)) **2)/ dim**2
    
def hilbert_schmidt_inner_product(kraus_tensor_1:jnp.ndarray, kraus_tensor_2:jnp.ndarray)->float:
    """Compute the Hilbert Schmidt inner product between the estimate and target superoperators defined by the kraus tensors.
    
    This is equivalent to the entanglement fidelity when one of the channels is unital.
    
    Note: this sums over the num_gates dimension!
    
    Args:
        kraus_tensor_est: kraus tensor of the estimate of dimensions (..., kraus_rank, dim_out, dim_in)
    """
    # This path corresponds to contracting the upper tensors together, the lower tensors together, scaling of d**2 * rank_2 * rank_1 - each
    # and then contracting these together, scaling of rank_1 * rank_2
    # see notes to compare against the scaling of d**5 * rank_1 of naive contraction
    optimal_path_trace = [(0, 1), (0, 1), (0, 1)]
    return jnp.einsum(
        "...ajk, ...bjk, ...alm, ...blm->", 
        kraus_tensor_2.conj(), kraus_tensor_1, 
        kraus_tensor_2, kraus_tensor_1.conj(), optimize=optimal_path_trace
    )

def average_process_fidelity(kraus_tensor_est:jnp.ndarray, kraus_tensor_target:jnp.ndarray, unital:bool=False)->float:
    """Compute the average fidelity between the estimate and target Kraus operators.
    Namely: F_avg(target, estimate) = (1/d * (d+1)) * ( tr(S(target)^dagger * S(estimate)) + <<1|| S(target)^dagger * S(estimate) ||1>>)
    
    If the composition target^\dagger * estimate is not trace preserving, this quantity is not guaranteed to be 1 even when target == estimate. Therefore, cannot be used reliably as a distance between the channels and an infidelity measure derived from it might be ill defined.
    
    Args:
        kraus_tensor_est: kraus tensor of the estimate of dimensions (..., kraus_rank, dim_out, dim_in)
        kraus_tensor_target: kraus tensor of the target of dimensions (..., kraus_rank, dim_out, dim_in)
        unital: If True, at least one of the quantum channels is assumed to be unital.  
            Then, the composition of the channels is trace preserving, and the second term can be simplified to be dim(Hilbert space).
            If False, we compute all the contractions.
    """
    dim = kraus_tensor_est.shape[-1]
    if unital:
        # <<1|| S(target)^dagger * S(estimate) ||1>> = dim
        # Would also work if S(target)^dagger * S(estimate) is trace preserving.
        inner_product_identity = dim
    else:
        # <<1|| S(target)^dagger * S(estimate) ||1>>
        optimal_path = [(0, 2), (0, 1), (0, 1)]
        inner_product_identity = jnp.einsum(
            "...ajk, ...bjm, ...alk, ...blm->", 
            kraus_tensor_target.conj(), kraus_tensor_est, 
            kraus_tensor_target, kraus_tensor_est.conj(),
            optimize=optimal_path).real
        
    return (hilbert_schmidt_inner_product(kraus_tensor_target, kraus_tensor_est).real + inner_product_identity) / (dim * (dim + 1))

    
def process_infidelity(kraus_tensor_est:jnp.ndarray, kraus_tensor_target:jnp.ndarray, unitary:bool=True)->float:
    """Compute the process infidelity between the estimate and target Kraus operators.

    This quantity is always well defined regarldess of whether the channels are unital or not.
    This is because we are the process fidelity instead of the average process fidelity.

    Args:
        kraus_tensor_est: kraus tensor of the estimate of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        kraus_tensor_target: kraus tensor of the target of dimensions (num_gates, kraus_rank, dim_out, dim_in)
    Returns:
        The infidelities between the two arays quantum channels, of dimension (num_gates).
    """
    num_gates = kraus_tensor_est.shape[0]
    return num_gates - process_fidelity(kraus_tensor_est, kraus_tensor_target, unitary=unitary)
    
def process_fidelity_povm(povm_tensor_est:jnp.ndarray, povm_tensor_target:jnp.ndarray)->float:
    """Compute the entanglement fidelity between the estimate and target classical quantum channels arising from the POVM's.
    
    This is the process fidelity between the quantum channels that map the input state to a diagonal state in the computational basis with amplitudes as the probabilities tr(F_i * rho) = p(i|rho).

    Args:
        povm_tensor_est: estimate povm tensor of dimensions (num_povm, povm_rank, dim_in)
        povm_tensor_target: estimate povm tensor of dimensions (num_povm, povm_rank, dim_in)

    Returns:
        The fidelity between the classical quantum channels.
    """
    return _povm_fidelity_from_block_matrices(povm_tensor_est, povm_tensor_target)
    
def _povm_fidelity_from_block_matrices(povm_tensor_est, povm_tensor_target):
    """
    Computes the fidelity between two POVMs using the block matrix formula.
    
    See notes in ipad from 03.04.25 to se why this works.
    The idea is that the svd of a block diagonal is composed of the the svd of the blocks.
    
    Args:
        povm_est: The estimated POVM tensor of shape (num_povm, rank_povm, dim_in).
        povm_target: The target POVM tensor of shape (num_povm, rank_povm, dim_in).
    
    Returns:
        The fidelity between the two POVMs.
    """
    dim = povm_tensor_est.shape[-1]
    choi_product = jnp.einsum("iaj, ibj -> iab", povm_tensor_target.conj(), povm_tensor_est)
    norms_i = jnp.linalg.norm(choi_product, ord="nuc", axis=(-2, -1)) # num_povm
    return jnp.sum(norms_i)**2/dim**2 # normalizing choi state (dim**2).
    
def process_infidelity_povm(povm_tensor_est:jnp.ndarray, povm_tensor_target:jnp.ndarray)->float:
    """Compute the entanglement infidelity between the estimate and target classical quantum channels arising from the POVM's.

    Args:
        povm_tensor_est: estimate povm tensor of dimensions (..., povm_rank, dim_in)
        povm_tensor_target: estimate povm tensor of dimensions (..., povm_rank, dim_in)

    Returns:
        The infidelity between the classical quantum channels.
    """
    return 1 - process_fidelity_povm(povm_tensor_est, povm_tensor_target)

def cost_function_jax_mps_regularized(kraus_tensor:jnp.ndarray, 
                                      povm_psd:jnp.ndarray,
                                      state_psd:jnp.ndarray,
                                      indices_list:list[list[int]],
                                      prob_matrix:jnp.ndarray,
                                      target_kraus_tensor:jnp.ndarray,
                                      target_povm_psd:jnp.ndarray,
                                      target_state_psd:jnp.ndarray,
                                      num_samples:int,
                                      regularization_parameter: float =  None,
                                      metric_function:callable = None,
                                      jit:bool=False,
                                      verbose:bool=True)->jnp.ndarray:
    
    loss_value = 0.5 * cost_function_jax_mps(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, jit, verbose)
    
    regularized_value = compute_regularized_value_all_operators(
        kraus_tensor_est=kraus_tensor,
        povm_psd_est=povm_psd,
        state_psd_est=state_psd,
        kraus_tensor_target=target_kraus_tensor,
        povm_psd_target=target_povm_psd,
        state_psd_target=target_state_psd,
        state_metric=metric_function,
        povm_metric=metric_function,
        kraus_metric=metric_function
    )
    
    if regularization_parameter is None:
        regularization_parameter = 10 / num_samples
    # Adding the 1/2 factor to the regularization term, following Eq. 7 of Sugiyama's paper.
    return loss_value + 0.5 * regularization_parameter * regularized_value

def compute_regularized_value_all_operators(kraus_tensor_est, povm_psd_est, state_psd_est, kraus_tensor_target, povm_psd_target, state_psd_target, state_metric:callable=None, povm_metric:callable=None, kraus_metric:callable=None):
    """_summary_

    Args:
        kraus_tensor_est: kraus tensor estimate of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: cholesky factor of the povm estimate of dimensions (num_povm, rank_povm, dim)
        state_psd: cholesky factor of the state estimate of dimensions (dim, rank_state)
        kraus_tensor_target: kraus tensor target of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd_target: cholesky factor of the povm target of dimensions (num_povm, rank_povm, dim)
        state_psd_target: cholesky factor of the state target of dimensions (dim, rank_state)
    """
    # regularization
    regularized_value = 0
    
    # State
    if state_metric is None:
        state_metric = state_infidelity
    regularized_value += state_metric(state_psd_est, state_psd_target)
    
    # POVM
    if povm_metric is None:
        povm_metric = process_infidelity_povm
    regularized_value += povm_metric(povm_psd_est, povm_psd_target)
    
    # Kraus
    if kraus_metric is None:
        kraus_metric = process_infidelity
    
    regularized_value += kraus_metric(kraus_tensor_est, kraus_tensor_target)
    
    return regularized_value

def cost_function_jax_mps(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, jit:bool=False, verbose:bool=False, use_log_likelihood:bool=False, num_shots:int=None):
    """Compute the cost function using jax and mps contraction strategy.
    
    Optimized using the most scalable jnp functions.

    Args:
        kraus: kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) factor of the state of dimensions: (dim, rank_state)
            (see B in Eq. 9 of mGST paper)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        use_log_likelihood: If True, use the log-likelihood cost function instead of the least squares cost function.
    """
    cost_value = 0
    num_gate_sequences = len(indices_list)
    num_povm = povm_psd.shape[0]
    num_gates, kraus_rank, dim_out, dim_in = kraus_tensor.shape    
    
    if use_log_likelihood and num_shots is None:
        raise ValueError("num_shots must be provided when using log-likelihood cost function.")
    
    if kraus_rank == 1: # coherent channels. Can use specialized contraction path.
        if verbose:
            warnings.warn("Kraus rank = 1 detected. Using the coherent version of the cost function for better performance.")
        if use_log_likelihood:
            # TODO: implement the coherent version once I know this works well.
            inner_function = log_likelihood_mps_single_gate_sequence_jit if jit else log_likelihood_mps_single_gate_sequence
        else:
            inner_function = least_squares_single_gate_sequence_coherent_jit if jit else least_squares_mps_single_gate_sequence_coherent
    else:
        
        if use_log_likelihood:
            inner_function = log_likelihood_mps_single_gate_sequence_jit if jit else log_likelihood_mps_single_gate_sequence
        else:
            inner_function = least_squares_mps_single_gate_sequence_jit if jit else least_squares_mps_single_gate_sequence
      
    if jit:
        previous_count = inner_function._cache_size()
        if verbose:
            print(f'Initial count: {previous_count}')
    
    for idx, gates_indices in enumerate(indices_list):
        cost_value += inner_function(kraus_tensor, povm_psd, state_psd, gates_indices, prob_matrix[:,idx])
        
        if jit:
            new_count = inner_function._cache_size()
            if new_count != previous_count and verbose:
                print(f'at iteration {idx} the new count changed to: {new_count}')
                previous_count = new_count

    if use_log_likelihood:
        factor = -num_shots
    else:
        factor = 1 / (num_gate_sequences * num_povm)
                
    return factor * cost_value

def cost_function_jax(kraus_contracted, povm_matrix, state_vector, indices_list, prob_matrix):
    """Implementation of the cost_function_jax_mps used with the input of the rest of the mGST code.

    This is the code implemented in cost_function_jax_mps without fragmenting into separate functions since we don't care about JIT.

    Parameters
    ----------
    kraus_contracted : Contracted Kraus tensor of dimensions (num_gates, dim_out, dim_in)
    povm_tensor : A 2D array representing the POVM elements, of dimensions (num_povm, dim**2).
    state_vector : A 1D array representing the density matrix of dimensions (dim**2).
    indices_list : A list of lists representing the indices for which the objective function will be evaluated.
    prob_matrix: A 2D array of shape (num_povm, num_gate_sequences) containing the measured probabilities.

    Returns
    -------
    The cost function value.
    """
    
    cost_value = 0
    num_gate_sequences = len(indices_list)
    num_povm = povm_matrix.shape[0]
        
    for idx, gates_indices in enumerate(indices_list):
        
        prob_vector = prob_matrix[:,idx]
        
        right_tensor = state_vector  # dim_up_in, dim_down_in
        # Iterate through the Kraus tensors in reverse order
        for gate_idx in reversed(gates_indices):
            k = kraus_contracted[gate_idx]
            right_tensor = k @ right_tensor
       
        inner_prod_vector = povm_matrix.conj() @  right_tensor # -> num_povm
                
        cost_vector = jnp.abs(inner_prod_vector - prob_vector)**2        
        cost_value += jnp.sum(cost_vector) # num_povm ->
    return cost_value / (num_gate_sequences * num_povm)
    
def contract_jax(X, j_vec):
    """Contract a sequence of matrices in the given order using JAX.

    This function computes the product of a sequence of matrices specified by
    the indices in `j_vec`. The result is the contracted product of the matrices
    in the given order.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
    j_vec : numpy.ndarray
        A 1D array of indices specifying the order in which to contract the matrices in X.

    Returns
    -------
    numpy.ndarray
        The contracted product of the matrices specified by the indices in `j_vec`.
    """
    res = jnp.eye(X[0].shape[0], dtype=jnp.complex128)
    for j in j_vec:
        # I can actually erase this comparison here if I preprocess J beforehand
        res = jnp.where(j >= 0, res.dot(X[j]), res)
    return res

contract_jax_jit = jax.jit(contract_jax)
# This function will only get compiled when the type of length of j_vec or X changes.

def gradient_k_mps(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=0)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_k_mps_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=0)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_k_mps_jit_reg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples):
    "Calculate the Euclidean gradient"
    return jax.grad(fun=cost_function_jax_mps_regularized, argnums=0)(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples, jit=True)

def gradient_povm_mps(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the state"
    return jax.grad(fun=cost_function_jax_mps, argnums=1)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_povm_mps_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the state"
    return jax.grad(fun=cost_function_jax_mps, argnums=1)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_povm_mps_jit_reg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples):
    "Calculate the Euclidean gradient"
    return jax.grad(fun=cost_function_jax_mps_regularized, argnums=1)(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples, jit=True)

def gradient_state_mps(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the state"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=2)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_state_mps_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the state"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=2)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_state_mps_jit_reg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples):
    "Calculate the Euclidean gradient"
    return jax.grad(fun=cost_function_jax_mps_regularized, argnums=2)(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, target_kraus_tensor, target_povm_psd, target_state_psd, num_samples, jit=True)

def gradient_all_3_mps(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the kraus, sate, and povm"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=(0, 1, 2))(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_all_3_mps_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "Calculate the Euclidean gradient with respect to the kraus, sate, and povm"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_mps, argnums=(0, 1, 2))(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_all_3_and_value(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "computes both the value of the function and gradient"
    return jax.value_and_grad(fun=cost_function_jax_mps, argnums=(0,1,2))(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_all_3_and_value_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "computes both the value of the function and gradient"
    return jax.value_and_grad(fun=cost_function_jax_mps, argnums=(0,1,2))(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_k_and_value_jit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "computes both the value of the function and gradient"
    return jax.value_and_grad(fun=cost_function_jax_mps, argnums=0)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=True)

def gradient_k_and_value_nojit(kraus, povm_psd, state_psd, indices_list, prob_matrix):
    "computes both the value of the function and gradient"
    return jax.value_and_grad(fun=cost_function_jax_mps, argnums=0)(kraus, povm_psd, state_psd, indices_list, prob_matrix, jit=False)

def gradient_k_numba(K, E, rho, J, y):
    num_gates = K.shape[0]
    kraus_rank = K.shape[1]
    dim = K.shape[2]
    # einsum is not supported by numba
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((num_gates, dim**2, dim**2))
    return dK(X, K, E, rho, J, y, d=num_gates, r=dim**2, rK=kraus_rank) 

def cost_function_jax_for_gradient(K, E, rho, J, y):
    num_gates, _, dim, dim = K.shape
    X = jnp.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((num_gates, dim**2, dim**2))
    return cost_function_jax(X, E, rho, J, y)
    

def dK_jax(K, E, rho, J, y):
    "Calculate the Euclidean derivative wrt the Gate tensor K using JAX"
    print('Using JAX power')
    return jax.grad(fun=cost_function_jax_for_gradient, argnums=0)(K, E, rho, J, y)

@njit(cache=True)
def MVE_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm):
    """Compute the lower bound of the mean value error (MVE) between true and estimated parameters.

    This function calculates the lower bound of the MVE between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Parameters
    ----------
    X_true : numpy.ndarray
        A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
    E_true : numpy.ndarray
        A 2D array representing the true POVM elements, of shape (n_povm, r).
    rho_true : numpy.ndarray
        A 1D array representing the true density matrix.
    X : numpy.ndarray
        A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the estimated POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the estimated density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    n_povm : int
        The number of POVM elements.

    Returns
    -------
    tuple of float
        A tuple containing the lower bound of the mean value error and the maximum distance.
    """
    m = len(J)
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        curr = 0
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            curr += np.abs(y_t - y)
        curr = curr / 2
        dist += curr
        max_dist = max(max_dist, curr)
    return dist / m, max_dist


@njit(cache=True)
def Mp_norm_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm, p):
    """Compute the Mp-norm lower bound of the distance between true and estimated parameters.

    This function calculates the lower bound of the Mp-norm between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Parameters
    ----------
    X_true : numpy.ndarray
        A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
    E_true : numpy.ndarray
        A 2D array representing the true POVM elements, of shape (n_povm, r).
    rho_true : numpy.ndarray
        A 1D array representing the true density matrix.
    X : numpy.ndarray
        A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the estimated POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the estimated density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    n_povm : int
        The number of POVM elements.
    p : float
        The order of the Mp-norm (p > 0).

    Returns
    -------
    tuple of float
        A tuple containing the Mp-norm lower bound and the maximum distance.
    """
    m = len(J)
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            dist += np.abs(y_t - y) ** p
        max_dist = max(max_dist, curr)
    return dist ** (1 / p) / m / n_povm, max_dist ** (1 / p)


@njit(cache=True)  # , parallel=True)
def dK(X, K, E, rho, J, y, d, r, rK, mle=False):
    """Compute the derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the derivative of the Kraus operator K, based on the
    input matrices X, E, and rho, as well as the isometry condition.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    K : numpy.ndarray
        The Kraus operator K, reshaped to (d, rK, -1).
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    d : int
        The number of Kraus operators.
    r : int
        The rank of the problem.
    rK : int
        The number of rows in the reshaped Kraus operator K.
    mle : bool
        If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns
    -------
    numpy.ndarray
        The derivative objective function with respect to the Kraus tensor K,
        reshaped to (d, rK, pdim, pdim), and scaled by 2/m/n_povm.
    """
    # pylint: disable=too-many-nested-blocks
    K = K.reshape(d, rK, -1)
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r))
    dK_ = np.ascontiguousarray(dK_.astype(np.complex128))
    m = len(J)

    for k in prange(d):  # pylint: disable=not-an-iterable
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i, j_curr in enumerate(j):
                if j_curr == k:
                    R = rho.copy()
                    for ind in j[i + 1 :][::-1]:
                        R = X[ind] @ R
                    for o in range(n_povm):
                        L = E[o].conj()
                        for ind in j[:i]:
                            L = L @ X[ind]
                        if mle:
                            p_ind = L @ X[k] @ R
                            dK_[k] -= (
                                K[k].conj()
                                @ np.kron(L.reshape(pdim, pdim).T, R.reshape(pdim, pdim).T)
                                * y[o, n]
                                / p_ind
                            )
                        else:
                            D_ind = L @ X[k] @ R - y[o, n]
                            dK_[k] += (
                                D_ind
                                * K[k].conj()
                                @ np.kron(L.reshape(pdim, pdim).T, R.reshape(pdim, pdim).T)
                                * 2
                                / m
                                / n_povm
                            )
    return dK_.reshape(d, rK, pdim, pdim)


@njit(cache=True)  # , parallel=False)
def dK_dMdM(X, K, E, rho, J, y, d, r, rK, mle=False):
    """Compute the derivatives of the objective function with respect to K and the
    product of derivatives of the measurement map with respect to K.

    This function calculates the derivatives of K, dM10, and dM11 based on the input matrices X,
    matrix K, POVM elements E, density matrix rho, and target values y.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing input matrices, of shape (n_matrices, n_rows, n_columns).
    K : numpy.ndarray
        A 1D array representing the matrix K.
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing target values.
    d : int
        The number of dimensions for the matrix K.
    r : int
        The number of rows for the matrix K.
    rK : int
        The number of columns for the matrix K.
    mle : bool
        If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the derivatives of K, dM10, and dM11, each of which is a numpy.ndarray.
    """
    K = K.reshape(d, rK, -1)
    pdim = int(np.sqrt(r))
    n = d * rK * r
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r)).astype(np.complex128)
    dM11 = np.zeros(n**2).astype(np.complex128)
    dM10 = np.zeros(n**2).astype(np.complex128)
    m = len(J)
    for n in range(m):
        j = J[n][J[n] >= 0]
        dM = np.ascontiguousarray(np.zeros((n_povm, d, rK, r)).astype(np.complex128))
        p_ind_array = np.zeros(n_povm).astype(np.complex128)
        for o in range(n_povm):
            for i, k in enumerate(j):
                R = rho.copy()
                for ind in j[i + 1 :][::-1]:
                    R = X[ind] @ R
                L = E[o].conj().copy()
                for ind in j[:i]:
                    L = L @ X[ind]
                dM_loc = K[k].conj() @ np.kron(L.reshape((pdim, pdim)).T, R.reshape((pdim, pdim)).T)
                p_ind = L @ X[k] @ R
                if mle:
                    dM[o, k] += dM_loc
                    dK_[k] -= dM_loc * y[o, n] / p_ind
                else:
                    dM[o, k] += dM_loc
                    D_ind = p_ind - y[o, n]
                    dK_[k] += D_ind * dM_loc * 2 / m / n_povm
            if len(j) == 0:
                p_ind_array[o] = E[o].conj() @ rho
            else:
                p_ind_array[o] = p_ind
        for o in range(n_povm):
            if mle:
                dM11 += np.kron(dM[o].conj().reshape(-1), dM[o].reshape(-1)) * y[o, n] / p_ind_array[o] ** 2
                dM10 += np.kron(dM[o].reshape(-1), dM[o].reshape(-1)) * y[o, n] / p_ind_array[o] ** 2
            else:
                dM11 += np.kron(dM[o].conj().reshape(-1), dM[o].reshape(-1)) * 2 / m / n_povm
                dM10 += np.kron(dM[o].reshape(-1), dM[o].reshape(-1)) * 2 / m / n_povm
    return (dK_.reshape((d, rK, pdim, pdim)), dM10, dM11)


@njit(cache=True)  # , parallel=False)
def ddM(X, K, E, rho, J, y, d, r, rK, mle=False):
    """Compute the second derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the second derivative of the objective function for a given
    set of input parameters.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    K : ndarray
        Array of Kraus operators.
    E : ndarray
        Array of measurement operators.
    rho : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    d : int
        Number of Kraus operators.
    r : int
        Dimension of the local basis.
    rK : int
        Number of rows in the Kraus operator matrix.
    mle : bool
        If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns
    -------
    ddK : ndarray, shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
        Second derivative of the objective function with respect to matrix elements, reshaped
        for easier manipulation.
    dconjdK : ndarray, shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
        Conjugate of the second derivative of the objective function with respect to matrix
        elements, reshaped for easier manipulation.
    """
    # pylint: disable=too-many-branches, too-many-nested-blocks
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    ddK = np.zeros((d**2, rK**2, r, r))
    ddK = np.ascontiguousarray(ddK.astype(np.complex128))
    dconjdK = np.zeros((d**2, rK**2, r, r))
    dconjdK = np.ascontiguousarray(dconjdK.astype(np.complex128))
    m = len(J)
    for k in range(d**2):
        k1, k2 = local_basis(k, d, 2)
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i1, j_1 in enumerate(j):
                if j_1 == k1:
                    for i2, j_2 in enumerate(j):
                        if j_2 == k2:
                            L0 = contract(X, j[: min(i1, i2)])
                            C = contract(X, j[min(i1, i2) + 1 : max(i1, i2)]).reshape(pdim, pdim, pdim, pdim)
                            R = contract(X, j[max(i1, i2) + 1 :]) @ rho
                            for o in range(n_povm):
                                L = E[o].conj() @ L0
                                if i1 == i2:
                                    p_ind = L @ X[k1] @ R
                                elif i1 < i2:
                                    p_ind = L @ X[k1] @ C.reshape(r, r) @ X[k2] @ R
                                else:
                                    p_ind = L @ X[k2] @ C.reshape(r, r) @ X[k1] @ R
                                D_ind = p_ind - y[o, n]

                                ddK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                dconjdK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                for rk1 in range(rK):
                                    for rk2 in range(rK):
                                        if i1 < i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim) @ K[k2, rk2].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 3, 2, 1)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim).T @ K[k2, rk2].T,
                                            ) @ np.ascontiguousarray(C.transpose(1, 2, 3, 0)).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 2, 3, 1)
                                            ).reshape(r, r)

                                        elif i1 == i2:
                                            dconjdK_loc[rk1 * rK + rk2] = np.outer(L, R)

                                        elif i1 > i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k2, rk2].conj(),
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(3, 0, 1, 2)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim).T @ K[k2, rk2],
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose((0, 3, 2, 1))).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(2, 0, 1, 3)
                                            ).reshape(r, r)
                                if mle:
                                    ddK[k1 * d + k2] -= ddK_loc * y[o, n] / p_ind
                                    dconjdK[k1 * d + k2] -= dconjdK_loc * y[o, n] / p_ind
                                else:
                                    ddK[k1 * d + k2] += D_ind * ddK_loc * 2 / m / n_povm
                                    dconjdK[k1 * d + k2] += D_ind * dconjdK_loc * 2 / m / n_povm
    return (
        ddK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim),
        dconjdK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim),
    )


@njit(cache=True)  # , parallel=True)
def dA(X, A, B, J, y, r, pdim, n_povm):
    """Compute the derivative of to the objective function with respect to the POVM tensor A

    This function calculates the gradient of A for a given set of input parameters.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    A : ndarray
        Array of measurement operators.
    B : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    r : int
        Number of elements in each measurement operator.
    pdim : int
        Dimension of the density matrices.
    n_povm : int
        Number of measurement operators.

    Returns
    -------
    dA : ndarray
        Derivative of the objective function with respect to A.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
    m = len(J)
    # pylint: disable=not-an-iterable
    for n in prange(m):
        j = J[n][J[n] >= 0]
        inner_deriv = contract(X, j) @ rho
        dA_step = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
        for o in range(n_povm):
            D_ind = E[o].conj() @ inner_deriv - y[o, n]
            dA_step[o] += D_ind * A[o].conj() @ inner_deriv.reshape(pdim, pdim).T
        dA_ += dA_step
    return dA_ * 2 / m / n_povm


@njit(cache=True)  # , parallel=True)
def dB(X, A, B, J, y, pdim):
    """Compute the derivative of the objective function with respect to the state tensor B.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    A : ndarray
        Array of measurement operators.
    B : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    pdim : int
        Dimension of the density matrices.

    Returns
    -------
    dB : ndarray
        Derivative of the objective function with respect to the state tensor B.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = (A.T.conj() @ A).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim))
    dB_ = dB_.astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        jE = J[n][J[n] >= 0][0]
        j = J[n][J[n] >= 0][1:]
        inner_deriv = E[jE].conj().dot(contract(X, j))
        D_ind = inner_deriv.dot(rho) - y[n]
        dB_ += D_ind * inner_deriv.reshape(pdim, pdim).conj() @ B
    return dB_


@njit(cache=True)  # , parallel=True)
def ddA_derivs(X, A, B, J, y, r, pdim, n_povm, mle=False):
    """Calculate all nonzero terms of the second derivatives with respect to the POVM tensor A.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    A : numpy.ndarray
        A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
    B : numpy.ndarray
        A 2D array of shape (pdim, pdim) representing the isometry matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    r : int
        The rank of the problem.
    pdim : int
        The dimension of the input matrices A and B.
    n_povm : int
        The number of POVM elements.
    mle : bool
        If True, the log-likelihood objective function is used, otherwise the least squares objective function is used

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the computed derivatives:
        - dA: The derivative w.r.t. A
        of shape (n_povm, pdim, pdim).
        - dMdM: The product of the measurement map derivatives dM and dM, of shape (n_povm, r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (n_povm, r, r).
        - dconjdA: The product of the conjugate of dA, of shape (n_povm, r, r).
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
    # dM: derivative of probability wrt to Z
    # D_ind: evaluation of probability
    # dMdM: product of two derivatives in the last line of equation of the derivatives (page 26)
    dMdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dMconjdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dconjdA = np.zeros((n_povm, r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        j = J[n][J[n] >= 0]
        R = contract(X, j) @ rho
        dA_step = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
        dMdM_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        dMconjdM_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        dconjdA_step = np.zeros((n_povm, r, r)).astype(np.complex128)
        for o in range(n_povm):
            dM = A[o].conj() @ R.reshape(pdim, pdim).T
            if mle:
                p_ind = E[o].conj() @ R
                dMdM_step[o] += np.outer(dM, dM) * y[o, n] / p_ind**2
                dMconjdM_step[o] += np.outer(dM.conj(), dM) * y[o, n] / p_ind**2
                dA_step[o] -= dM * y[o, n] / p_ind
                dconjdA_step[o] -= (
                    np.kron(np.eye(pdim).astype(np.complex128), R.reshape(pdim, pdim).T) * y[o, n] / p_ind
                )
            else:
                D_ind = E[o].conj() @ R - y[o, n]
                dMdM_step[o] += np.outer(dM, dM) * 2 / m / n_povm
                dMconjdM_step[o] += np.outer(dM.conj(), dM) * 2 / m / n_povm
                dA_step[o] += D_ind * dM * 2 / m / n_povm
                dconjdA_step[o] += (
                    D_ind * np.kron(np.eye(pdim).astype(np.complex128), R.reshape(pdim, pdim).T) * 2 / m / n_povm
                )
        dA_ += dA_step
        dMdM += dMdM_step
        dMconjdM += dMconjdM_step
        dconjdA += dconjdA_step
    return dA_, dMdM, dMconjdM, dconjdA


@njit(cache=True)  # , parallel=True)
def ddB_derivs(X, A, B, J, y, r, pdim, mle=False):
    """Calculate all nonzero terms of the second derivative with respect to the state tensor B.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    A : numpy.ndarray
        A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
    B : numpy.ndarray
        A 2D array of shape (pdim, pdim) representing the isometry matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    r : int
        The rank of the problem.
    pdim : int
        The dimension of the input matrices A and B.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the computed derivatives:
        - dB: The derivative w.r.t. B, of shape (pdim, pdim).
        - dMdM: The product of the derivatives dM and dM, of shape (r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (r, r).
        - dconjdB: The mixed second derivative of by dB and dB*, of shape (r, r).
    """
    n_povm = A.shape[0]
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim)).astype(np.complex128)
    dM = np.zeros((pdim, pdim)).astype(np.complex128)
    dMdM = np.zeros((r, r)).astype(np.complex128)
    dMconjdM = np.zeros((r, r)).astype(np.complex128)
    dconjdB = np.zeros((r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        j = J[n][J[n] >= 0]
        C = contract(X, j)
        for o in range(n_povm):
            L = E[o].conj() @ C
            dM = L.reshape(pdim, pdim) @ B.conj()
            if mle:
                p_ind = L @ rho
                dMdM += np.outer(dM, dM) * y[o, n] / p_ind**2
                dMconjdM += np.outer(dM.conj(), dM) * y[o, n] / p_ind**2
                dB_ -= dM * y[o, n] / p_ind
                dconjdB -= np.kron(L.reshape(pdim, pdim), np.eye(pdim).astype(np.complex128)) * y[o, n] / p_ind
            else:
                D_ind = L @ rho - y[o, n]
                dMdM += np.outer(dM, dM) * 2 / m / n_povm
                dMconjdM += np.outer(dM.conj(), dM) * 2 / m / n_povm
                dB_ += D_ind * dM * 2 / m / n_povm
                dconjdB += D_ind * np.kron(L.reshape(pdim, pdim), np.eye(pdim).astype(np.complex128)) * 2 / m / n_povm
    return dB_, dMdM, dMconjdM, dconjdB.T
