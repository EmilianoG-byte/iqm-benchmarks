"""Functions used to simulate samples for testing purposes"""
import time

from mGST import additional_fns
from mGST.low_level_jit import contract_mps_all_povm
from mGST.utility_functions_comparisons import kraus_tensor_to_mgst, factorize_psd_truncated, get_mgst_tensors_from_psd_representation, generate_target_state, generate_target_povm, get_compressed_rep_from_mgst_output

from mGST.typing import Tensor, Matrix, TrustRegionOptions, Vector
import jax.numpy as jnp
import numpy as np

from mGST.typing import OperatorSchedule, TrustRegionOptions
from mGST.trust_region import run_riemannian_optimization, estimate_noise_floor_threshold
from mGST.low_level_jit import cost_function_jax_mps
from mGST.reporting.reporting import gauge_opt
from mGST.compatibility import arrays_to_pygsti_model

from typing import Any

import warnings

def generate_sequence_indices(num_gates:int, num_circuits:int, seq_len_list:list[int]) -> dict[str, list[list[int]]]:
    """Generate a sequence of gate indices

    This code is adpated from `compressive_gst.generate_meas_circuits`

    Args:
        num_gates: Number of gates in the circuit.
        num_circuits: Number of circuits to generate.
        seq_len_list: List of three integers representing the minimum, cut, and maximum sequence lengths.

    Returns:
        A dictionary containing the gate indices and the gate indices with negative values.
    """

    # Calculate number of short and long circuits
    N_short = int(jnp.ceil(num_circuits / 2))
    N_long = int(jnp.floor(num_circuits / 2))
    L_MIN, L_CUT, L_MAX = seq_len_list

    gate_indices_with_negs = additional_fns.random_seq_design(num_gates, L_MIN, L_CUT, L_MAX, N_short, N_long)
    gate_indices = [list(seq[seq >= 0]) for seq in gate_indices_with_negs]
    return {"gate_indices": gate_indices, "gate_indices_with_negs": gate_indices_with_negs}

def add_negative_padding(
    gate_indices: list[list[int]],
    target_length: int,
):
    """Reconstruct fixed-length sequences by right-padding with -1.

    This inverts:
        [list(seq[seq >= 0]) for seq in gate_indices_with_negs]
    when the original had only trailing -1 padding.
    """
    pad_value = -1
    if target_length < 0:
        raise ValueError("target_length must be non-negative.")

    restored = []
    for seq in gate_indices:
        if len(seq) > target_length:
            raise ValueError(
                f"Found sequence of length {len(seq)} > target_length={target_length}."
            )
        restored.append(seq + [pad_value] * (target_length - len(seq)))

    return jnp.array(restored)

def compute_probability_matrices(kraus_tensor:Tensor, povm_psd:Tensor, state_psd:Tensor, gate_indices: list[list[int]], num_shots:int, seed:int=42) -> dict[str, Matrix]:
    """Compute the exact and sampled probability matrices for a given set of gate indices and gate set (kraus, povm, and state).
    
    Args:
        gate_indices: A list of lists, where each inner list contains the gate indices for a circuit.
        kraus_tensor: Kraus tensor of shape (num_gates, kraus_rank, dim, dim).
        povm_psd: POVM tensor of shape (num_povm_elements, povm_rank, dim).
        state_psd: State tensor of shape (dim, state_rank).
        num_shots: The number of shots to use for sampling.
        seed: The random seed for sampling.

    Returns:
        A dictionary containing the exact and sampled probability matrices.
    """
    prob_matrix_exact = []
    for indices in gate_indices:
        prob_matrix_exact.append(jnp.real(contract_mps_all_povm(kraus_tensor, povm_psd, state_psd, indices)))
        
    prob_matrix_exact = jnp.array(prob_matrix_exact).T
    prob_matrix_sampled = additional_fns.sampled_measurements(prob_matrix_exact, num_shots, seed=seed)
    return {"exact": prob_matrix_exact, "sampled": prob_matrix_sampled}


def get_perturbed_state_matrix(state_matrix: Matrix, rank:int, epsilon: float, seed: int=42) -> Vector:
    """Get a perturbed state vector by applying a random Kraus operator to the original state vector.
    
    Args:
        state_matrix: The original state matrix of shape (dim_in, dim_in*).
        rank: The rank of the Kraus operator to be applied. This will be the new rank of the perturbed state as the original state is assumed to be rank 1.
        epsilon: The perturbation strength.
        seed: The random seed for generating the Kraus operator.
        
    Returns:
        The perturbed state matrix of shape (dim_in, dim_in*).
    """
    dim = state_matrix.shape[0]
    kraus_perturbed = additional_fns.randKrausSet(1, dim, rank_kraus=rank, a=epsilon, seed=seed)
    kraus_perturbed_superop = kraus_tensor_to_mgst(kraus_perturbed) # num_gates, dim_out^2, dim_in^2
    # squeeze the num_gates out
    kraus_perturbed_superop = kraus_perturbed_superop.squeeze(axis=0) # dim^2, dim^2
    return apply_single_superop_to_state_matrix(state_matrix, kraus_perturbed_superop)

def apply_single_superop_to_state_matrix(state_matrix: Matrix, superop: Matrix | Tensor) -> Matrix:
    """Apply a single superoperator to a state matrix.

    Args:
        state_matrix: The original state matrix of shape (dim_in, dim_in*).
        superop: The superoperator to apply of shape (dim_in*dim_in, dim_in*dim_in).

    Returns:
        The transformed state matrix of shape (dim_in, dim_in*).
    """
    if superop.ndim == 3:
        superop = superop.squeeze(axis=0) # dim^2, dim^2
    
    dim = state_matrix.shape[0]
    state_vect = state_matrix.flatten()  # dim^2
    state_perturbed = jnp.dot(superop, state_vect)  # dim^2
    return state_perturbed.reshape((dim, dim))  # dim_in, dim_in*

def get_perturbed_compressed_state_tensor(state_matrix: Matrix, rank:int, epsilon: float, seed: int=42) -> Matrix:
    """Get a perturbed compressed sttate tensor by applying a random Kraus operator to the original state tensor.
    
    Args:
        state_matrix: The original state matrix of shape (dim_in, dim_in*).
        rank: The rank of the Kraus operator to be applied. This will be the new rank of the perturbed state as the original state is assumed to be rank 1.
        epsilon: The perturbation strength.
        seed: The random seed for generating the Kraus operator.
    Returns:
        The perturbed compressed state tensor of shape (dim_in, rank_state).
    """
    perturbed_state_matrix = get_perturbed_state_matrix(state_matrix, rank, epsilon, seed)
    state_psd_perturbed = factorize_psd_truncated(perturbed_state_matrix, max_rank=rank) # dim_in, rank_state
    return state_psd_perturbed

def get_perturbed_povm_tensor(povm_tensor: Tensor, rank:int, epsilon: float, seed: int=42) -> Tensor:
    """Get a perturbed POVM tensor by applying a random Kraus operator to the original POVM tensor.
    
    Args:
        povm_tensor: The original POVM tensor of shape (num_povm, dim_out, dim_out*).
        rank: The rank of the Kraus operator to be applied. This will be the new rank of the perturbed POVM as the original POVM is assumed to be rank 1.
        epsilon: The perturbation strength.
        seed: The random seed for generating the Kraus operator.
        
    Returns:
        The perturbed POVM tensor of shape (num_povm, dim_out, dim_out*).
    """
    num_povm, dim, dim = povm_tensor.shape
    kraus_perturbed = additional_fns.randKrausSet(1, dim, rank_kraus=rank, a=epsilon, seed=seed)
    kraus_perturbed_superop = kraus_tensor_to_mgst(kraus_perturbed) # 1, dim_out^2, dim_in^2*
    # squeeze the num_gates out
    kraus_perturbed_superop = kraus_perturbed_superop.squeeze(axis=0) # dim_out^2, dim_in^2*
    return apply_superops_to_povm_tensor(povm_tensor, kraus_perturbed_superop)

def apply_superops_to_povm_tensor(povm_tensor: Tensor, superop: Tensor) -> Tensor:
    """Apply a single superoperator to a POVM tensor.

    This now works with superop with batched dimensions.

    Args:
        povm_tensor: The original POVM tensor of shape (num_povm, dim_out, dim_out*).
        superop: The superoperator to apply of shape (..., dim_out x dim_out*, dim_in x dim_in*).

    Returns:
        The transformed POVM tensor of shape (num_povm, dim_out, dim_out*).
    """
    num_povm, dim, _ = povm_tensor.shape
    povm_vect = povm_tensor.reshape((num_povm, dim * dim))  # num_povm, dim_out * dim_out
    povm_vect = jnp.einsum('...ij, ...jk -> ...ik', povm_vect, superop)  # num_povm, dim_out * dim_out
    return povm_vect.reshape((num_povm, dim, dim))  # num_povm, dim_out, dim_out*

def get_perturbed_compressed_povm_tensor(povm_tensor: Tensor, rank:int, epsilon: float, seed: int=42) -> Tensor:
    """Get a perturbed compressed POVM tensor by applying a random Kraus operator to the original POVM tensor.
    
    Args:
        povm_tensor: The original POVM tensor of shape (num_povm, dim_out, dim_out*).
        rank: The rank of the Kraus operator to be applied. This will be the new rank of the perturbed POVM as the original POVM is assumed to be rank 1.
        epsilon: The perturbation strength.
        seed: The random seed for generating the Kraus operator.
    Returns:
        The perturbed compressed POVM tensor of shape (num_povm, rank_povm, dim_out).
    """
    perturbed_povm_tensor = get_perturbed_povm_tensor(povm_tensor, rank, epsilon, seed)
    povm_psd_perturbed = factorize_psd_truncated(perturbed_povm_tensor, max_rank=rank).transpose(0, 2, 1).conj() # num_povm, rank_povm, dim_out
    return povm_psd_perturbed


def depolarizing_kraus_operators(p:float, num_qubits:int)-> Tensor:
    """
    Generate Kraus operators for the depolarizing channel.
    
    Args:
        p: Error rate (depolarizing probability)
        num_qubits: Number of qubits (1 or 2)
    
    Returns:
        A kraus tensor of shape (kraus_rank, dim_out, dim_in) where dim_out = dim_in = 2^num_qubits
    """
    if num_qubits == 1:
        # Single qubit Pauli matrices
        I = jnp.array([[1, 0], [0, 1]], dtype=complex)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        
        # Kraus operators
        K0 = jnp.sqrt(1 - 3*p/4) * I
        K1 = jnp.sqrt(p/4) * X
        K2 = jnp.sqrt(p/4) * Y
        K3 = jnp.sqrt(p/4) * Z
        
        return jnp.array([K0, K1, K2, K3])
    
    elif num_qubits == 2:
        # Two-qubit Pauli matrices (tensor products)
        I = jnp.array([[1, 0], [0, 1]], dtype=complex)
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        
        paulis = [I, X, Y, Z]
        two_qubit_paulis = [jnp.kron(P1, P2) for P1 in paulis for P2 in paulis]
        
        # For 2-qubit depolarizing: 1 identity + 15 non-identity Paulis
        K0 = jnp.sqrt(1 - 15*p/16) * two_qubit_paulis[0]  # Identity
        kraus_ops = [K0]
        for pauli in two_qubit_paulis[1:]:  # Non-identity Paulis
            kraus_ops.append(jnp.sqrt(p/16) * pauli)
        
        return jnp.array(kraus_ops)

def amplitude_damping_kraus_operators(gamma: float, num_qubits:int) -> Tensor:
    """
    Generate Kraus operators for the amplitude damping channel.
    
    Args:
        gamma: Decay rate (amplitude damping parameter)
        num_qubits: Number of qubits (1 or 2)
    
    Returns:
        A kraus tensor of shape (kraus_rank, dim_out, dim_in) where dim_out = dim_in = 2^num_qubits
    """
    if num_qubits == 1:
        # Single qubit amplitude damping
        K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=complex)
        K1 = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=complex)
        
        return jnp.array([K0, K1])
    
    elif num_qubits == 2:
        # Two-qubit amplitude damping (independent on each qubit)
        K0_1q = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=complex)
        K1_1q = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=complex)
        
        # Tensor products for both qubits
        kraus_ops = []
        for K_a in [K0_1q, K1_1q]:
            for K_b in [K0_1q, K1_1q]:
                kraus_ops.append(jnp.kron(K_a, K_b))
        
        return jnp.array(kraus_ops)
    
def contract_noiseless_and_noisy_kraus_tensors(kraus_unitary_gates: Tensor, kraus_tensor_noisy: Tensor) -> Tensor:
    """
    Contract a noiseless Kraus tensor with a noisy Kraus tensor to produce a new Kraus tensor that represents the combined effect of both channels.
    
    Namely: Gate o Noise -> Noisy Gate

    Args:
        kraus_unitary_gates: The noiseless Kraus tensor of shape (num_gates, dim_out, dim_in) or (num_gates, 1, dim_out, dim_in).
        kraus_tensor_noisy: The noisy Kraus tensor of shape (kraus_rank, dim_out, dim_in) or (num_gates, kraus_rank, dim_out, dim_in).

    Returns:
        A new Kraus tensor of shape (num_gates, kraus_rank, dim_out, dim_in) representing the noisy gate.
    """
    tensor_rank_noiseless = kraus_unitary_gates.ndim    
    tensor_rank_noisy = kraus_tensor_noisy.ndim
    # check they are either 3 or 4 dimensional
    if tensor_rank_noiseless not in [3, 4]:
        raise ValueError(f"Noiseless Kraus tensor must be 3 or 4 dimensional, but got {tensor_rank_noiseless}.")
    if tensor_rank_noisy not in [3, 4]:
        raise ValueError(f"Noisy Kraus tensor must be 3 or 4 dimensional, but got {tensor_rank_noisy}.")
    
    # prune the first kraus dimension if it's 1
    if kraus_unitary_gates.ndim == 4:
        # this will raise an error if the first dimension is not 1
        kraus_unitary_gates = jnp.squeeze(kraus_unitary_gates, axis=1)
    
    return jnp.einsum("...kij,...jm->...kim", kraus_tensor_noisy, kraus_unitary_gates) # num_gates, kraus_rank, dim_out, dim_in

def get_random_noisy_kraus_tensors_for_gate_set(num_gates:int, dim:int, rank_kraus:int, noise_strength:float, seed:int|tuple=None) -> dict[str, Tensor]:
    """Generate 3 random Kraus tensors of the same rank
    
    These are meant to be used as the noisy kraus tensors to be applied to a target gate set
    
    Args:
        num_gates: Number of gates in the gate set.
        dim: Dimension of the Hilbert space the Kraus operators acts on. This is dim = 2**num_qubits
        rank_kraus: Number of Kraus operators per gate ("Kraus rank")
        noise_strength: Parameter to control the norm of the hermitian generator and thereby
            the strength of the noise.
        seed: Random seed for reproducibility.
    Returns:
        A dictionary containing the random Kraus tensors for the gate set.
    """
    if seed is None:
        seed = 42
    if isinstance(seed, int):
        seed = (seed, seed+1, seed+2)
        
    tensor_for_kraus = additional_fns.randKrausSet(num_gates=num_gates, dim=dim, rank_kraus=rank_kraus, a=noise_strength, seed=seed[0]) # num_gates, kraus_rank, dim_out, dim_in
    
    tensor_for_povm = additional_fns.randKrausSet(num_gates=1, dim=dim, rank_kraus=rank_kraus, a=noise_strength, seed=seed[1]).squeeze(axis=0) # kraus_rank, dim_out, dim_out
    
    tensor_for_state = additional_fns.randKrausSet(num_gates=1, dim=dim, rank_kraus=rank_kraus, a=noise_strength, seed=seed[2]).squeeze(axis=0) # kraus_rank, dim_out, dim_out
    
    return {"kraus": tensor_for_kraus, "povm": tensor_for_povm, "state": tensor_for_state}

def _unvectorize_target_gate_set(gate_set:dict[str, Tensor])->dict[str, Tensor]:
    """Unvectorize the superoperators from a given gate set.
    
    Args:
        gate_set: A dictionary containing the superoperators for Kraus, POVM, and State in vectorized form.
    Returns:
        A dictionary containing the unvectorized superoperators for Kraus, POVM, and State.
    """
    # we get the dimension from the kraus tensor
    # assume shape (num_gates, 1, dim, dim)
    dim = gate_set["kraus"].shape[-1]
    gate_set_unvect = gate_set.copy()
    gate_set_unvect["povm"] = gate_set_unvect["povm"].reshape(dim, dim, dim)
    gate_set_unvect["state"] = gate_set_unvect["state"].reshape(dim, dim)
    return gate_set_unvect

def apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy:dict[str, Tensor], target_gate_set:dict[str, Tensor], is_target_vectorized:bool=False)-> dict[str, Tensor]:
    """Apply noisy Kraus tensors to a target gate set to produce the noisy gate set.
    
    Args:
        kraus_tensors_noisy: A dictionary containing the noisy Kraus tensors for the gate set.
        target_gate_set: A dictionary containing the target gate set (kraus, povm, state).
            The POVM and State are assumed to be in the "full representation".
            Namely, of shapes (num_povm, dim_out*, dim_out) and (dim_in, dim_in*) respectively.
            On the other hand, the kraus should be the PSD representation of shape (num_gates, kraus_rank, dim_out, dim_in).
    Returns:
        A dictionary containing the noisy gate set (kraus, povm, state) in superop/vectorized form.
        Shapes:
        * Kraus: (num_gates, dim_out x dim_out, dim_in x dim_in)
        * POVM: (num_povm, dim_out x dim_out)
        * State: (dim_in x dim_in)
    """
    noise_tensor_kraus = kraus_tensors_noisy["kraus"] # (num_gates, kraus_rank, dim_out, dim_in)
    noise_tensor_povm = kraus_tensors_noisy["povm"] # (kraus_rank, dim_out*, dim_out)
    noise_tensor_state = kraus_tensors_noisy["state"] # (kraus_rank, dim_in, dim_in*)
    
    noise_superop_povm = kraus_tensor_to_mgst(noise_tensor_povm) # (kraus_rank, dim_out*dim_out, dim_out*dim_out)
    noise_superop_state = kraus_tensor_to_mgst(noise_tensor_state) # (kraus_rank, dim_in*dim_in, dim_in*dim_in)
    
    # check expected tensor ranks
    expected_povm_ndim = 2 if is_target_vectorized else 3
    expected_state_ndim = 1 if is_target_vectorized else 2
    
    if target_gate_set["povm"].ndim != expected_povm_ndim or target_gate_set["state"].ndim != expected_state_ndim:
        raise ValueError(f"Target gate set POVM and State tensors have unexpected dimensions. Expected POVM ndim={expected_povm_ndim}, got {target_gate_set['povm'].ndim}. Expected State ndim={expected_state_ndim}, got {target_gate_set['state'].ndim}.")
    
    if is_target_vectorized:
        target_gate_set = _unvectorize_target_gate_set(target_gate_set)
    
    kraus_tensor_target = target_gate_set["kraus"] # (num_gates, 1, dim_out, dim_in)
    povm_tensor_target = target_gate_set["povm"] # (num_povm, dim_out*, dim_out)
    state_tensor_target = target_gate_set["state"] # (dim_in, dim_in*)        
        
    # apply the noisy Kraus tensors to the target gate set
    kraus_tensor_noisy = contract_noiseless_and_noisy_kraus_tensors(kraus_tensor_target, noise_tensor_kraus) # (num_gates, kraus_rank, dim_out, dim_in)
    
    kraus_superop_noisy = kraus_tensor_to_mgst(kraus_tensor_noisy) # (num_gates, dim_out*dim_out, dim_in*dim_in)
    
    povm_tensor_noisy = apply_superops_to_povm_tensor(povm_tensor_target, noise_superop_povm) # (num_povm, dim_out, dim_out)
    num_povm = povm_tensor_noisy.shape[0]
    povm_tensor_noisy = povm_tensor_noisy.reshape((num_povm, -1)) # (num_povm, dim_out x dim_out)
        
    state_matrix_noisy = apply_single_superop_to_state_matrix(state_tensor_target, noise_superop_state).reshape((-1)) # (dim_in x dim_in)
    
    
    return {"kraus": kraus_superop_noisy, "povm": povm_tensor_noisy, "state": state_matrix_noisy}
    
def get_initial_state_and_measurement(dim: int) -> tuple[Matrix, Tensor]:
    """Get the initial density matrix and POVM for a GST experiment given a physical dimension

    Args:
        dim: The physical dimension of the system (e.g., 2 for a qubit, 4 for two qubits).

    Returns:
        tuple[Matrix, Tensor]: The initial density matrix and the POVM for the GST experiment.
    """
    state = generate_target_state(dim)
    
    state = state.reshape((dim, dim))  # Reshape to (dim, dim) for density matrix            
    # Computational basis measurement:
    povm = generate_target_povm(dim)

    num_povm_elements = povm.shape[0]
    povm = povm.reshape((num_povm_elements, dim, dim))  # Reshape to (num_povm_elements, dim_out, dim_in)
    return state, povm

def get_default_optimization_options() -> tuple[dict[str, OperatorSchedule], int, float, float, float, float, bool, bool]:
    """Get default optimization options for the Riemannian optimization workflow."""
    num_iterations_outer = 5
    num_iterations_tr = 10
    num_iterations_cg = 10
    tol_grad = 1e-4
    kappa = 1/10
    theta = 1 # before by mistake we set it to 0.5 (WRONG)
    
    tr_schedule_verbose = OperatorSchedule.default(num_iterations_outer, TrustRegionOptions(num_iterations=num_iterations_tr, verbose=True, verbose_cg=True, num_iterations_cg=num_iterations_cg, tol_grad=tol_grad, kappa_cg=kappa, theta_cg=theta))

    optimization_schedule_all_tr = {
        "kraus": tr_schedule_verbose,
        "povm": tr_schedule_verbose,
        "state": tr_schedule_verbose
    }
    
    relative_precision = 1e-5
    global_gradient_norm_tol = 1e-6

    threshold_multiplier = None
    noise_threshold = None
    optimization_verbose = False
    compute_least_squares = False

    return optimization_schedule_all_tr, num_iterations_outer, relative_precision, global_gradient_norm_tol, threshold_multiplier, noise_threshold, optimization_verbose, compute_least_squares


def check_operators_dictionaries(*dicts)-> None:
    """
    Check if all dictionaries have valid keys.
    """
    EXPECTED_KEYS = {"kraus", "povm", "state"}

    for dict in dicts:
        if set(dict.keys()) != EXPECTED_KEYS:
            raise ValueError(f"Dictionary keys {dict.keys()} do not match expected keys {EXPECTED_KEYS}.")

def generate_random_initial_operators(num_gates:int, dim:int, kraus_rank:int, povm_rank:int, state_rank:int, seeds:tuple[int, int, int]|None=None) -> dict[str, jnp.ndarray]:
    """Generate random initial operators (kraus, povm, state) for the GST experiment."""
    rndm_gate_set_superop = get_random_gate_set_superop(num_gates=num_gates, dim=dim, kraus_rank=kraus_rank, num_povm=dim, seeds=seeds)
    rndm_gate_set_psd = get_compressed_rep_from_mgst_output(*rndm_gate_set_superop.values(), kraus_rank=kraus_rank, state_rank=state_rank, povm_rank=povm_rank)
    return {
        "kraus": rndm_gate_set_psd[0],
        "povm": rndm_gate_set_psd[1],
        "state": rndm_gate_set_psd[2],
    }

def get_parameters_from_gate_set(gate_set:dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Get the num_gates, dim, kraus_rank, povm_rank, state_rank from a gate set dictionary."""
    num_gates, kraus_rank, dim, _ = gate_set["kraus"].shape
    num_povm, povm_rank, _ = gate_set["povm"].shape
    _, state_rank = gate_set["state"].shape
    
    return {
        "num_gates": num_gates,
        "dim": dim,
        "kraus_rank": kraus_rank,
        "povm_rank": povm_rank,
        "state_rank": state_rank,
    }
    
def generate_random_gate_set_from_gate_set(gate_set:dict[str, jnp.ndarray], seeds:tuple[int, int, int]|None=None) -> dict[str, jnp.ndarray]:
    """Generate a random gate set by applying random Kraus operators to a given gate set."""
    params = get_parameters_from_gate_set(gate_set)
    return generate_random_initial_operators(**params, seeds=seeds)

def run_optimization_workflow_multi_init(num_shots:int, operators_psd_true:dict[str, jnp.ndarray], target_superops:dict[str, jnp.ndarray], optimization_options:dict[str, Any]=None, use_log_likelihood:bool=False, num_sequences:int|None=None, indices_dict:dict[str, list[list[int]]]|None=None, num_max_iterations_random_inits:int=5, num_max_random_inits:int=10, batch_size:int=0.1, seq_len_list:list[int]|None=None, threshold_multiplier:float=5.0):    
    # Generate the random sequences already here to fix it for all random inits.
    num_gates = target_superops["kraus"].shape[0]
    indices_dict = _process_indices_generation(num_gates=num_gates, num_sequences=num_sequences, seq_len_list=seq_len_list, indices_dict=indices_dict)
    
    probability_matrices = _process_probability_matrices(operators_psd_true=operators_psd_true, indices_list=indices_dict["gate_indices"], num_shots=num_shots, probability_matrices=None)
        
    noise_threshold = estimate_noise_floor_threshold(multiplier=threshold_multiplier, num_shots=num_shots, probability_matrix=probability_matrices["sampled"])
    
    if optimization_options is None:
        optimization_options_random_init = {}
    else:
        optimization_options_random_init = optimization_options.copy()
    
    optimization_options_random_init["noise_threshold"] = noise_threshold
    optimization_options_random_init["num_iterations_outer"] = num_max_iterations_random_inits
    converged_below_noise_threshold = False
    
    for init_idx in range(num_max_random_inits):
        print(f"🎲👾 Running optimization workflow with random initialization {init_idx+1}/{num_max_random_inits} 🎲👾")
        
        batched_probability_matrices, batched_indices_dict = batch_probability_matrices_and_indices(probability_matrices=probability_matrices, indices_dict=indices_dict, batch_size=batch_size, seed=None) # no seed to make it random every time
        
        random_init_result = run_optimization_workflow(
            num_shots=num_shots,
            init_operators=None, # generate random initial operators
            operators_psd_true=operators_psd_true,
            target_superops=target_superops,
            use_exact_probabilities=False, # noise threshold only makes sense with sampled probabilities
            optimization_options=optimization_options_random_init,
            warmup_run=False, # is gonna be ran so many times that we don't need to warmup every time
            use_log_likelihood=use_log_likelihood,
            indices_dict=batched_indices_dict, # batched version
            probability_matrices=batched_probability_matrices, # batched version
            verbose=False, # we don't want to print every time
            gauge_optimize=False, # no need to optimize for every run.
        )
        
        if "below noise threshold" in random_init_result["convergence_reason"]:
            converged_below_noise_threshold = True
            break
    
    if converged_below_noise_threshold:
        cost_to_report = {cost_type: history[-1] for cost_type, history in random_init_result["cost_fn_history"].items()}
        print(f"✅ Converged below noise threshold after {init_idx+1} random initializations."
              f"Last cost function values: {cost_to_report}")
    else:
        print(f"⚠️ Did not converge below noise threshold after {num_max_random_inits} random initializations."
              "Using the result from the last random initialization.")
        
    # now we again run a full optimization with either the last result (if max iters is hit) or the result that converged below the noise threshold
    
    final_result = run_optimization_workflow(
                num_shots=num_shots,
                init_operators=random_init_result["optimized_operators"], # use the result from the last random initialization
                operators_psd_true=operators_psd_true,
                target_superops=target_superops,
                use_exact_probabilities=False, # noise threshold only makes sense with sampled probabilities
                optimization_options=optimization_options,
                warmup_run=False,
                use_log_likelihood=True, # force this for "refinement" optimization
                indices_dict=indices_dict, # batched version
                probability_matrices=probability_matrices, # batched version
                verbose=False, # we don't want to print every time
                gauge_optimize=True, # we want to optimize the gauge for the final result
            )
    
    return final_result, random_init_result
    
def _process_optimization_options(optimization_options:dict[str, Any]|None, num_shots:int, probability_matrix:jnp.ndarray) -> dict[str, Any]:
    """Process the optimization options and return the relevant parameters for the optimization workflow.
    
    1. If noise threshold is provided, it is used in the optimization workflow.
    2. If threshold_multiplier is provided, it is used to compute the noise threshold from the probability matrix.
    3. If neither is provided, the noise threshold is None and means it won't be used in the optimization.
    
    """
    if optimization_options is None:
            optimization_schedule, num_iterations_outer, relative_precision, global_gradient_norm_tol, threshold_multiplier, noise_threshold, optimization_verbose, compute_least_squares = get_default_optimization_options()
    else:
        expected_keys = set(["schedule", "num_iterations_outer", "relative_precision", "global_gradient_norm_tol", "threshold_multiplier", "noise_threshold", "optimization_verbose"])
        optimization_keys = set(optimization_options.keys())
        if not optimization_keys.issubset(expected_keys):
            raise ValueError(f"Optimization options must contain keys: {expected_keys}, but got {optimization_keys}.")
        
        optimization_schedule = optimization_options["schedule"]
        num_iterations_outer = optimization_options["num_iterations_outer"]
        relative_precision = optimization_options["relative_precision"]
        global_gradient_norm_tol = optimization_options["global_gradient_norm_tol"]
        
        noise_threshold = optimization_options.get("noise_threshold", None)
        threshold_multiplier = optimization_options.get("threshold_multiplier", None)
        
        optimization_verbose = optimization_options.get("optimization_verbose", False)
        compute_least_squares = optimization_options.get("compute_least_squares", False)
    
    if noise_threshold is None and threshold_multiplier is not None:
        noise_threshold = estimate_noise_floor_threshold(multiplier=threshold_multiplier, num_shots=num_shots, probability_matrix=probability_matrix)
        
    return {
        "optimization_schedule": optimization_schedule,
        "num_iterations": num_iterations_outer,
        "relative_precision": relative_precision,
        "global_gradient_norm_tol": global_gradient_norm_tol,
        "noise_threshold": noise_threshold,
        "verbose": optimization_verbose,
        "compute_least_squares": compute_least_squares,
    }

def _process_indices_generation(num_gates:int, num_sequences:int|None, seq_len_list:list[int]|None, indices_dict:dict[str, list[list[int]]]|None) -> dict[str, list[list[int]]]:
    """
    Process the generation of sequence indices for the optimization workflow.
    
    Args:
        num_gates: Number of gates in the circuit.
        num_sequences: Number of sequences to generate.
        seq_len_list: List of three integers representing the minimum, cut, and maximum sequence lengths.
        indices_dict: Optional dictionary containing precomputed sequence indices. If None, new indices will be generated.
    Returns:
        A dictionary containing the gate indices and the gate indices with negative values.
    """
    if indices_dict is None:
        if num_sequences is None:
            raise ValueError("If indices_dict is None, num_sequences must be provided to generate new sequence indices.")
        if seq_len_list is None:
            seq_len_list = [1, 8, 14]  # default sequence length list
        indices_dict = generate_sequence_indices(num_gates=num_gates, num_circuits=num_sequences, seq_len_list=seq_len_list)
    else:
        if set(indices_dict) != {"gate_indices", "gate_indices_with_negs"}:
            raise ValueError(f"indices_dict must contain keys: ['gate_indices', 'gate_indices_with_negs'], but got {list(indices_dict.keys())}.")
        if num_sequences is not None:
            warnings.warn("num_sequences is ignored when indices_dict is provided. Using the provided indices_dict.")
    return indices_dict
    
def _process_probability_matrices(operators_psd_true:dict[str, jnp.ndarray], indices_list:list[list[int]], num_shots:int, probability_matrices:dict[str, jnp.ndarray]|None) -> jnp.ndarray:
    """
    Process the generation of probability matrices for the optimization workflow.
    
    Args:
        operators_psd_true: Dictionary containing the true operators (kraus, povm, state).
        indices_list: List of lists containing the gate indices for each sequence.
        num_shots: Number of shots for sampling.
        probability_matrices: Optional dictionary containing precomputed probability matrices. If None, new probability matrices will be generated.
    
    Returns:
        A dictionary containing the exact and sampled probability matrices.
    """
    if probability_matrices is None:
        probability_matrices = compute_probability_matrices(*operators_psd_true.values(), gate_indices=indices_list, num_shots=num_shots, seed=42)
    else:
        if set(probability_matrices) != {"exact", "sampled"}:
            raise ValueError(f"probability_matrices must contain keys: ['exact', 'sampled'], but got {list(probability_matrices.keys())}.")
        
    return probability_matrices

def run_optimization_workflow(num_shots:int, init_operators:dict[str, jnp.ndarray]|None, operators_psd_true:dict[str, jnp.ndarray], target_superops:dict[str, jnp.ndarray], use_exact_probabilities:bool=False, optimization_options:dict[str, Any]=None, warmup_run:bool=False, use_log_likelihood:bool=False, num_sequences:int|None=None, indices_dict:dict[str, list[list[int]]]|None=None, seq_len_list:list[int]|None=None, probability_matrices:dict[str, jnp.ndarray]|None=None, verbose:bool=False, gauge_optimize:bool=True) -> dict[str, Any]:
    """
    Run the optimization workflow for the Riemannian optimization of the GST operators.
    
    Args:
        num_sequences: Number of sequences to generate.
        num_shots: Number of shots for sampling.
        init_operators: Dictionary containing the initial operators (kraus, povm, state).
        operators_psd_true: Dictionary containing the true operators (kraus, povm, state), namely the operators that generate the probability matrices used for the cost function optimization. In other words, these are the operators that generate the "experimental" data.
        target_superops: Dictionary containing the target superoperators (kraus, povm, state).
        use_exact_probabilities: Whether to use exact probabilities or sampled probabilities for the cost function optimization. If True, exact probabilities are used (infinite shots); if False, sampled probabilities are used.
        optimization_options: Dictionary containing optimization options. If None, default options are used.
        use_log_likelihood: Whether to use log-likelihood or least-squares for the cost function optimization. If True, log-likelihood is used; if False, least-squares is used.
        warmup_run: Whether to perform a warmup run of the cost function before the optimization. This can help with JIT compilation and caching.
        optimization_verbose: Whether to print information during the optimization.
        compute_least_squares: Whether to compute the least squares value during the optimization in addition to the cost function. This can be useful for debugging and analysis, but may add additional computational overhead.
        gauge_optimize: Whether to perform gauge optimization after the main optimization.
        indices_dict: Optional dictionary containing precomputed sequence indices. If None, new indices will be generated.

    Returns:
        A dictionary containing the optimized operators, gauged superoperators, target model, cost function history, probability matrices, times for optimization and gauging, and indices dictionary.
    """
    if indices_dict is not None and probability_matrices is not None and verbose:
        warnings.warn("When both `indices_dict` and `probability_matrices` are provided, make sure these are compatible.")
    
    if init_operators is None:
        if verbose:
            print("🎲 Generating random initial operators ...")
        init_operators = generate_random_gate_set_from_gate_set(gate_set=operators_psd_true)
    
    check_operators_dictionaries(operators_psd_true, target_superops, init_operators)    
    
    # we can get the number of gates from any of the kraus tensors
    num_gates = target_superops["kraus"].shape[0]
    
    # return this one
    indices_dict = _process_indices_generation(num_gates=num_gates, num_sequences=num_sequences, seq_len_list=seq_len_list, indices_dict=indices_dict)
        
    indices_list = indices_dict["gate_indices"]

    # return this one
    probability_matrices = _process_probability_matrices(operators_psd_true=operators_psd_true, indices_list=indices_list, num_shots=num_shots, probability_matrices=probability_matrices)
    
    if use_exact_probabilities:
        if verbose:
            print("Using exact probabilities 🔪")
            warnings.warn("Setting num_shots=1 when using exact probs.")
        num_shots = 1
        probability_matrix_kwarg = probability_matrices["exact"]
    else:
        probability_matrix_kwarg = probability_matrices["sampled"]
    
    cost_fn_kwargs = {
        "indices_list": indices_list,
        "prob_matrix": probability_matrix_kwarg,
        "jit": True,
        "use_log_likelihood": use_log_likelihood,
        "verbose": False,  # we don't want to print during the cost function evaluation
    }
    
    if use_log_likelihood:
        if verbose:
            print("Using log-likelihood for the cost function optimization 🧮")
        cost_fn_kwargs["num_shots"] = num_shots
    
    optimization_options = _process_optimization_options(optimization_options=optimization_options, num_shots=num_shots, probability_matrix=probability_matrix_kwarg)
        
    if warmup_run:
        start_compilation_time = time.time()
        if verbose:
            print("🏎️ Warming up the cost function ...")
        initial_cost_value = cost_function_jax_mps(*init_operators.values(), **cost_fn_kwargs)
        if verbose:
            print(f"Initial cost value: {initial_cost_value}")
    
    # start timer
    start_time = time.time()
    # return these ones
    optimized_operators, cost_fn_history, convergence_reason = run_riemannian_optimization(
        *init_operators.values(),
        cost_function=cost_function_jax_mps,
        cost_fn_kwargs=cost_fn_kwargs,
        **optimization_options,
    )
    # end timer
    time_after_opt = time.time()
    optimization_time = time_after_opt - start_time
    
    times = {"optimization_time": optimization_time}
    
    gauge_dict = {}
    if gauge_optimize:
        kraus_superop_gauged, povm_vect_gauged, state_vect_gauged, target_model_pygsti = perform_gauge(optimized_operators=optimized_operators, target_superops=target_superops, num_gates=num_gates)
        
        time_after_gauge = time.time()
        gauge_time = time_after_gauge - time_after_opt

        superops_gauged = {
            "kraus": kraus_superop_gauged,
            "povm": povm_vect_gauged,
            "state": state_vect_gauged
        }
        
        times["gauge_time"] = gauge_time
 
        gauge_dict["superops_gauged"] = superops_gauged
        gauge_dict["target_model_pygsti"] = target_model_pygsti
    
    if warmup_run:
        compilation_time = start_time - start_compilation_time
        times["compilation_time"] = compilation_time
    
    return {"optimized_operators": optimized_operators, "cost_fn_history": cost_fn_history, "probability_matrices": probability_matrices, "indices_dict": indices_dict, "times": times, "convergence_reason": convergence_reason, "init_operators":init_operators} | gauge_dict
    
def perform_gauge(optimized_operators:dict[jnp.ndarray], target_superops:dict[jnp.ndarray], num_gates:int):
    """
    Perform gauge optimization on the optimized operators.
    """
    optimized_superops = get_mgst_tensors_from_psd_representation(*optimized_operators.values())
    
    gauge_weights = dict({f"G%i" % i: 1 for i in range(num_gates)}, **{"spam": 0.1})

    target_model_pygsti = arrays_to_pygsti_model(*target_superops.values(), basis="std")

    kraus_superop_gauged, povm_vect_gauged, state_vect_gauged = gauge_opt(X=optimized_superops["kraus"], E=optimized_superops["povm"], rho=optimized_superops["state"], target_mdl=target_model_pygsti, weights=gauge_weights)
    
    return kraus_superop_gauged, povm_vect_gauged, state_vect_gauged, target_model_pygsti

def get_random_gate_set_superop(num_gates:int, dim:int, kraus_rank:int, num_povm:int, seeds:list[int]=None) -> dict[str, Tensor]:
    """Get a random gate set in superoperator form of the specified dimensions."""     
    kraus_psd, kraus_superop, povm_superop, state_superop = additional_fns.random_gs(
        d=num_gates,
        r=int(dim**2),
        rK=kraus_rank,
        n_povm=num_povm,
        seeds=seeds
    )
    return {"kraus": kraus_superop, "povm": povm_superop, "state": state_superop}

def batch_probability_matrices_and_indices(
    probability_matrices: dict[str, Matrix],
    indices_dict: dict[str, Any],
    batch_size: int | float,
    seed: int | None = None,
) -> tuple[dict[str, Matrix], dict[str, Any]]:
    """
    Subsample a consistent batch of circuits from probability matrices and indices.

    Expected shapes:
    - probability_matrices["exact"]:   (num_povm, num_circuits)
    - probability_matrices["sampled"]: (num_povm, num_circuits)
    - indices_dict["gate_indices"]: length num_circuits
    - indices_dict["gate_indices_with_negs"]: (num_circuits, max_len)
    """
    required_prob_keys = {"exact", "sampled"}
    required_idx_keys = {"gate_indices", "gate_indices_with_negs"}

    if set(probability_matrices.keys()) != required_prob_keys:
        raise ValueError(
            f"probability_matrices must contain keys {required_prob_keys}, "
            f"got {set(probability_matrices.keys())}."
        )
    if set(indices_dict.keys()) != required_idx_keys:
        raise ValueError(
            f"indices_dict must contain keys {required_idx_keys}, "
            f"got {set(indices_dict.keys())}."
        )

    prob_exact = probability_matrices["exact"]
    prob_sampled = probability_matrices["sampled"]
    num_circuits = prob_exact.shape[1]

    gate_indices = indices_dict["gate_indices"]
    gate_indices_with_negs = indices_dict["gate_indices_with_negs"]

    if len(gate_indices) != len(gate_indices_with_negs) or len(gate_indices) != num_circuits:
        raise ValueError(
            f"Mismatch in gate indices lengths: "
            f"len(gate_indices)={len(gate_indices)}, "
            f"len(gate_indices_with_negs)={len(gate_indices_with_negs)}, "
            f"num_circuits={num_circuits}."
        )

    # Same semantics as additional_fns.batch
    if num_circuits <= batch_size:
        return probability_matrices, indices_dict

    # if given as a fraction, convert to absolute number of circuits
    if batch_size < 1:
        batch_size = int(batch_size * num_circuits // 1)

    batch_size = max(1, min(int(batch_size), num_circuits))

    rng = np.random.default_rng(seed)
    selected = rng.choice(num_circuits, size=batch_size, replace=False)
    selected.sort()

    batched_probability_matrices = {
        "exact": prob_exact[:, selected],
        "sampled": prob_sampled[:, selected],
    }

    batched_indices_dict = {
        "gate_indices": [gate_indices[i] for i in selected],
        "gate_indices_with_negs": [gate_indices_with_negs[i] for i in selected]
    }

    return batched_probability_matrices, batched_indices_dict