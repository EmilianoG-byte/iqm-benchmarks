"""Functions used to simulate samples for testing purposes"""
import time

from mGST import additional_fns
from mGST.low_level_jit import contract_mps_all_povm
from mGST.utility_functions_comparisons import kraus_tensor_to_mgst, factorize_psd_truncated, get_mgst_tensors_from_psd_representation, generate_target_state, generate_target_povm, get_compressed_rep_from_mgst_output

from mGST.typing import Tensor, Matrix, TrustRegionOptions, Vector
import jax.numpy as jnp
import numpy as np

from mGST.typing import OperatorSchedule, TrustRegionOptions, OptimizationResult
from mGST.trust_region import (
    run_riemannian_optimization,
    estimate_noise_floor_threshold,
    compute_cost_floor,
    convergence_criteria_from_noise_threshold,
    convergence_criteria_from_relative_precision,
    convergence_criteria_from_absolute_tolerance,
)
from mGST.jitted_objective import JittedObjective
from mGST.low_level_jit import cost_function_jax_mps
from mGST.reporting.reporting import gauge_opt
from mGST.compatibility import arrays_to_pygsti_model

from typing import Any
from collections.abc import Sequence

import warnings

def generate_sequence_indices(num_gates:int, num_circuits:int, seq_len_list:list[int], seed:int|None=None) -> dict[str, list[list[int]]]:
    """Generate a sequence of gate indices

    This code is adpated from `compressive_gst.generate_meas_circuits`

    Args:
        num_gates: Number of gates in the circuit.
        num_circuits: Number of circuits to generate.
        seq_len_list: List of three integers representing the minimum, cut, and maximum sequence lengths.
        seed: The random seed for sequence generation.

    Returns:
        A dictionary containing the gate indices and the gate indices with negative values.
    """

    # Calculate number of short and long circuits
    N_short = int(jnp.ceil(num_circuits / 2))
    N_long = int(jnp.floor(num_circuits / 2))
    L_MIN, L_CUT, L_MAX = seq_len_list

    gate_indices_with_negs = additional_fns.random_seq_design(num_gates, L_MIN, L_CUT, L_MAX, N_short, N_long, seed=seed)
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

def run_optimization_workflow_multi_init(num_shots:int, operators_psd_true:dict[str, jnp.ndarray], target_superops:dict[str, jnp.ndarray], optimization_options:dict[str, Any], num_sequences:int|None=None, seq_len_list:list[int]|None=None, indices_dict:dict[str, list[list[int]]]|None=None, probability_matrices:dict[str, Matrix]|None=None, num_max_iterations_random_inits:int=5, num_max_random_inits:int=10, batch_size:int=0.1, threshold_multiplier:float=5.0, num_batches:int|None=None, seed:int|None=None, verbose:bool=True, gauge_optimize:bool=False)-> tuple[OptimizationResult, OptimizationResult]:
    """Search for a good starting point with random initializations, then refine it on the full data.

    The random-init phase mirrors the batch phase of `mGST.algorithm.run_mGST`: a different batch of
    circuits at every outer iteration, with the noise threshold evaluated on the full data set. The
    batches are drawn once into a pool (see `build_batch_pool`) and reused across initializations,
    which bounds the jit compilation cost to `num_batches` instead of one per outer iteration.

    Args:
        num_shots: Number of measurement shots per circuit.
        operators_psd_true: The true positive semidefinite operators for the system.
        target_superops: The target superoperators for the system.
        optimization_options: Options for the optimization algorithm.
        num_sequences: Number of random sequences to generate. If None, it will be determined from `seq_len_list`.
        seq_len_list: List of sequence lengths to generate. If None, it will be determined from `num_sequences`.
        indices_dict: Pre-generated sequence indices. If None, they will be generated.
        probability_matrices: Pre-computed probability matrices. If None, they will be computed.
        num_max_iterations_random_inits: Maximum number of iterations for each random initialization.
        num_max_random_inits: Maximum number of random initializations to try.
        batch_size: Fraction of sequences to use in each batch.
        threshold_multiplier: Multiplier for the noise threshold estimation.
        num_batches: Size of the batch pool. Defaults to `num_max_iterations_random_inits`, so that each
            initialization sees a different batch at every outer iteration. Larger values give more batch
            diversity across initializations at the price of proportionally more jit compilation.
        seed: Seed for the batch draws and the random initial operators. If None, they are not reproducible.
        verbose: Whether to print progress and debug information.
        gauge_optimize: Whether to perform gauge optimization after the final log-likelihood evaluation.
    """
    # Generate the random sequences already here to fix it for all random inits.
    num_gates = target_superops["kraus"].shape[0]
    indices_dict = _process_indices_generation(num_gates=num_gates, num_sequences=num_sequences, seq_len_list=seq_len_list, indices_dict=indices_dict)
    
    probability_matrices = _process_probability_matrices(operators_psd_true=operators_psd_true, indices_list=indices_dict["gate_indices"], num_shots=num_shots, probability_matrices=probability_matrices)
        
    noise_threshold = estimate_noise_floor_threshold(multiplier=threshold_multiplier, num_shots=num_shots, probability_matrix=probability_matrices["sampled"])
    print(f"🔊 Estimated noise threshold: {noise_threshold:.6e} 🔊")
    # The random-init phase fits the least squares on the sampled probabilities: that is the cost the
    # noise threshold is expressed in.
    cost_fn_kwargs_random_init = {
        "indices_list": indices_dict["gate_indices"],
        "prob_matrix": probability_matrices["sampled"],
        "jit": True,
        "use_log_likelihood": False,
        "verbose": False,
    }

    if num_batches is None:
        num_batches = num_max_iterations_random_inits

    rng = np.random.default_rng(seed)
    init_operators_per_init = [
        generate_random_gate_set_from_gate_set(
            gate_set=operators_psd_true,
            seeds=[int(value) for value in rng.integers(0, 2**31, size=3)],
        )
        for _ in range(num_max_random_inits)
    ]

    # Raises if the schedule does not cover the random-init iterations.
    schedule_random_init = {
        op_name: optimization_options["schedule"][op_name].shorten_to(num_max_iterations_random_inits)
        for op_name in OPERATOR_NAMES
    }

    batch_pool = build_batch_pool(
        probability_matrices=probability_matrices,
        indices_dict=indices_dict,
        batch_size=batch_size,
        num_batches=num_batches,
        cost_fn_kwargs=cost_fn_kwargs_random_init,
        seed=rng,
        validate_at=init_operators_per_init[0],
    )
    
    jitted_objective_full = JittedObjective.from_cost_fn_kwargs(cost_fn_kwargs_random_init)
    jitted_objective_full.validate(cost_function_jax_mps, cost_fn_kwargs_random_init, **_as_objective_operators(init_operators_per_init[0]))

    converged_below_noise_threshold = False
    best_random_result = None
    best_random_idx = None
    for init_idx in range(num_max_random_inits):
        print(f"🎲👾 Running optimization workflow with random initialization {init_idx+1}/{num_max_random_inits} 🎲👾")

        random_init_result = run_optimization_workflow_batched(
            init_operators=init_operators_per_init[init_idx],
            batch_pool=batch_pool,
            cost_fn_kwargs=cost_fn_kwargs_random_init,
            num_iterations=num_max_iterations_random_inits,
            optimization_schedule=schedule_random_init,
            jitted_objective_full=jitted_objective_full,
            init_offset=init_idx,
            noise_threshold=noise_threshold,
            relative_precision=None, #optimization_options.get("relative_precision"),
            # for the least squares, and particularly in the case of random inits, we care more about
            # fitting to the estimated noise floor threshold.
            absolute_tolerance=None,
            verbose=verbose, # would say setting to True makes sense.
        )
        
        if "below noise threshold" in random_init_result["convergence_reason"]:
            converged_below_noise_threshold = True
            best_random_result = random_init_result
            best_random_idx = init_idx
            break
        # if its the first iteration this is the best by default
        if best_random_result is None:
            best_random_result = random_init_result
            best_random_idx = init_idx
        # check if the cost function is better than the previous best
        else:
            current_cost = random_init_result["cost_fn_history"]["main"][-1]
            best_cost = best_random_result["cost_fn_history"]["main"][-1]
            if current_cost < best_cost:
                best_random_result = random_init_result
                best_random_idx = init_idx
                
    if converged_below_noise_threshold:
        print(f"✅ Converged below noise threshold after {init_idx+1} random initializations. ")
    else:
        print(f"⚠️ Did not converge below noise threshold after {num_max_random_inits} random initializations. "
              f"Using the BEST result from random initialization {best_random_idx}")
        
    first_costs = {cost_type: history[0] for cost_type, history in best_random_result["cost_fn_history"].items()}
    final_costs = {cost_type: history[-1] for cost_type, history in best_random_result["cost_fn_history"].items()}  
    
    print(f" Rnd. Init: Initial cost fn values: {first_costs} \n"
          f" Rnd. Init: Final cost fn values: {final_costs}")
    
    # now we again run a full optimization with either the last result (if max iters is hit) or the result that converged below the noise threshold
    
    final_result = run_optimization_workflow(
                num_shots=num_shots,
                init_operators=best_random_result["optimized_operators"], # use the result from the BEST random initialization
                operators_psd_true=operators_psd_true,
                target_superops=target_superops,
                use_exact_probabilities=False, # noise threshold only makes sense with sampled probabilities
                optimization_options=optimization_options, # includes verbosity of optimization itself
                warmup_run=False,
                use_log_likelihood=True, # force this for "refinement" optimization
                indices_dict=indices_dict,
                probability_matrices=probability_matrices,
                verbose=False, # mostly boilerplat verbosity, not for the optimization
                gauge_optimize=gauge_optimize,
            )
    
    return final_result, best_random_result
    
def _process_optimization_options(optimization_options:dict[str, Any]|None, num_shots:int, probability_matrix:jnp.ndarray) -> dict[str, Any]:
    """Process the optimization options and return the relevant parameters for the optimization workflow.
    
    1. If noise threshold is provided, it is used in the optimization workflow.
    2. If threshold_multiplier is provided, it is used to compute the noise threshold from the probability matrix.
    3. If neither is provided, the noise threshold is None and means it won't be used in the optimization.
    
    """
    if optimization_options is None:
            optimization_schedule, num_iterations_outer, relative_precision, global_gradient_norm_tol, threshold_multiplier, noise_threshold, optimization_verbose, compute_least_squares = get_default_optimization_options()
    else:
        expected_keys = set(["schedule", "num_iterations_outer", "relative_precision", "global_gradient_norm_tol", "threshold_multiplier", "noise_threshold", "optimization_verbose", "compute_least_squares", "carry_over_radius", "absolute_tolerance"])
        optimization_keys = set(optimization_options.keys())
        if not optimization_keys.issubset(expected_keys):
            raise ValueError(f"Optimization options must contain keys: {expected_keys}, but got {optimization_keys}.")
        
        # TODO: need to automate this retrieval of options from the dictionary to avoid manual assignment.
        optimization_schedule = optimization_options["schedule"]
        num_iterations_outer = optimization_options["num_iterations_outer"]
        relative_precision = optimization_options["relative_precision"]
        global_gradient_norm_tol = optimization_options["global_gradient_norm_tol"]
        
        noise_threshold = optimization_options.get("noise_threshold", None)
        threshold_multiplier = optimization_options.get("threshold_multiplier", None)
        
        optimization_verbose = optimization_options.get("optimization_verbose", False)
        compute_least_squares = optimization_options.get("compute_least_squares", False)
        carry_over_radius = optimization_options.get("carry_over_radius", True)
        absolute_tolerance = optimization_options.get("absolute_tolerance", None)
    
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
        "carry_over_radius": carry_over_radius,
        "absolute_tolerance": absolute_tolerance,
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
    
def _process_probability_matrices(operators_psd_true:dict[str, jnp.ndarray], indices_list:list[list[int]], num_shots:int, probability_matrices:dict[str, jnp.ndarray]|None) -> dict[str, jnp.ndarray]:
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

def run_optimization_workflow(num_shots:int, init_operators:dict[str, jnp.ndarray]|None, operators_psd_true:dict[str, jnp.ndarray], target_superops:dict[str, jnp.ndarray], use_exact_probabilities:bool=False, optimization_options:dict[str, Any]=None, warmup_run:bool=False, use_log_likelihood:bool=False, num_sequences:int|None=None, indices_dict:dict[str, list[list[int]]]|None=None, seq_len_list:list[int]|None=None, probability_matrices:dict[str, jnp.ndarray]|None=None, verbose:bool=False, gauge_optimize:bool=True) -> OptimizationResult:
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
    seed: int | np.random.Generator | None = None,
) -> tuple[dict[str, Matrix], dict[str, Any]]:
    """
    Subsample a consistent batch of circuits from probability matrices and indices.

    Expected shapes:
    - probability_matrices["exact"]:   (num_povm, num_circuits)
    - probability_matrices["sampled"]: (num_povm, num_circuits)
    - indices_dict["gate_indices"]: length num_circuits
    - indices_dict["gate_indices_with_negs"]: (num_circuits, max_len)

    `seed` may also be a `numpy.random.Generator`, which is then used (and advanced) as is,
    so that successive draws of a pool of batches stay reproducible.
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

OPERATOR_NAMES = ("kraus", "povm", "state")

def build_batch_pool(
    probability_matrices: dict[str, Matrix],
    indices_dict: dict[str, Any],
    batch_size: int | float,
    num_batches: int,
    cost_fn_kwargs: dict[str, Any],
    which_probabilities: str = "sampled",
    seed: int | np.random.Generator | None = None,
    jit_contraction: str = "prefix",
    validate_at: dict[str, jnp.ndarray] | None = None,
) -> list[tuple[dict[str, Any], JittedObjective]]:
    """Draw a fixed pool of batches and compile one `JittedObjective` for each.

    `mGST.jitted_objective.make_vectorized_cost` bakes the circuits and probabilities into the jit
    closure, so a new batch means recompiling the cost and the (gradient, HVP) pair of all three
    operators. Drawing the batches once and cycling through the pool bounds that cost to
    `num_batches` compilations instead of one per outer iteration of every random initialization.

    Args:
        probability_matrices: Full exact and sampled probability matrices.
        indices_dict: Full gate indices, as returned by `generate_sequence_indices`.
        batch_size: Absolute number of circuits per batch, or a fraction of the total.
        num_batches: Number of batches in the pool.
        cost_fn_kwargs: Keyword arguments of the cost function over the full data. `indices_list` and
            `prob_matrix` are replaced by the batched ones; everything else is kept.
        which_probabilities: Which probability matrix the batches are taken from, "sampled" or "exact".
        seed: Seed or generator for the batch draws. A generator is advanced across the draws.
        jit_contraction: Contraction strategy of the jitted cost function, "prefix" or "padded".
        validate_at: Operators at which each objective is checked against `cost_function_jax_mps`.
            If None, the objectives are not validated.

    Returns:
        A list of `num_batches` (cost function keyword arguments, jitted objective) pairs.
    """
    if which_probabilities not in probability_matrices:
        raise ValueError(f"which_probabilities must be one of {sorted(probability_matrices)}, got {which_probabilities}.")

    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(num_batches):
        batched_probability_matrices, batched_indices_dict = batch_probability_matrices_and_indices(
            probability_matrices=probability_matrices,
            indices_dict=indices_dict,
            batch_size=batch_size,
            seed=rng,
        )
        batch_cost_fn_kwargs = cost_fn_kwargs | {
            "indices_list": batched_indices_dict["gate_indices"],
            "prob_matrix": batched_probability_matrices[which_probabilities],
        }
        objective = JittedObjective.from_cost_fn_kwargs(batch_cost_fn_kwargs, contraction=jit_contraction)
        if validate_at is not None:
            objective.validate(cost_function_jax_mps, batch_cost_fn_kwargs, **_as_objective_operators(validate_at))
        pool.append((batch_cost_fn_kwargs, objective))

    return pool

def _as_objective_operators(operators: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Rename a gate set dictionary to the argument names used by `JittedObjective`."""
    return {"kraus_tensor": operators["kraus"], "povm_psd": operators["povm"], "state_psd": operators["state"]}

def run_optimization_workflow_batched(
    init_operators: dict[str, jnp.ndarray],
    batch_pool: list[tuple[dict[str, Any], JittedObjective]],
    cost_fn_kwargs: dict[str, Any],
    num_iterations: int,
    optimization_schedule: dict[str, OperatorSchedule],
    jitted_objective_full: JittedObjective | None = None,
    init_offset: int = 0,
    noise_threshold: float | None = None,
    relative_precision: float | None = None,
    absolute_tolerance: float | None = None,
    cost_floor: float | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Optimize with a different batch of circuits at every outer iteration.

    This reproduces the batch phase of `mGST.algorithm.run_mGST`: each outer iteration updates
    povm -> kraus -> state on a fresh batch, and the stopping criteria are then evaluated on the
    **full** data set, not on the batch. `run_riemannian_optimization` is therefore called one outer
    iteration at a time with all of its own criteria disabled, since those would compare batch values.
    The trust region radius is not carried over: the objective changes between iterations.

    Args:
        init_operators: Initial operators (kraus, povm, state).
        batch_pool: Pool of (cost function keyword arguments, jitted objective) pairs, see `build_batch_pool`.
            Iteration `idx` uses `batch_pool[(init_offset + idx) % len(batch_pool)]`.
        cost_fn_kwargs: Keyword arguments of the cost function over the full data set.
        num_iterations: Maximum number of outer iterations.
        optimization_schedule: Per-operator schedules, sliced one iteration at a time.
        jitted_objective_full: Jitted objective over the full data, used for the convergence checks.
            Built here if None; pass it in to share one compilation across several calls.
        init_offset: Offset into the pool, so that different initializations see the batches in a different order.
        noise_threshold: See `estimate_noise_floor_threshold`. Disabled if None.
        relative_precision: See `convergence_criteria_from_relative_precision`. Disabled if None.
        absolute_tolerance: See `convergence_criteria_from_absolute_tolerance`. Disabled if None.
        cost_floor: Lower bound of the cost function. Computed from the full data if None.
        verbose: Whether to print the full-data cost after every outer iteration.

    Returns:
        A dictionary with the optimized operators, the cost function history and the convergence reason.
        `cost_fn_history["main"]` holds one full-data value per outer iteration (plus the initial one),
        and `cost_fn_history["batch"]` the per-operator values on the batches.
    """
    if not batch_pool:
        raise ValueError("batch_pool must contain at least one batch.")

    if jitted_objective_full is None:
        jitted_objective_full = JittedObjective.from_cost_fn_kwargs(cost_fn_kwargs)

    if cost_floor is None:
        cost_floor = compute_cost_floor(
            prob_matrix=cost_fn_kwargs["prob_matrix"],
            use_log_likelihood=cost_fn_kwargs.get("use_log_likelihood", False),
            num_shots=cost_fn_kwargs.get("num_shots"),
        )

    cost_on_full_data = lambda ops: float(jitted_objective_full.cost(ops["kraus"], ops["povm"], ops["state"]))

    operators = dict(init_operators)
    cost_fn_history = [cost_on_full_data(operators)]
    cost_fn_batch = []
    cost_decrease_previous = None
    convergence_reason = "Max iterations reached ⏳."

    start_time = time.time()
    for idx in range(num_iterations):
        batch_cost_fn_kwargs, batch_objective = batch_pool[(init_offset + idx) % len(batch_pool)]
        schedule_iteration = {
            operator: OperatorSchedule.default(1, optimization_schedule[operator].get_options_for_iteration(idx))
            for operator in OPERATOR_NAMES
        }

        operators, cost_dict_iteration, _ = run_riemannian_optimization(
            operators["kraus"],
            operators["povm"],
            operators["state"],
            cost_function=cost_function_jax_mps,
            cost_fn_kwargs=batch_cost_fn_kwargs,
            num_iterations=1, # we do one full iteration over all 3 operators for a given batch
            optimization_schedule=schedule_iteration,
            jitted_objective=batch_objective,
            carry_over_radius=False,
            noise_threshold=None,
            relative_precision=None,
            absolute_tolerance=None,
            global_gradient_norm_tol=0.0,
            compute_least_squares=False,
            verbose=False,
        )
        # The first entry is the cost at the point already recorded by the previous iteration.
        cost_fn_batch.extend(cost_dict_iteration["main"][1:])

        cost_fn_previous = cost_fn_history[-1]
        cost_fn_current = cost_on_full_data(operators)
        cost_fn_history.append(cost_fn_current)

        if verbose:
            print(f"🏁 Iteration {idx + 1}/{num_iterations}. f(x) on full data: {cost_fn_current:.6e}.")

        if noise_threshold is not None:
            converged, convergence_reason = convergence_criteria_from_noise_threshold(
                cost_fn_current, noise_threshold, operator="full data")
            if converged:
                break

        # NOTE: For the least squares, I might just want to check for the noise threshold
        # TODO: erase these two actually.
        if relative_precision is not None:
            converged, convergence_reason, _ = convergence_criteria_from_relative_precision(
                cost_fn_previous=cost_fn_previous, cost_fn_current=cost_fn_current,
                relative_precision=relative_precision, cost_floor=cost_floor)
            if converged:
                break

        if absolute_tolerance is not None:
            cost_decrease = cost_fn_previous - cost_fn_current
            converged, convergence_reason, _ = convergence_criteria_from_absolute_tolerance(
                cost_decrease=cost_decrease, cost_decrease_previous=cost_decrease_previous,
                absolute_tolerance=absolute_tolerance)
            if converged:
                break
            cost_decrease_previous = cost_decrease

    return {
        "optimized_operators": operators,
        "init_operators": init_operators,
        "cost_fn_history": {"main": cost_fn_history, "batch": cost_fn_batch},
        "convergence_reason": convergence_reason,
        "times": {"optimization_time": time.time() - start_time},
    }

def _process_batch_size(batch_size: int | float | Sequence[float|int] | None, num_sequences_list: list[int], dim:int) -> int | float | Sequence[float|int]:
    """
    Process the batch size parameter for the random-init phase of the optimization workflow.

    Args:
        batch_size: Can be an int (absolute number of circuits), a float (fraction of total circuits),
            or a list of ints/floats (one per sequence count).

    Returns:
        Processed batch size as an int, float, or sequence of ints/floats.
    """
    if batch_size is None:
        # See CompressiveGST class definition for the default batch size logic.
        batch_size = 30 * dim

    batch_size_is_sequence = isinstance(batch_size, Sequence)
    if batch_size_is_sequence:
        if len(batch_size) != len(num_sequences_list):
            raise ValueError(
                f"When batch_size is a list, it must have the same length as num_sequences_list. "
                f"Got len(batch_size)={len(batch_size)} and len(num_sequences_list)={len(num_sequences_list)}."
            )
            
    return batch_size, batch_size_is_sequence

def run_optimization_workflow_multi_init_realizations(
    num_shots: int,
    operators_psd_true: dict[str, jnp.ndarray],
    target_superops: dict[str, jnp.ndarray],
    optimization_options: dict[str, Any],
    num_sequences_list: list[int],
    num_realizations: int,
    seq_len_list: list[int] | None = None,
    num_max_iterations_random_inits: int = 5,
    num_max_random_inits: int = 10,
    batch_size: int | float | list[int] = 0.1,
    threshold_multiplier: float = 5.0,
    seed: int | None = None,
    debug: bool = False,
    gauge_optimize: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Run multiple independent realizations of the multi-init optimization workflow across
    a list of sequence counts, generating fresh indices and probability matrices each time.

    Args:
        num_shots: Number of shots for sampling.
        operators_psd_true: Dictionary containing the true operators (kraus, povm, state).
        target_superops: Dictionary containing the target superoperators (kraus, povm, state).
        optimization_options: Optimization options dictionary. Must include a 'schedule' whose
            OperatorSchedule covers at least num_max_iterations_random_inits iterations.
        gauge_optimize: Whether to perform gauge optimization after the final log-likelihood evaluation.
        num_sequences_list: List of sequence counts to run per realization.
        num_realizations: Number of independent realizations to run.
        seq_len_list: Sequence length [L_MIN, L_CUT, L_MAX]. Defaults to [1, 8, 14].
        num_max_iterations_random_inits: Max outer iterations for each random-init run.
        num_max_random_inits: Max number of random initializations to try.
        batch_size: Batch size (absolute int or fraction) for the random-init phase.
        threshold_multiplier: Multiplier for the noise floor threshold estimate.
        seed: Base random seed. If None, seeds are drawn randomly per call. If an int,
            seeds are derived deterministically as seed + realization * len(num_sequences_list) + seq_idx.
        debug: If True, also return partial (random-init phase) results in the same dict format.

    Returns:
        A tuple (final_results, partial_results) where:
        - final_results: list of length num_realizations, each a dict mapping str(num_seq) -> final_result.
        - partial_results: same structure for the random-init phase result, or None when debug=False.
    """
    num_gates, kraus_rank, dim, _ = operators_psd_true["kraus"].shape
    _seq_len_list = seq_len_list if seq_len_list is not None else [1, 8, 14]

    final_results_all: list[dict[str, Any]] = []
    partial_results_all: list[dict[str, Any]] | None = [] if debug else None

    batch_size, batch_size_is_sequence = _process_batch_size(batch_size, num_sequences_list, dim=dim)

    for realization_idx in range(num_realizations):
        print(f"🔁 Realization {realization_idx + 1}/{num_realizations}")

        final_results_realization: dict[str, Any] = {}
        partial_results_realization: dict[str, Any] = {}

        for seq_idx, num_seq in enumerate(num_sequences_list):
            print(f"  📊 num_sequences={num_seq}")

            batch_size_idx = batch_size[seq_idx] if batch_size_is_sequence else batch_size

            # Derive a reproducible seed per (realization, sequence-count) pair when requested.
            if seed is not None:
                prob_seed = seed + realization_idx * len(num_sequences_list) + seq_idx
            else:
                prob_seed = int(np.random.default_rng().integers(0, 2**31))

            indices_dict_r = generate_sequence_indices(
                num_gates=num_gates,
                num_circuits=num_seq,
                seq_len_list=_seq_len_list,
            )
            probability_matrices_r = compute_probability_matrices(
                *operators_psd_true.values(),
                gate_indices=indices_dict_r["gate_indices"],
                num_shots=num_shots,
                seed=prob_seed,
            )

            final_result, partial_result = run_optimization_workflow_multi_init(
                num_shots=num_shots,
                operators_psd_true=operators_psd_true,
                target_superops=target_superops,
                optimization_options=optimization_options,
                indices_dict=indices_dict_r,
                probability_matrices=probability_matrices_r,
                num_max_iterations_random_inits=num_max_iterations_random_inits,
                num_max_random_inits=num_max_random_inits,
                batch_size=batch_size_idx,
                threshold_multiplier=threshold_multiplier,
                seed=prob_seed,
                gauge_optimize=gauge_optimize,
            )

            final_results_realization[str(num_seq)] = final_result
            if debug:
                partial_results_realization[str(num_seq)] = partial_result

        final_results_all.append(final_results_realization)
        if debug:
            partial_results_all.append(partial_results_realization)

    return final_results_all, partial_results_all