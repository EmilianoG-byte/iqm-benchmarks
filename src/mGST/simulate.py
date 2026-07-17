"""Functions used to simulate samples for testing purposes"""
import time

from mGST import additional_fns
from mGST.low_level_jit import contract_mps_all_povm
from mGST.utility_functions_comparisons import kraus_tensor_to_mgst, factorize_psd_truncated, get_mgst_tensors_from_psd_representation, generate_target_state, generate_target_povm

from mGST.typing import Tensor, Matrix, TrustRegionOptions, Vector
import jax.numpy as jnp
import numpy as np

from mGST.typing import OperatorSchedule, TrustRegionOptions
from mGST.trust_region import run_riemannian_optimization
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
    kraus_perturbed_superop = kraus_tensor_to_mgst(kraus_perturbed) # num_gates, dim_out^2, dim_in^2*
    # squeeze the num_gates out
    kraus_perturbed_superop = kraus_perturbed_superop.squeeze(axis=0) # dim_out^2, dim_in^2*
    return apply_single_superop_to_povm_tensor(povm_tensor, kraus_perturbed_superop)

def apply_single_superop_to_povm_tensor(povm_tensor: Tensor, superop: Matrix | Tensor) -> Tensor:
    """Apply a single superoperator to a POVM tensor.

    Args:
        povm_tensor: The original POVM tensor of shape (num_povm, dim_out, dim_out*).
        superop: The superoperator to apply of shape (dim_out*dim_out, dim_out*dim_out).

    Returns:
        The transformed POVM tensor of shape (num_povm, dim_out, dim_out*).
    """
    if superop.ndim == 3:
        superop = superop.squeeze(axis=0) # dim^2, dim^2
    num_povm, dim, _ = povm_tensor.shape
    povm_vect = povm_tensor.reshape((num_povm, dim * dim))  # num_povm, dim_out * dim_out
    povm_vect = jnp.einsum('ij, jk -> ik', povm_vect, superop)  # num_povm, dim_out * dim_out
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
        List of Kraus operators as JAX arrays
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
        
        return [K0, K1, K2, K3]
    
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
        List of Kraus operators as JAX arrays
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

def get_default_optimization_options() -> tuple[dict[str, OperatorSchedule], int, float]:
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

    return optimization_schedule_all_tr, num_iterations_outer, relative_precision, global_gradient_norm_tol


def check_operators_dictionaries(*dicts)-> None:
    """
    Check if all dictionaries have valid keys.
    """
    EXPECTED_KEYS = {"kraus", "povm", "state"}

    for dict in dicts:
        if set(dict.keys()) != EXPECTED_KEYS:
            raise ValueError(f"Dictionary keys {dict.keys()} do not match expected keys {EXPECTED_KEYS}.")


def run_optimization_workflow(num_sequences:int, num_shots:int, init_operators:dict[str, jnp.ndarray], operators_psd_true:dict[str, jnp.ndarray], target_superops:dict[str, jnp.ndarray], use_exact_probabilities:bool=False, optimization_options:dict[str, Any]=None, warmup_run:bool=False, use_log_likelihood:bool=False, optimization_verbose:bool=True) -> dict[str, Any]:
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

    Returns:
        A dictionary containing the optimized operators, gauged superoperators, target model, cost function history, probability matrices, times for optimization and gauging, and indices dictionary.
    """
    check_operators_dictionaries(operators_psd_true, target_superops, init_operators)    
    
    seq_len_list = [1, 8, 14]
    kraus_tensor_true = operators_psd_true["kraus"]
    povm_psd_true = operators_psd_true["povm"]
    state_psd_true = operators_psd_true["state"]
    
    # we can get the number of gates from any of the kraus tensors
    num_gates = kraus_tensor_true.shape[0]
    
    # return this one
    indices_dict = generate_sequence_indices(num_gates=num_gates, num_circuits=num_sequences, seq_len_list=seq_len_list)
    indices_list = indices_dict["gate_indices"]

    # return this one
    probability_matrices = compute_probability_matrices(kraus_tensor=kraus_tensor_true, povm_psd=povm_psd_true, state_psd=state_psd_true, gate_indices=indices_list, num_shots=num_shots, seed=42)
    
    if use_exact_probabilities:
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
    }
    
    if use_log_likelihood:
        print("Using log-likelihood for the cost function optimization 🧮")
        cost_fn_kwargs["num_shots"] = num_shots
    
    if optimization_options is None:
        optimization_schedule_all_tr, num_iterations_outer, relative_precision, global_gradient_norm_tol = get_default_optimization_options()
    else:
        if ["schedule", "num_iterations_outer", "relative_precision", "global_gradient_norm_tol"] != list(optimization_options.keys()):
            raise ValueError(f"Optimization options must contain keys: ['schedule', 'num_iterations_outer', 'relative_precision', 'global_gradient_norm_tol'], but got {list(optimization_options.keys())}.")
        optimization_schedule_all_tr = optimization_options["schedule"]
        num_iterations_outer = optimization_options["num_iterations_outer"]
        relative_precision = optimization_options["relative_precision"]
        global_gradient_norm_tol = optimization_options["global_gradient_norm_tol"]
    
    if warmup_run:
        start_compilation_time = time.time()
        print("🏎️ Warming up the cost function ...")
        initial_cost_value = cost_function_jax_mps(*init_operators.values(), **cost_fn_kwargs)
        print(f"Initial cost value: {initial_cost_value}")
    
    # start timer
    start_time = time.time()
    
    # return these ones
    optimized_operators, cost_fn_history = run_riemannian_optimization(
        *init_operators.values(),
        cost_function=cost_function_jax_mps,
        cost_fn_kwargs=cost_fn_kwargs,
        num_iterations=num_iterations_outer,
        optimization_schedule=optimization_schedule_all_tr,
        save_intermediate_cost_values=True,
        noise_threshold=None,
        relative_precision=relative_precision,
        global_gradient_norm_tol=global_gradient_norm_tol,
        verbose=optimization_verbose,
    )
    
    time_after_opt = time.time()
    optimization_time = time_after_opt - start_time
    
    kraus_superop_gauged, povm_vect_gauged, state_vect_gauged, target_model_pygsti = perform_gauge(optimized_operators=optimized_operators, target_superops=target_superops, num_gates=num_gates)
    
    time_after_gauge = time.time()
    gauge_time = time_after_gauge - time_after_opt
    
    superops_gauged = {
        "kraus": kraus_superop_gauged,
        "povm": povm_vect_gauged,
        "state": state_vect_gauged
    }
    
    times = {"optimization_time": optimization_time, "gauge_time": gauge_time}
    
    if warmup_run:
        compilation_time = start_time - start_compilation_time
        times["compilation_time"] = compilation_time
    
    return {"optimized_operators": optimized_operators, "superops_gauged": superops_gauged, "target_model_pygsti": target_model_pygsti, "cost_fn_history": cost_fn_history, "probability_matrices": probability_matrices, "indices_dict": indices_dict, "times": times}
    
def perform_gauge(optimized_operators:dict[jnp.ndarray], target_superops:dict[jnp.ndarray], num_gates:int):
    """
    Perform gauge optimization on the optimized operators.
    """
    optimized_superops = get_mgst_tensors_from_psd_representation(*optimized_operators.values())
    
    gauge_weights = dict({f"G%i" % i: 1 for i in range(num_gates)}, **{"spam": 0.1})

    target_model_pygsti = arrays_to_pygsti_model(*target_superops.values(), basis="std")

    kraus_superop_gauged, povm_vect_gauged, state_vect_gauged = gauge_opt(X=optimized_superops["kraus"], E=optimized_superops["povm"], rho=optimized_superops["state"], target_mdl=target_model_pygsti, weights=gauge_weights)
    
    return kraus_superop_gauged, povm_vect_gauged, state_vect_gauged, target_model_pygsti