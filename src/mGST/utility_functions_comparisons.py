# Needs iqm-benchmarks from the github repo to access all the mGST functions: https://github.com/iqm-finland/iqm-benchmarks
from cvxpy import PSD
from mGST import additional_fns
from iqm.benchmarks.compressive_gst.compressive_gst import GSTConfiguration, CompressiveGST
from iqm.benchmarks.compressive_gst.gst_analysis import dataset_counts_to_mgst_format

from mGST.qiskit_interface import qiskit_gate_to_operator
from mGST.low_level_jit import (
    gradient_all_3_and_value_jit,
    cost_function_jax_mps,
    gradient_povm_mps_jit,
    gradient_k_mps_jit,
    gradient_state_mps_jit,
    cost_function_jax_mps_regularized,
    gradient_povm_mps_jit_reg,
    gradient_k_mps_jit_reg,
    gradient_state_mps_jit_reg)

from mGST.typing import Tensor, Matrix, Scalar

from mGST.algorithm import B_SFN_riem_Hess, A_SFN_riem_Hess, SFN_riem_Hess_full
from mGST.additional_fns import random_gs, perturbed_target_init

from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from qiskit.circuit.library import CZGate, RGate

from scipy.optimize import minimize

from typing_extensions import Literal

import jax.numpy as jnp
import numpy as np
import jax

from typing import Sequence

backend = "iqmfakeapollo"

def get_mgst_parameters_from_dataset(dataset, qubit_layout, rK):
    y = dataset_counts_to_mgst_format(dataset, qubit_layout)
    J = dataset.attrs["J"]
    l = dataset.attrs["seq_len_list"][-1]
    d = dataset.attrs["num_gates"]
    pdim = dataset.attrs["pdim"]
    r = pdim ** 2
    n_povm = dataset.attrs["num_povm"]
    bsize = dataset.attrs["batch_size"]
    meas_samples = dataset.attrs["shots"]
    # Setting some additional matrix shape parameters for the first and second derivatives
    n = rK * pdim
    nt = rK * r
    return y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt


## Preparing an initialization (random gate set or target gate set)

def initialize_mgst_parameters(dataset, target_init = True, seed:int = 42):
    d = dataset.attrs["num_gates"]
    pdim = dataset.attrs["pdim"]
    r = pdim ** 2
    n_povm = dataset.attrs["num_povm"]
    rK = dataset.attrs["rank"]
    
    if target_init:
        K_target = qiskit_gate_to_operator(dataset.attrs["gate_set"])
        X_target = jnp.einsum("ijkl,ijnm -> iknlm", K_target, K_target.conj()).reshape(
            (dataset.attrs["num_gates"], dataset.attrs["pdim"] ** 2, dataset.attrs["pdim"] ** 2)
        )  # tensor of superoperators
        
        rho = (
            jnp.kron(additional_fns.basis(dataset.attrs["pdim"], 0).T.conj(), additional_fns.basis(dataset.attrs["pdim"], 0))
            .reshape(-1)
            .astype(jnp.complex128)
        )
        
        # Computational basis measurement:
        E = jnp.array(
            [
                jnp.kron(
                    additional_fns.basis(dataset.attrs["pdim"], i).T.conj(), additional_fns.basis(dataset.attrs["pdim"], i)
                ).reshape(-1)
                for i in range(dataset.attrs["pdim"])
            ]
        ).astype(jnp.complex128)
        
        
        K = additional_fns.perturbed_target_init(X_target, dataset.attrs["rank"], seed=seed)
        X = jnp.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    else:
        K, X, E, rho = random_gs(d, r, rK, n_povm)
        
    return K, X, E, rho

def get_full_mgst_parameters_from_configuration(configuration:GSTConfiguration, backend, seed:int = 42, only_jax_variables:bool = False):
    """
    Get the full set of parameters required to run mGST from a given configuration.
    
    Note: when using `only_jax_variables=True`, the returned indices J will be a list of arrays where each array contains only the valid indices (i.e., indices that are not -1).
    """
    benchmark = CompressiveGST(backend, configuration)
    result = benchmark.run()
    
    rK = configuration.rank
    qubit_layout = configuration.qubit_layouts[0]
    dataset = result.dataset
    y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt = get_mgst_parameters_from_dataset(dataset, qubit_layout=qubit_layout, rK=rK)
    K, X, E, rho = initialize_mgst_parameters(dataset=dataset, target_init=True, seed=seed)
    
    if only_jax_variables:
        indices_list = [indices[indices != -1] for indices in J]
        return K, X, E, rho, y, indices_list
    return K, X, E, rho, y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt, rK

def create_4q_gst_config(kraus_rank:int=1, max_gates_per_batch:int | None = None):
    """Create the configuration to run a 4 qubit Gate set tomography protocol.

    NOTE: Garnet allows for a maximum of 500 circuits per job. Under this configuration, setting 500 gates per batch will create 87 jobs of each 23 circuits (except the last one with 22 circuits).

    Args:
         kraus_rank: Rank of the Kraus operators in the compressed representation. Defaults to 1.
         max_gates_per_batch: Maximum number of gates per batch to be sent to the backend. If None, no limit is set. Defaults to None.

    Returns:
       The configuration used for 4Q GST
    """

    cz_cz = QuantumCircuit(4)
    cz_cz.append(CZGate(), [0,1])
    cz_cz.append(CZGate(), [2,3])

    gate_list = [
        RGate(0.5 * jnp.pi, 0),
        RGate(0.5 * jnp.pi, 0),
        RGate(0.5 * jnp.pi, 0),
        RGate(0.5 * jnp.pi, 0),
        RGate(0.5 * jnp.pi, jnp.pi / 2),
        RGate(0.5 * jnp.pi, jnp.pi / 2),
        RGate(0.5 * jnp.pi, jnp.pi / 2),
        RGate(0.5 * jnp.pi, jnp.pi / 2),
        cz_cz,
    ]
    gates = [QuantumCircuit(4, 0) for _ in range(len(gate_list))]
    gate_qubits = [[0], [1], [2], [3], [0], [1], [2], [3], [0, 1, 2, 3]]
    for i, gate in enumerate(gate_list):
        if isinstance(gate, QuantumCircuit):
            gates[i].compose(gate, gate_qubits[i], inplace = True)
        else:
            gates[i].append(gate, gate_qubits[i])
            
    gate_labels = ["Rx(pi/2)", "Rx(pi/2)", "Rx(pi/2)", "Rx(pi/2)", 
                "Ry(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "Ry(pi/2)", 
                "CZ-CZ"]

    Q4_GST = GSTConfiguration(
        qubit_layouts=[[0,1,3,4]],
        gate_set=gates,
        gate_labels=gate_labels,
        num_circuits=2000,
        shots=1000,
        rank=kraus_rank,
        max_gates_per_batch=max_gates_per_batch,
    )

    return Q4_GST

def create_2q_emerald_gst_config():
    qubit_layouts = [[2,8], [4,10], [6,12], [15,23], [19,25], [21,27], [31,39], [30,35], [42,46], [44,48]]

    gates = [QuantumCircuit(2, 0) for _ in range(7)]
    gates[0].append(RGate(1e-9, 0), [0])
    gates[1].append(RGate(0.5 * jnp.pi, 0), [0])
    gates[2].append(RGate(0.5 * jnp.pi, 0), [1])
    gates[3].append(RGate(0.5 * jnp.pi, jnp.pi / 2), [0])
    gates[4].append(RGate(0.5 * jnp.pi, jnp.pi / 2), [1])
    gates[5].append(RGate(0.5 * jnp.pi, 0), [0])
    gates[5].append(RGate(0.5 * jnp.pi, 0), [1])
    gates[6].append(CZGate(), [[0], [1]])
    gate_labels = [
        "Idle",
        "Rx(pi/2):0",
        "Rx(pi/2):1",
        "Ry(pi/2):0",
        "Ry(pi/2):1",
        "Rx(pi/2)-Rx(pi/2):0-1",
        "CZ",
    ]
    
    Q2_GST_EMERALD = GSTConfiguration(
        qubit_layouts=qubit_layouts,
        gate_set=gates,
        gate_labels = gate_labels,
        num_circuits=1000,
        shots=1000,
        rank=16,
        opt_method="GD",
        max_iterations = [140, 250],
        convergence_criteria=[4, 1e-4],
        parallel_execution = True,
        max_gates_per_batch = 35000
    )

    return Q2_GST_EMERALD

def get_x_from_k(k, depth=None, dim_squared=None):
    """Get the superoperator representation from the Kraus operators.
    
    DEPRECATED: use kraus_tensor_to_mgst instead.
    """
    if not depth or not dim_squared:
        depth = k.shape[0]
        dim_squared = k.shape[-1]**2
    return jnp.einsum("ijkl,ijnm -> iknlm", k, k.conj()).reshape((depth, dim_squared, dim_squared))

def get_compressed_rep_from_mgst_output(kraus_mgst, povm_mgst, state_mgst, kraus_rank:int = 1, state_rank:int = 1, povm_rank:int = 1)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get the compressed representation of the MGST operators.
    
    Args:
        kraus_mgst: Kraus operators from MGST. Dimensions: (num_gates, dim_out x dim_out*, dim_in x dim_in*)
        povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
        state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)
        kraus_rank: Rank of the Kraus operators in the compressed representation. Defaults to 1.
        state_rank: Rank of the state operator in the compressed representation. Defaults to 1.
        povm_rank: Rank of the POVM operators in the compressed representation. Defaults to 1.
    
    Returns:
        A tuple containing the compressed representation of the Kraus, POVM, and State.
            * kraus_tensor: dimensions (num_gates, kraus_rank, dim_out, dim_in)
            * povm_psd: dimensions (num_povm, povm_rank, dim_in)
            * state_psd: dimensions (dim_out, state_rank)
            
    Note:
        To compare the resulting POVM with the one used in the mGST code, we have to take into account that povm_jax = povm_mgst *. Which means, also the factorizations follow this relation: povm_psd_jax = povm_psd_mgst *. Therefore, in order to compute if the resulting POVM is the same we should do:
        >>> povm_jax = povm_psd.transpose(0, 2, 1).conj() @ povm_psd
        >>> jnp.allclose(povm_jax.conj(), povm_mgst)
    """
    # POVM
    num_povm, dim_squared = povm_mgst.shape
    dim = int(jnp.sqrt(dim_squared))
    povm_tensor = jnp.reshape(povm_mgst, shape=(num_povm, dim, dim)) # num_povm, dim_in, dim_in*
    povm_psd  = factorize_psd_truncated(psd=povm_tensor, max_rank=povm_rank).transpose(0, 2, 1).conj() # num_povm, rank_povm, dim_in
    # STATE
    state = jnp.reshape(state_mgst, shape=(dim, dim)) # dim_out, dim_out*
    state_psd = factorize_psd_truncated(psd=state, max_rank=state_rank) # dim_out, rank_state
    # KRAUS
    kraus_tensor = get_kraus_psd_from_mgst(kraus_mgst, kraus_rank)
    return kraus_tensor, povm_psd, state_psd

def povm_psd_to_mgst(povm_psd:jnp.ndarray)->jnp.ndarray:
    """Convert the POVM from its PSD representation to the MGST representation.
    
    Args:
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
    Returns:
        povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
    """
    num_povm, rank_povm, dim_in = povm_psd.shape
    return (povm_psd.conj().transpose(0, 2, 1) @ povm_psd).reshape(num_povm, dim_in**2) # (num_povm, dim_out, dim_out*)

def state_psd_to_mgst(state_psd:jnp.ndarray)->jnp.ndarray:
    """Convert the State from its PSD representation to the MGST representation.
    
    Args:
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
    Returns:
        state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)
    """
    return (state_psd @ state_psd.conj().T).reshape(-1) # (dim_in * dim_in*)

def kraus_tensor_to_mgst(kraus_tensor:jnp.ndarray)->jnp.ndarray:
    """Convert the Kraus operators from their PSD representation to the MGST representation.
    
    Args:
        kraus_tensor: Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
    Returns:
        kraus_mgst: Kraus operators from MGST. Dimensions: (num_gates, dim_out x dim_out*, dim_in x dim_in*)
    """
    num_gates, kraus_rank, dim_out, dim_in = kraus_tensor.shape
    return jnp.einsum("ijkl,ijnm -> iknlm", kraus_tensor, kraus_tensor.conj()).reshape((num_gates, dim_in**2, dim_in**2))

def get_mgst_tensors_from_psd_representation(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the MGST representation of the operators from their PSD representation.
    
    Args:
        kraus_tensor: Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
    Returns:
        A tuple containing the MGST representation of the Kraus, POVM, and State.
            * kraus_mgst: Kraus operators from MGST. Dimensions: (num_gates, dim_out x dim_out*, dim_in x dim_in*)
            * povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
            * state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)    
    """
    # POVM
    povm_mgst = povm_psd_to_mgst(povm_psd) # (num_povm, dim_out, dim_out*)
    
    # Kraus
    kraus_superop_mgst = kraus_tensor_to_mgst(kraus_tensor) # (num_gates, dim_out x dim_out*, dim_in x dim_in*)

    # State
    state_vect_mgst = state_psd_to_mgst(state_psd) # (dim_in * dim_in*)

    return {"kraus": np.array(kraus_superop_mgst), "povm": np.array(povm_mgst), "state": np.array(state_vect_mgst)}

def get_kraus_psd_from_mgst(kraus_mgst, rank:int)->jnp.ndarray:
    """ 
    Get the PSD representation of the Kraus operators from the MGST output.
    
    Args:
        kraus_mgst: Kraus operators from MGST. Dimensions: (num_gates, dim_out x dim_out*, dim_in x dim_in*)
        rank: Rank of the Kraus operators in the compressed representation.
    Returns:
        kraus_tensor: dimensions (num_gates, kraus_rank, dim_out, dim_in)
    """
    dim_squared = kraus_mgst.shape[-1]
    dim = int(jnp.sqrt(dim_squared))
    kraus_mgst_trans = kraus_mgst.transpose(0, 2, 1) # num_gates, dim_in x dim_in*, dim_out x dim_out*
    num_gates, *_ = kraus_mgst_trans.shape
    choi_kraus = superop2choi(kraus_mgst_trans) # num_gates, dim_in x dim_out, dim_in* x dim_out*
    choi_psd = factorize_psd_truncated(choi_kraus, max_rank=rank) # num_gates, dim_in x dim_out, rank_kraus
    kraus_tensor = jnp.reshape(choi_psd, shape=(num_gates, dim, dim, rank)) # num_gates, dim_in, dim_out, rank_kraus
    kraus_tensor = jnp.transpose(kraus_tensor, (0, 3, 2, 1)) # num_gates, rank_kraus, dim_out, dim_in
    return kraus_tensor

def get_compressed_perturbed_rep_from_mgst(povm_mgst, state_mgst)->tuple[jnp.ndarray, jnp.ndarray]:
    """Get the compressed representation of the MGST operators using cholesky factorization.
    
    This is the implementation used in the original mGST code.
    
    Args:
        povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
        state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)
    Returns:
        A tuple containing the compressed representation of the POVM and State as JAX arrays.
    """
    num_povm, dim_sqrd = povm_mgst.shape
    dim = int(jnp.sqrt(dim_sqrd))
    povm_psd = jnp.array([jnp.linalg.cholesky(povm_mgst[k].reshape(dim, dim) + 1e-14 * jnp.eye(dim)).T.conj() for k in range(num_povm)])
    state_mgst_offset = state_mgst + 1e-14 * jnp.eye(dim).reshape(-1)
    state_psd = jnp.linalg.cholesky(state_mgst_offset.reshape(dim, dim))
    return povm_psd, state_psd

def get_compressed_perturbed_rep_from_mgst_numpy(povm_mgst, state_mgst)->tuple[np.ndarray, np.ndarray]:
    """Get the compressed representation of the MGST operators using cholesky factorization.
    
    This is implementation usedin the original mGST code.
    
    Args:
        povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
        state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)
    Returns:
        A tuple containing the compressed representation of the POVM and State.
    """
    num_povm, dim_sqrd =povm_mgst.shape
    dim = int(np.sqrt(dim_sqrd))
    povm_psd = np.array([np.linalg.cholesky(povm_mgst[k].reshape(dim, dim) + 1e-14 * np.eye(dim)).T.conj() for k in range(num_povm)])
    state_mgst_offset = state_mgst + 1e-14 * np.eye(dim).reshape(-1)
    state_psd = np.linalg.cholesky(state_mgst_offset.reshape(dim, dim))
    return povm_psd, state_psd

def superop2choi(superop:jnp.ndarray)->jnp.ndarray:
    """
    Convert a superoperator into its choi matrix representation.

    Args:
        superop: Superoperator representation of the quantum
            channel with dimensions (..., dim x dim, dim x dim)
            This assumes an order: (dim_in x dim_in*, dim_out x dim_out*)
        
    Returns:
        Choi Matrix representation of the quantum channel with dimensions (..., dim^2, dim^2): (..., dim_in x dim_out, dim_in* x dim_out*)
    """
    *batch_shape, m, n = superop.shape  # Extract batch dimensions
    if m != n:
        raise ValueError(f"Input must be square in the last two dimensions. Instead got dimensions {m} and {n}")
    
    dim = int(jnp.sqrt(m))
    if dim * dim != m:
        raise ValueError(f"Invalid input size: {m}. Expected dim^2 for some integer dim.")
            
    superop_shape = tuple(batch_shape) + (dim,)*4
        
    superop_tensor = superop.reshape(superop_shape)  # (..., dim_in, dim_in*, dim_out, dim_out*)
    original_shape = superop.shape
    return superop_tensor.swapaxes(-2, -3).reshape(original_shape) # (..., dim_in x dim_out, dim_in* x dim_out*)

def run_gds_jax(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray, max_iter:int=200, target_rel_prec=1e-3, step_size:float=1, optimize_step:bool=True, ls_max_iter:int = 20, use_geodesic:bool=True, dmrg_like:bool=True, use_hessian:bool=False, regularized:bool=False, target_operators:Sequence[jnp.ndarray]= None, num_samples:int = None, return_operators_list:bool = False, **hessian_kwargs)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[float]]:
    """Run a simple gradient descent optimization on the gates using JAX.

    Args:
        kraus0: Kraus tensor to start the optimization from. Dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        max_iter: Max number of iterations to run GDS for. Defaults to 200.
        target_rel_prec: Relative precision used to decide whether to terminate optimization early. Defaults to 1e-3.
        step_size: Step size used throughought the optimization. Use only if line search is not desired. Defaults to 1.
        optimize_step: Whether to optimize the step_size using line search or not. Defaults to True.
        ls_max_iter: Max number of iterations used in line search. Defaults to 20.
        use_geodesic: Whether to use the geodesic to compute to the updated tensors after following the gradient direction. Defaults to True. If False, use the polar decomposition.
        dmrg_like: Whether to use a DMRG-like optimization for the step size (alternating). Defaults to False for backwards compatibility.
        use_hessian: Whether to use a second order method to udpate the operators at each step. Defaults to False, so we use a first order method.
        regularized: Whether to use a regularized cost function. Defaults to False.
        target_operators: Target operators to use in the regularized cost function. Defaults to None.
        num_samples: Number of samples to use in the regularized cost function. Defaults to None.
        return_operators_list: Whether to return the list of operators at each step. Defaults to False.

    Returns:
        krausi: Optimized Kraus tensor.
        povm_i: Optimized POVM tensor.
        state_i: Optimized state tensor.
        cost_function_history: History of the cost function values at each iteration.
    """
    
   
    cost_function_history = [] # the cost function will be evaluated in the first step
    
    if return_operators_list:
        kraus_list = [kraus_tensor]
        povm_list = [povm_psd]
        state_list = [state_psd]
    
    kraus_i = kraus_tensor
    state_i = state_psd
    povm_i = povm_psd
    opt_step_size = jnp.array([step_size, step_size, step_size])
    
    try:
        for i in range(max_iter):
            print('iteration: ', i)
            if use_hessian:
                kraus_i, povm_i, state_i, opt_step_size, cost_i = optimization_step_hessian(kraus_i, povm_i, state_i, indices_list, prob_matrix, initial_step_size=opt_step_size, ls_max_iter=ls_max_iter, **hessian_kwargs)
            else:
                kraus_i, povm_i, state_i, opt_step_size, cost_i = gradient_descent_step(kraus_i, povm_i, state_i, indices_list, prob_matrix, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step_size=opt_step_size, use_geodesic=use_geodesic, dmrg_like=dmrg_like, regularized=regularized, target_operators=target_operators, num_samples=num_samples)
                
            cost_function_history.append(cost_i)
            print('cost: ', cost_function_history[-1])
            
            if return_operators_list:
                kraus_list.append(kraus_i)
                povm_list.append(povm_i)
                state_list.append(state_i)
            
            if i > 1 and jnp.abs(cost_function_history[-2] - cost_function_history[-1])/cost_function_history[-2] <   target_rel_prec:
                print('Success threshold reached prematurely.')
                break
    except KeyboardInterrupt:
        print(f"Optimization was stopped prematurely at iteration: {i}")
        
        if return_operators_list:
            kraus_i = kraus_list
            povm_i = povm_list
            state_i = state_list
            
        return kraus_i, povm_i, state_i, cost_function_history
    
    if return_operators_list:
            kraus_i = kraus_list
            povm_i = povm_list
            state_i = state_list
    
    return kraus_i, povm_i, state_i, cost_function_history

def update_state_via_saddle_free_newton(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray)->jnp.ndarray:
    """Update the state tensor using the saddle-free newton method.
    
    Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
    Returns:
        The updated state tensor.
    """
    num_gates, _, dim, dim = kraus_tensor.shape
    num_povm = povm_psd.shape[0]
    
    return B_SFN_riem_Hess(
        K=kraus_tensor,
        A=povm_psd,
        B=state_psd,
        y=prob_matrix,
        J=indices_list,
        d=num_gates,
        r=dim**2,
        n_povm=num_povm,
        lam=1e-3)
    
def update_kraus_via_saddle_free_newton(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray)->jnp.ndarray:
    """Update the Kraus tensor using the saddle-free newton method.
    
    Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
    Returns:
        The updated Kraus tensor.
    """
    num_gates, kraus_rank, dim, dim = kraus_tensor.shape
    num_povm = povm_psd.shape[0]
    
    rho_mgst = (state_psd @ state_psd.T.conj()).reshape(-1)
    povm_mgst = (povm_psd.conj().transpose(0, 2, 1) @ povm_psd).reshape(num_povm, -1)
    
    updated_kraus_np = SFN_riem_Hess_full(
        K=np.array(kraus_tensor),
        E=np.array(povm_mgst),
        rho=np.array(rho_mgst),
        y=prob_matrix,
        J=indices_list,
        d=num_gates,
        r=dim**2,
        rK=kraus_rank,
        lam=1e-3,
        ls="COBYLA",
    )
    return jnp.array(updated_kraus_np)
    
def update_povm_via_saddle_free_newton(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray)->jnp.ndarray:
    """Update the POVM tensor using the saddle-free newton method.
    
    Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
    Returns:
        The updated POVM tensor.
    """
    num_gates, _, dim, dim = kraus_tensor.shape
    num_povm = povm_psd.shape[0]
    
    updated_povm_np = A_SFN_riem_Hess(
        K=kraus_tensor,
        A=np.array(povm_psd),
        B=np.array(state_psd),
        y=prob_matrix,
        J=indices_list,
        d=num_gates,
        r=dim**2,
        n_povm=num_povm,
        lam=1e-3
    )
    return jnp.array(updated_povm_np)
    


def optimization_step_hessian(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray, initial_step_size:jnp.ndarray, ls_method:str="COBYLA", ls_max_iter:int=200, which_hessian:list = None, metric:Literal["canonical", "euclidean"]="canonical")->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """Perform a Hessian optimization step on the Kraus, POVM and State operators
    
    Currently, it uses
    
     Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        initial_step_size: Initial step sizes for the update of the gradient descent step. It is expected to be [step_size_kraus, step_size_povm, step_size_state].
        ls_method: Method to use in line search optimization. Defaults to "COBYLA".
        ls_max_iter: Max number of iterations used in line search. Defaults to 200.
    Returns:
        A tuple containing:
            The updated Kraus tensor
            The updated POVM tensor
            The updated state tensor
            The optimized step size array in the order [step_size_kraus, step_size_povm, step_size_state].
            The cost function evaluated at the intial kraus, povm and state tensors.
    """
    if which_hessian is None:
        which_hessian = ["povm", "state"]
        
    # Get the individual step sizes
    initial_step_kraus, initial_step_povm, initial_step_state = initial_step_size
    # Initial cost value
    initial_cost_value = cost_function_jax_mps(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, jit=True)
    
    if "povm" in which_hessian:
        # First sweep over the POVM tensor        
        new_povm_psd = update_povm_via_saddle_free_newton(
            kraus_tensor=kraus_tensor,
            povm_psd=povm_psd,
            state_psd=state_psd,
            indices_list=indices_list,
            prob_matrix=prob_matrix,
        )
    else:
        new_povm_psd, optimized_step_povm = _update_tensor_via_gradient("povm", kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, initial_step=initial_step_povm, metric=metric)        
    
    # Then over the kraus tensor
    new_kraus_tensor, optimized_step_kraus = _update_tensor_via_gradient("kraus", kraus_tensor, new_povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, initial_step=initial_step_kraus, metric=metric)
        
    # And finally we optimize the state tensor
    if "state" in which_hessian:
        new_state_psd = update_state_via_saddle_free_newton(
            kraus_tensor=new_kraus_tensor,
            povm_psd=new_povm_psd,
            state_psd=state_psd,
            prob_matrix=prob_matrix,
            indices_list=indices_list,
            )
    else:
        new_state_psd, optimized_step_state = _update_tensor_via_gradient("state", new_kraus_tensor, new_povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, initial_step=initial_step_state, metric=metric)
        
        
    # optimized_step = initial_step_size.at[0].set(optimized_step_kraus[0])
    optimized_step = initial_step_size
    
    return new_kraus_tensor, new_povm_psd, new_state_psd, optimized_step, initial_cost_value 
    
def gradient_descent_step(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray, ls_method:str="COBYLA", ls_max_iter:int=200, optimize_step:bool=True, dmrg_like:bool = False, initial_step_size:float|jnp.ndarray=1, use_geodesic:bool=True, regularized:bool=False, target_operators:Sequence[jnp.ndarray]= None, num_samples:int = None)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """Perform a gradient descent step on all operators (Kraus, POVM and State)

    Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, rank_povm, dim)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        ls_method: Method to use in line search optimization. Defaults to "COBYLA".
        ls_max_iter: Max number of iterations used in line search. Defaults to 200.
        optimize_step: Whether to optimize the step size using line search. Defaults to True.
        dmrg_like: Whether to use a DMRG-like optimization for the step size (alternating). Defaults to False.
        step_size: Step size for the update of the gradient descent step. Defaults to 1.
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to True.
            If false, we use the polar decomposition.

    Returns:
        A tuple containing:
            The updated Kraus tensor
            The updated POVM tensor
            The updated state tensor
            The updated step size (if optimize_step is False this is just the same as input step_size)
            The cost function evaluated at the intial kraus, povm and state tensors.
    """
    
    if dmrg_like:
        return _gradient_descent_step_dmrg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step_size=initial_step_size, use_geodesic=use_geodesic, regularized=regularized, target_operators=target_operators, num_samples=num_samples)
        
    return _gradient_descent_step_no_dmrg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step_size=initial_step_size, use_geodesic=use_geodesic)
    
    
GRADIENT_FUNCTIONS = {
        "povm": gradient_povm_mps_jit,
        "kraus": gradient_k_mps_jit,
        "state": gradient_state_mps_jit
    }
    
GRADIENT_FUNCTIONS_REGULARIZED = {
    "povm": gradient_povm_mps_jit_reg,
    "kraus": gradient_k_mps_jit_reg,
    "state": gradient_state_mps_jit_reg
}    

def _update_tensor_via_gradient(
    operator_type: str, kraus_tensor: jnp.ndarray, povm_psd: jnp.ndarray, state_psd: jnp.ndarray,
    indices_list: list[list[int]], prob_matrix: jnp.ndarray, ls_method="COBYLA", ls_max_iter=200,
    optimize_step: bool = True, initial_step: float = 1, use_geodesic: bool = True,
    regularized:bool=False, target_operators:Sequence[jnp.ndarray]= None, num_samples:int = None,
    metric:Literal["canonical", "euclidean"]="canonical", verbose:bool=False, return_cost_fn_value:bool=False)->tuple[jnp.ndarray, float]:
    """Update a given operator tensor (POVM, Kraus, or State) following the gradient direction.

    Args:
        operator_type (str): Type of the operator to update ('povm', 'kraus', or 'state').
        kraus_tensor (jnp.ndarray): The current Kraus tensor.
        povm_psd (jnp.ndarray): The current POVM tensor.
        state_psd (jnp.ndarray): The current state tensor.
        indices_list (list[list[int]]): List of indices corresponding to gate sequences.
        prob_matrix (jnp.ndarray): Probability matrix of shape (num_povm, num_gate_sequences).
        ls_method (str, optional): The method to use in line search optimization. Defaults to "COBYLA".
        ls_max_iter (int, optional): The max number of iterations used in line search. Defaults to 200.
        optimize_step (bool, optional): Whether to optimize the step size using line search. Defaults to True.
        initial_step (float, optional): The initial step size for the update of the gradient descent step. Defaults to 1.
        use_geodesic (bool, optional): Whether to use the geodesic to compute the updated tensor. Defaults to True.

    Returns:
        * The updated tensor.
        * The optimized step size.
        * (optional) The cost function value at the updated tensor.
    """
    
    operator_tensors = {
        "povm": povm_psd,
        "kraus": kraus_tensor,
        "state": state_psd
    }
    
    if operator_type not in GRADIENT_FUNCTIONS:
        raise ValueError(f"Invalid operator type: {operator_type}. Choose from 'povm', 'kraus', or 'state'.")
    
    if regularized:
        target_kraus, target_povm, target_state = target_operators
        ambient_gradient = GRADIENT_FUNCTIONS_REGULARIZED[operator_type](
            kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix,
            target_kraus, target_povm, target_state, num_samples
            )    
    else:
        ambient_gradient = GRADIENT_FUNCTIONS[operator_type](kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    
    # NOTE: Take the conjugate because jax returns 2df/dx and the gradient is 2df/dx*.
    # NOTE: No need to divide by 2. This is just for agreement with mGST. 
    # Correct one has factor of 2x.
    stiefel_gradient_matrix, isometry = euclidean_gradients_to_stiefel(
        gradient_tensor=ambient_gradient.conj()/2, operator_tensor=operator_tensors[operator_type], operator_type=operator_type, # (..., n, p)
        metric=metric
    )
    
    previous_shape = operator_tensors[operator_type].shape
    
    if optimize_step:
        optimization_result = minimize(
            cost_function_from_updated_operator, initial_step,
            args=(isometry, stiefel_gradient_matrix, previous_shape, operator_type,
                  kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, use_geodesic),
            method=ls_method, options={"maxiter": ls_max_iter}
        )
        optimized_step = optimization_result.x
        cost_fn_value = optimization_result.fun
        if verbose:
            print(f"Optimized step size for {operator_type.capitalize()}: {optimized_step}")
    else:
        optimized_step = initial_step
    
    updated_tensor = _update_isometry_and_back_to_tensor(optimized_step, isometry, stiefel_gradient_matrix, previous_shape, operator_type, use_geodesic)
    if return_cost_fn_value:
        return updated_tensor, optimized_step, cost_fn_value
    return updated_tensor, optimized_step

    
def _gradient_descent_step_dmrg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method="COBYLA", ls_max_iter=200, optimize_step:bool=True, initial_step_size:jnp.ndarray | None = None, use_geodesic:bool=True, regularized:bool=False, target_operators:Sequence[jnp.ndarray]= None, num_samples:int = None)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Perform a gradient descent step on the Kraus operators using a DMRG-like optimization."""
    
    if initial_step_size is None:
        initial_step_size = jnp.array([1, 1, 1])
        
    # Get the individual step sizes
    initial_step_kraus, initial_step_povm, initial_step_state = initial_step_size
    # Initial cost value
    if not regularized:
        initial_cost_value = cost_function_jax_mps(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, jit=True)
    else:
        kraus_target, povm_target, state_target = target_operators
        initial_cost_value = cost_function_jax_mps_regularized(
            kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix,
            kraus_target, povm_target, state_target, num_samples,
            jit=True)

    # First sweep over the POVM tensor        
    new_povm_psd, optimized_step_povm = _update_tensor_via_gradient("povm", kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step=initial_step_povm, use_geodesic=use_geodesic, regularized=regularized, target_operators=target_operators, num_samples=num_samples)
    # Then over the kraus tensor
    new_kraus_tensor, optimized_step_kraus = _update_tensor_via_gradient("kraus", kraus_tensor, new_povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step=initial_step_kraus, use_geodesic=use_geodesic, regularized=regularized, target_operators=target_operators, num_samples=num_samples)
    # And finally we optimize the state tensor
    new_state_psd, optimized_step_state = _update_tensor_via_gradient("state", new_kraus_tensor, new_povm_psd, state_psd, indices_list, prob_matrix, ls_method=ls_method, ls_max_iter=ls_max_iter, optimize_step=optimize_step, initial_step=initial_step_state, use_geodesic=use_geodesic, regularized=regularized, target_operators=target_operators, num_samples=num_samples)
        
    optimized_step = jnp.array([optimized_step_kraus, optimized_step_povm, optimized_step_state])
        
    return new_kraus_tensor, new_povm_psd, new_state_psd, optimized_step, initial_cost_value 
    
def _gradient_descent_step_no_dmrg(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method="COBYLA", ls_max_iter=200, optimize_step:bool=True, initial_step_size:float|jnp.ndarray=1, use_geodesic:bool=True)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """Perform a gradient descent step on the Kraus, POVM and State operators updating all elements at once."""
    
    cost_value, ambient_gradients = gradient_all_3_and_value_jit(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    ambient_grad_kraus, ambient_grad_povm, ambient_grad_state = ambient_gradients
    
    # We would want to project 2 * (conjugate_wirtinger_derivative)
    # However, JAX returns already 2 * wirtinger_derivative, so we just need to take the conjugate
    
    # Kraus tensor to stiefel
    kraus_stiefel_gradient, kraus_isometries =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_kraus.conj(), operator_tensor=kraus_tensor, operator_type="kraus") # num_gates, rank_kraus*dim, dim
    
    # State tensor to stiefel
    state_stiefel_gradient, state_isometry =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_state.conj(), operator_tensor=state_psd, operator_type="state") # dim*rank_state, 1
    
    # POVM tensor to stiefel
    povm_stiefel_gradient, povm_isometry =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_povm.conj(), operator_tensor=povm_psd, operator_type="povm") # num_povm*rank_povm, dim
    
    # Gathering isometies, gradients and shapes of tensors
    shapes_of_tensors = (kraus_tensor.shape, povm_psd.shape, state_psd.shape)
    stiefel_gradients = (kraus_stiefel_gradient, povm_stiefel_gradient, state_stiefel_gradient)
    stiefel_isometries = (kraus_isometries, povm_isometry, state_isometry)
    
    if optimize_step:
        optimization_result = minimize(cost_function_from_updated_isometries_individual_step_size, initial_step_size, args=(stiefel_gradients, stiefel_isometries, indices_list, prob_matrix, shapes_of_tensors, use_geodesic), method=ls_method, options={"maxiter": ls_max_iter})
        
        optimized_step_size = optimization_result.x
        # number_of_iterations = optimization_result.nit
        print(f"optimized step size: {optimized_step_size}")
    else:
        optimized_step_size = initial_step_size
            
    new_kraus_tensor, new_povm_psd, new_state_psd = _update_all_isometries_and_back_to_tensors(optimized_step_size, stiefel_gradients, stiefel_isometries, shapes_of_tensors=shapes_of_tensors, use_geodesic=use_geodesic)
    
    return new_kraus_tensor, new_povm_psd, new_state_psd, optimized_step_size, cost_value
    
def cost_function_from_updated_operator(
    step_size: float,
    initial_isometry: Matrix,
    update_direction: jnp.ndarray,
    tensor_shape: tuple[int],
    operator_type: str,
    kraus_tensor: Tensor = None,
    povm_psd: Tensor = None,
    state_psd: Tensor = None,
    indices_list: list[list[int]] = None,
    prob_matrix: Matrix = None,
    use_geodesic: bool = False,
    regularized:bool=False,
    target_operators:Sequence[jnp.ndarray]=  None,
    num_samples:int = None,
) -> Scalar:
    """Compute the objective function after updating an operator.

    Args:
        step_size: Gradient descent step size to be optimized.
        initial_isometry: Initial operator on the isometry manifold.
        update_direction: Tangent vector corresponding to the update direction.
        tensor_shape: Original shape of the operator tensor.
        operator_type: The name of the operator ('povm', 'state', 'kraus').
        kraus_tensor: Kraus tensor of shape (num_gates, kraus_rank, dim_out, dim_in).
        povm_psd: POVM tensor of shape (num_povm, rank_povm, dim).
        state_psd: State tensor of shape (dim, rank_state).
        indices_list: List of indices corresponding to gate sequences.
        prob_matrix: Probability matrix of shape (num_povm, num_gate_sequences).
        use_geodesic: Whether to use geodesic as a retraction.

    Returns:
        The cost function value at the updated operator tensor.
    """    
    
    updated_tensor = _update_isometry_and_back_to_tensor(step_size, initial_isometry, update_direction, tensor_shape, operator_type, use_geodesic)
    
    if regularized:
        kraus_target, povm_target, state_target = target_operators
        cost_function = lambda *args: cost_function_jax_mps_regularized(
            *args,
            target_kraus_tensor=kraus_target, target_povm_psd=povm_target, target_state_psd=state_target, num_samples=num_samples,
            jit=True)
    else:
        cost_function = lambda *args: cost_function_jax_mps(*args, jit=True)
    
    if operator_type == "povm":
        return cost_function(kraus_tensor, updated_tensor, state_psd, indices_list, prob_matrix)
    elif operator_type == "state":
        return cost_function(kraus_tensor, povm_psd, updated_tensor, indices_list, prob_matrix)
    elif operator_type == "kraus":
        return cost_function(updated_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    else:
        raise ValueError(f"Unsupported operator_type: {operator_type}")
    
def _update_isometry_and_back_to_tensor(
    step_size: float,
    isometries: Sequence[Matrix] | Matrix,
    update_directions: Sequence[Matrix] | Matrix,
    tensor_shape: tuple[int],
    operator_type: str,
    use_geodesic: bool = True) -> Tensor:
    """Helper function to update tensor of isometries and bring them back to its original tensor shape to be used in the next optimization step.
    
    Args:
        step_size: The step size of the update.
        isometries: The isometries to be updated. Dimensions are (n, p) or (num_gates, n, p) for kraus operators
        update_directions: The tangent vectors corresponding to the update direction (at point x in manifold). 
            Must have the same dimensions as isometries.
        tensor_shape: The shape of the original tensor.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to True.
    
    Returns:
        The updated tensor of the isometries each with `tensor_shape` shape.
    """
    from mGST.riemannian import update_isometry_tensors
    
    updated_isometry = update_isometry_tensors(
        isometries=isometries,
        update_directions=update_directions,
        step_size=step_size,
        operator_type=operator_type,
        use_geodesic=use_geodesic
    )
    
    return isometry_to_tensor(updated_isometry, tensor_shape)

def cost_function_from_updated_isometries_individual_step_size(step_sizes:jnp.array, stiefel_gradients:tuple[jnp.ndarray], stiefel_isometries:tuple[jnp.ndarray], indices_list:list[list[int]], prob_matrix:jnp.ndarray, shapes_of_tensors:tuple[tuple[int]], use_geodesic:bool=False)->float:
    """Compute objective function at position on geodesic
    
    Args:
        step_size: Geodesic curve parameter
        tanget_vector: Element of the tangent space at K and local direction of the geodesic
        kraus: Current position. Dimensions are (num_gates, kraus_rank, dim_out, dim_in)
        povm_tensor: Current POVM estimate. Dimensions are (num_povm, dim_out, dim_out)
        state_psd: Positive semidefinite matrix representing the initial state. Dimensions are (dim_out, dim_out)
        indices_list: 2D array where each row contains the gate indices of a gate sequence
        prob_matrix: 2D array of measurement outcomes for sequences in J.
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to False.
    Returns:
        Objective function value at new position along the geodesic
    """
    
    new_kraus_tensor, new_povm_psd, new_state_psd = _update_all_isometries_and_back_to_tensors(step_sizes, stiefel_gradients, stiefel_isometries, shapes_of_tensors=shapes_of_tensors, use_geodesic=use_geodesic)
    
    return cost_function_jax_mps(new_kraus_tensor, new_povm_psd, new_state_psd, indices_list, prob_matrix, jit=True)

def update_all_isometries(step_sizes: jnp.ndarray | float, stiefel_gradients:tuple[jnp.ndarray], stiefel_isometries:tuple[jnp.ndarray], use_geodesic:bool=False)->tuple[jnp.ndarray]:
    """Compute the updated isometries in the direction of gradients using step sizes

    Args:
        steps_size: Array of step sizes indicating magnitude of updated for each component. Order is kraus, povm, state.
        stiefel_gradients: Sequence of stiefel gradients dictating diretion to move in. Order is kraus, povm, state.
        stiefel_isometries: Sequence of initial isometries to be updated. Order is kraus, povm, state.
        use_geodesic: Whether to use the geodesic after moving in gradient direction. Defaults to False, which uses the polar decomposition.

    Returns:
        A tuple containing the updated isometries in the direction of the gradients.
    """
    from mGST.riemannian import update_isometry_tensors
    
    kraus_gradients, povm_gradient, state_gradient = stiefel_gradients
    kraus_isometries, povm_isometry, state_isometry = stiefel_isometries
    
    if isinstance(step_sizes, float):
        step_kraus, step_povm, step_state = step_sizes, step_sizes, step_sizes
    elif isinstance(step_sizes, jnp.ndarray | np.ndarray | Sequence):
        assert len(step_sizes) == 3, f"Wrong length of step size array. Must be 3. Intead got {len(step_sizes)}"
        step_kraus, step_povm, step_state = step_sizes
    else:
        raise ValueError(f"Step size must be a float or a sequence of length 3. Instead got {type(step_sizes)}")
        
    # update kraus tensor
    kraus_isometries_updated = update_isometry_tensors(isometries = kraus_isometries, update_directions = kraus_gradients, step_size = step_kraus, operator_type="kraus", use_geodesic=use_geodesic)        
    # update state
    state_isometry_updated = update_isometry_tensors(isometries = state_isometry, update_directions = state_gradient, step_size = step_povm, operator_type="state", use_geodesic=use_geodesic)
    # update povm
    povm_isometry_updated = update_isometry_tensors(isometries = povm_isometry, update_directions = povm_gradient, step_size = step_state, operator_type="povm", use_geodesic=use_geodesic)

    return kraus_isometries_updated, povm_isometry_updated, state_isometry_updated

def reshape_all_isometries_to_tensors(stiefel_isometries:tuple[jnp.ndarray], shapes_of_tensors:tuple[tuple[int]]):
    """Reshape the isometries back to tensors of indicated shapes
    
    Args:
        stiefel_isometries: Tuple of isometries to be reshaped. Order is kraus, povm, state.
        shapes_of_tensors: Tuple of shapes to reshape the isometries to. Order is kraus, povm, state.
    Returns:
        A tuple containing the reshaped tensors.
    """
    kraus_isometries, povm_isometry, state_isometry = stiefel_isometries
    previous_kraus_shape, previous_povm_shape, previous_state_shape = shapes_of_tensors
    
    new_kraus_tensor = isometry_to_tensor(kraus_isometries, tensor_shape=previous_kraus_shape) # num_gates, rank_kraus, dim, dim
    
    new_state_psd = isometry_to_tensor(state_isometry, tensor_shape=previous_state_shape) # dim, rank_state
    
    new_povm_psd = isometry_to_tensor(povm_isometry, tensor_shape=previous_povm_shape) # num_povm, rank_povm, dim
        
    return new_kraus_tensor, new_povm_psd, new_state_psd



def cost_function_from_updated_isometries_single_step_size(step_size:float, stiefel_gradients:tuple[jnp.ndarray], stiefel_isometries:tuple[jnp.ndarray], indices_list:list[list[int]], prob_matrix:jnp.ndarray, shapes_of_tensors:tuple[tuple[int]], use_geodesic:bool=False)->float:
    """Compute objective function at position on geodesic
    
    Args:
        step_size: Geodesic curve parameter
        tanget_vector: Element of the tangent space at K and local direction of the geodesic
        kraus: Current position. Dimensions are (num_gates, kraus_rank, dim_out, dim_in)
        povm_tensor: Current POVM estimate. Dimensions are (num_povm, dim_out, dim_out)
        state_psd: Positive semidefinite matrix representing the initial state. Dimensions are (dim_out, dim_out)
        indices_list: 2D array where each row contains the gate indices of a gate sequence
        prob_matrix: 2D array of measurement outcomes for sequences in J.
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to False.
    Returns:
        Objective function value at new position along the geodesic
    """
    
    new_kraus_tensor, new_povm_psd, new_state_psd = _update_all_isometries_and_back_to_tensors(step_size, stiefel_gradients, stiefel_isometries, shapes_of_tensors=shapes_of_tensors, use_geodesic=use_geodesic)
        
    return cost_function_jax_mps(new_kraus_tensor, new_povm_psd, new_state_psd, indices_list, prob_matrix, jit=True)

def _update_all_isometries_and_back_to_tensors(step_sizes: jnp.ndarray | float, stiefel_gradients:tuple[jnp.ndarray], stiefel_isometries:tuple[jnp.ndarray], shapes_of_tensors:tuple[tuple[int]], use_geodesic:bool=False)->tuple[jnp.ndarray]:
    """Compute the updated isometries in the direction of gradients using step sizes"""
    kraus_isometries_updated, povm_isometry_updated, state_isometry_updated = update_all_isometries(step_sizes, stiefel_gradients, stiefel_isometries, use_geodesic)
    
    return reshape_all_isometries_to_tensors(stiefel_isometries=(kraus_isometries_updated, povm_isometry_updated, state_isometry_updated), shapes_of_tensors=shapes_of_tensors)
    
def check_kraus_tensor_is_isometry(kraus_tensor:jnp.ndarray)->bool: 
    """Check if the Kraus tensor is an isometry.

    Args:
        kraus_tensor (jnp.ndarray): The Kraus tensor to check. Dimensions: rank, dim, dim

    Returns:
        bool: True if the Kraus tensor is an isometry, False otherwise.
    """
    return jnp.allclose(jnp.eye(kraus_tensor.shape[-1]), jnp.einsum("ijk,ijl->kl", kraus_tensor, kraus_tensor.conj()))

# Extracted from openTN
def split_matrix_svd(op: jnp.ndarray, max_rank: int = 2):
    """
    Perform batched singular value decomposition (SVD) on an operator (matrix),
    truncating small singular values based on the given rank.

    Supports input tensors of shape (..., N, M), where the SVD is applied independently
    to each (N, M) matrix along the batch dimensions.

    Args:
        op (jnp.ndarray): Input tensor of shape (..., N, M).
        max_rank (int): Maximum number of singular values to keep.

    Returns:
        u (jnp.ndarray): Left singular vectors of shape (..., N, min(sv_keep, M)).
        s (jnp.ndarray): Singular values of shape (..., min(max_rank, M)).
    """
    if not max_rank >=1:
        raise ValueError(f"max_rank={max_rank} must be at least 1")
    if not op.ndim >= 2:
        raise ValueError(f"op.ndim={op.ndim} must be at least 2")
    
    # Compute batched SVD
    u, s, _ = jnp.linalg.svd(op, hermitian=True)
    
    # Truncate singular values and vectors accordingly
    u = u[..., :max_rank]  # Keep only the top singular vectors
    s = s[..., :max_rank]  # Keep only the top singular values
    return u, s


def factorize_psd_truncated(psd: jnp.ndarray, max_rank: int | None = None, unique_srt:bool = False) -> jnp.ndarray:
    """
    Factorizes a batch of positive semi-definite (PSD) matrices by truncating singular values.

    More robust to small values than cholesky decomposition from numpy.

    Returns x' such that psd ≈ x' @ x'.conj().T
    
    Args:
        psd: Input tensor of shape (..., N, N) (must be Hermitian).
        max_rank: Maximum number of singular values to keep.
        unique_srt: Whether to return the unique square root of the factorized matrix.
            This is the hermitian (square) matrix satisfying x^2 = x' @ x' = psd.
            If False, the factorized matrix is x' @ x'.conj().T.
    
    Returns:
        jnp.ndarray: The factorized matrix `x'` of shape (..., N, min(max_rank, N)).
    """
    if max_rank is None:
        max_rank = psd.shape[-1]  # Assume full rank by default
        
    u, s, = split_matrix_svd(psd, max_rank)
    
    factorization = u * jnp.sqrt(s)[..., None, :]
    if unique_srt:
        return factorization @ u.conj().swapaxes(-1, -2)
    return factorization
    #  s[..., None, :] reshapes s into shape (..., 1, min(max_rank, N)), allowing elementwise multiplication with x ((..., N, min(max_rank, N))).

def canonical_gradient(x:Matrix, z:Matrix)->Matrix:
    """ Compute the riemmanian gradient at the point x on the stiefel manifold using the canonical metric

    Handles batch dimensions.

    Args:
        x: The base point of the tangent space. Shape (..., n, p)
        z: The euclidean gradient at x. Shape (..., n, p)

    Returns:
        The riemannian gradient using the canonical metric
    """
    return z - x @ transpose(z.conj()) @ x


def project_onto_tangent_space(x: Matrix, z: Matrix)->Matrix:
    """ Project a matrix z onto the tangent space of the manifold at x

    Handles batch dimensions.

    Args:
        x: The base point of the tangent space. Shape (..., n, p)
        z: The matrix to project onto the tangent space. Shape (..., n, p)

    Returns:
        A matrix projected onto the tangent space of the manifold at x.
    """
    return z - x @ symmetrize(transpose(x).conj() @ z)

def transpose(A:Matrix)->Matrix:
    """
    Transpose a matrix, swapping its last two dimensions.

    Handles batch dimensions.

    Args:
        A: Matrix to be transposed. Shape (..., n, p)
        
    Returns:
        Transposed matrix
    """
    return A.swapaxes(-1, -2)

def symmetrize(A:Matrix)->Matrix:
    """
    Symmetrize a matrix by projecting it onto the symmetric subspace.
    
    Handles batch dimensions.
    
    Args:
        A: square matrix to be symmetrized. Shape (..., n, n)
    Returns:
        Symmetrized matrix. Shape (..., n, n)
    """
    return 0.5 * (A + transpose(A).conj())

def tensor_to_isometry(tensor: Tensor, n:int, p:int)-> Matrix:
    """
    Reshape a tensor into an isometry matrix of dimensions n and p.
    
    Handles batch dimensions.
    e.g. for kraus operators, the first dimension is the number of gates.

    Args:
        x: tensor to be reshaped
        n: Row dimension of the new matrix
        p: Column dimension of the new matrix

    Returns:
        Matrix of dimensions (bath_dim, n, p) or (n, p) if no batch dimension.
    """
    mat = jnp.reshape(tensor, (-1, n, p))
    # Safe squeeze of axis = 0 only.
    return mat[0] if mat.shape[0] == 1 else mat

def isometry_to_tensor(isometry: Matrix, tensor_shape: tuple[int]) -> Tensor:
    """Reshape the updated isometry based on the operator type.

    Handles batch dimensions. (as long as the tensor shape contains the batch dimension)

    Args:
        isometry: The updated isometry tensor. Shape (..., n, p)
        tensor_shape: The target shape for reshaping.

    Returns:
       The reshaped tensor.
    """
    return jnp.reshape(isometry, shape=tensor_shape)

def tensors_to_isometries(*tensors: Tensor, operator_type: str = None, n:int = None, p:int = None) -> tuple[Matrix, ...]:
    """Convert multiple tensors to isometries with the same dimensions.
    
    Handles batch dimensions.
    
    Args:
        *tensors: Variable number of tensors to convert.
        operator_type: The type of the input operator used to infer the dimension of 
            the isometry manifold (n, p). Can be: ('kraus', 'state', or 'povm')
        n: First dimension of isometry if known
        p: Second dimension of isometry if known
        
    Returns:
        Tuple of isometry matrices in the same order as input tensors.
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided.")
    
    if operator_type is None and (n is None or p is None):
        raise ValueError("Either operator_type or both n and p must be provided.")

    if n is None or p is None:
        n, p = get_isometry_dimensions_from_tensor(tensors[0], operator_type=operator_type)

    # Handle POVM and state cases
    return tuple(tensor_to_isometry(tensor, n=n, p=p) for tensor in tensors)


# TODO: move this and all the acompannying functions to Riemannian file and use stiefel_shape_from_... functions.
def euclidean_gradients_to_stiefel(gradient_tensor: Tensor, operator_tensor: Tensor, operator_type:str="kraus", metric:Literal["canonical", "euclidean"] = "canonical")-> tuple[Matrix, Matrix]:
    """ Project a sequence of tensor of Euclidean gradients onto the Tangent space of the Stiefel manifold

    Args:
        gradient_tensor: The Euclidean gradient tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        operator_tensor: The tensor corresponding to either the Kraus, state or POVM operators.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        metric: The metric to use for the tangent space ('canonical' or 'euclidean')
    Returns:
        A tuple containing matrices of dimensions (n, p) or arrays thereof in the case of Kraus operators:
            An array of gradients projected onto the Tanget space of the Stiefel manifold
            An array of stiefel isometries
    """

    n, p = get_isometry_dimensions_from_tensor(operator_tensor, operator_type=operator_type)

    if operator_type == "kraus":
        
        kraus_isometries = []
        gradients_stiefel = []
        
        for gradient, kraus in zip(gradient_tensor, operator_tensor):
            
            gradient_stiefel, kraus_stiefel = gradient_and_operator_tensors_to_stiefel(gradient = gradient, operator = kraus, n=n, p=p, metric=metric)
            
            gradients_stiefel.append(gradient_stiefel)
            kraus_isometries.append(kraus_stiefel)
                
        return jnp.array(gradients_stiefel), jnp.array(kraus_isometries)
        
    return gradient_and_operator_tensors_to_stiefel(gradient = gradient_tensor, operator = operator_tensor, n=n, p=p, metric=metric)

def gradient_and_operator_tensors_to_stiefel(gradient:Tensor, operator:Tensor, n:int, p:int, metric:Literal["canonical", "euclidean"] = "canonical")->tuple[Matrix, Matrix]:
    """ Convert a pair of gradient and operator tensors to the Stiefel manifold matrices

    We first reshape the operator and gradient tensors into isometries and 
    project the gradient onto the tangent space of the manifold at the operator.
    
    We assume the tensors are ordered in a way that we can group the first indices into a single
    index of dimension n and the rest into another one of dimension p.
    
    Args:
        gradient: The gradient tensor
        operator: The operator tensor
        n: The row dimension of stiefel manifold
        p: The column dimension of the stiefel manifold
        metric: 

    Returns:
        A tuple containing matrices of dimensions (n, p):
            The gradient projected onto the Stiefel manifold
            The operator projected onto the Stiefel manifold
    """
    operator_stiefel = tensor_to_isometry(tensor=operator, n=n, p=p)
    gradient_np = tensor_to_isometry(tensor=gradient, n=n, p=p)
    
    if metric == "euclidean":
        gradient_stiefel = project_onto_tangent_space(x = operator_stiefel, z = gradient_np)
    elif metric == "canonical":
        gradient_stiefel = canonical_gradient(x = operator_stiefel, z = gradient_np)
    else:
        raise ValueError(f"Metric {metric} is not recognized. Please use one of the following: 'euclidean', 'canonical'")
    
    return gradient_stiefel, operator_stiefel
        
def povm_from_psd(povm_psd:jnp.ndarray) -> jnp.ndarray:
    """
    Get the full POVM tensor from its factorization
    Args:
        povm_psd: Factorization of the POVM of shape (num_povm, rank_povm, dim)
    """
    return jnp.einsum("irk, irm -> ikm", povm_psd.conj(), povm_psd)

def state_from_psd(state_psd:jnp.ndarray) -> jnp.ndarray:
    """
    Get the full state tensor from its factorization
    Args:
        state_psd: Factorization of the state of shape (dim, rank_state)
    """
    return state_psd @ state_psd.conj().T

def kraus_from_psd(kraus_psd:jnp.ndarray)->jnp.ndarray:
    """Get the full kraus tensor from its factorization

    Args:
        kraus_psd: Factorization of the kraus tensor of shape (num_gates, rank_kraus, dim_out, dim_in)

    Returns:
        Full kraus tensor of shape (num_gates, dim_out, dim_out*, dim_in, dim_in*)
    """
    # num_gates, rank_kraus, dim, dim = kraus_psd.shape
    return jnp.einsum("ijkl,ijnm -> iknlm", kraus_psd, kraus_psd.conj())

def calculate_finite_sampling_error(prob_matrix_exact:jnp.ndarray, prob_matrix_sampled:jnp.ndarray):
    """ Calculate the finite sampling error between the exact and sampled probability matrices.
    
    Args:
        prob_matrix_exact: Exact probability matrix of shape (num_povm, num_gate_sequences).
        prob_matrix_sampled: Sampled probability matrix of shape (num_povm, num_gate_sequences).
    Returns:
        float: The finite sampling error.
    """
    num_povm, num_gate_sequences = prob_matrix_exact.shape
    return jnp.sum(jnp.abs(prob_matrix_exact - prob_matrix_sampled)**2) / (num_gate_sequences * num_povm)

def random_tangent_vector(x:Matrix, seed=42)->Matrix:
    """
    Generate a random tangent vector at the point x on the stiefel manifold.
    
    Args:
        x: Point on the stiefel manifold of dimensions (n, p)
        n: Row dimension of the tangent vector
        p: Column dimension of the tangent vector
    
    Returns:
        A random tangent vector of dimensions (n, p) at the point x
    """
    key = jax.random.PRNGKey(seed)
    # Split the key to generate independent random parts
    key_real, key_imag = jax.random.split(key)

    # Generate real and imaginary parts
    real_part = jax.random.normal(key_real, x.shape)
    imag_part = jax.random.normal(key_imag, x.shape)

    # Combine into a complex matrix
    z = real_part + 1j * imag_part
    return project_onto_tangent_space(x=x, z=z)

def get_isometry_dimensions_from_tensor(tensor:Tensor, operator_type:str)->tuple[int, int]:
    """
    Get the Stiefel dimensions n and p from the given tensor.
    
    Args:
        tensor: The tensor from where dimensions will be inferred.
        operator_type: The type of the tensor. Can be one of the following: 'state', 'povm', 'kraus'.
            Shape according to the type should be:
            * POVM: (num_povm, povm_rank, dim)
            * State: (dim, rank_state)
            * Kraus: (num_gates, kraus_rank, dim, dim)
    Returns:
        n: The Stiefel n dimension
        p: The Stiefel p dimension
        
    Raises:
        ValueError: If the tensor_type is not recognized.
    """
    if operator_type == "state":
        dim, rank_state = tensor.shape # dim, rank_state
        n = dim * rank_state
        p = 1

    elif operator_type == "povm":
        num_povm, rank_povm, dim = tensor.shape # num_povm, rank_povm, dim
        n = num_povm * rank_povm
        p = dim
    elif operator_type == "kraus":
        _, rank_kraus, dim, _ = tensor.shape # num_gates, kraus_rank, dim_out, dim_in
        n = rank_kraus * dim
        p = dim
    else:
        raise ValueError(f"Operator name '{operator_type}' is not recognized. Please use one of the following: 'state', 'povm' or 'kraus'.")
    return n, p
    
def riemannian_hessian_vector_mgst(rhessian_tensor:Tensor, vector:Matrix)->Matrix:
    """Compute the Riemannian Hessian-vector product using the mGST implementation
    
    Args:
        rhessian_tensor: The Riemannian Hessian tensor of shape (2, np, 2np) where (n,p) are the dimensions of the stiefel manifold.
        vector: The vector to be multiplied with the Riemannian Hessian tensor. Shape is (n, p).

    Returns:
        The result of the Riemannian Hessian-vector product.
    """
    np = rhessian_tensor.shape[1]
    rhessian_matrix = rhessian_tensor.reshape(2 * np, 2 * np)
    vector_and_conjugate = jnp.vstack((vector, vector.conj())).reshape(-1)
    rhessian_vector = rhessian_matrix @ vector_and_conjugate
    return rhessian_vector[:np].reshape(vector.shape)

from mGST.low_level_jit import dK_dMdM, ddM

def compute_euclidean_hessian_kraus(kraus_tensor:jnp.ndarray, povm_mgst:jnp.ndarray, state_mgst:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray)->jnp.ndarray:
    """Compute the Euclidean Hessian tensor for the Kraus operators.
    
    Args:
        kraus_tensor: Kraus tensor of shape (num_gates, kraus_rank, dim_out, dim_in).
        povm_mgst: POVM tensor of shape (num_povm, dim^2).
        state_mgst: State tensor of shape (dim^2).
        indices_list: List of indices corresponding to gate sequences.
        prob_matrix: Probability matrix of shape (num_povm, num_gate_sequences).
        
    Returns:
        Hessian 
    """
    # making sure we are using numpy for the hessian calculation
    kraus_tensor = np.array(kraus_tensor)
    povm_mgst = np.array(povm_mgst)
    state_mgst = np.array(state_mgst)
    prob_matrix = np.array(prob_matrix)
    
    num_gates = kraus_tensor.shape[0]
    kraus_rank = kraus_tensor.shape[1]
    dim = kraus_tensor.shape[2]
    dim_sqrd = dim**2
    # vectorized dimension
    n = num_gates * kraus_rank * dim_sqrd
    H = np.zeros((2 * n, 2 * n)).astype(np.complex128)
    kraus_mgst = np.einsum("ijkl,ijnm -> iknlm", kraus_tensor, kraus_tensor.conj()).reshape((num_gates, dim_sqrd, dim_sqrd))


    _, dM10, dM11 = dK_dMdM(X=kraus_mgst, K=kraus_tensor, E=povm_mgst, rho=state_mgst, J=indices_list, y=prob_matrix, d=num_gates, r=dim_sqrd, rK=kraus_rank)
    dd, dconjd = ddM(X=kraus_mgst, K=kraus_tensor, E=povm_mgst, rho=state_mgst, J=indices_list, y=prob_matrix, d=num_gates, r=dim_sqrd, rK=kraus_rank)

    # Based on the mGST implementation we see that actually here we use the hermitian order.
    A00 = dM11.reshape(n, n) + jnp.einsum("ijklmnop->ikmojlnp", dconjd).reshape(n, n)
    A10 = dM10.reshape(n, n) + jnp.einsum("ijklmnop->ikmojlnp", dd).reshape(n, n)
    A11 = A00.conj()
    A01 = A10.conj()

    H[:n, :n] = A00
    H[:n, n:] = A01
    H[n:, :n] = A10
    H[n:, n:] = A11
    return H

def compute_euclidean_derivatives_kraus(kraus_tensor:jnp.ndarray, povm_mgst:jnp.ndarray, state_mgst:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray):
    
    # making sure we are using numpy for the hessian calculation
    kraus_tensor = np.array(kraus_tensor)
    povm_mgst = np.array(povm_mgst)
    state_mgst = np.array(state_mgst)
    prob_matrix = np.array(prob_matrix)
    
    num_gates, kraus_rank, dim, dim = kraus_tensor.shape
    dim_sqrd = dim**2
    # vectorized dimension
    kraus_mgst = np.einsum("ijkl,ijnm -> iknlm", kraus_tensor, kraus_tensor.conj()).reshape((num_gates, dim_sqrd, dim_sqrd))
    
    dK_, dM10, dM11 = dK_dMdM(X=kraus_mgst, K=kraus_tensor, E=povm_mgst, rho=state_mgst, J=indices_list, y=prob_matrix, d=num_gates, r=dim_sqrd, rK=kraus_rank)
    dd, dconjd = ddM(X=kraus_mgst, K=kraus_tensor, E=povm_mgst, rho=state_mgst, J=indices_list, y=prob_matrix, d=num_gates, r=dim_sqrd, rK=kraus_rank)
    return {"dK_": dK_, "dM10": dM10, "dM11": dM11, "dd": dd, "dconjd": dconjd}

from mGST.additional_fns import transp

def compute_riemannian_hessian_kraus(kraus_tensor_mgst:jnp.ndarray, povm_mgst:jnp.ndarray, state_mgst:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray)->tuple[jnp.ndarray, dict]:
    """ Compute the Riemannian Hessian tensor for the Kraus operators under the canonical metric.
    
    Args:
        kraus_tensor_mgst: Kraus tensor of shape (num_gates, kraus_rank, dim_out, dim_in).
        povm_mgst: POVM tensor of shape (num_povm, dim^2).
        state_mgst: State tensor of shape (dim^2).
        indices_list: List of indices corresponding to gate sequences.
        prob_matrix: Probability matrix of shape (num_povm, num_gate_sequences).
    Returns:
        * Riemannian Hessian tensor of shape (2, num_gates, vect_dim, 2, num_gates, vect_dim) where vect_dim = kraus_rank * dim_out * dim_in.
        * Dictionary containing Euclidean derivatives used in the computation.
    """
    # Euclidean derivatives
    print("Computing 1st and 2nd order Euclidean derivatives ⏳")
    derivatives_kraus = compute_euclidean_derivatives_kraus(
        kraus_tensor=kraus_tensor_mgst,
        povm_mgst=povm_mgst,
        state_mgst=state_mgst,
        indices_list=indices_list,
        prob_matrix=prob_matrix,
    )
    
    print("Computing Riemannian Hessian ⏳")
    dM11 = derivatives_kraus['dM11']
    dM10 = derivatives_kraus['dM10']
    dd = derivatives_kraus['dd']
    dconjd = derivatives_kraus['dconjd']
    dK_ = derivatives_kraus['dK_']
    
    num_gates, kraus_rank, dim, dim =  kraus_tensor_mgst.shape
    n = kraus_rank * dim
    p = dim
    vect_dim = n * p

    # Reshaping second derivatives
    Fyconjy = dM11.reshape(num_gates, vect_dim, num_gates, vect_dim) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((num_gates, vect_dim, num_gates, vect_dim))
    Fyy = dM10.reshape(num_gates, vect_dim, num_gates, vect_dim) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((num_gates, vect_dim, num_gates, vect_dim))
    
    rhessian_kraus_mgst = np.zeros((2, num_gates, vect_dim, 2, num_gates, vect_dim)).astype(np.complex128)
    G = np.zeros((2, num_gates, vect_dim)).astype(np.complex128)

    for k in range(num_gates):
    # Reshaping into isometry
        Fy = dK_[k].reshape((n, dim))
        Y = kraus_tensor_mgst[k].reshape((n, dim))
        # riemannian gradient under canonical metric
        rGrad = Fy.conj() - Y @ Fy.T @ Y

        # saving rgrad for gate k as a vector
        G[0, k, :] = rGrad.reshape(-1)
        # saving (rgrad)* for gate k as a vector
        G[1, k, :] = rGrad.conj().reshape(-1)

        # projector onto orthogonal complement of stiefel manifold. I = X X^dag + X_perp X_perp^dag
        P = np.eye(n) - Y @ Y.T.conj()
        # transpose superoperator
        T = transp(n, dim)
        # Hessian assembly
        # See theorem 1, equation A27
        # This is already the riemannian hessian elements
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(dim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        # Full hessian on Z and Z*
        # See relation in equation A37 of individual hessian submatrices
        # Riemannian Hessian with correction terms
        rhessian_kraus_mgst[0, k, :, 0, k, :] = H00
        rhessian_kraus_mgst[0, k, :, 1, k, :] = H01 
        rhessian_kraus_mgst[1, k, :, 0, k, :] = H01.conj()
        rhessian_kraus_mgst[1, k, :, 1, k, :] = H00.conj()
        
        # These are the cross terms of the Hessian between different gates.
        for k2 in range(num_gates):
            if k2 != k:
                Yk2 = kraus_tensor_mgst[k2].reshape(n, p)
                rhessian_kraus_mgst[0, k2, :, 0, k, :] = Fyconjy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyy[k, :, k2, :].T
                rhessian_kraus_mgst[0, k2, :, 1, k, :] = Fyy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyconjy[k, :, k2, :].T
                rhessian_kraus_mgst[1, k2, :, 0, k, :] = rhessian_kraus_mgst[0, k2, :, 1, k, :].conj()
                rhessian_kraus_mgst[1, k2, :, 1, k, :] = rhessian_kraus_mgst[0, k2, :, 0, k, :].conj()
                
    return rhessian_kraus_mgst, derivatives_kraus