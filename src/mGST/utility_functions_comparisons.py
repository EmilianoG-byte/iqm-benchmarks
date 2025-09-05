# Needs iqm-benchmarks from the github repo to access all the mGST functions: https://github.com/iqm-finland/iqm-benchmarks
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

from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from qiskit.circuit.library import CZGate, RGate

from scipy.optimize import minimize

from typing_extensions import Literal

import jax.numpy as jnp
import numpy as np
import jax

from typing import Sequence, Callable
import warnings

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
from mGST.additional_fns import random_gs

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

def get_full_mgst_parameters_from_configuration(configuration:GSTConfiguration, backend, seed:int = 42):
    
    benchmark = CompressiveGST(backend, configuration)
    result = benchmark.run()
    
    rK = configuration.rank
    qubit_layout = configuration.qubit_layouts[0]
    dataset = result.dataset
    y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt = get_mgst_parameters_from_dataset(dataset, qubit_layout=qubit_layout, rK=rK)
    K, X, E, rho = initialize_mgst_parameters(dataset=dataset, target_init=True, seed=seed)
    
    return K, X, E, rho, y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt, rK

def create_4q_gst_config():
    """Create the configuration to run a 4 qubit Gate set tomography protocol.

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
        rank=1,
    )

    return Q4_GST

def get_x_from_k(k, depth=None, dim_squared=None):
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
    kraus_mgst_trans = kraus_mgst.transpose(0, 2, 1) # num_gates, dim_in x dim_in*, dim_out x dim_out*
    num_gates, *_ = kraus_mgst_trans.shape
    choi_kraus = superop2choi(kraus_mgst_trans) # num_gates, dim_in x dim_out, dim_in* x dim_out*
    choi_psd = factorize_psd_truncated(choi_kraus, max_rank=kraus_rank) # num_gates, dim_in x dim_out, rank_kraus
    kraus_tensor = jnp.reshape(choi_psd, shape=(num_gates, dim, dim, kraus_rank)) # num_gates, dim_in, dim_out, rank_kraus
    kraus_tensor = jnp.transpose(kraus_tensor, (0, 3, 2, 1)) # num_gates, rank_kraus, dim_out, dim_in
    return kraus_tensor, povm_psd, state_psd

def get_compressed_rep_mgst_cholesky(povm_mgst, state_mgst)->tuple[jnp.ndarray, jnp.ndarray]:
    """Get the compressed representation of the MGST operators using cholesky factorization.
    
    This is implementation usedin the original mGST code.
    
    Args:
        povm_mgst: POVM operators from MGST. Dimensions: (num_povm, dim_in x dim_in*)
        state_mgst: State operator from MGST. Dimensions: (dim_out x dim_out*)
    Returns:
        A tuple containing the compressed representation of the POVM and State.
    """
    num_povm, dim_sqrd =povm_mgst.shape
    dim = int(jnp.sqrt(dim_sqrd))
    povm_psd = jnp.array([jnp.linalg.cholesky(povm_mgst[k].reshape(dim, dim) + 1e-14 * jnp.eye(dim)).T.conj() for k in range(num_povm)])
    state_mgst_offset = state_mgst + 1e-14 * jnp.eye(dim).reshape(-1)
    state_psd = jnp.linalg.cholesky(state_mgst_offset.reshape(dim, dim))
    return povm_psd, state_psd

def get_compressed_rep_mgst_cholesky_numpy(povm_mgst, state_mgst)->tuple[np.ndarray, np.ndarray]:
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

from mGST.algorithm import B_SFN_riem_Hess, A_SFN_riem_Hess

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
    """Perform a gradient descent step on the Kraus operators

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
    
    
def _update_tensor_via_gradient(
    operator_type: str, kraus_tensor: jnp.ndarray, povm_psd: jnp.ndarray, state_psd: jnp.ndarray,
    indices_list: list[list[int]], prob_matrix: jnp.ndarray, ls_method="COBYLA", ls_max_iter=200,
    optimize_step: bool = True, initial_step: float = 1, use_geodesic: bool = True,
    regularized:bool=False, target_operators:Sequence[jnp.ndarray]= None, num_samples:int = None,
    metric:Literal["canonical", "euclidean"]="canonical")->tuple[jnp.ndarray, float]:
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
        jnp.ndarray: The updated tensor.
        float: The optimized step size.
    """
    gradient_functions = {
        "povm": gradient_povm_mps_jit,
        "kraus": gradient_k_mps_jit,
        "state": gradient_state_mps_jit
    }
    
    gradient_functions_regularized = {
        "povm": gradient_povm_mps_jit_reg,
        "kraus": gradient_k_mps_jit_reg,
        "state": gradient_state_mps_jit_reg
    }
    
    operator_tensors = {
        "povm": povm_psd,
        "kraus": kraus_tensor,
        "state": state_psd
    }
    
    if operator_type not in gradient_functions:
        raise ValueError(f"Invalid operator type: {operator_type}. Choose from 'povm', 'kraus', or 'state'.")
    
    if regularized:
        target_kraus, target_povm, target_state = target_operators
        ambient_gradient = gradient_functions_regularized[operator_type](
            kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix,
            target_kraus, target_povm, target_state, num_samples
            )    
    else:
        ambient_gradient = gradient_functions[operator_type](kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    
    # NOTE: should we divide by 2 to obtain *just* df/dz* instead of 2df/dz*
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
        print(f"Optimized step size for {operator_type.capitalize()}: {optimized_step}")
    else:
        optimized_step = initial_step
    
    return _update_isometry_and_back_to_tensor(optimized_step, isometry, stiefel_gradient_matrix, previous_shape, operator_type, use_geodesic), optimized_step

    
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
    initial_isometry: jnp.ndarray,
    update_direction: jnp.ndarray,
    tensor_shape: tuple[int],
    operator_type: str,
    kraus_tensor: jnp.ndarray = None,
    povm_psd: jnp.ndarray = None,
    state_psd: jnp.ndarray = None,
    indices_list: list[list[int]] = None,
    prob_matrix: jnp.ndarray = None,
    use_geodesic: bool = False,
    regularized:bool=False,
    target_operators:Sequence[jnp.ndarray]=  None,
    num_samples:int = None,
) -> float:
    """Compute the objective function after updating an operator.

    Args:
        step_size (float): Gradient descent step size to be optimized.
        initial_isometry (jnp.ndarray): Initial operator on the isometry manifold.
        update_direction (jnp.ndarray): Tangent vector corresponding to the update direction.
        tensor_shape (tuple[int]): Original shape of the operator tensor.
        operator_type (str): The name of the operator ('povm', 'state', 'kraus').
        kraus_tensor (jnp.ndarray, optional): Kraus tensor of shape (num_gates, kraus_rank, dim_out, dim_in).
        povm_psd (jnp.ndarray, optional): POVM tensor of shape (num_povm, rank_povm, dim).
        state_psd (jnp.ndarray, optional): State tensor of shape (dim, rank_state).
        indices_list (list[list[int]], optional): List of indices corresponding to gate sequences.
        prob_matrix (jnp.ndarray, optional): Probability matrix of shape (num_povm, num_gate_sequences).
        use_geodesic (bool, optional): Whether to use geodesic as a retraction.

    Returns:
        float: The cost function value at the updated operator tensor.
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
    isometries: jnp.ndarray,
    update_directions: jnp.ndarray,
    tensor_shape: tuple[int],
    operator_type: str,
    use_geodesic: bool = True) -> jnp.ndarray:
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

def is_isometry(x:jnp.ndarray)->bool:
    "check if `x` belongs to the stiefel manifold"
    return jnp.allclose(x.conj().T @ x, jnp.eye(x.shape[1]))

def is_in_tangent_space(x:jnp.ndarray, z:jnp.ndarray)->bool:
    """
    Checks if the matrix z is in the tangent space of isometry x 
    
    Checks tangent space condition x^H z + z^H x = 0
    Args:
        x: Stiefel matrix of dimensions (n, p)
        z: Any matrix of dimensions (n, p)
    """
    return jnp.allclose(x.conj().T @ z, - z.conj().T @ x)

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
    

def update_isometry_using_polar_decomposition(x:jnp.ndarray, z:jnp.ndarray, step_size:float = 1):
    """
    Retraction based on canonical polar decomposition of scipy. Uses the SVD decomposition to obtain the isometry corresponding to z.
    
    Args:
        x: The base point of the retraction
        z: Tangent vector at x, corresponding to the update direction. 
        step_size: The step size of the retraction
    Returns:
        The retracted matrix

    References:
        [1] https://page.math.tu-berlin.de/~mehl/papers/hmt1.pdf
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html
    """
    return jax.scipy.linalg.polar(x - step_size * z)[0]

def canonical_gradient(x: jnp.ndarray, z:jnp.ndarray)->jnp.ndarray:
    """ Compute the riemmanian gradient at the point x on the stiefel manifold using the canonical metric

    Args:
        x: The base point of the tangent space
        z: The euclidean gradient at x

    Returns:
        The riemannian gradient using the canonical metric
    """
    return z - x @ z.conj().T @ x


def project_onto_tangent_space(x: jnp.ndarray, z: jnp.ndarray)->jnp.ndarray:
    """ Project a matrix z onto the tangent space of the manifold at x

    Args:
        x: The base point of the tangent space
        z: The matrix to project onto the tangent space

    Returns:
        A matrix projected onto the tangent space of the manifold at x
    """
    return z - x @ symmetrize(x.T.conj() @ z)

def symmetrize(A:jnp.ndarray)->jnp.ndarray:
    """
    Symmetrize a matrix by projecting it onto the symmetric subspace.
    
    Args:
        A: square matrix to be symmetrized
    Returns:
        Symmetrized matrix
    """
    return 0.5 * (A + A.T.conj())

def tensor_to_isometry(tensor: jnp.ndarray, n:int, p:int)-> jnp.ndarray:
    """ Reshape a tensor into an isometry matrix of dimensions n and p

    Args:
        x: tensor to be reshaped
        row_dim: Row dimension of the new matrix
        col_dim: Column dimension fo the new matrix

    Returns:
        Matrix of dimensions (row_dim, col_dim)
    """
    return jnp.reshape(tensor, shape=(n, p))

def isometry_to_tensor(isometry: jnp.ndarray, tensor_shape: tuple[int]) -> jnp.ndarray:
    """Reshape the updated isometry based on the operator type.

    Args:
        isometry (jnp.ndarray): The updated isometry tensor.
        tensor_shape (tuple[int]): The target shape for reshaping.

    Returns:
        jnp.ndarray: The reshaped tensor.
    """
    reshaped_tensor = jnp.reshape(isometry, shape=tensor_shape)
    return reshaped_tensor

def euclidean_gradients_to_stiefel(gradient_tensor: jnp.ndarray, operator_tensor: jnp.ndarray, operator_type:str="kraus", metric:Literal["canonical", "euclidean"] = "canonical")-> tuple[jnp.ndarray, jnp.ndarray]:
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
    
    if operator_type == "kraus":
        _, rank_kraus, dim, _ = operator_tensor.shape # num_gates, kraus_rank, dim_out, dim_in
        n = rank_kraus * dim
        p = dim
        
        kraus_isometries = []
        gradients_stiefel = []
        
        for gradient, kraus in zip(gradient_tensor, operator_tensor):
            
            gradient_stiefel, kraus_stiefel = gradient_and_operator_tensors_to_stiefel(gradient = gradient, operator = kraus, n=n, p=p, metric=metric)
            
            gradients_stiefel.append(gradient_stiefel)
            kraus_isometries.append(kraus_stiefel)
                
        return jnp.array(gradients_stiefel), jnp.array(kraus_isometries)
    
    if operator_type == "state":
        dim, rank_state = operator_tensor.shape # dim, rank_state
        n = dim * rank_state
        p = 1
        
    elif operator_type == "povm":
        num_povm, rank_povm, dim = operator_tensor.shape # num_povm, rank_povm, dim
        n = num_povm * rank_povm
        p = dim
    else:
        raise ValueError(f"Operator name {operator_type} is not recognized. Please use one of the following: 'kraus', 'state', 'povm'")
        
    return gradient_and_operator_tensors_to_stiefel(gradient = gradient_tensor, operator = operator_tensor, n=n, p=p, metric=metric)

        
def gradient_and_operator_tensors_to_stiefel(gradient:jnp.ndarray, operator:jnp.ndarray, n:int, p:int, metric:Literal["canonical", "euclidean"] = "canonical")->tuple[jnp.ndarray, jnp.ndarray]:
    """ Convert a pair of gradient and operator tensors to the Stiefel manifold

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

def update_isometry_tensors(isometries:jnp.ndarray, update_directions:jnp.ndarray, step_size:float, operator_type:str = "kraus", use_geodesic:bool =  True) ->  jnp.ndarray:
    """Update a tensor of isometries in the direction of the tensor of tangent vectors scaled by step size
    
    In order to retract back to the stiefel manifold we either follow the geodesic or use a first order retraction.
    
    Args:
        isometries: The isometries to be updated. Dimensions are (n, p) or (num_gates, n, p) for kraus operators
        tangent_vectors: The tangent vectors at the point x on the stiefel manifold. 
            Must have the same dimensions as isometries.
        step_size: The step size of the update.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to True.
    """
    
    if operator_type!= "kraus":
        isometries = [isometries] # for povm and state isometries we only have one isometry
        update_directions = [update_directions]
        
    new_isometries = []
    for isometry, vector in zip(isometries, update_directions):
        if use_geodesic:
            new_isometry = update_isometry_using_geodesic(x=isometry, z=vector, step_size=step_size)
        else:
            new_isometry = update_isometry_using_polar_decomposition(x = isometry, z = vector, step_size=step_size)
            
        new_isometries.append(new_isometry)
        
    if operator_type != "kraus":
        return new_isometries[0]
    
    return jnp.array(new_isometries)
    

def update_isometry_using_geodesic(x: jnp.ndarray, z: jnp.ndarray, step_size: float = 1) -> jnp.ndarray:
    """Compute a new point following the geodesic for a single isometry
    
    Source: Eq. 27 of https://arxiv.org/pdf/2112.05176

    Args:
        x: Current isometry of dimension (n, p)
        z: Element of the tangent space at x corresponding to the update direction.
            For instance, the riemannian gradient of the cost function. Dimensions are (n, p)
        step_size: Geodesic curve parameter
    Returns:
        x_new: New position given by x_new = g(a) with g(a) being a geodesic with g(0) = x, [dg/dt](0) = z
    """
    
    n, p = x.shape
    dim = p
    
    Q, R = jnp.linalg.qr((jnp.eye(n) - x @ x.T.conj()) @ z)
    
    # Construct AR_mat directly using jnp.block
    AR_mat = jnp.block([
        [x.T.conj() @ z, -R.T.conj()],
        [R, jnp.zeros((dim, dim), dtype=jnp.complex128)]
    ])
    
    MN = eigy_expm_jax(-step_size * AR_mat) @ jnp.eye(2 * dim, dim)
    
    return x @ MN[:dim, :] + Q @ MN[dim:, :]
    
def eigy_expm_jax(A:jnp.ndarray):
    """Custom Matrix exponential using the eigendecomposition of jax.linalg

    Args:
        A: Matrix to be exponentiated

    Returns:
        Matrix exponential of A
    """
    eigvals, eigvects = jnp.linalg.eig(A)
    return jnp.einsum("...ik, ...k, ...kj -> ...ij", eigvects, jnp.exp(eigvals), jnp.linalg.inv(eigvects))
        
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

def riemannian_connection(x:jnp.ndarray, w_x:jnp.ndarray, z:jnp.ndarray, Dw_in_z_at_x:jnp.ndarray, alpha0:float=1, alpha1:float=0.5)-> jnp.ndarray:
    """
    General parametrized riemannian connection of tangent spaces

    From equation 5.4 of https://arxiv.org/abs/2009.10159
    
    Args:
        x: Base point isometry of the tangent space
        w_x: Riemannian vector field evaluated at x.
        z: Riemannian tangent vector equivalent to the "direction" of the derivative
        Dw_in_z_at_x: Riemannian derivative of the vector field w in the direction of z evaluated at x
        alpha0: First parameter of the riemannian connection (see equation 5.4)
        alpha1: Second parameter of the riemannian connection (see equation 5.4)
    Returns:
        The riemannian connection of the vector field w in the direction of z at x
    """
    In = jnp.eye(x.shape[0])
    return Dw_in_z_at_x + 0.5 * x @ (z.conj().T @ w_x + w_x.conj().T @ z) + ((alpha0-alpha1)/alpha0)*(In - x @ x.conj().T) @ (z @ w_x.conj().T + w_x @ z.conj().T) @ x
    
def riemannian_metric(z1:jnp.ndarray, z2:jnp.ndarray, x:jnp.ndarray = None, metric:str = "euclidean")-> jnp.ndarray:
    """
    Compute the riemannian metric on the stiefel manifold at the point x for two tangent vectors z1 and z2.
    
    Args:
        z1: First tangent vector of dimensions (n, p)
        z2: Second tangent vector of dimensions (n, p)
        x: Point on the stiefel manifold of dimensions (n, p)
        metric: The type of the metric to use ('euclidean' or 'canonical'). Defaults to 'euclidean'.
    
    Returns:
        The inner product of the two tangent vectors at the point x
    """
    n, p = z1.shape
    if metric == "euclidean":
        gamma = jnp.eye(n)
    elif metric == "canonical":
        if x is None:
            raise ValueError("To use the canonical metric, the point x on the stiefel manifold must be provided.")
        gamma = jnp.eye(n) - 0.5 * (x@x.conj().T)
    else:
        raise ValueError(f"Metric: {metric} is not recognized. Please use one of the following: 'euclidean', 'canonical'")
    return jnp.trace(z1.conj().T @ gamma @ z2).real

def random_tangent_vector(x:jnp.ndarray, n:int, p:int, seed=42)->jnp.ndarray:
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
    real_part = jax.random.normal(key_real, (n, p))
    imag_part = jax.random.normal(key_imag, (n, p))

    # Combine into a complex matrix
    z = real_part + 1j * imag_part
    return project_onto_tangent_space(x=x, z=z)

def truncated_cg(rgrad:jnp.ndarray, rhess_vect_fn:Callable, radius:float, isometry:jnp.ndarray = None, metric:str = "euclidean", **kwargs):
    """
    Truncated CG (tCG) method for the trust-region subproblem:
        minimize   <grad, z> + 1/2 <z, H z>
        subject to <z, z> <= radius^2
        
    Args:
        rgrad: Riemannian gradient matrix at the current point
        rhess_vect_fn: Function that computes the Hessian-vector product and returns a matrix in the tangent space.
            This should be a function that takes as single input a tangent vector.
        radius: Trust region radius
        **kwargs: Additional keyword arguments:
            maxiter: Maximum number of iterations (default: 2 * len(rgrad))
            abstol: Absolute tolerance for stopping criterion (default: 1e-8)
            reltol: Relative tolerance for stopping criterion (default: 1e-6)

    References:
      - Algorithm 11 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
      - Trond Steihaug
        The conjugate gradient method and trust regions in large scale optimization
        SIAM Journal on Numerical Analysis 20, 626-637 (1983)
    """
    maxiter = kwargs.get("maxiter", 2 * len(rgrad))
    abstol  = kwargs.get("abstol", 1e-8)
    reltol  = kwargs.get("reltol", 1e-6)
    r_vector = rgrad.copy()
    
    rsq = riemannian_metric(r_vector, r_vector, x=isometry, metric=metric)
    stoptol = max(abstol, reltol * jnp.sqrt(rsq))
    z_vector = jnp.zeros_like(r_vector)
    z_vectors = [z_vector]
    delta_vector = -r_vector
    for iter in range(1, maxiter+1):
        Hessian_delta = rhess_vect_fn(delta_vector)
        # Hessian_delta = hess @ delta_vector
        delta_Hessian_delta = riemannian_metric(delta_vector, Hessian_delta, x=isometry, metric=metric)
        t = _move_to_boundary(z_vector, delta_vector, radius, isometry=isometry, metric=metric)
        alpha = rsq / delta_Hessian_delta
        if delta_Hessian_delta <= 0 or alpha > t:
            # return with move to boundary
            print(f"Finished successfully after iters: {iter} with alpha={alpha}, delta_Hessian_delta={delta_Hessian_delta}, t={t}")
            z_vector += t * delta_vector
            z_vectors.append(z_vector)
            return z_vectors, True
        # update iterates
        r_vector += alpha * Hessian_delta
        z_vector += alpha * delta_vector
        z_vectors.append(z_vector)
        rsq_next = riemannian_metric(r_vector, r_vector, x=isometry, metric=metric)
        if jnp.sqrt(rsq_next) <= stoptol:
            # early stopping
            print(f"Finished early after iters: {iter}")
            return z_vectors, False
        beta = rsq_next / rsq
        delta_vector = -r_vector + beta * delta_vector
        rsq = rsq_next
    # maxiter reached
    print(f"Did not converge after max iterations: {maxiter}")
    return z_vectors, False

def _move_to_boundary(eta_j:jnp.ndarray, delta_j:jnp.ndarray, radius:float, isometry:jnp.ndarray = None, metric:str = "euclidean")-> float:
    """
    Move to the unit ball boundary by solving
    ||eta_sol|| = || eta_j + t * delta_j || == radius
    for t with t > 0.
    """
    dsq = riemannian_metric(delta_j, delta_j, x=isometry, metric=metric)
    if jnp.allclose(dsq, 0):
        warnings.warn("tangent vector 'delta_j' has norm zero")
        return 0 # t =0 such that the next iteration is the same eta_j
    p = riemannian_metric(eta_j, delta_j, x=isometry, metric=metric) / dsq
    q = (riemannian_metric(eta_j, eta_j, x=isometry, metric=metric) - radius**2) / dsq
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn("encountered t < 0")
    return t


def solve_quadratic_equation(p:float, q:float)->tuple[float, float]:
    """
    Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
    
    The solution should be  -p ± sqrt(p**2 - q).
    
    Args:
        p: Coefficient of the linear term (half of the coefficient of x).
        q: Constant term of the quadratic equation.
    
    Returns:
        A tuple containing the two solutions of the quadratic equation, (negative, positive).
        
    Raises:
        ValueError: If the discriminant is negative, i.e., p**2 - q < 0.
    """
    if (p**2 - q) < 0:
        raise ValueError("require non-negative discriminant")
    if jnp.isclose(p, 0):
        x = jnp.sqrt(-q)
        return (-x, x)
    x1 = -(p + jnp.sign(p)*jnp.sqrt(p**2 - q))
    x2 = q / x1
    return tuple(sorted((x1, x2)))

def riemannian_gradient_fn_povm(x:jnp.ndarray, kraus_tensor:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray, metric:str = "canonical", )->jnp.ndarray:
    """Calculate the riemannian gradient of the cost function wrt to the POVM tensor
    
    Args:
        x: The POVM tensor to calculate the gradient at.
        metric: The metric to use for the gradient calculation. Can be "canonical" or "euclidean".
    Returns:
        The riemannian gradient (tensor) of the cost function wrt to the POVM tensor.
    """
    gradient_ambient_jax = gradient_povm_mps_jit(
        kraus_tensor, x, state_psd,
        indices_list, prob_matrix) # 2df/dx
    
    gradient_ambient = gradient_ambient_jax.conj()/2 # df/dx*
    gradient_stiefel_matrix, _ = euclidean_gradients_to_stiefel(
        gradient_tensor=gradient_ambient,
        operator_tensor=x, operator_type="povm", metric=metric,
    ) # Riemannian gradient
    return gradient_stiefel_matrix.reshape(x.shape)

def vhp(function:callable, x:jnp.ndarray, z:jnp.ndarray)-> tuple[jnp.ndarray, jnp.ndarray]:
    function_at_x, vjp_function = jax.vjp(function, x)
    (vjp_vector, ) = vjp_function(z)
    return function_at_x, vjp_vector

def hvp(function:callable, x:jnp.ndarray, z:jnp.ndarray)-> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Hessian-vector product using JAX's jvp function.
    Args:
        function: The function for which to compute the Hessian-vector product.
        x: The point at which to evaluate the function and its gradient.
        z: The vector with which to compute the Hessian-vector product.
    Returns:
        A tuple containing the function value at x and the Hessian-vector product.
    """
    return jax.jvp(function, (x,), (z,))

def riemannian_hessian_vector_povm_jax(tangent_vector:jnp.ndarray, kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, indices_list:list[list[int]], prob_matrix:jnp.ndarray, metric:str="canonical")->jnp.ndarray:
    """Compute the Riemannian Hessian-vector product for the POVM tensor using JAX.
    
    Args:
        povm_psd: POVM factor tensor of shape (num_povm, povm_rank, dim).
        tangent_vector: Tangent vector determining direction of covariant derivative of shape (num_povm, povm_rank, dim).
        metric: The metric to use for the Hessian-vector product. Can be "canonical" or "euclidean".
    """

    num_povm, povm_rank, dim = povm_psd.shape
    n = num_povm * povm_rank
    p = dim
    riemannian_gradient_function = lambda x: riemannian_gradient_fn_povm(
        x, kraus_tensor=kraus_tensor, state_psd=state_psd, indices_list=indices_list, prob_matrix=prob_matrix, metric=metric) # Riemannian gradient (from df/dx*)
    # NOTE: here, we should use df/dx* as this is the actual gradient. See Corollary 4.0.1. of An introduction to complex differentials and complex differentiability - Hunger.
    grad_at_x_tensor, Dgrad_to_z_tensor = hvp(function=riemannian_gradient_function, x=povm_psd, z=tangent_vector)
    grad_at_x_matrix = tensor_to_isometry(grad_at_x_tensor, n, p)
    Dgrad_to_z_matrix = tensor_to_isometry(Dgrad_to_z_tensor, n, p)
    povm_psd_matrix = tensor_to_isometry(povm_psd, n, p)
    tangent_vector_matrix = tensor_to_isometry(tangent_vector, n, p)
    
    if metric == "euclidean":
        alpha0, alpha1 = 1, 1
    elif metric == "canonical":
        alpha0, alpha1 = 1, 0.5
    else:
        raise ValueError("Metric must be either 'euclidean' or 'canonical'")
    return riemannian_connection(x = povm_psd_matrix, w_x=grad_at_x_matrix, z=tangent_vector_matrix, Dw_in_z_at_x=Dgrad_to_z_matrix, alpha0=alpha0, alpha1=alpha1)

def riemannian_trust_region_optimize(f, retract, gradfunc, hessfunc, x_init, save_x=False, check_convergence:bool=False, show_quotient:bool=True, **kwargs):
    """
    Optimization via the Riemannian trust-region (RTR) algorithm.

    Reference:
        Algorithm 10 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)

    args:
    ---------
    f: real valued function representing the optimization problem.
        it should accept as single input a list of elements of the manifold to optimze over.
    retract:
        retraction from tanget space at x to original manifold.
        signature: x_list:list of elements of manifold, eta: array containing the parametrization of the tangent elements

    returns:
    ---------
    x_iter:
        if `save_x = True`, it is a list with all the x's used in the `niter` iterations (list of lists)
        else, it returns the last x. Note that the last x is not used to compute f(x).
    f_iter:
        evaluation of the cost function across the `niter` iterations
    g_iter:
        evaluation of the error function. Not used
    radius:
        last radius used in the trust region algorithm
    """
    rho_trust   = kwargs.get("rho_trust", 0.125)
    radius_init = kwargs.get("radius_init", 0.01)
    maxradius   = kwargs.get("maxradius",   0.1)
    niter       = kwargs.get("niter", 20)
    gfunc       = kwargs.get("gfunc", None)
    tol         = kwargs.get("tol", 1e-10)
    # transfer keyword arguments for truncated_cg
    tcg_kwargs = {}
    for key in ["maxiter", "abstol", "reltol"]:
        if ("tcg_" + key) in kwargs.keys():
            tcg_kwargs[key] = kwargs["tcg_" + key]
    assert 0 <= rho_trust < 0.25
    x = x_init
    radius = radius_init
    f_iter = []
    g_iter = []
    x_iter = [x]

    if gfunc is not None:
        g_iter.append(gfunc(x))
    try:
        for k in range(niter):
            print(f'iteration: {k}')
            grad = gradfunc(x)
            hess = hessfunc(x)
            eta, on_boundary = truncated_cg(grad, hess, radius, **tcg_kwargs)
            x_next = retract(x, eta)
            fx = f(x)
            f_iter.append(fx)
            print(f'f(x{k}): {fx}')
            # Eq. (7.7)
            rho = (f(x_next) - fx) / (np.dot(grad, eta) + 0.5 * np.dot(eta, hess @ eta))
            if rho < 0.25:
                # reduce radius
                radius *= 0.25
            elif rho > 0.75 and on_boundary:
                # enlarge radius
                radius = min(2 * radius, maxradius)
            if show_quotient:
                print('rho:', rho, 'updated radius:', radius)
            if rho > rho_trust:
                x = x_next
            if gfunc is not None:
                g_iter.append(gfunc(x))
            if save_x or k == (niter - 1): # if save = False, this will save only the last iteration
                x_iter.append(x)
            if check_convergence:
                if np.abs(f_iter[-1]-f_iter[-2])/f_iter[-1] <= tol:
                    print(f"optimization converged prematurely at iteration: {k}")
                    break
        return x_iter, f_iter, radius # x_iter will have 1 more element f_iter
    except KeyboardInterrupt:
        print(f"optimization was stopped prematurely at iteration: {k}")
        return x_iter, f_iter, radius
