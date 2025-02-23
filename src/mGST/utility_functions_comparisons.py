# Needs iqm-benchmarks from the github repo to access all the mGST functions: https://github.com/iqm-finland/iqm-benchmarks
from mGST import additional_fns
from iqm.benchmarks.compressive_gst.compressive_gst import GSTConfiguration, CompressiveGST
from iqm.benchmarks.compressive_gst.gst_analysis import dataset_counts_to_mgst_format

from mGST.qiskit_interface import qiskit_gate_to_operator
from mGST.low_level_jit import gradient_all_3_and_value_jit, cost_function_jax_mps

from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from qiskit.circuit.library import CZGate, RGate

from scipy.optimize import minimize

import jax.numpy as jnp
import jax

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

from mGST.algorithm import gd

def get_x_from_k(k, depth=None, dim_squared=None):
    if not depth or not dim_squared:
        depth = k.shape[0]
        dim_squared = k.shape[-1]**2
    return jnp.einsum("ijkl,ijnm -> iknlm", k, k.conj()).reshape((depth, dim_squared, dim_squared))

def compute_new_x(K, E, rho, y, J, d, r, rK, fixed_gates, gds_kwargs={}):
    K_gds = gd(K, E, rho, y, J, d, r, rK, fixed_gates=fixed_gates, ls="COBYLA", **gds_kwargs)
    return get_x_from_k(k=K_gds, depth=d, dim_squared=r)

from mGST.low_level_jit import objf

def run_simple_gds_on_gates(K0, E, rho, y, J, d, r, rK, fixed_gates, max_iter=200, gds_kwargs={}, threshold_multiplier=3, target_rel_prec=1e-4):
    """Function emulating what run_mGST does, but focusing only on the optimization of the gates using GDS.
    """
    # n_povm = E.shape[0]
    # delta = threshold_multiplier * (1 - y.reshape(-1)) @ y.reshape(-1) / len(J) / n_povm / shots
    
    X0 = get_x_from_k(K0, d, r)
    cost_function_history = [objf(X0, E, rho, J, y)]
    
    Ki = K0
    
    for i in range(max_iter):
        print('iteration: ', i)
        print('cost: ', cost_function_history[-1])
        Ki = gd(Ki, E, rho, y, J, d, r, rK, fixed_gates=fixed_gates, ls="COBYLA", **gds_kwargs)
        Xi = get_x_from_k(Ki, d, r)
        cost_function_history.append(objf(Xi, E, rho, J, y))
        
        if jnp.abs(cost_function_history[-2] - cost_function_history[-1])/cost_function_history[-2] < target_rel_prec:
            print('Success threshold reached prematurely.')
            break
        
    return Ki, cost_function_history

def run_simple_gds_on_gates_jax(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, max_iter:int=200, target_rel_prec=1e-3, step_size:float=1, optimize_step:bool=True, use_geodesic:bool=True):
    """Run a simple gradient descent optimization on the gates using JAX.

    Args:
        kraus0: Kraus tensor to start the optimization from. Dimensions: (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, dim, rank_povm)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        max_iter: Max number of iterations to run GDS for. Defaults to 200.
        target_rel_prec: Relative precision used to decide whether to terminate optimization early. Defaults to 1e-3.
        step_size: Step size used throughought the optimization. Use only if line search is not desired. Defaults to 1.
        optimize_step: Whether to optimize the step_size using line search or not. Defaults to True.

    Returns:
        krausi: Optimized Kraus tensor.
        cost_function_history: History of the cost function values at each iteration.
    """
    
   
    cost_function_history = [] # the cost function will be evaluated in the first step
    
    kraus_i = kraus_tensor
    state_i = state_psd
    povm_i = povm_psd
    opt_step_size = step_size
    
    try:
        for i in range(max_iter):
            print('iteration: ', i)

            kraus_i, povm_i, state_i, opt_step_size, cost_i = gradient_descent_step(kraus_i, povm_i, state_i, indices_list, prob_matrix, ls_method="COBYLA", ls_max_iter=20, optimize_step=optimize_step, step_size=opt_step_size, use_geodesic=use_geodesic)
                
            cost_function_history.append(cost_i)
            print('cost: ', cost_function_history[-1])
            
            if i > 1 and jnp.abs(cost_function_history[-2] - cost_function_history[-1])/cost_function_history[-2] <   target_rel_prec:
                print('Success threshold reached prematurely.')
                break
    except KeyboardInterrupt:
        print(f"Optimization was stopped prematurely at iteration: {i}")
        return kraus_i, cost_function_history
    
    return kraus_i, povm_i, state_i, cost_function_history

def gradient_descent_step(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix, ls_method="COBYLA", ls_max_iter=200, optimize_step:bool=True, step_size:float=1, use_geodesic:bool=True)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """Perform a gradient descent step on the Kraus operators

    Args:
        kraus_tensor: The current Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        povm_psd: Positive-semidefinite (PSD) root of the POVM tensor of dimensions: (num_povm, dim, rank_povm)
        state_psd: Positive-semidefinite (PSD) root of the state tensor of dimensions: (dim, rank_state)
        indices_list: list of length num_gate_sequences, where each elements is a list of indices corresponding to a gate sequence.
        prob_matrix: tensor of dimensions (num_povm, num_gate_sequences)
        ls_method: Method to use in line search optimization. Defaults to "COBYLA".
        ls_max_iter: Max number of iterations used in line search. Defaults to 200.
        optimize_step: Whether to optimize the step size using line search. Defaults to True.
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
    # cost_value, ambient_grad_kraus = gradient_k_and_value_jit(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    cost_value, ambient_gradients = gradient_all_3_and_value_jit(kraus_tensor, povm_psd, state_psd, indices_list, prob_matrix)
    ambient_grad_kraus, ambient_grad_povm, ambient_grad_state = ambient_gradients
    
    # We would want to project 2 * (conjugate_wirtinger_derivative)
    # However, JAX returns already 2 * wirtinger_derivative, so we just need to take the conjugate
    
    # Kraus tensor to stiefel
    kraus_stiefel_gradient, kraus_isometries =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_kraus.conj(), operator_tensor=kraus_tensor, operator_name="kraus") # num_gates, rank_kraus*dim, dim
    
    # State tensor to stiefel
    state_stiefel_gradient, state_isometry =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_state.conj(), operator_tensor=state_psd, operator_name="state") # dim*rank_state, 1
    
    # POVM tensor to stiefel
    povm_stiefel_gradient, povm_isometry =  euclidean_gradients_to_stiefel(gradient_tensor=ambient_grad_povm.conj(), operator_tensor=povm_psd, operator_name="povm") # num_povm*rank_povm, dim
    previous_povm_shape = povm_psd.shape[0], povm_psd.shape[2], povm_psd.shape[1] # num_povm, rank_povm, dim
    
    
    if optimize_step:
        
        stiefel_gradients = (kraus_stiefel_gradient, povm_stiefel_gradient, state_stiefel_gradient)
        stiefel_isometries = (kraus_isometries, povm_isometry, state_isometry)
        shapes_of_tensors = (kraus_tensor.shape, previous_povm_shape, state_psd.shape)
        
        optimization_result = minimize(cost_function_from_updated_isometries, step_size, args=(stiefel_gradients, stiefel_isometries, indices_list, prob_matrix, shapes_of_tensors, use_geodesic), method=ls_method, options={"maxiter": ls_max_iter})
        
        step_size = optimization_result.x
        print('optimized step size: ', step_size)
            
    # update kraus tensor
    kraus_isometries_updated = update_isometries(isometries = kraus_isometries, riemannian_gradients = kraus_stiefel_gradient, step_size = step_size, operator_name="kraus", use_geodesic=use_geodesic)    
    new_kraus_tensor = jnp.reshape(kraus_isometries_updated, shape=kraus_tensor.shape)
    
    # update state
    state_isometry_updated = update_isometries(isometries = state_isometry, riemannian_gradients = state_stiefel_gradient, step_size = step_size, operator_name="state", use_geodesic=use_geodesic)
    new_state_psd = jnp.reshape(state_isometry_updated, shape=state_psd.shape)
    
    # update povm
    povm_isometry_updated = update_isometries(isometries = povm_isometry, riemannian_gradients = povm_stiefel_gradient, step_size = step_size, operator_name="povm", use_geodesic=use_geodesic)
    new_povm_psd = jnp.reshape(povm_isometry_updated, shape=previous_povm_shape) # num_povm, rank_povm, dim
    new_povm_psd = jnp.transpose(new_povm_psd, axes=(0, 2, 1)) # num_povm, dim, rank_povm
    
    return new_kraus_tensor, new_povm_psd, new_state_psd, step_size, cost_value

def cost_function_from_updated_isometries(step_size:float, stiefel_gradients:tuple[jnp.ndarray], stiefel_isometries:tuple[jnp.ndarray], indices_list:list[list[int]], prob_matrix:jnp.ndarray, shapes_of_tensors:tuple[tuple[int]], use_geodesic:bool=False)->float:
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
    
    kraus_gradients, povm_gradient, state_gradient = stiefel_gradients
    kraus_isometries, povm_isometry, state_isometry = stiefel_isometries
    previous_kraus_shape, previous_povm_shape, previous_state_shape = shapes_of_tensors
    
    # update kraus tensor
    kraus_isometries_updated = update_isometries(isometries = kraus_isometries, riemannian_gradients = kraus_gradients, step_size = step_size, operator_name="kraus", use_geodesic=use_geodesic)    
    new_kraus_tensor = jnp.reshape(kraus_isometries_updated, shape=previous_kraus_shape) # num_gates, rank_kraus, dim, dim
    
    # update state
    state_isometry_updated = update_isometries(isometries = state_isometry, riemannian_gradients = state_gradient, step_size = step_size, operator_name="state", use_geodesic=use_geodesic)
    new_state_psd = jnp.reshape(state_isometry_updated, shape=previous_state_shape)
    
    # update povm
    povm_isometry_updated = update_isometries(isometries = povm_isometry, riemannian_gradients = povm_gradient, step_size = step_size, operator_name="povm", use_geodesic=use_geodesic)
    new_povm_psd = jnp.reshape(povm_isometry_updated, shape=previous_povm_shape)
    new_povm_psd = jnp.transpose(new_povm_psd, axes=(0, 2, 1)) # num_povm, dim, rank_povm
    
    return cost_function_jax_mps(new_kraus_tensor, new_povm_psd, new_state_psd, indices_list, prob_matrix, jit=True)

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


def factorize_psd_truncated(psd: jnp.ndarray, max_rank: int | None = None) -> jnp.ndarray:
    """
    Factorizes a batch of positive semi-definite (PSD) matrices by truncating singular values.

    More robust to small values than cholesky decomposition from numpy.

    Returns x' such that psd ≈ x' @ x'.conj().T
    
    Args:
    psd (jnp.ndarray): Input tensor of shape (..., N, N) (must be Hermitian).
    max_rank (int, optional): Maximum number of singular values to keep.
    
    Returns:
        jnp.ndarray: The factorized matrix `x'` of shape (..., N, min(max_rank, N)).
    """
    if max_rank is None:
        max_rank = psd.shape[-1]  # Assume full rank by default
        
    x, s, = split_matrix_svd(psd, max_rank)
    
    return x * jnp.sqrt(s)[..., None, :]
    #  s[..., None, :] reshapes s into shape (..., 1, min(max_rank, N)), allowing elementwise multiplication with x ((..., N, min(max_rank, N))).

def update_isometry_using_polar_decomposition(x:jnp.ndarray, z:jnp.ndarray, step_size:float = 1):
    """
    Retraction based on canonical polar decomposition of scipy. Uses the SVD decomposition to obtain the isometry corresponding to z.
    
    Args:
        x: The base point of the retraction
        z: The matrix to retract
        step_size: The step size of the retraction
    Returns:
        The retracted matrix

    References:
        [1] https://page.math.tu-berlin.de/~mehl/papers/hmt1.pdf
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html
    """
    return jax.scipy.linalg.polar(x - step_size * z)[0]


def project_onto_tangent_space(x: jnp.ndarray, z: jnp.ndarray)->jnp.ndarray:
    """ Project a matrix z onto the tangent space of the manifold at x

    Args:
        x: The base point of the tangent space
        z: The matrix to project onto the tangent space

    Returns:
        A matrix projected onto the tangent space of the manifold at x
    """
    return 0.5 * (z - x @ z.conj().T @ x)

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


def euclidean_gradients_to_stiefel(gradient_tensor: jnp.ndarray, operator_tensor: jnp.ndarray, operator_name:str="kraus")-> tuple[jnp.ndarray, jnp.ndarray]:
    """ Project a sequence of tensor of Euclidean gradients onto the Stiefel manifold

    Args:
        gradient_tensor: The Euclidean gradient tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        operator_tensor: The tensor corresponding to either the Kraus, state or POVM operators.
    Returns:
        A tuple containing:
            An array of gradients projected onto the Stiefel manifold
            An array of stiefel isometries
    """
    
    if operator_name == "kraus":
        _, rank_kraus, dim, _ = operator_tensor.shape # num_gates, kraus_rank, dim_out, dim_in
        n = rank_kraus * dim
        p = dim
        
        kraus_isometries = []
        gradients_stiefel = []
        
        for gradient, kraus in zip(gradient_tensor, operator_tensor):
            
            gradient_stiefel, kraus_stiefel = gradient_and_operator_tensors_to_stiefel(gradient = gradient, operator = kraus, n=n, p=p)
            
            gradients_stiefel.append(gradient_stiefel)
            kraus_isometries.append(kraus_stiefel)
                
        return jnp.array(gradients_stiefel), jnp.array(kraus_isometries)
    
    elif operator_name == "state":
        dim, rank_state = operator_tensor.shape # dim, rank_state
        n = dim * rank_state
        p = 1
        
    elif operator_name == "povm":
        num_povm, dim, rank_povm = operator_tensor.shape # num_povm, dim, rank_povm
        n = num_povm * rank_povm
        p = dim
        operator_tensor = jnp.transpose(operator_tensor, axes=(0, 2, 1)) # num_povm, rank_povm, dim
        
    else:
        raise ValueError(f"Operator name {operator_name} is not recognized. Please use one of the following: 'kraus', 'state', 'povm'")
        
    return gradient_and_operator_tensors_to_stiefel(gradient = gradient_tensor, operator = operator_tensor, n=n, p=p)

        
def gradient_and_operator_tensors_to_stiefel(gradient:jnp.ndarray, operator:jnp.ndarray, n:int, p:int)->tuple[jnp.ndarray, jnp.ndarray]:
    """ Convert a pair of gradient and operator tensors to the Stiefel manifold

    We first reshape the operator and gradient tensors into isometries and 
    project the gradient onto the tangent space of the manifold at the operator.
    
    We assume the tensors are ordered in a way that we can group the first indices into a single
    index of dimension n and the rest into another one of dimension p.
    
    Args:
        gradient: The gradient tensor
        operator: The operator tensor

    Returns:
        A tuple containing:
            The gradient projected onto the Stiefel manifold
            The operator projected onto the Stiefel manifold
    """
    operator_stiefel = tensor_to_isometry(tensor=operator, n=n, p=p)
    gradient_np = tensor_to_isometry(tensor=gradient, n=n, p=p)
    
    return project_onto_tangent_space(x = operator_stiefel, z = gradient_np), operator_stiefel

def update_isometries(isometries:jnp.ndarray, riemannian_gradients:jnp.ndarray, step_size:float,operator_name:str = "kraus", use_geodesic:bool =  True) ->  jnp.ndarray:
    
    if operator_name!= "kraus":
        isometries = [isometries] # for povm and state isometries we only have one isometry
        riemannian_gradients = [riemannian_gradients]
        
    new_isometries = []
    for isometry, gradient in zip(isometries, riemannian_gradients):
        if use_geodesic:
            new_isometry = update_isometry_using_geodesic(x=isometry, z=gradient, step_size=step_size)
        else:
            new_isometry = update_isometry_using_polar_decomposition(x = isometry, z = gradient, step_size=step_size)
            
        new_isometries.append(new_isometry)
        
    if operator_name != "kraus":
        return new_isometries[0]
    
    return jnp.array(new_isometries)
    

def update_isometry_using_geodesic(x: jnp.ndarray, z: jnp.ndarray, step_size: float = 1) -> jnp.ndarray:
    """Compute a new point on the geodesic for a single isometry
    
    Args:
        x: Current isometry of dimension (n, p)
        z: Element of the tangent space at x corresponding to the riemannian gradient of the cost function. Dimensions are (n, p)
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
    
def eigy_expm_jax(A):
    """Custom Matrix exponential using the eigendecomposition of numpy.linalg

    Parameters
    ----------
    A : numpy array
        Matrix to be exponentiated

    Returns
    -------
    M: numpy array
        Matrix exponential of A
    """
    vals, vects = jnp.linalg.eig(A)
    return jnp.einsum("...ik, ...k, ...kj -> ...ij", vects, jnp.exp(vals), jnp.linalg.inv(vects))
        

