# Needs iqm-benchmarks from the github repo to access all the mGST functions: https://github.com/iqm-finland/iqm-benchmarks
from mGST import additional_fns
from iqm.benchmarks.compressive_gst.compressive_gst import GSTConfiguration, CompressiveGST
from iqm.benchmarks.compressive_gst.gst_analysis import dataset_counts_to_mgst_format

from mGST.qiskit_interface import qiskit_gate_to_operator

from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from qiskit.circuit.library import CZGate, RGate

import jax.numpy as jnp

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

def initialize_mgst_parameters(dataset, target_init = True):
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
        
        
        K = additional_fns.perturbed_target_init(X_target, dataset.attrs["rank"])
        X = jnp.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    else:
        K, X, E, rho = random_gs(d, r, rK, n_povm)
        
    return K, X, E, rho

def get_full_mgst_parameters_from_configuration(configuration:GSTConfiguration, backend):
    
    benchmark = CompressiveGST(backend, configuration)
    result = benchmark.run()
    
    rK = configuration.rank
    qubit_layout = configuration.qubit_layouts[0]
    dataset = result.dataset
    y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt = get_mgst_parameters_from_dataset(dataset, qubit_layout=qubit_layout, rK=rK)
    K, X, E, rho = initialize_mgst_parameters(dataset=dataset, target_init=True)
    
    return K, X, E, rho, y, J, l, d, pdim, r, n_povm, bsize, meas_samples, n, nt, rK

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