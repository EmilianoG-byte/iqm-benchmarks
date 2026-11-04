"""How much MVE is carried by the incoherent part of the true gates?

Replace every true gate by the unitary closest to its dominant Kraus operator (SPAM untouched) and
compute the MVE against the true gate set with the same settings as the notebook (length 10, 1000 circuits).
Also print the singular-value gap of the tangent-space Jacobian around the rank-1 model dimension.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set
from mGST.analysis import compute_average_and_std_variational_error, compute_average_and_std_shot_noise_variational_error
from mGST.gauge_analysis import min_num_parameters

noise_strength = float(sys.argv[1]) if len(sys.argv) > 1 else 5e-2
num_gates, kraus_rank, dim = 6, 4, 4

target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=noise_strength, seed=42)
true_superops = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, _, _ = get_compressed_rep_from_mgst_output(*true_superops.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim)
K = np.asarray(K)

unitaries = []
for g in range(num_gates):
    choi_vecs = K[g].reshape(kraus_rank, -1)  # rows = vec(K_i)
    _, _, vh = np.linalg.svd(choi_vecs, full_matrices=False)
    dominant = vh[0].conj().reshape(dim, dim)  # top Choi eigenvector (up to phase/scale)
    w, _, v = np.linalg.svd(dominant)
    unitaries.append(w @ v)  # polar factor -> closest unitary
coherent_kraus = np.array(unitaries)[:, None]  # (num_gates, 1, dim, dim)

coherent_superops = dict(true_superops)
coherent_superops["kraus"] = kraus_tensor_to_mgst(coherent_kraus)

mve_coh = compute_average_and_std_variational_error(true_superops=true_superops, gst_superops=coherent_superops, length=10, samples=1000, num_runs=10)
target_like = dict(true_superops)
target_like["kraus"] = kraus_tensor_to_mgst(target_operators["kraus"])
mve_target = compute_average_and_std_variational_error(true_superops=true_superops, gst_superops=target_like, length=10, samples=1000, num_runs=10)
mve_shots = compute_average_and_std_shot_noise_variational_error(true_superops=true_superops, num_shots=1000, length=10, num_circuits_sampled=1000, num_repetitions=10)

print(f"noise_strength = {noise_strength}")
print(f"MVE(true vs. true with gates replaced by closest unitaries): {float(mve_coh[0]):.4f}")
print(f"MVE(true vs. true SPAM + ideal target gates):               {float(mve_target[0]):.4f}")
print(f"MVE(shot noise only, 1000 shots):                           {float(mve_shots[0]):.4f}")
print("rank-1 model (unitary gates, pure state, rank-1 POVM) min_num_parameters:",
      min_num_parameters(num_qubits=2, num_gates=num_gates, kraus_rank=1, state_rank=1, povm_rank=1))
