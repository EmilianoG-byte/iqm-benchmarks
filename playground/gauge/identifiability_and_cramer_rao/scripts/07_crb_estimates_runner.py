"""Run the notebook's maximum-likelihood refinement on R independent shot-noise datasets (same design) and save the estimates.

Usage: python 07_crb_estimates_runner.py <first_realization> <last_realization_exclusive> <tag> <absolute_tolerance> <num_iterations_outer> [<relative_precision>]

Settings used for the saved data (data/crb_estimates):
    python 07_crb_estimates_runner.py 0 40 default 1 20            # about 22 s per realization
    python 07_crb_estimates_runner.py 0 20 converged 1e-4 60 1e-8  # 39-77 s per realization
"""
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst, get_initial_gate_set_for_optimization
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices, compute_probability_matrices, run_optimization_workflow
from mGST.typing import TrustRegionOptions, OperatorSchedule

tag, absolute_tolerance, num_outer = sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
relative_precision = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
OUT = Path(__file__).resolve().parent.parent / "data" / "crb_estimates"
OUT.mkdir(parents=True, exist_ok=True)
num_gates, kraus_rank, dim, num_shots, N = 6, 4, 4, 1000, 400

target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=5e-2, seed=42)
noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = get_compressed_rep_from_mgst_output(*noisy_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim)
true_psd = {"kraus": K, "povm": A, "state": B}
target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
init = get_initial_gate_set_for_optimization(target_superops=target_superops, kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim,
                                             spam_depolarization=1e-2, project_kraus_to_isometry=True)

tr_options = TrustRegionOptions(num_iterations=5, verbose=False, verbose_cg=False, num_iterations_cg=10, tol_grad=1e-8, radius_heuristic="typical_distance")
schedule = OperatorSchedule.default(num_outer, tr_options)
options = {"schedule": {op: schedule for op in ["povm", "kraus", "state"]}, "num_iterations_outer": num_outer, "relative_precision": relative_precision,
           "global_gradient_norm_tol": 1e-10, "noise_threshold": None, "compute_least_squares": 1, "optimization_verbose": False, "absolute_tolerance": absolute_tolerance}

indices_dict = generate_sequence_indices(num_gates=num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)
for r in range(int(sys.argv[1]), int(sys.argv[2])):
    t0 = time.time()
    probability_matrices = compute_probability_matrices(K, A, B, gate_indices=indices_dict["gate_indices"], num_shots=num_shots, seed=1000 + r)
    result = run_optimization_workflow(num_shots=num_shots, init_operators=init, operators_psd_true=true_psd, target_superops=target_superops,
                                       optimization_options=options, use_log_likelihood=True, indices_dict=indices_dict,
                                       probability_matrices=probability_matrices, gauge_optimize=False)
    op = result["optimized_operators"]
    np.savez(OUT / f"{tag}_{r}.npz", kraus=np.asarray(op["kraus"]), povm=np.asarray(op["povm"]), state=np.asarray(op["state"]),
             sampled=np.asarray(probability_matrices["sampled"]), final_cost=float(result["cost_fn_history"]["main"][-1]),
             reason=str(result["convergence_reason"]))
    print(f"realization {r}: {time.time() - t0:.0f}s, final cost {float(result['cost_fn_history']['main'][-1]):.6e}, {result['convergence_reason']}", flush=True)
