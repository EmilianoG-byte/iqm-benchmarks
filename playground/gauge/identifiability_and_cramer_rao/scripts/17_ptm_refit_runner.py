"""Test T3 of PTM_PRECISION.md, part 1: refit independent shot-noise datasets at one candidate design and save the
estimates. Script 07 with the truth and the design as arguments.

Usage (one design; realizations first..last-1 can be split over several processes):
    python 17_ptm_refit_runner.py --N 1000 --m 1100 --L-max 23 --first 0 --last 40 --dry-run
    python 17_ptm_refit_runner.py --N 1000 --m 1100 --L-max 23 --first 0 --last 40

Truth: target gates followed by random noise channels (`--noise-strength`, `--noise-seed`; 0.3 and 42 are the
notebook's truth). Design: generate_sequence_indices(..., seq_len_list=[1, L_max // 2 + 1, L_max], seed=--design-seed).
Shot noise of realization r: seed 1000 + r, as in script 07. Optimizer: the settings of script 07 ("default" tag:
absolute tolerance 1, 20 outer iterations, relative precision 1e-3), same initial point (perturbed target).
Output: data/ptm_refits/<tag>_s<noise>_N<N>_m<m>_L<L_max>_d<design seed>_r<r>.npz with the keys of script 07 (kraus,
povm, state, sampled, final_cost, reason). Existing files are skipped, so the run can be resumed.
--dry-run prints the design, the output files and a rough time estimate, and fits nothing.
Runtime: about 22 s per realization at N = 400, L_max = 14 (script 07); more for larger N and L_max.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output, kraus_tensor_to_mgst, get_initial_gate_set_for_optimization
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices, compute_probability_matrices, run_optimization_workflow
from mGST.typing import TrustRegionOptions, OperatorSchedule

import ptm_design as pdz

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--noise-strength", type=float, default=0.3)
parser.add_argument("--noise-seed", type=int, default=42)
parser.add_argument("--N", type=int, required=True, help="number of circuits")
parser.add_argument("--m", type=int, required=True, help="shots per circuit")
parser.add_argument("--L-max", type=int, required=True, help="maximum sequence length (< 24)")
parser.add_argument("--design-seed", type=int, default=12345)
parser.add_argument("--first", type=int, default=0)
parser.add_argument("--last", type=int, default=40)
parser.add_argument("--tag", default="default")
parser.add_argument("--absolute-tolerance", type=float, default=1.0)
parser.add_argument("--num-outer", type=int, default=20)
parser.add_argument("--relative-precision", type=float, default=1e-3)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

OUT = Path(__file__).resolve().parent.parent / "data" / "ptm_refits"
num_gates, kraus_rank, dim = 6, 4, 4
stem = f"{args.tag}_s{args.noise_strength}_N{args.N}_m{args.m}_L{args.L_max}_d{args.design_seed}"

indices_dict = generate_sequence_indices(num_gates=num_gates, num_circuits=args.N, seq_len_list=pdz.seq_len_list(args.L_max), seed=args.design_seed)
lengths = np.array([len(s) for s in indices_dict["gate_indices"]])
todo = [r for r in range(args.first, args.last) if not (OUT / f"{stem}_r{r}.npz").exists()]
print(f"design: {len(lengths)} circuits, seq_len_list {pdz.seq_len_list(args.L_max)}, mean length {lengths.mean():.1f}, "
      f"length histogram {dict(zip(*np.unique(lengths, return_counts=True)))}")
print(f"output: {OUT}/{stem}_r<r>.npz; {len(todo)} of {args.last - args.first} realizations to run")
reference_cost = 400 * np.mean([len(s) for s in generate_sequence_indices(num_gates=num_gates, num_circuits=400, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"]])
print(f"rough time: {22 * lengths.sum() / reference_cost:.0f} s per realization (22 s at N = 400, L_max = 14, scaled by the total "
      f"number of gates), {22 * lengths.sum() / reference_cost * len(todo) / 60:.0f} min in total")
if args.dry_run:
    raise SystemExit

OUT.mkdir(parents=True, exist_ok=True)
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=args.noise_strength, seed=args.noise_seed)
noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
K, A, B = get_compressed_rep_from_mgst_output(*noisy_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim)
true_psd = {"kraus": K, "povm": A, "state": B}
target_superops = {"kraus": kraus_tensor_to_mgst(target_operators["kraus"]), "povm": target_operators["povm"], "state": target_operators["state"]}
init = get_initial_gate_set_for_optimization(target_superops=target_superops, kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim,
                                             spam_depolarization=1e-2, project_kraus_to_isometry=True)

tr_options = TrustRegionOptions(num_iterations=5, verbose=False, verbose_cg=False, num_iterations_cg=10, tol_grad=1e-8, radius_heuristic="typical_distance")
schedule = OperatorSchedule.default(args.num_outer, tr_options)
options = {"schedule": {op: schedule for op in ["povm", "kraus", "state"]}, "num_iterations_outer": args.num_outer, "relative_precision": args.relative_precision,
           "global_gradient_norm_tol": 1e-10, "noise_threshold": None, "compute_least_squares": 1, "optimization_verbose": False, "absolute_tolerance": args.absolute_tolerance}

for r in todo:
    t0 = time.time()
    probability_matrices = compute_probability_matrices(K, A, B, gate_indices=indices_dict["gate_indices"], num_shots=args.m, seed=1000 + r)
    result = run_optimization_workflow(num_shots=args.m, init_operators=init, operators_psd_true=true_psd, target_superops=target_superops,
                                       optimization_options=options, use_log_likelihood=True, indices_dict=indices_dict,
                                       probability_matrices=probability_matrices, gauge_optimize=False)
    op = result["optimized_operators"]
    np.savez(OUT / f"{stem}_r{r}.npz", kraus=np.asarray(op["kraus"]), povm=np.asarray(op["povm"]), state=np.asarray(op["state"]),
             sampled=np.asarray(probability_matrices["sampled"]), final_cost=float(result["cost_fn_history"]["main"][-1]),
             reason=str(result["convergence_reason"]))
    print(f"realization {r}: {time.time() - t0:.0f}s, final cost {float(result['cost_fn_history']['main'][-1]):.6e}, {result['convergence_reason']}", flush=True)
