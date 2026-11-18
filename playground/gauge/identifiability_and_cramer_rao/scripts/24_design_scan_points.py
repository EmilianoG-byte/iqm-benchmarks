"""Design scans of Part C (notebook ptm_design_pipeline_2q.ipynb): per evaluation point, the Fisher-based widths for a
grid of (L_max, N), one design seed by default.

Points (target 2QXYICZ followed by random noise channels of Kraus rank 4):
- noise 0.3,  "truth" (noise seed 42) and "prior" (noise seed 7): N in {400, 600, 1000}, L_max in {14, 18, 23, 28};
- noise 0.05, "truth" (noise seed 42) and "prior" (noise seed 7): N in {300, 400, 600},  L_max in {14, 18, 23, 28, 32}.
The depths come from depth_check_2q.ipynb (the gain stops at about 23 for noise 0.3 and continues for noise 0.05). The van
Trees widths use tau = 2 tau_0 and tau_0, tau_0 from the reference design (N = 400, L_max = 14, m = 1000), as in the old
notebook ptm_precision_design_2q.ipynb. Caches: data/ptm_design_scan_v2_noise<s>_<point>.npz. The noise-0.3 truth reuses
the rows of the old cache data/ptm_design_scan_noise0.3_truth.npz (identical code and design seeds).

Usage: python 24_design_scan_points.py [--points 0.3:truth 0.3:prior 0.05:prior 0.05:truth] [--seeds 0]
Runtime: about 20-60 s per design (more for larger N); about 25 min for all four points with one seed.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import ptm_design as pdz
from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--points", nargs="+", default=["0.3:truth", "0.3:prior", "0.05:prior", "0.05:truth"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
args = parser.parse_args()

DATA = Path(__file__).resolve().parent.parent / "data"
GRID = {0.3: ([400, 600, 1000], [14, 18, 23, 28]), 0.05: ([300, 400, 600], [14, 18, 23, 28, 32])}
NOISE_SEED = {"truth": 42, "prior": 7}
M_GRID, EPS = np.logspace(1.5, 6.5, 101), 1e-2

target, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
target_ptms = pdz.pauli_transfer_matrices(jnp.asarray(target["kraus"]))


def random_noise_gate_set(noise_strength, seed):
    noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=6, dim=4, rank_kraus=4, noise_strength=noise_strength, seed=seed)
    superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target, is_target_vectorized=True)
    return tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=4, state_rank=4, povm_rank=4))


for spec in args.points:
    s, point_name = spec.split(":")
    s = float(s)
    N_list, L_list = GRID[s]
    t0 = time.time()
    point = pdz.prepare_point(random_noise_gate_set(s, NOISE_SEED[point_name]), target_ptms)
    kraus, povm, state = point["operators"]
    ref = generate_sequence_indices(num_gates=6, num_circuits=400, seq_len_list=pdz.seq_len_list(14), seed=0)["gate_indices"]
    tau0 = pdz.prior_width(kraus, pdz.fisher_decomposition(kraus, povm, state, ref, basis=point["T"]), eps=EPS)
    old = DATA / f"ptm_design_scan_noise{s}_{point_name}.npz"
    out = DATA / f"ptm_design_scan_v2_noise{s}_{point_name}.npz"
    print(f"=== noise {s} {point_name}: tau0 = {tau0:.4e}, N {N_list}, L_max {L_list}, seeds {args.seeds}; reuse {old.name if old.exists() else 'nothing'}", flush=True)
    pdz.extend_scan(out, point, N_list, L_list, args.seeds, M_GRID, [2 * tau0, tau0], EPS, tau0, f"noise {s}, {point_name}",
                    cache_in=old if old.exists() else None)
    print(f"    saved {out.name}, {time.time() - t0:.0f}s", flush=True)
