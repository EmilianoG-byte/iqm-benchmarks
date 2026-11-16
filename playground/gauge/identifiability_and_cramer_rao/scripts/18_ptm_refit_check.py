"""Test T3 of PTM_PRECISION.md, part 2: do the refits of script 17 at one design reach the target, and are the Fisher
predictions of ptm_design (the notebook ptm_precision_design_2q.ipynb) right for this estimator?

Usage (same truth and design arguments as script 17):
    python 18_ptm_refit_check.py --N 1000 --m 1100 --L-max 23 --delta 1e-2

For the gauge-fixed Pauli transfer matrix elements (gauge fixing Pi as in ptm_design, applied to truth and estimates):
1) per element: rmse over datasets, rmse / sigma and coverage of +-1.96 sigma for the LAN width and the van Trees widths
   (tau = 2 tau0 and tau0, tau0 from the reference design as in the notebook), as in script 15 part 1;
2) the goal itself: fraction of (element, dataset) pairs with |error| <= delta, and fraction of datasets with all
   elements within delta;
3) the small Choi eigenvalues (gauge-invariant, no gauge fixing): rmse / sigma_LAN(mu) and the fraction of estimates
   at the positivity boundary (mu < 1e-6), which tests the rate-resolution criterion of the notebook;
4) the fraction of converged runs (assumption A4; "Max iterations reached" counts as not converged).
LAN = local asymptotic normality. With R datasets an rmse is known to about +-1 / sqrt(2 (R - 1)) (README 6.4).
Runtime: about 1 s per estimate (gauge fixing) plus one Fisher evaluation.
"""
import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import ptm_design as pdz
from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--noise-strength", type=float, default=0.3)
parser.add_argument("--noise-seed", type=int, default=42)
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--m", type=int, required=True)
parser.add_argument("--L-max", type=int, required=True)
parser.add_argument("--design-seed", type=int, default=12345)
parser.add_argument("--tag", default="default")
parser.add_argument("--delta", type=float, default=1e-2)
args = parser.parse_args()

OUT = Path(__file__).resolve().parent.parent / "data" / "ptm_refits"
stem = f"{args.tag}_s{args.noise_strength}_N{args.N}_m{args.m}_L{args.L_max}_d{args.design_seed}"
files = sorted(OUT.glob(f"{stem}_r*.npz"), key=lambda f: int(f.stem.split("_r")[-1]))
if not files:
    raise SystemExit(f"no estimates {OUT}/{stem}_r*.npz; run 17_ptm_refit_runner.py with the same arguments first")
num_gates, kraus_rank, dim, m = 6, 4, 4, args.m

target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
target_ptms = pdz.pauli_transfer_matrices(jnp.asarray(target_operators["kraus"]))
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=args.noise_strength, seed=args.noise_seed)
superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target_operators, is_target_vectorized=True)
truth = tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim))

point = pdz.prepare_point(truth, target_ptms)
K0, A0, B0 = point["operators"]
R_true = np.asarray(pdz.pauli_transfer_matrices(K0)).reshape(-1)[point["keep"]]
indices = generate_sequence_indices(num_gates=num_gates, num_circuits=args.N, seq_len_list=pdz.seq_len_list(args.L_max), seed=args.design_seed)["gate_indices"]
fd = pdz.fisher_decomposition(K0, A0, B0, indices, basis=point["T"])
ref = generate_sequence_indices(num_gates=num_gates, num_circuits=400, seq_len_list=pdz.seq_len_list(14), seed=0)["gate_indices"]
tau0 = pdz.prior_width(K0, pdz.fisher_decomposition(K0, A0, B0, ref, basis=point["T"]))
taus = [2 * tau0, tau0]
w = pdz.element_widths(point["D"], fd, np.array([float(m)]), taus)
sig_lan, sig_vt = w["sigma_lan1"] / np.sqrt(m), w["sigma_vt"][:, :, 0]
sig_rate = pdz.lan_sigma1(point["D_rate"], fd) / np.sqrt(m)

errors, mu_hat, converged = [], [], []
for f in files:
    d = np.load(f)
    ops = tuple(jnp.asarray(d[k]) for k in ("kraus", "povm", "state"))
    (K, _, _), _ = pdz.gauge_fix_to_target(*ops, target_ptms)
    errors.append(np.asarray(pdz.pauli_transfer_matrices(K)).reshape(-1)[point["keep"]] - R_true)
    mu_hat.append(np.asarray(pdz.choi_eigenvalues(ops[0]))[:, 1:].reshape(-1))
    converged.append(not str(d["reason"]).startswith("Max iterations"))
err, mu_hat, R = np.array(errors), np.array(mu_hat), len(files)
rmse = np.sqrt((err ** 2).mean(axis=0))
print(f"{stem}: {R} estimates, {100 * np.mean(converged):.0f}% converged; rank {fd['rank']}; tau0 = {tau0:.3e}; "
      f"an rmse is known to about +-{100 / np.sqrt(2 * (R - 1)):.0f}%")
print(f"actual error: median rmse over elements {np.median(rmse):.2e}, max {rmse.max():.2e}; max |error| {np.abs(err).max():.2e}")


def report(name, sig):
    ratio = rmse / sig
    print(f"  {name:<26} max sigma {sig.max():.2e}  predicted 1.96 max sigma {1.96 * sig.max():.2e}  median rmse/sigma {np.median(ratio):5.2f}  "
          f"5%-95% {np.quantile(ratio, 0.05):4.2f}-{np.quantile(ratio, 0.95):4.2f}  coverage(95%) {100 * np.mean(np.abs(err) <= 1.96 * sig):5.1f}%")


print("\n1) Fisher predictions versus the actual errors of the Pauli transfer matrix elements")
report("LAN", sig_lan)
for t, tau in enumerate(taus):
    report(f"van Trees tau = {tau:.3e}", sig_vt[t])

print(f"\n2) the goal, delta = {args.delta:.0e}")
print(f"  (element, dataset) pairs with |error| <= delta: {100 * np.mean(np.abs(err) <= args.delta):.1f}%   (target: >= 95%)")
print(f"  datasets with all {err.shape[1]} elements within delta: {100 * np.mean((np.abs(err) <= args.delta).all(axis=1)):.1f}%")
print(f"  worst element: rmse {rmse.max():.2e}, 1.96 rmse {1.96 * rmse.max():.2e}")

print("\n3) small Choi eigenvalues (incoherent rates)")
mu_true = point["mu_small"]
rate_rmse = np.sqrt(((mu_hat - mu_true) ** 2).mean(axis=0))
print(f"  predicted sigma(mu)/mu: median {np.median(sig_rate / mu_true):.2f}, max {(sig_rate / mu_true).max():.2f}")
print(f"  rmse(mu)/sigma_LAN(mu): median {np.median(rate_rmse / sig_rate):.2f}, range {np.min(rate_rmse / sig_rate):.2f}-{np.max(rate_rmse / sig_rate):.2f}")
print(f"  estimates at the positivity boundary (mu < 1e-6): {100 * np.mean(mu_hat < 1e-6):.1f}% of all (rate, dataset) pairs")
