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
4) the fraction of converged runs (assumption A4; only runs stopped by a "below ..." criterion count as converged).
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
parser.add_argument("--reference-tag", default="tight", help="tag of better-converged refits of the same datasets (section 5)")
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
    converged.append("below" in str(d["reason"]))         # stopped by a convergence criterion, not by the iteration cap
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

print("\n4) likelihood (VALIDITY 4.5, 4.7, 5; computed as in script 19)")
from mGST import gauge_analysis as ga
gates = pdz.pad_sequences(indices)
probabilities = lambda ops: np.asarray(ga._circuit_probabilities(*ops, gates))                       # shape (N, 4)
negative_log_likelihood = lambda f, p: -m * np.sum(np.where(f > 0, f * np.log(np.where(f > 0, p, 1.0)), 0.0))
p_true = probabilities(truth)
n_id = fd["rank"]
k = 3 * len(indices) - n_id
dev_est, dev_true, cost_gap = [], [], []
for f in files:
    d = np.load(f)
    freq = np.asarray(d["sampled"]).T
    saturated = negative_log_likelihood(freq, freq)
    cost = negative_log_likelihood(freq, probabilities(tuple(jnp.asarray(d[q]) for q in ("kraus", "povm", "state"))))
    cost_gap.append(abs(cost - float(d["final_cost"])) / abs(float(d["final_cost"])))
    dev_est.append(2 * (cost - saturated))
    dev_true.append(2 * (negative_log_likelihood(freq, p_true) - saturated))
dev_est, dev_true = np.array(dev_est), np.array(dev_true)
lam_stat = dev_true - dev_est
print(f"  check: |our cost - optimizer final cost| / |final cost|: max {max(cost_gap):.1e} (the optimizer's cost must be this negative log-likelihood)")
print(f"  outcomes with expected count m p < 5: {100 * np.mean(m * p_true < 5):.0f}%")
print(f"  likelihood-ratio statistic Lambda = D(truth) - D(estimate): mean {lam_stat.mean():.0f} +- {lam_stat.std(ddof=1) / np.sqrt(R):.0f} "
      f"(Wilks: {n_id} +- {np.sqrt(2 * n_id / R):.0f}); effective number of fitted parameters {lam_stat.mean() / n_id:.0%} of {n_id}")
print(f"  deviance at the estimate: mean {dev_est.mean():.0f} (Wilks: {k}); N_sigma = (D - k) / sqrt(2 k): mean {((dev_est - k) / np.sqrt(2 * k)).mean():.1f}")
print(f"  deviance at the truth: mean {dev_true.mean():.0f} (Wilks: {3 * len(indices)})")

tight = sorted(OUT.glob(f"{args.reference_tag}{stem[len(args.tag):]}_r*.npz")) if args.reference_tag != args.tag else []
if tight:
    print(f"\n5) optimizer: {args.tag} against {args.reference_tag} estimates of the same datasets (in LAN widths of each element)")
    for t in tight:
        r_ = t.stem.split("_r")[-1]
        default_file = OUT / f"{stem}_r{r_}.npz"
        if not default_file.exists():
            continue
        R_pair = []
        for f in (default_file, t):
            d = np.load(f)
            (K, _, _), _ = pdz.gauge_fix_to_target(*(jnp.asarray(d[q]) for q in ("kraus", "povm", "state")), target_ptms)
            R_pair.append(np.asarray(pdz.pauli_transfer_matrices(K)).reshape(-1)[point["keep"]])
        shift = np.abs(R_pair[0] - R_pair[1]) / sig_lan
        print(f"  dataset {r_}: max shift {shift.max():.3f} sigma_LAN, median {np.median(shift):.4f} (VALIDITY 4.6: worst case 1.41 at 1 nat)")
