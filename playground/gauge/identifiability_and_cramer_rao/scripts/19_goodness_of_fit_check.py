"""Goodness-of-fit and likelihood-ratio statistics of the saved maximum-likelihood estimates (VALIDITY.md Section 5).
No fitting: it only evaluates the log-likelihood at the saved estimates of script 07 and at the truth.

Usage: python 19_goodness_of_fit_check.py > ../data/19_goodness_of_fit_check_output.txt

Notation: n_ci = counts, f_ci = n_ci / m = frequencies, p_ci = model probabilities, N circuits, m shots, 4 outcomes.
    log-likelihood                 l(theta) = sum_ci n_ci log p_ci(theta)
    saturated log-likelihood       l_sat    = sum_ci n_ci log f_ci             (the model p = f; the code's "cost floor")
    deviance                       D(theta) = 2 [l_sat - l(theta)] = 2 m sum_c KL(f_c || p_c(theta))
    likelihood-ratio statistic     LR       = 2 [l(estimate) - l(truth)] = D(truth) - D(estimate)

If Wilks' theorem applied (it needs the same conditions as asymptotic normality of the estimate), then
    D(truth) ~ chi2 with 3 N degrees of freedom, D(estimate) ~ chi2 with k = 3 N - 624, LR ~ chi2 with 624,
and pyGSTi's model-violation measure N_sigma = (D(estimate) - k) / sqrt(2 k) would be of order 1 (Nielsen et al. 2021,
Sec. 6.1). Here the data are simulated from the model itself, so any large N_sigma is a failure of the reference
distribution, not a model violation.
Runtime: about 30 s.
"""
import glob
import numpy as np
from scipy import stats

import ptm_tools as pt
from mGST import gauge_analysis as ga
from mGST.simulate import generate_sequence_indices

m, N, num_identifiable = 1000, 400, 624
gates = pt.pad(generate_sequence_indices(num_gates=6, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"])
probabilities = lambda operators: np.asarray(ga._circuit_probabilities(*operators, gates))          # shape (N, 4)
# negative log-likelihood -sum n log p, with 0 log 0 = 0 for outcomes that were never observed
negative_log_likelihood = lambda f, p: -m * np.sum(np.where(f > 0, f * np.log(np.where(f > 0, p, 1.0)), 0.0))

p_true = probabilities(pt.truth)
print(f"truth: smallest outcome probability {p_true.min():.2e}; outcomes with expected count m p < 5: {(m * p_true < 5).sum()} of {p_true.size}; "
      f"with m p < 1: {(m * p_true < 1).sum()}")

k = 3 * N - num_identifiable
for tag in ("default", "converged"):
    files = sorted(glob.glob(str(pt.DATA / "crb_estimates" / f"{tag}_*.npz")))
    deviance_estimate, deviance_truth, cost_difference, zero_counts = [], [], [], []
    for file in files:
        data = np.load(file)
        f = data["sampled"].T                                                                        # shape (N, 4)
        saturated = negative_log_likelihood(f, f)
        cost = negative_log_likelihood(f, probabilities(pt.load(file)))
        cost_difference.append(abs(cost - float(data["final_cost"])))
        deviance_estimate.append(2 * (cost - saturated))
        deviance_truth.append(2 * (negative_log_likelihood(f, p_true) - saturated))
        zero_counts.append((f == 0).sum())
    deviance_estimate, deviance_truth = np.array(deviance_estimate), np.array(deviance_truth)
    likelihood_ratio = deviance_truth - deviance_estimate
    n_sigma = (deviance_estimate - k) / np.sqrt(2 * k)
    quantile = stats.chi2.ppf(0.95, num_identifiable)
    print(f"\n[{tag}] {len(files)} estimates. Check: |recomputed cost - saved final_cost| <= {max(cost_difference):.1e}. "
          f"Outcomes never observed, per dataset: {np.mean(zero_counts):.1f} of {4 * N}")
    print(f"  deviance at the truth    : mean {deviance_truth.mean():7.1f}, standard deviation {deviance_truth.std(ddof=1):5.1f}   "
          f"(chi2 with {3 * N} degrees of freedom: {3 * N}, {np.sqrt(6 * N):.1f})")
    print(f"  deviance at the estimate : mean {deviance_estimate.mean():7.1f}, standard deviation {deviance_estimate.std(ddof=1):5.1f}   "
          f"(Wilks, chi2 with {k}: {k}, {np.sqrt(2 * k):.1f})")
    print(f"  N_sigma = (deviance - {k}) / sqrt(2 * {k}): mean {n_sigma.mean():.1f}, min {n_sigma.min():.1f}, max {n_sigma.max():.1f}")
    print(f"  likelihood-ratio statistic: mean {likelihood_ratio.mean():7.1f}, standard deviation {likelihood_ratio.std(ddof=1):5.1f}   "
          f"(Wilks, chi2 with {num_identifiable}: {num_identifiable}, {np.sqrt(2 * num_identifiable):.1f})")
    print(f"  effective number of fitted parameters = mean likelihood-ratio statistic = {likelihood_ratio.mean():.0f} of {num_identifiable}")
    print(f"  datasets with the truth inside the Wilks 95% likelihood region (statistic <= {quantile:.0f}): {(likelihood_ratio <= quantile).mean():.0%}")
