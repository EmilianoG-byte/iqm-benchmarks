"""How do weakly determined Fisher directions enter the error of the gauge-fixed Pauli transfer matrix elements?

For each gauge-fixed Pauli transfer matrix element f o Pi, the LAN (local asymptotic normality) variance is
sigma^2 = d^T F^+ d = sum_k c_k, c_k = (d . v_k)^2 / lambda_k, with d = T^T grad(f o Pi). At the truth (N = 400,
m = 1000, design seed 12345) this script
1) compares truncated sums (first K directions) and the van Trees variant with the actual error of the 40 default
   estimates of script 07;
2) measures how much each Fisher direction moves the Pauli transfer matrices, |D v_k|, and its standard deviation in
   Pauli-transfer-matrix units, |D v_k| / sqrt(lambda_k);
3) diagonalizes the covariance of the Pauli transfer matrices, Sigma_R = D F^+ D^T, and tests the estimates along its
   principal directions;
4) checks that sigma^2 does not depend on the metric on the tangent space, while n_eff(eps) does;
5) attributes the actual errors to Fisher directions with the linear reconstruction delta R ~ D V diag(lambda)^-1/2 e,
   where e is the per-direction statistic of script 09;
6) splits each v_k into dominant Kraus operators, small (incoherent) Kraus operators and SPAM, and evaluates van Trees
   with the a priori width tau^2 = |small Kraus part of the truth|^2 / (number of directions with k > n_eff(0.01)).
Needs the estimates of 07_crb_estimates_runner.py (default set). Runtime: a few minutes.
"""
import glob
import numpy as np
from scipy.stats import norm

import ptm_tools as pt
from mGST.simulate import generate_sequence_indices
from mGST import gauge_analysis as ga

m, N = 1000, 400
train = pt.pad(generate_sequence_indices(num_gates=pt.num_gates, num_circuits=N, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"])
files = sorted(glob.glob(f"{pt.DATA}/crb_estimates/default_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))

theta_truth, _, _ = pt.gauge_fix(pt.to_vec(*pt.truth))
fi = pt.fisher(theta_truth, train, m)
lam, V = fi["lam"], fi["V"]
D_all = pt.ptm_gradient(theta_truth) @ fi["T"]                                 # 1536 x 815, tangent coordinates
keep = np.linalg.norm(D_all, axis=1) > 1e-10                                    # drops the constant first rows
D = D_all[keep]
G = D @ V                                                                       # entries d . v_k
C = G ** 2 / lam                                                                # c_k for every element
var_full = C.sum(axis=1)
n_eff = int((lam > 1e4).sum())
print(f"rank {fi['rank']}, {keep.sum()} non-constant elements, n_eff(0.01) = {n_eff}, "
      f"max |kernel component of d| / |d| = {np.max(np.linalg.norm(D @ fi['kernel'], axis=1) / np.linalg.norm(D, axis=1)):.1e}")

R_true = np.asarray(pt.ptm_vector(theta_truth))[keep]
errors, E = [], []
for f in files:
    theta, _, _ = pt.gauge_fix(pt.to_vec(*pt.load(f)))
    errors.append(np.asarray(pt.ptm_vector(theta))[keep] - R_true)
    K, A, B = pt.to_ops(theta)
    w = np.sqrt(m) * (np.asarray(ga._circuit_probabilities(K, A, B, train)).reshape(-1) - fi["p"]) / np.sqrt(fi["p"])
    E.append(fi["U"].T @ w)
err, E = np.array(errors), np.array(E)                                          # (40, 1440), (40, 624)
rmse = np.sqrt((err ** 2).mean(axis=0))
print(f"{len(files)} estimates; actual rmse over elements: median {np.median(rmse):.2e}, max {rmse.max():.2e}; "
      f"max |error| over all elements and estimates {np.abs(err).max():.2e}")


def report(name, var):
    sig = np.sqrt(var)
    print(f"  {name:<28} median sigma {np.median(sig):.2e}  max sigma {sig.max():.2e}  median rmse/sigma {np.median(rmse / sig):5.2f}  "
          f"5%-95% rmse/sigma {np.quantile(rmse / sig, 0.05):4.2f}-{np.quantile(rmse / sig, 0.95):4.2f}  coverage(95%) {100 * np.mean(np.abs(err) <= 1.96 * sig):5.1f}%")


print("\n1) truncated sums and van Trees versus the actual error")
for K in [50, 100, n_eff, 200, 300, 400, 500, len(lam)]:
    report(f"first {K} directions", C[:, :K].sum(axis=1))
for tau in [1.0, 0.1, 0.05, 0.02, 0.01]:
    report(f"van Trees tau = {tau}", (G ** 2 / (lam + 1 / tau ** 2)).sum(axis=1))
share = C[:, n_eff:].sum(axis=1) / var_full
print(f"  share of sigma^2 from k > n_eff(0.01): quartiles {np.round(np.quantile(share, [0.25, 0.5, 0.75]), 3)}")

print("\n2) per Fisher direction (bins as in README 5.5). |Dv| = Frobenius norm of the change of all 1440 elements per unit step along v_k")
print(f"  random unit tangent direction (orthogonal to ker F), for scale: median |Dv| over 200 draws "
      f"{np.median(np.linalg.norm(G @ np.random.default_rng(0).standard_normal((len(lam), 200)), axis=0) / np.sqrt(len(lam))):.2e}")
print(f"  {'k':<10} {'1/sqrt(lam)':>12} {'|Dv_k|':>10} {'|Dv_k|/sqrt(lam)':>17} {'share of tr Sigma_R':>20} {'mean e_k^2':>11}")
for a, b in [(0, 50), (50, 100), (100, n_eff), (n_eff, 200), (200, 300), (300, 400), (400, 500), (500, len(lam))]:
    sl = slice(a, b)
    nrm = np.linalg.norm(G[:, sl], axis=0)
    print(f"  {a + 1:>3}-{b:<6} {np.median(1 / np.sqrt(lam[sl])):12.2e} {np.median(nrm):10.2e} {np.median(nrm / np.sqrt(lam[sl])):17.2e} "
          f"{C[:, sl].sum() / C.sum():20.3f} {np.mean(E[:, sl] ** 2):11.2f}")

print("\n3) covariance of the Pauli transfer matrices, Sigma_R = D F^+ D^T, and the estimates along its principal directions")
Q, s_R, _ = np.linalg.svd(G / np.sqrt(lam), full_matrices=False)               # Sigma_R = Q diag(s_R^2) Q^T
for delta in [1e-2, 5e-3, 3e-3, 1e-3]:
    print(f"  principal standard deviations of Sigma_R above {delta:.0e}: {(s_R > delta).sum():4d}   "
          f"(Fisher directions with 1/sqrt(lam) above {delta:.0e}: {(1 / np.sqrt(lam) > delta).sum()})")
eff_R = ((err @ Q) ** 2).mean(axis=0) / s_R ** 2
print(f"  mean (q_j . delta R)^2 / s_j^2 over the estimates (1 = efficient along q_j):")
for a, b in [(0, 10), (10, 30), (30, 100), (100, 200), (200, 400), (400, 624)]:
    print(f"    j = {a + 1:>3}-{b:<4} s_j {s_R[a]:.1e}..{s_R[b - 1]:.1e}   median {np.median(eff_R[a:b]):5.2f}")
print(f"  E|delta R|_F^2: predicted tr Sigma_R = {s_R @ s_R:.3e}, actual mean {np.mean((err ** 2).sum(axis=1)):.3e}")
z_sim = norm.ppf(1 - 0.05 / (2 * keep.sum()))
print(f"  simultaneous 95% over {keep.sum()} elements (Bonferroni): z = {z_sim:.2f}; predicted max half-width {z_sim * np.sqrt(var_full.max()):.2e}")

print("\n4) metric dependence: tangent coordinates z = Q z' with Q = O diag(exp(g)) O^T, g ~ N(0, 1), O random orthogonal")
rng = np.random.default_rng(1)
O, _ = np.linalg.qr(rng.standard_normal((815, 815)))
Qm = O @ np.diag(np.exp(rng.standard_normal(815))) @ O.T
W = fi["U"] * np.sqrt(lam / m) @ V.T                                            # whitened Jacobian, W = U S V^T
u2, s2, vt2 = np.linalg.svd(np.sqrt(m) * W @ Qm, full_matrices=False)
r2 = len(lam)
lam2, V2 = s2[:r2] ** 2, vt2[:r2].T
var2 = ((D @ Qm @ V2) ** 2 / lam2).sum(axis=1)                                  # d' = Q^T d, F' = Q^T F Q
print(f"  max relative change of sigma^2 over elements: {np.max(np.abs(var2 / var_full - 1)):.1e}")
print(f"  n_eff(0.01): {n_eff} -> {(lam2 > 1e4).sum()};  n_eff(0.003): {(lam > 1 / 0.003 ** 2).sum()} -> {(lam2 > 1 / 0.003 ** 2).sum()}")

print("\n5) attribution of the actual errors: delta R_lin = D V diag(lambda)^-1/2 e")
lin = (E / np.sqrt(lam)) @ G.T
print(f"  |delta R - delta R_lin| / |delta R| per estimate: median {np.median(np.linalg.norm(err - lin, axis=1) / np.linalg.norm(err, axis=1)):.2f}")
for a, b in [(0, 100), (100, n_eff), (n_eff, 400), (400, len(lam))]:
    part = (E[:, a:b] / np.sqrt(lam[a:b])) @ G[:, a:b].T
    print(f"  k = {a + 1:>3}-{b:<4}: mean |part|^2 / mean |delta R|^2 = {np.mean((part ** 2).sum(axis=1)) / np.mean((err ** 2).sum(axis=1)):.3f}   "
          f"predicted share of tr Sigma_R {C[:, a:b].sum() / C.sum():.3f}")

print("\n6) composition of the Fisher directions (median weight |v_k|^2 per bin) and an a priori van Trees width")
K_true = np.asarray(pt.to_ops(theta_truth)[0])
norms2 = np.linalg.norm(K_true, axis=(2, 3)) ** 2                               # canonical Kraus operators: orthogonal
dominant = np.zeros_like(norms2, dtype=bool)
dominant[np.arange(pt.num_gates), norms2.argmax(axis=1)] = True
in_dominant = np.concatenate([np.repeat(dominant.reshape(-1), pt.dim ** 2)] * 2)   # real and imaginary blocks of K
w = (fi["T"] @ V) ** 2
n_K = in_dominant.size
print(f"  {'k':<10} {'dominant Kraus':>15} {'small Kraus':>12} {'SPAM':>6}")
for a, b in [(0, 50), (50, 100), (100, n_eff), (n_eff, 200), (200, 300), (300, 400), (400, 500), (500, len(lam))]:
    print(f"  {a + 1:>3}-{b:<6} {np.median(w[:n_K][in_dominant, a:b].sum(axis=0)):15.3f} "
          f"{np.median(w[:n_K][~in_dominant, a:b].sum(axis=0)):12.3f} {np.median(w[n_K:, a:b].sum(axis=0)):6.3f}")
small2 = norms2[~dominant].sum()
tau0 = np.sqrt(small2 / (len(lam) - n_eff))
print(f"  |small Kraus part|^2 = {small2:.2e} (= d x sum over gates of the non-dominant Choi weight); tau0 = {tau0:.1e}")
print(f"  largest single-element change from removing all small Kraus operators of one gate, bound max_g sum_r |K_r|^2 / d = {norms2[~dominant].reshape(pt.num_gates, -1).sum(axis=1).max() / pt.dim:.1e}")
for tau in [tau0, 2 * tau0]:
    report(f"van Trees tau = {tau:.1e}", (G ** 2 / (lam + 1 / tau ** 2)).sum(axis=1))
