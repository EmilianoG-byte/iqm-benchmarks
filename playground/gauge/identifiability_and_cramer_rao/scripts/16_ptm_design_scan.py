"""Experiment design for Pauli transfer matrix precision: predicted error of the gauge-fixed elements versus the number of
circuits N, the maximum depth L_max and the shots per circuit m, without fitting anything.

Usage: python 16_ptm_design_scan.py [prior|truth] [summary]
  prior (default) = target + other noise, seed 7 (the prior guess of script 08); truth = the truth of scripts 07-15.
  summary = only print the tables from the saved data/16_ptm_design_scan_<point>.npz.

Designs: generate_sequence_indices(..., seq_len_list=[1, max(3, L_max // 2 + 1), L_max]) (L_max = 14 gives the
[1, 8, 14] of the notebook), 3 design seeds per (N, L_max). The gradient D of the elements f o Pi depends only on the
evaluation point; the Fisher information is computed per design at m = 1 and rescaled (lambda_k is proportional to m).
Per element, at shots m:
  LAN       sigma^2 = sum_k (d . v_k)^2 / (m lambda_k), infinite if d has a component in ker F outside the gauge;
  van Trees sigma^2 = sum_k (d . v_k)^2 / (m lambda_k + 1 / tau^2) + tau^2 |kernel part of d|^2.
Required m for a target 95% half-width delta: smallest m on a grid with z * max over elements of sigma <= delta, with
z = 1.96 (each element separately) or the Bonferroni z for all elements at once.
Runtime: about 15 minutes per evaluation point; summary: seconds.
"""
import sys
import time
import numpy as np
from scipy.stats import norm

import ptm_tools as pt
from mGST.simulate import generate_sequence_indices

point = sys.argv[1] if len(sys.argv) > 1 else "prior"
OUT = pt.DATA / f"16_ptm_design_scan_{point}.npz"
N_list, L_list, seeds = [100, 150, 200, 250, 300, 400, 600, 800], [6, 10, 14, 18, 23], [0, 1, 2]
m_grid = np.unique(np.round(np.logspace(2, 6, 81)))
taus, deltas, eps = [0.01, 0.02], [2e-2, 1e-2, 5e-3, 2e-3], 0.01


def scan():
    theta, _, _ = pt.gauge_fix(pt.to_vec(*(pt.gate_set(5e-2, 7) if point == "prior" else pt.truth)))
    D_amb = pt.ptm_gradient(theta)
    keep = np.linalg.norm(D_amb @ np.asarray(pt.ga.tangent_space_basis(*pt.to_ops(theta))), axis=1) > 1e-10  # constant first rows vanish on the tangent space only
    D_amb = D_amb[keep]
    rows = []
    for L in L_list:
        for N in N_list:
            for seed in seeds:
                t0 = time.time()
                gates = pt.pad(generate_sequence_indices(num_gates=pt.num_gates, num_circuits=N, seq_len_list=[1, max(3, L // 2 + 1), L], seed=seed)["gate_indices"])
                fi = pt.fisher(theta, gates, 1)
                D = D_amb @ fi["T"]
                G2 = (D @ fi["V"]) ** 2
                ker2 = np.linalg.norm(D @ fi["kernel"], axis=1) ** 2 / np.linalg.norm(D, axis=1) ** 2   # gauge part is 0
                estimable = ker2 < 1e-12
                lam1 = fi["lam"]
                var_lan1 = np.where(estimable, (G2 / lam1).sum(axis=1), np.inf)                         # at m = 1
                ker_abs = ker2 * np.linalg.norm(D, axis=1) ** 2
                max_vt = np.array([[np.sqrt(((G2 / (m * lam1 + 1 / tau ** 2)).sum(axis=1) + tau ** 2 * ker_abs).max()) for m in m_grid] for tau in taus])
                strong = lambda m: m * lam1 > 1 / eps ** 2
                weak_share = np.median(np.where(estimable, (G2 * ~strong(1000) / lam1).sum(axis=1) / (G2 / lam1).sum(axis=1), 1.0))
                rows.append(dict(L=L, N=N, seed=seed, rank=fi["rank"], n_circ=gates.shape[0], estimable=estimable.mean(),
                                 max_sig_lan1=np.sqrt(var_lan1.max()), med_sig_lan1=np.sqrt(np.median(var_lan1)), max_vt=max_vt,
                                 n_eff_1000=int(strong(1000).sum()), weak_share_1000=weak_share, lam1=lam1))
                print(f"L_max {L:2d} N {N:3d} seed {seed}: {gates.shape[0]} circuits, rank {fi['rank']}, estimable {100 * estimable.mean():5.1f}%, "
                      f"max sigma_LAN(m=1000) {np.sqrt(var_lan1.max() / 1000):.2e}, max sigma_vT(0.01, m=1000) {max_vt[0, np.searchsorted(m_grid, 1000)]:.2e}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    np.savez(OUT, rows=np.array(rows, dtype=object), m_grid=m_grid, taus=taus, z_all=norm.ppf(1 - 0.05 / (2 * keep.sum())), n_elements=keep.sum())


def summary():
    data = np.load(OUT, allow_pickle=True)
    rows, z_all, n_el = list(data["rows"]), float(data["z_all"]), int(data["n_elements"])
    i1000 = np.searchsorted(m_grid, 1000)
    cell = lambda L, N: [r for r in rows if r["L"] == L and r["N"] == N]
    med = lambda rs, fn: np.median([fn(r) for r in rs])

    def m_required(sig_of_m, z, delta):
        ok = z * sig_of_m <= delta
        return m_grid[np.argmax(ok)] if ok.any() else np.inf

    print(f"evaluation point: {point}. Medians over {len(seeds)} design seeds. sigma = standard deviation of a gauge-fixed element at m = 1000.")
    print(f"{'L_max':>5} {'N':>4} {'rank':>5} {'estim.':>7} {'n_eff(.01)':>10} {'weak share':>10} {'max sig LAN':>12} {'med sig LAN':>12} {'max sig vT.01':>14} {'seed spread':>11}")
    for L in L_list:
        for N in N_list:
            rs = cell(L, N)
            lan = [r["max_sig_lan1"] / np.sqrt(1000) for r in rs]
            print(f"{L:5d} {N:4d} {med(rs, lambda r: r['rank']):5.0f} {100 * med(rs, lambda r: r['estimable']):6.1f}% {med(rs, lambda r: r['n_eff_1000']):10.0f} "
                  f"{med(rs, lambda r: r['weak_share_1000']):10.2f} {np.median(lan):12.2e} {med(rs, lambda r: r['med_sig_lan1']) / np.sqrt(1000):12.2e} "
                  f"{med(rs, lambda r: r['max_vt'][0, i1000]):14.2e} {max(lan) / min(lan) if np.isfinite(lan).all() else np.nan:11.2f}")

    for z, label in [(1.96, "each element separately"), (z_all, f"all {n_el} elements at once (Bonferroni)")]:
        print(f"\nrequired shots per circuit m (and total shots N m) for a 95% half-width delta, {label}, z = {z:.2f}")
        print(f"{'L_max':>5} {'N':>4} " + " ".join(f"{'LAN ' + f'{d:.0e}':>18} {'vT.01 ' + f'{d:.0e}':>18}" for d in deltas))
        for L in L_list:
            for N in N_list:
                rs, cells = cell(L, N), []
                for d in deltas:
                    m_lan = med(rs, lambda r: (z * r["max_sig_lan1"] / d) ** 2)
                    m_vt = med(rs, lambda r: m_required(r["max_vt"][0], z, d))
                    cells += [f"{m_lan:8.1e} ({N * m_lan:7.1e})", f"{m_vt:8.1e} ({N * m_vt:7.1e})"]
                print(f"{L:5d} {N:4d} " + " ".join(f"{c:>18}" for c in cells))

    for z, label in [(1.96, "each element"), (z_all, "all elements at once")]:
        print(f"\ncompact, delta = 1e-2, {label} (z = {z:.2f}): m required by LAN | van Trees tau = 0.01 | tau = 0.02, and total shots N m in millions")
        for L in L_list:
            out = []
            for N in [N for N in N_list if N >= 250]:
                rs = cell(L, N)
                ms = [med(rs, lambda r: (z * r["max_sig_lan1"] / 1e-2) ** 2)] + [med(rs, lambda r, t=t: m_required(r["max_vt"][t], z, 1e-2)) for t in (0, 1)]
                out.append(f"N{N}: {ms[0]:.1e}|{ms[1]:.0f}|{ms[2]:.0f} ({N * ms[0] / 1e6:.2g}|{N * ms[1] / 1e6:.2g}|{N * ms[2] / 1e6:.2g})")
            print(f"  L_max {L:2d}  " + "  ".join(out))

    print("\nscaling: slope of log(max sigma at m = 1000) versus log L_max (L_max = 10..23) and versus log N (N = 400..800)")
    L_fit, N_fit = [10, 14, 18, 23], [400, 600, 800]
    lan = lambda L, N: med(cell(L, N), lambda r: r["max_sig_lan1"])
    vt = lambda L, N: med(cell(L, N), lambda r: r["max_vt"][0, i1000])
    for N in N_fit:
        print(f"  N {N}: versus L_max: LAN {np.polyfit(np.log(L_fit), np.log([lan(L, N) for L in L_fit]), 1)[0]:5.2f}, van Trees {np.polyfit(np.log(L_fit), np.log([vt(L, N) for L in L_fit]), 1)[0]:5.2f}")
    for L in [10, 14, 23]:
        print(f"  L_max {L}: versus N: LAN {np.polyfit(np.log(N_fit), np.log([lan(L, N) for N in N_fit]), 1)[0]:5.2f}, van Trees {np.polyfit(np.log(N_fit), np.log([vt(L, N) for N in N_fit]), 1)[0]:5.2f}")
    bins = [(0, 50), (50, 100), (135, 400), (400, 624)]
    print("  median per-shot lambda_k at N = 400 in bins of k; last column: slope of log lambda versus log L_max (10..23)")
    lam_bins = {L: np.median([r["lam1"] for r in cell(L, 400)], axis=0) for L in L_list}
    for a, b in bins:
        vals = [np.median(lam_bins[L][a:b]) for L in L_list]
        slope = np.polyfit(np.log(L_fit), np.log([np.median(lam_bins[L][a:b]) for L in L_fit]), 1)[0]
        print(f"    k {a + 1:>3}-{b:<4} " + " ".join(f"L{L}: {v:.2e}" for L, v in zip(L_list, vals)) + f"   slope {slope:.2f}")


if "summary" not in sys.argv:
    scan()
summary()
