"""Checks and first use of the matrix-free Fisher analysis `fisher_matrix_free.py` (VALIDITY.md Section 8).

Usage:
    python 20_matrix_free_check.py validate                      # two qubits: matrix-free versus the dense `ptm_design`
    python 20_matrix_free_check.py four_qubits --N 2000 --steps 50 [--kappa] [--contraction grouped]
    python 20_matrix_free_check.py saved_run                     # deviance and expected counts of the saved 4-qubit fit

validate (about 2 minutes): at the two-qubit reference design (N = 400, truth of scripts 07-19) every matrix-free
object is compared with its dense counterpart: projector, Fisher product, gauge cost and gauge-fixed gradients,
van Trees and LAN widths (LAN = local asymptotic normality), merged-outcome width, nonlinearity ratio, and the
optimizer certificate and deviance at saved estimates.

four_qubits: the four-qubit gate set of `create_4q_gst_config` (4 Rx(pi/2), 4 Ry(pi/2), CZ-CZ), truth = target followed
by random noise channels of Kraus rank 4, state and POVM rank 4. No fitting. Runtime is about
(number of quantities) * steps * (0.6 s per 1000 circuits) with the grouped contraction, plus about a minute of compilation.

saved_run: needs playground/4q_gst/data (the 10 000-circuit, 500-shot run). No Fisher products, only probabilities.
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reverse-mode derivatives of real functions through complex intermediates drop an imaginary part by design.
warnings.filterwarnings("ignore", message="Casting complex values to real")

import fisher_matrix_free as fmf
import ptm_design as pdz
from mGST.simulate import generate_sequence_indices, apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("mode", choices=["validate", "four_qubits", "saved_run"])
parser.add_argument("--N", type=int, default=2000, help="number of circuits (four_qubits)")
parser.add_argument("--L-max", type=int, default=19, help="maximum sequence length (four_qubits)")
parser.add_argument("--m", type=int, default=500, help="shots per circuit (four_qubits)")
parser.add_argument("--steps", type=int, default=50, help="Lanczos steps per quantity (four_qubits)")
parser.add_argument("--noise-strength", type=float, default=0.05)
parser.add_argument("--design-seed", type=int, default=12345)
parser.add_argument("--contraction", default="grouped", choices=fmf.CONTRACTIONS)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--kappa", action="store_true", help="also solve for the direction of the first quantity and compute its nonlinearity ratio")
args = parser.parse_args()
rng = np.random.default_rng(0)


def report_widths(name, quadrature, m, taus, value=None):
    """Print the van Trees interval for each tau and the LAN lower bound of one quantity."""
    lan, kernel_share = fmf.lan_sigma(quadrature, m)
    parts = []
    for tau in taus:
        lower, upper = fmf.van_trees_sigma(quadrature, m, tau)
        parts.append(f"van Trees(tau={tau:g}) in [{lower:.3e}, {upper:.3e}]")
    relative = "" if value is None else f"  value {value:.3e}, value / LAN bound {value / lan:.2f}"
    print(f"  {name:38s} {', '.join(parts)}; LAN >= {lan:.3e} (kernel share {kernel_share:.1e}, {quadrature['num_steps']} steps{', exact' if quadrature['exact'] else ''}){relative}", flush=True)


# ================================================================ two qubits: validation against the dense route
if args.mode == "validate":
    import ptm_tools as pt

    m, taus = 1000, (0.01, 0.1)
    target_kraus = jnp.asarray(pt.target_operators["kraus"])
    indices = generate_sequence_indices(num_gates=6, num_circuits=400, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"]
    point = pdz.prepare_point(pt.truth, pt.R_target)                      # dense: gauge-fixed truth, T, gradients
    K0, A0, B0 = point["operators"]
    T = point["T"]
    fd = pdz.fisher_decomposition(K0, A0, B0, indices, num_shots=1, basis=T)
    dense = pdz.element_widths(point["D"], fd, np.array([float(m)]), taus)
    problem = fmf.fisher_problem(K0, A0, B0, indices)
    print(f"two qubits: tangent dimension {fmf.num_tangent_directions(problem.shapes)} (dense {T.shape[1]}), "
          f"identifiable {fmf.num_identifiable_parameters(problem.shapes)} (dense rank {fd['rank']})")

    print("\n1. building blocks (relative differences)")
    v = jnp.asarray(rng.normal(size=T.shape[0]))
    print(f"  projector P v versus T T^T v: {np.linalg.norm(np.asarray(problem.project(v)) - T @ (T.T @ np.asarray(v))) / np.linalg.norm(v):.1e}")
    z = rng.normal(size=T.shape[1])
    dense_product = fd["V"] @ (fd["lam"] * (fd["V"].T @ z))
    print(f"  Fisher product T^T F_1 T z: {np.linalg.norm(T.T @ np.asarray(problem.matvec(jnp.asarray(T @ z))) - dense_product) / np.linalg.norm(dense_product):.1e}")
    grouped = fmf.fisher_problem(K0, A0, B0, indices, contraction="grouped", batch_size=150)
    print(f"  grouped contraction in 3 batches: probabilities {np.abs(grouped.probabilities - problem.probabilities).max():.1e}, "
          f"Fisher product {float(jnp.linalg.norm(grouped.matvec(v) - problem.matvec(v)) / jnp.linalg.norm(problem.matvec(v))):.1e}")
    h = jnp.asarray(0.1 * rng.normal(size=15))
    theta_truth, shapes = pdz.to_vec(*pt.truth), problem.shapes
    print(f"  gauge cost (traces) versus Pauli-transfer-matrix cost at a random gauge: "
          f"{abs(float(fmf._gauge_cost(h, theta_truth, shapes, target_kraus)) - float(pdz._gauge_cost(h, theta_truth, shapes, pt.R_target))):.1e}")
    fixed, info = fmf.gauge_fix_to_target(*pt.truth, target_kraus)
    print(f"  gauge fixing: Pauli transfer matrices versus `ptm_design`: {float(jnp.abs(fmf.pauli_transfer_matrices(fixed[0]) - pdz.pauli_transfer_matrices(K0)).max()):.1e}")
    gauge = fmf.gauge_context(problem, target_kraus)
    print(f"  smallest eigenvalue of the gauge Hessian H_hh (assumption G1): {gauge['min_eigenvalue']:.3f}")

    print(f"\n2. widths at m = {m}: Lanczos (matrix-free) versus dense")
    kept = np.cumsum(point["keep"]) - 1                                    # row of an element among the non-constant ones
    worst = int(np.argmax(dense["sigma_lan1"]))
    elements = [(0, 1, 1), (5, 7, 9), tuple(int(x) for x in np.unravel_index(np.nonzero(point["keep"])[0][worst], (6, 16, 16)))]
    for g, i, j in elements:
        row = kept[(g * 16 + i) * 16 + j]
        d = fmf.quantity_gradient(problem, fmf.ptm_element(g, i, j), gauge)
        print(f"  element R[{g}][{i},{j}]: gradient versus dense {np.linalg.norm(T.T @ np.asarray(d) - point['D'][row]) / np.linalg.norm(point['D'][row]):.1e}; "
              f"dense van Trees {dense['sigma_vt'][0][row, 0]:.4e} (tau {taus[0]}), {dense['sigma_vt'][1][row, 0]:.4e} (tau {taus[1]}), LAN {dense['sigma_lan1'][row] / np.sqrt(m):.4e}")
        for steps in (30, 100, 300, 700):
            report_widths(f"    {steps} Lanczos steps", fmf.lanczos_quadrature(problem, d, steps), m, taus)

    print("\n3. rate resolution: smallest Choi eigenvalue of each gate")
    rate_dense = pdz.lan_sigma1(point["D_rate"], fd) / np.sqrt(m)
    for g in range(6):
        d = fmf.quantity_gradient(problem, fmf.choi_eigenvalue(g, 3))
        row = g * 3 + 2
        quadrature = fmf.lanczos_quadrature(problem, d, 300)
        lan, _ = fmf.lan_sigma(quadrature, m)
        lower, upper = fmf.van_trees_sigma(quadrature, m, 0.1)
        print(f"  gate {g}: mu = {point['mu_small'][row]:.2e}; gradient versus dense {np.linalg.norm(T.T @ np.asarray(d) - point['D_rate'][row]) / np.linalg.norm(point['D_rate'][row]):.1e}; "
              f"sigma(mu) dense LAN {rate_dense[row]:.2e}, Lanczos LAN >= {lan:.2e}, van Trees(0.1) in [{lower:.2e}, {upper:.2e}]; u = mu / sigma = {point['mu_small'][row] / rate_dense[row]:.2f}")

    print("\n4. merged-outcome width (outcomes with m p < 5 merged per circuit), element", elements[2])
    g, i, j = elements[2]
    d = fmf.quantity_gradient(problem, fmf.ptm_element(g, i, j), gauge)
    rare = fmf.rare_outcomes(problem, m)
    J = (fd["U"] * np.sqrt(fd["lam"])) @ fd["V"].T * np.sqrt(fd["p"])[:, None]            # dense J_T
    J3, p2, rare2 = J.reshape(400, 4, -1), fd["p"].reshape(400, 4), rare
    rows = [J3[c, ~rare2[c]] / np.sqrt(p2[c, ~rare2[c]])[:, None] for c in range(400)]
    rows += [J3[c, rare2[c]].sum(axis=0, keepdims=True) / np.sqrt(p2[c, rare2[c]].sum()) for c in range(400) if rare2[c].any()]
    F_merged = np.concatenate(rows).T @ np.concatenate(rows)
    dz = T.T @ np.asarray(d)
    dense_merged = np.sqrt(dz @ np.linalg.solve(m * F_merged + np.eye(len(dz)) / taus[0] ** 2, dz))
    lower, upper = fmf.van_trees_sigma(fmf.lanczos_quadrature(problem, d, 300, rare=rare), m, taus[0])
    row = kept[(g * 16 + i) * 16 + j]
    print(f"  {rare.mean():.1%} of the outcomes are rare. van Trees(tau {taus[0]}): all outcomes {dense['sigma_vt'][0][row, 0]:.4e}, "
          f"merged dense {dense_merged:.4e}, merged Lanczos in [{lower:.4e}, {upper:.4e}]")

    print(f"\n5. direction of a quantity, nonlinearity ratio kappa (m = {m}, tau = {taus[0]})")
    for name, d in [("smallest Choi eigenvalue, gate 0", fmf.quantity_gradient(problem, fmf.choi_eigenvalue(0, 3))),
                    (f"element R[{g}][{i},{j}]", d), ("element R[0][1,1]", fmf.quantity_gradient(problem, fmf.ptm_element(0, 1, 1), gauge))]:
        v, residual = fmf.solve_direction(problem, d, m, taus[0])
        ratio = fmf.nonlinearity_ratio(problem, v, m)
        print(f"  {name:34s}: conjugate-gradient residual {residual:.1e}, sqrt(d . v) = {np.sqrt(float(d @ v)):.4e}, "
              f"kappa = {ratio['kappa']:.3f} (second derivatives), {fmf.nonlinearity_ratio_finite_step(problem, v, m):.3f} (finite step; "
              f"displacement of the gate set {ratio['step'] * float(jnp.linalg.norm(v)):.1e})")

    print("\n6. at saved estimates: deviance, and optimizer certificate for a gauge-invariant quantity (probability of outcome 0 of a test circuit)")
    for tag in ("default_0", "converged_0"):
        data = np.load(pt.DATA / "crb_estimates" / f"{tag}.npz")
        estimate = fmf.fisher_problem(*pt.load(pt.DATA / "crb_estimates" / f"{tag}.npz"), indices)
        frequencies = data["sampled"].T
        gradient = fmf.log_likelihood_gradient(estimate, frequencies, m)
        d = fmf.quantity_gradient(estimate, fmf.circuit_probability([0, 3, 5, 1, 2], 0))
        v, _ = fmf.solve_direction(estimate, d, m, taus[0])
        fit = fmf.deviance(estimate, frequencies, m)
        print(f"  {tag}: deviance {fit['deviance']:.1f} (script 19: 2 (final cost - floor)), N_sigma {fit['n_sigma']:.1f}; |Riemannian gradient of the log-likelihood| {float(jnp.linalg.norm(gradient)):.2e}; "
              f"shift of a test-circuit probability to the exact maximum {fmf.optimizer_shift(v, gradient):+.2e}, its van Trees width {np.sqrt(float(d @ v)):.2e}")


# ================================================================ four qubits
def four_qubit_gate_set(noise_strength, seed=42, ranks=(4, 4, 4)):
    """Target Kraus operators (9, 1, 16, 16) of `create_4q_gst_config` and a noisy gate set of the given Kraus, state
    and POVM ranks (target followed by random noise channels, as `ptm_tools.gate_set` for two qubits)."""
    from mGST.utility_functions_comparisons import (create_4q_gst_config, target_kraus_tensor_from_configuration, generate_target_povm,
                                                    generate_target_state, get_compressed_rep_from_mgst_output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target_kraus, _ = target_kraus_tensor_from_configuration(create_4q_gst_config(kraus_rank=1, num_gate_sequences=100, shots=100), backend="iqmfakeapollo")
    target = {"kraus": target_kraus, "povm": generate_target_povm(16), "state": generate_target_state(16)}
    noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=9, dim=16, rank_kraus=ranks[0], noise_strength=noise_strength, seed=seed)
    superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target, is_target_vectorized=True)
    gate_set = tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=ranks[0], state_rank=ranks[1], povm_rank=ranks[2]))
    return jnp.asarray(target_kraus), gate_set


if args.mode == "four_qubits":
    t0 = time.time()
    m, taus = args.m, (0.01, 0.03)
    target_kraus, truth = four_qubit_gate_set(args.noise_strength)
    shapes = pdz._shapes(*truth)
    n_id = fmf.num_identifiable_parameters(shapes)
    print(f"four qubits, 9 gates, ranks (Kraus, state, POVM) = (4, 4, 4), noise strength {args.noise_strength}")
    print(f"  tangent dimension {fmf.num_tangent_directions(shapes)}, identifiable parameters {n_id}, minimum number of circuits {int(np.ceil(n_id / 15))}")
    mu = np.asarray(pdz.choi_eigenvalues(truth[0]))
    sci = lambda x: " ".join(f"{y:.1e}" for y in x)
    print(f"  incoherent weight 1 - mu_1 per gate: {sci(1 - mu[:, 0])}\n  smallest Choi eigenvalue per gate:   {sci(mu[:, -1])}")
    fixed, info = fmf.gauge_fix_to_target(*truth, target_kraus)
    print(f"  gauge fixing: cost {info['cost']:.4e}, gradient {info['grad']:.1e} ({time.time() - t0:.0f} s)")

    indices = generate_sequence_indices(num_gates=9, num_circuits=args.N, seq_len_list=pdz.seq_len_list(args.L_max), seed=args.design_seed)["gate_indices"]
    problem = fmf.fisher_problem(*fixed, indices, batch_size=args.batch_size, contraction=args.contraction)
    counts = fmf.expected_count_summary(problem, m)
    print(f"design: {len(indices)} circuits = {len(indices) / np.ceil(n_id / 15):.2f} x minimum, seq_len_list {pdz.seq_len_list(args.L_max)}, m = {m}")
    print(f"  expected counts (assumption R1): smallest probability {counts['min_probability']:.1e}; {counts['share_below_threshold']:.1%} of the outcomes have m p < 5 "
          f"and {counts['share_below_one']:.1%} have m p < 1; the former carry {counts['probability_mass_below_threshold']:.1%} of the probability")
    v = jnp.asarray(rng.normal(size=problem.theta.shape[0]))
    t1 = time.time(); problem.matvec(v).block_until_ready(); compile_time = time.time() - t1
    t1 = time.time(); problem.matvec(v).block_until_ready(); product_time = time.time() - t1
    print(f"  one Fisher product: {product_time:.2f} s (compilation {compile_time:.0f} s, contraction '{args.contraction}')")
    gauge = fmf.gauge_context(problem, target_kraus)
    print(f"  smallest eigenvalue of the gauge Hessian H_hh (assumption G1): {gauge['min_eigenvalue']:.3f}")

    cz = 8
    coefficients, n_elements = fmf.random_ptm_coefficients(shapes, seed=1, gate=cz)
    quantities = [("smallest Choi eigenvalue, CZ-CZ", fmf.choi_eigenvalue(cz, 3), None, float(mu[cz, 3])),
                  ("smallest Choi eigenvalue, Rx on qubit 0", fmf.choi_eigenvalue(0, 3), None, float(mu[0, 3])),
                  ("entanglement infidelity, CZ-CZ", fmf.entanglement_infidelity(cz, target_kraus), gauge, float(fmf.entanglement_infidelity(cz, target_kraus)(*fixed))),
                  (f"random combination of the {n_elements} elements of CZ-CZ", fmf.ptm_combination(coefficients), gauge, None)]
    print(f"\nwidths at m = {m} ({args.steps} Lanczos steps per quantity, about {args.steps * product_time:.0f} s each). For a rate, 'value / LAN bound' is an upper limit on the rate resolution u = mu / sigma(mu).")
    first = None
    for name, quantity, gauge_arg, value in quantities:
        t1 = time.time()
        d = fmf.quantity_gradient(problem, quantity, gauge_arg)
        quadrature = fmf.lanczos_quadrature(problem, d, args.steps)
        report_widths(name, quadrature, m, taus, value)
        if first is None:
            first = (name, d)
        if "random combination" in name:
            lower, upper = fmf.van_trees_sigma(quadrature, m, taus[0])
            print(f"    -> root-mean-square van Trees width of an element of CZ-CZ (one random combination; average several for accuracy): "
                  f"[{lower / np.sqrt(n_elements):.2e}, {upper / np.sqrt(n_elements):.2e}]")
        print(f"    ({time.time() - t1:.0f} s)", flush=True)

    if args.kappa:
        name, d = first
        t1 = time.time()
        direction, residual = fmf.solve_direction(problem, d, m, taus[0], tol=1e-4, maxiter=150)
        ratio = fmf.nonlinearity_ratio(problem, direction, m)
        print(f"\nnonlinearity ratio of '{name}' along its van Trees direction (tau = {taus[0]}): kappa = {ratio['kappa']:.2f}, "
              f"finite step {fmf.nonlinearity_ratio_finite_step(problem, direction, m):.2f}; conjugate-gradient residual {residual:.1e} ({time.time() - t1:.0f} s)")
    print(f"\ntotal {time.time() - t0:.0f} s")


# ================================================================ the saved four-qubit run
if args.mode == "saved_run":
    folder = Path(__file__).resolve().parents[3] / "4q_gst" / "data"
    data = np.load(folder / "gst_tensors_4q_10000seq_500shots.npz", allow_pickle=True)
    result = np.load(folder / "optimization_results_4q_10000seq_500shots_rank_4.npz")
    indices, frequencies, m = [list(s) for s in data["indices_list"]], data["prob_matrix"].T, 500
    estimate = tuple(jnp.asarray(result[k]) for k in ("kraus_tensor", "povm_psd", "state_psd"))
    problem = fmf.fisher_problem(*estimate, indices, batch_size=2500)
    fit, counts = fmf.deviance(problem, frequencies, m), fmf.expected_count_summary(problem, m)
    mu = np.asarray(pdz.choi_eigenvalues(estimate[0]))
    print(f"saved four-qubit run: {len(indices)} circuits, {m} shots, estimate of shapes {[tuple(x.shape) for x in estimate]}")
    print(f"  frequencies sum to 1 per circuit: {np.abs(frequencies.sum(axis=1) - 1).max():.1e}; outcomes never observed: {(frequencies == 0).mean():.0%}")
    print(f"  expected counts at the estimate: smallest probability {counts['min_probability']:.1e}; {counts['share_below_threshold']:.0%} of the outcomes have m p < 5, {counts['share_below_one']:.0%} have m p < 1")
    print(f"  deviance {fit['deviance']:.0f}; Wilks reference k = 15 N - n_id = {fit['wilks_degrees_of_freedom']}; N_sigma = {fit['n_sigma']:.1f} (reference not calibrated, VALIDITY Section 5)")
    sci = lambda x: " ".join(f"{y:.1e}" for y in x)
    print(f"  incoherent weight 1 - mu_1 of the estimated gates: {sci(1 - mu[:, 0])}")
    print(f"  smallest Choi eigenvalue of the estimated gates:   {sci(mu[:, -1])}")
