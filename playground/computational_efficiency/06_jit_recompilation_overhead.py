"""Measure the JIT (re)compilation overhead of rebuilding a `JittedObjective` per outer iteration.

Background: `make_vectorized_cost` bakes the batch (gates, targets) into the jit closure, and
`JittedObjective.derivatives` caches the jitted grad/hvp on the instance. So resampling the batch at
every outer iteration of `run_riemannian_optimization` means a fresh objective and therefore a fresh
compilation of the cost plus the (rgrad, rhvp) pair of each of the three operators.

This script answers: how does that compilation cost compare to one outer iteration of actual work?

Run:
    python playground/computational_efficiency/06_jit_recompilation_overhead.py
    python playground/computational_efficiency/06_jit_recompilation_overhead.py --gate-set 3QXYCZ --num-sequences 1974
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mGST.jitted_objective import JittedObjective
from mGST.simulate import (
    apply_noisy_kraus_tensors_to_target_gate_set,
    batch_probability_matrices_and_indices,
    compute_probability_matrices,
    generate_random_gate_set_from_gate_set,
    generate_sequence_indices,
    get_random_noisy_kraus_tensors_for_gate_set,
)
from mGST.trust_region import _optimize_single_operator
from mGST.typing import TrustRegionOptions
from mGST.utility_functions_comparisons import (
    get_compressed_rep_from_mgst_output,
    get_target_operators_gate_set_name,
)

# Defaults match playground/reconstruction_test/2q_random_init_vs_sequences.ipynb
NOISE_STRENGTH = 5e-2
NUM_SHOTS = 1000
SEQ_LEN_LIST = [1, 8, 14]

# Random-init phase settings
NUM_ITERATIONS_TR = 5
NUM_ITERATIONS_CG = 10
TOL_GRAD = 1e-8
METRIC = "euclidean"
OPERATORS = ("povm", "kraus", "state")

# Random-init phase budget of `run_optimization_workflow_multi_init`
NUM_MAX_ITERATIONS_RANDOM_INITS = 5
NUM_MAX_RANDOM_INITS = 10

NUM_DISTINCT_BATCHES = 4


def build_true_gate_set(gate_set_name, kraus_rank):
    """Noisy gate set in PSD representation, as in the notebook."""
    target_operators, _ = get_target_operators_gate_set_name(gate_set_name=gate_set_name)
    num_gates, _, dim, _ = target_operators["kraus"].shape
    random_kraus_tensors = get_random_noisy_kraus_tensors_for_gate_set(
        num_gates=num_gates, dim=dim, rank_kraus=kraus_rank, noise_strength=NOISE_STRENGTH, seed=42
    )
    noisy_gate_set_superop = apply_noisy_kraus_tensors_to_target_gate_set(
        kraus_tensors_noisy=random_kraus_tensors, target_gate_set=target_operators, is_target_vectorized=True
    )
    kraus_tensor_noisy, povm_psd_noisy, state_psd_noisy = get_compressed_rep_from_mgst_output(
        *noisy_gate_set_superop.values(), kraus_rank=kraus_rank, state_rank=dim, povm_rank=dim
    )
    noisy_gate_set_psd = {"kraus": kraus_tensor_noisy, "povm": povm_psd_noisy, "state": state_psd_noisy}
    return noisy_gate_set_psd, num_gates, dim


def block(tree):
    """Wait for every array in `tree` so the timings include the actual device work."""
    jax.block_until_ready(tree)


def time_compilation(cost_fn_kwargs, operators, contraction):
    """Time a full rebuild: cost trace + (rgrad, rhvp) trace for each of the three operators."""
    kraus, povm, state = operators["kraus"], operators["povm"], operators["state"]

    start = time.perf_counter()
    objective = JittedObjective.from_cost_fn_kwargs(cost_fn_kwargs, contraction=contraction)
    block(objective.cost(kraus, povm, state))
    cost_time = time.perf_counter() - start

    per_operator = {}
    for operator_type in OPERATORS:
        tangent = 1e-3 * jnp.ones_like(operators[operator_type])
        start = time.perf_counter()
        rgrad_fn, rhvp_fn = objective.derivatives(operator_type, METRIC)
        block(rgrad_fn(kraus, povm, state))
        block(rhvp_fn(kraus, povm, state, tangent))
        per_operator[operator_type] = time.perf_counter() - start

    return objective, cost_time, per_operator


def time_steady_state(objective, operators, cost_fn_kwargs, num_repetitions=3):
    """Time one outer iteration (povm -> kraus -> state) with an already compiled objective.

    Every repetition restarts from the same operators, otherwise the adaptive CG/TR work per call
    would differ and the timings would not be comparable.
    """
    options = TrustRegionOptions(
        num_iterations=NUM_ITERATIONS_TR,
        num_iterations_cg=NUM_ITERATIONS_CG,
        tol_grad=TOL_GRAD,
        radius_heuristic="typical_distance",
        verbose=False,
        verbose_cg=False,
    )

    def one_outer_iteration():
        current = dict(operators)
        for operator_type in OPERATORS:
            updated, _, _, _ = _optimize_single_operator(
                kraus_tensor=current["kraus"],
                povm_psd=current["povm"],
                state_psd=current["state"],
                optimization_options=options,
                operator_type=operator_type,
                cost_function=None,
                cost_fn_kwargs=cost_fn_kwargs,
                jitted_objective=objective,
            )
            current[operator_type] = updated
        block(current)

    one_outer_iteration()  # warm up: the first pass still triggers the compilations

    timings = []
    for _ in range(num_repetitions):
        start = time.perf_counter()
        one_outer_iteration()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-set", default="2QXYICZ")
    parser.add_argument("--kraus-rank", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=650)
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to 30 * dim.")
    args = parser.parse_args()

    noisy_gate_set_psd, num_gates, dim = build_true_gate_set(args.gate_set, args.kraus_rank)
    batch_size = args.batch_size if args.batch_size is not None else 30 * dim

    indices_dict = generate_sequence_indices(
        num_gates=num_gates, num_circuits=args.num_sequences, seq_len_list=SEQ_LEN_LIST
    )
    probability_matrices = compute_probability_matrices(
        *noisy_gate_set_psd.values(), gate_indices=indices_dict["gate_indices"], num_shots=NUM_SHOTS, seed=42
    )
    init_operators = generate_random_gate_set_from_gate_set(gate_set=noisy_gate_set_psd, seeds=(1, 2, 3))

    print(
        f"{args.gate_set} | dim {dim} | {num_gates} gates | {args.num_sequences} sequences "
        f"| batch {batch_size} | kraus rank {args.kraus_rank}"
    )
    print(f"TR: {NUM_ITERATIONS_TR} iterations, CG: {NUM_ITERATIONS_CG} iterations\n")

    for contraction in ("prefix", "padded"):
        print(f"=== contraction: {contraction} ===")
        compile_times = []
        steady_times = []

        for batch_idx in range(NUM_DISTINCT_BATCHES):
            batched_probs, batched_indices = batch_probability_matrices_and_indices(
                probability_matrices=probability_matrices,
                indices_dict=indices_dict,
                batch_size=batch_size,
                seed=100 + batch_idx,
            )
            cost_fn_kwargs = {
                "indices_list": batched_indices["gate_indices"],
                "prob_matrix": batched_probs["sampled"],
                "jit": True,
                "use_log_likelihood": False,  # the random-init phase uses least squares
                "verbose": False,
            }

            objective, cost_time, per_operator = time_compilation(cost_fn_kwargs, init_operators, contraction)
            total_compile = cost_time + sum(per_operator.values())
            compile_times.append(total_compile)

            steady = time_steady_state(objective, init_operators, cost_fn_kwargs)
            steady_times.append(steady)

            detail = " ".join(f"{name}={value:6.2f}s" for name, value in per_operator.items())
            print(
                f" batch {batch_idx}: compile total={total_compile:6.2f}s "
                f"(cost={cost_time:5.2f}s {detail}) | outer iteration={steady:6.3f}s"
            )

        compile_mean = float(np.mean(compile_times))
        steady_mean = float(np.mean(steady_times))
        rebuilds = NUM_MAX_ITERATIONS_RANDOM_INITS * NUM_MAX_RANDOM_INITS
        print(
            f"\n mean compile {compile_mean:.2f}s vs mean outer iteration {steady_mean:.3f}s "
            f"-> compile is {compile_mean / steady_mean:.1f}x one iteration of work"
        )
        print(
            f" random-init phase ({NUM_MAX_RANDOM_INITS} inits x {NUM_MAX_ITERATIONS_RANDOM_INITS} iterations): "
            f"{rebuilds} rebuilds = {rebuilds * compile_mean / 60:.1f} min compile "
            f"+ {rebuilds * steady_mean / 60:.1f} min work"
        )
        print(
            f" with a pool of {NUM_MAX_ITERATIONS_RANDOM_INITS} reused batches: "
            f"{NUM_MAX_ITERATIONS_RANDOM_INITS * compile_mean / 60:.1f} min compile "
            f"+ {rebuilds * steady_mean / 60:.1f} min work\n"
        )


if __name__ == "__main__":
    main()
