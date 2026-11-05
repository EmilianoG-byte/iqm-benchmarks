"""Checks for the per-outer-iteration batch resampling of the random-init phase.

Run:
    python playground/computational_efficiency/07_check_batch_pool.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mGST.jitted_objective import JittedObjective
from mGST.low_level_jit import cost_function_jax_mps
from mGST.simulate import (
    apply_noisy_kraus_tensors_to_target_gate_set,
    build_batch_pool,
    compute_probability_matrices,
    generate_random_gate_set_from_gate_set,
    generate_sequence_indices,
    get_random_noisy_kraus_tensors_for_gate_set,
    run_optimization_workflow_batched,
    run_optimization_workflow_multi_init,
)
from mGST.trust_region import run_riemannian_optimization
from mGST.typing import OperatorSchedule, TrustRegionOptions
from mGST.utility_functions_comparisons import (
    get_compressed_rep_from_mgst_output,
    get_target_operators_gate_set_name,
    kraus_tensor_to_mgst,
)

GATE_SET_NAME = "2QXYICZ"
KRAUS_RANK = 4
NOISE_STRENGTH = 5e-2
NUM_SHOTS = 1000
SEQ_LEN_LIST = [1, 8, 14]
NUM_SEQUENCES = 400
BATCH_SIZE = 120
NUM_ITERATIONS = 3

OPERATORS = ("kraus", "povm", "state")


def build_problem():
    target_operators, _ = get_target_operators_gate_set_name(gate_set_name=GATE_SET_NAME)
    num_gates, _, dim, _ = target_operators["kraus"].shape
    noisy_superop = apply_noisy_kraus_tensors_to_target_gate_set(
        kraus_tensors_noisy=get_random_noisy_kraus_tensors_for_gate_set(
            num_gates=num_gates, dim=dim, rank_kraus=KRAUS_RANK, noise_strength=NOISE_STRENGTH, seed=42
        ),
        target_gate_set=target_operators,
        is_target_vectorized=True,
    )
    kraus, povm, state = get_compressed_rep_from_mgst_output(
        *noisy_superop.values(), kraus_rank=KRAUS_RANK, state_rank=dim, povm_rank=dim
    )
    operators_psd_true = {"kraus": kraus, "povm": povm, "state": state}
    target_superops = {
        "kraus": kraus_tensor_to_mgst(target_operators["kraus"]),
        "povm": target_operators["povm"],
        "state": target_operators["state"],
    }

    indices_dict = generate_sequence_indices(
        num_gates=num_gates, num_circuits=NUM_SEQUENCES, seq_len_list=SEQ_LEN_LIST
    )
    probability_matrices = compute_probability_matrices(
        *operators_psd_true.values(), gate_indices=indices_dict["gate_indices"], num_shots=NUM_SHOTS, seed=42
    )
    cost_fn_kwargs = {
        "indices_list": indices_dict["gate_indices"],
        "prob_matrix": probability_matrices["sampled"],
        "jit": True,
        "use_log_likelihood": False,
        "verbose": False,
    }
    return operators_psd_true, target_superops, indices_dict, probability_matrices, cost_fn_kwargs


def make_schedule(num_iterations):
    options = TrustRegionOptions(num_iterations=3, num_iterations_cg=5, tol_grad=1e-8, radius_heuristic="typical_distance")
    schedule = OperatorSchedule.default(num_iterations, options)
    return {operator: schedule for operator in OPERATORS}


def check(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    return passed


def main():
    operators_psd_true, target_superops, indices_dict, probability_matrices, cost_fn_kwargs = build_problem()
    init_operators = generate_random_gate_set_from_gate_set(gate_set=operators_psd_true, seeds=[1, 2, 3])
    schedule = make_schedule(NUM_ITERATIONS)
    results = []

    # 2. A caller-built JittedObjective must give the same result as one built internally.
    objective = JittedObjective.from_cost_fn_kwargs(cost_fn_kwargs)
    common = dict(
        cost_function=cost_function_jax_mps,
        cost_fn_kwargs=cost_fn_kwargs,
        num_iterations=NUM_ITERATIONS,
        optimization_schedule=schedule,
        carry_over_radius=False,
        relative_precision=None,
        global_gradient_norm_tol=0.0,
    )
    _, internal_history, _ = run_riemannian_optimization(*init_operators.values(), **common)
    _, external_history, _ = run_riemannian_optimization(
        *init_operators.values(), jitted_objective=objective, **common
    )
    results.append(
        check(
            "prebuilt JittedObjective matches internally built one",
            np.allclose(np.array(internal_history["main"]), np.array(external_history["main"])),
            f"{internal_history['main'][-1]:.12e} vs {external_history['main'][-1]:.12e}",
        )
    )

    # 3. With a pool of one batch, the driver must reproduce a single multi-iteration call on that batch.
    pool = build_batch_pool(
        probability_matrices=probability_matrices,
        indices_dict=indices_dict,
        batch_size=BATCH_SIZE,
        num_batches=1,
        cost_fn_kwargs=cost_fn_kwargs,
        seed=7,
        validate_at=init_operators,
    )
    batch_cost_fn_kwargs, batch_objective = pool[0]
    reference_operators, reference_history, _ = run_riemannian_optimization(
        *init_operators.values(),
        cost_function=cost_function_jax_mps,
        cost_fn_kwargs=batch_cost_fn_kwargs,
        num_iterations=NUM_ITERATIONS,
        optimization_schedule=schedule,
        jitted_objective=batch_objective,
        carry_over_radius=False,
        relative_precision=None,
        global_gradient_norm_tol=0.0,
    )
    driver_result = run_optimization_workflow_batched(
        init_operators=init_operators,
        batch_pool=pool,
        cost_fn_kwargs=cost_fn_kwargs,
        num_iterations=NUM_ITERATIONS,
        optimization_schedule=schedule,
    )
    same_batch_values = np.allclose(
        np.array(reference_history["main"][1:]), np.array(driver_result["cost_fn_history"]["batch"])
    )
    same_operators = all(
        np.allclose(reference_operators[operator], driver_result["optimized_operators"][operator])
        for operator in OPERATORS
    )
    results.append(check("driver with a single-batch pool matches one multi-iteration call", same_batch_values and same_operators))

    # 4. The full-data history has one entry per outer iteration and matches a direct evaluation.
    history = driver_result["cost_fn_history"]["main"]
    final_cost = float(cost_function_jax_mps(*driver_result["optimized_operators"].values(), **cost_fn_kwargs))
    results.append(
        check(
            "full-data history length and final value",
            len(history) == NUM_ITERATIONS + 1 and np.isclose(history[-1], final_cost),
            f"len={len(history)}, {history[-1]:.12e} vs {final_cost:.12e}",
        )
    )

    # 5. The pool is compiled once: only num_batches objectives are built per multi-init call.
    num_batches = 2
    num_random_inits = 3
    num_random_iterations = 2
    built = []
    original_from_cost_fn_kwargs = JittedObjective.from_cost_fn_kwargs.__func__

    def counting_from_cost_fn_kwargs(cls, *args, **kwargs):
        built.append(1)
        return original_from_cost_fn_kwargs(cls, *args, **kwargs)

    JittedObjective.from_cost_fn_kwargs = classmethod(counting_from_cost_fn_kwargs)
    try:
        options_multi_init = {
            "schedule": make_schedule(num_random_iterations),
            "num_iterations_outer": num_random_iterations,
            "relative_precision": None,
            "global_gradient_norm_tol": 0.0,
            "carry_over_radius": False,
        }
        multi_init_kwargs = dict(
            num_shots=NUM_SHOTS,
            operators_psd_true=operators_psd_true,
            target_superops=target_superops,
            optimization_options=options_multi_init,
            indices_dict=indices_dict,
            probability_matrices=probability_matrices,
            num_max_iterations_random_inits=num_random_iterations,
            num_max_random_inits=num_random_inits,
            batch_size=BATCH_SIZE,
            threshold_multiplier=3.0,
            num_batches=num_batches,
        )
        final_a, partial_a = run_optimization_workflow_multi_init(seed=123, **multi_init_kwargs)
        # pool + full-data objective + one built by the final refinement run
        expected_builds = num_batches + 1 + 1
        results.append(
            check(
                "objectives compiled once per pool",
                len(built) == expected_builds,
                f"built={len(built)}, expected={expected_builds}",
            )
        )
    finally:
        JittedObjective.from_cost_fn_kwargs = classmethod(original_from_cost_fn_kwargs)

    # 6. Reproducibility across seeds.
    final_b, partial_b = run_optimization_workflow_multi_init(seed=123, **multi_init_kwargs)
    final_c, partial_c = run_optimization_workflow_multi_init(seed=321, **multi_init_kwargs)
    same_seed_matches = np.isclose(
        partial_a["cost_fn_history"]["main"][-1], partial_b["cost_fn_history"]["main"][-1]
    )
    different_seed_differs = not np.isclose(
        partial_a["cost_fn_history"]["main"][-1], partial_c["cost_fn_history"]["main"][-1]
    )
    results.append(
        check(
            "same seed reproduces, different seed differs",
            same_seed_matches and different_seed_differs,
            f"{partial_a['cost_fn_history']['main'][-1]:.6e} / "
            f"{partial_b['cost_fn_history']['main'][-1]:.6e} / "
            f"{partial_c['cost_fn_history']['main'][-1]:.6e}",
        )
    )

    # 7. The full-data cost decreases over the random-init phase despite the batch changing.
    full_data_history = partial_a["cost_fn_history"]["main"]
    results.append(
        check(
            "full-data cost decreases over the random-init phase",
            full_data_history[-1] < full_data_history[0],
            " -> ".join(f"{value:.3e}" for value in full_data_history),
        )
    )

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
