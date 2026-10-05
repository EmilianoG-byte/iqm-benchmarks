"""Per-call cost and compile time: current eager path vs `mGST.jitted_objective`, for 1q / 2q / 3q (Table 5 of README.md).

Usage:
    python 03_scaling_eager_vs_jitted.py --qubits 3q                     # prefix contraction (default)
    python 03_scaling_eager_vs_jitted.py --qubits 3q --contraction padded
"""
import argparse

import jax
import jax.numpy as jnp
from mGST.low_level_jit import cost_function_jax_mps
from mGST.trust_region_linearized import riemannian_and_euclidean_gradient_fn, linearize_riemannian_gradient_and_hvp
from mGST.trust_region import riemannian_hessian_vector_fn
from mGST.jitted_objective import make_vectorized_cost, make_jitted_derivatives

from common import load_data, time_call

parser = argparse.ArgumentParser()
parser.add_argument("--qubits", default="1q")
parser.add_argument("--contraction", default="prefix", choices=["prefix", "padded"])
args = parser.parse_args()

raw, init, kw = load_data(args.qubits)
K, P, S = init["kraus"], init["povm"], init["state"]
lengths = [len(i) for i in raw["indices_list"]]
print(f"{args.qubits}: kraus {K.shape}, povm {P.shape}, state {S.shape}, {len(lengths)} sequences, "
      f"length mean {sum(lengths) / len(lengths):.1f} max {max(lengths)}, contraction={args.contraction}")

cost_vectorized = make_vectorized_cost(raw["indices_list"], raw["prob_matrix"], use_log_likelihood=True,
                                       num_shots=raw["num_shots"], contraction=args.contraction)
print(f"  cost loop {float(cost_function_jax_mps(K, P, S, **kw)):.10e} | cost vectorized {float(cost_vectorized(K, P, S)):.10e}")
num_eager = 1 if K.shape[2] >= 8 else 3
operators = {"kraus": K, "povm": P, "state": S}

for operator_type in ["kraus", "povm"]:
    x = operators[operator_type]
    z = jax.random.normal(jax.random.PRNGKey(0), x.shape) * 1e-3 + 0j
    print(f" [{operator_type}]")

    def cost_x(y, operator_type=operator_type):
        arguments = dict(operators); arguments[operator_type] = y
        return cost_function_jax_mps(arguments["kraus"], arguments["povm"], arguments["state"], **kw)
    rgrad_eager = lambda y: riemannian_and_euclidean_gradient_fn(x=y, cost_fn=cost_x, operator_type=operator_type, metric="euclidean")
    time_call("current: rgrad (eager)", lambda: rgrad_eager(x), num_eager)
    time_call("current: linearize (eager, per TR iteration)", lambda: linearize_riemannian_gradient_and_hvp(x, rgrad_eager, operator_type, "euclidean")[:2], num_eager)
    linear_map = linearize_riemannian_gradient_and_hvp(x, rgrad_eager, operator_type, "euclidean")[2]
    time_call("current: apply linear map (per tCG iteration)", lambda: linear_map(z), num_eager)
    time_call("current: rHVP non-linearized (per tCG iteration)", lambda: riemannian_hessian_vector_fn(x, z, cost_x, operator_type, "euclidean"), num_eager)

    rgrad_fn, rhvp_fn = make_jitted_derivatives(cost_vectorized, operator_type, metric="euclidean")
    time_call("jitted: rgrad (first call = compile)", lambda: rgrad_fn(K, P, S), 20)
    time_call("jitted: rHVP (first call = compile)", lambda: rhvp_fn(K, P, S, z), 20)

    rgrad_new, _ = rgrad_fn(K, P, S); rgrad_old, _ = rgrad_eager(x)
    rhvp_new, _ = rhvp_fn(K, P, S, z); rhvp_old = riemannian_hessian_vector_fn(x, z, cost_x, operator_type, "euclidean")
    print(f"  rel. diff rgrad {float(jnp.linalg.norm(rgrad_new - rgrad_old) / jnp.linalg.norm(rgrad_old)):.1e} | "
          f"rel. diff rHVP {float(jnp.linalg.norm(rhvp_new - rhvp_old) / jnp.linalg.norm(rhvp_old)):.1e}")
