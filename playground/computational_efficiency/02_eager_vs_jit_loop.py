"""Eager autodiff vs naively jitting the existing loop-based cost function (Table 4 of README.md).

Shows the two problems that motivate the vectorized implementation:
  1. eager gradients / HVPs are dominated by dispatch overhead (~100 small jitted calls per evaluation),
  2. `jax.jit` around the Python loop over sequences unrolls it, so compilation is slow and grows with #circuits.

Usage: python 02_eager_vs_jit_loop.py --qubits 1q      (do not use 3q: the jitted HVP compile takes many minutes)
"""
import argparse
import time

import jax
import jax.numpy as jnp
from mGST.low_level_jit import cost_function_jax_mps
from mGST.trust_region_linearized import riemannian_and_euclidean_gradient_fn, linearize_riemannian_gradient_and_hvp
from mGST.trust_region import riemannian_hessian_vector_fn

from common import load_data, time_call

parser = argparse.ArgumentParser()
parser.add_argument("--qubits", default="1q")
args = parser.parse_args()

_, init, kw = load_data(args.qubits)
K, P, S = init["kraus"], init["povm"], init["state"]
Z = jax.random.normal(jax.random.PRNGKey(0), K.shape) * 1e-3 + 0j
operator_type, metric = "kraus", "euclidean"

# ---- eager (the current code path) ----
cost_kraus = lambda x: cost_function_jax_mps(kraus_tensor=x, povm_psd=P, state_psd=S, **kw)
rgrad_eager = lambda x: riemannian_and_euclidean_gradient_fn(x=x, cost_fn=cost_kraus, operator_type=operator_type, metric=metric)
time_call("eager cost", lambda: cost_kraus(K), 10)
time_call("eager rgrad", lambda: rgrad_eager(K), 10)
time_call("eager linearize (once per TR iteration)", lambda: linearize_riemannian_gradient_and_hvp(K, rgrad_eager, operator_type, metric)[:2], 10)
linear_map = linearize_riemannian_gradient_and_hvp(K, rgrad_eager, operator_type, metric)[2]
time_call("eager apply linear map (once per tCG iteration)", lambda: linear_map(Z), 10)
time_call("eager rHVP non-linearized (once per tCG iteration)", lambda: riemannian_hessian_vector_fn(K, Z, cost_kraus, operator_type, metric), 10)

# ---- jit around the loop-based cost (the loop over sequences is unrolled during tracing) ----
cost_full = lambda k, p, s: cost_function_jax_mps(kraus_tensor=k, povm_psd=p, state_psd=s, **kw)

@jax.jit
def rgrad_jit(k, p, s):
    return riemannian_and_euclidean_gradient_fn(x=k, cost_fn=lambda x: cost_full(x, p, s), operator_type=operator_type, metric=metric)

@jax.jit
def rhvp_jit(k, p, s, z):
    f = lambda x: riemannian_and_euclidean_gradient_fn(x=x, cost_fn=lambda y: cost_full(y, p, s), operator_type=operator_type, metric=metric)
    return jax.jvp(f, (k,), (z,))

time_call("jit(loop) rgrad (first call = compile)", lambda: rgrad_jit(K, P, S), 10)
time_call("jit(loop) rHVP (first call = compile)", lambda: rhvp_jit(K, P, S, Z), 10)
print("  max |diff| rgrad:", float(jnp.max(jnp.abs(rgrad_jit(K, P, S)[0] - rgrad_eager(K)[0]))))
