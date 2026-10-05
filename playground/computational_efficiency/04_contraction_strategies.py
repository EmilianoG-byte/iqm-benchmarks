"""Compare contraction strategies for the vectorized cost (Table 6 of README.md):
    * padded  : all sequences padded with identity gates, one `lax.scan` (mGST.jitted_objective, contraction="padded")
    * buckets : sequences grouped by length, padded within each bucket, one scan per bucket (implemented here only)
    * prefix  : sorted by length, only the still-active prefix is contracted, no identities (contraction="prefix", default)

Usage: python 04_contraction_strategies.py --qubits 3q
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from mGST.jitted_objective import make_vectorized_cost, make_jitted_derivatives, _apply_gate_batched, _povm_probabilities

from common import load_data, time_call

parser = argparse.ArgumentParser()
parser.add_argument("--qubits", default="1q")
parser.add_argument("--buckets", type=int, default=4)
args = parser.parse_args()

raw, init, _ = load_data(args.qubits)
K, P, S = init["kraus"], init["povm"], init["state"]
indices_list, num_shots = raw["indices_list"], raw["num_shots"]
lengths = np.array([len(i) for i in indices_list])
print(f"{args.qubits}: {len(lengths)} sequences, sum of lengths {lengths.sum()}, padded work {len(lengths) * lengths.max()} "
      f"({lengths.sum() / (len(lengths) * lengths.max()):.0%} of the padded work are real gates)")


def make_bucket_cost(num_buckets: int):
    """Log-likelihood with sequences grouped into length buckets, padded with identities within each bucket."""
    order = np.argsort(lengths, kind="stable")
    prepared = []
    for rows in np.array_split(order, num_buckets):
        max_length = lengths[rows].max()
        gates = np.full((len(rows), max_length), -1, dtype=np.int32)       # -1 -> identity appended to the Kraus tensor
        for row, n in enumerate(rows):
            gates[row, :lengths[n]] = np.asarray(indices_list[n])[::-1]    # application order
        prepared.append((jnp.asarray(gates), jnp.asarray(raw["prob_matrix"]).T[rows]))

    @jax.jit
    def cost(k, p, s):
        identity = jnp.zeros((1,) + k.shape[1:], k.dtype).at[0, 0].set(jnp.eye(k.shape[-1]))
        k_extended = jnp.concatenate([k, identity])
        total = 0.0
        for gates, targets in prepared:
            rho = jnp.broadcast_to(s @ s.conj().T, (gates.shape[0],) + (s.shape[0],) * 2)
            rho, _ = jax.lax.scan(lambda r, g: (_apply_gate_batched(k_extended[g], r), None), rho, gates.T)
            total = total - num_shots * jnp.sum(jnp.log(jnp.abs(_povm_probabilities(p, rho))) * targets)
        return total
    return cost


common = dict(indices_list=indices_list, prob_matrix=raw["prob_matrix"], use_log_likelihood=True, num_shots=num_shots)
costs = {
    "padded scan": make_vectorized_cost(**common, contraction="padded"),
    f"{args.buckets} length buckets": make_bucket_cost(args.buckets),
    "active prefix (no identities)": make_vectorized_cost(**common, contraction="prefix"),
}
reference = float(costs["padded scan"](K, P, S))
z = jax.random.normal(jax.random.PRNGKey(0), K.shape) * 1e-3 + 0j
for name, cost in costs.items():
    rgrad_fn, rhvp_fn = make_jitted_derivatives(cost, "kraus")
    print(f" {name}: rel. diff cost vs padded {abs(float(cost(K, P, S)) - reference) / abs(reference):.1e}")
    time_call("rgrad kraus (first call = compile)", lambda: rgrad_fn(K, P, S), 20)
    time_call("rHVP kraus (first call = compile)", lambda: rhvp_fn(K, P, S, z), 20)
