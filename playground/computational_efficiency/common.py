"""Shared helpers for the computational-efficiency benchmarks."""
import pickle
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DATA_DIR = Path(__file__).parent / "data"


def load_data(name: str) -> tuple[dict, dict, dict]:
    """Load data/data_<name>.pkl (see 00_make_data.py).

    Returns:
        - raw: the pickled dictionary (numpy arrays).
        - init: initial operators {"kraus", "povm", "state"} as jax arrays.
        - cost_fn_kwargs: keyword arguments for `cost_function_jax_mps` (log-likelihood).
    """
    with open(DATA_DIR / f"data_{name}.pkl", "rb") as f:
        raw = pickle.load(f)
    init = {k: jnp.asarray(v) for k, v in raw["init"].items()}
    cost_fn_kwargs = dict(indices_list=[jnp.asarray(i) for i in raw["indices_list"]], prob_matrix=jnp.asarray(raw["prob_matrix"]),
                          jit=True, use_log_likelihood=True, verbose=False, num_shots=raw["num_shots"])
    return raw, init, cost_fn_kwargs


def time_call(name: str, f, n: int) -> None:
    """Print the time of the first call (includes compilation) and the mean time of the next n calls."""
    t = time.perf_counter(); jax.block_until_ready(f()); first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(f())
    print(f"  {name:50s} first call {first:7.2f}s | {1e3 * (time.perf_counter() - t) / n:9.2f} ms/call")
