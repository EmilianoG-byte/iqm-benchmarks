"""End-to-end `run_riemannian_optimization` with use_jit=False / True (Table 7 of README.md).

Usage:
    python 05_end_to_end_use_jit.py --qubits 1q --outer 20 --mode prefix
    python 05_end_to_end_use_jit.py --qubits 1q --outer 20 --mode padded
    python 05_end_to_end_use_jit.py --qubits 1q --outer 20 --mode eager      # use_jit=False (slow)
    python 05_end_to_end_use_jit.py --qubits 3q --outer 2  --mode prefix
"""
import argparse
import contextlib
import io
import time

from mGST.trust_region import run_riemannian_optimization
from mGST.low_level_jit import cost_function_jax_mps
from mGST.typing import OperatorSchedule, TrustRegionOptions

from common import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--qubits", default="1q")
parser.add_argument("--outer", type=int, default=20)
parser.add_argument("--mode", choices=["eager", "prefix", "padded"], default="prefix")
parser.add_argument("--absolute-tolerance", type=float, default=None)
args = parser.parse_args()

_, init, kw = load_data(args.qubits)
options = lambda linearize: TrustRegionOptions(num_iterations=5, verbose=False, verbose_cg=False, num_iterations_cg=10, tol_grad=1e-8, linearize=linearize)
optimization_schedule = {"kraus": OperatorSchedule.default(args.outer, options(True)),     # linearize is ignored when use_jit=True
                         "povm": OperatorSchedule.default(args.outer, options(False)),
                         "state": OperatorSchedule.default(args.outer, options(False))}
jit_kwargs = {} if args.mode == "eager" else {"use_jit": True, "jit_contraction": args.mode}

t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    _, history, reason = run_riemannian_optimization(
        *init.values(), cost_function=cost_function_jax_mps, cost_fn_kwargs=kw,
        optimization_schedule=optimization_schedule, num_iterations=args.outer, relative_precision=None,
        absolute_tolerance=args.absolute_tolerance, global_gradient_norm_tol=1e-10, noise_threshold=None,
        compute_least_squares=1, carry_over_radius=True, **jit_kwargs)
print(f"{args.qubits}, {args.outer} outer iterations, {args.mode}: {time.perf_counter() - t0:.1f}s (incl. compilation) | "
      f"final f={history['main'][-1]:.8e} | least squares={history['least_squares'][-1]:.6e}")
print(f"  reason: {reason}")
