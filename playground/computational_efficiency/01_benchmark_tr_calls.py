"""Time `run_riemannian_optimization` and count the expensive calls (gradients, HVPs, linearizations, tCG).

Used to show that the tCG fix / projection / ratio regularization did not make an outer iteration slower,
and that the old relative-precision criterion stopped the optimization too early (Tables 1-3 of README.md).

Compare two code versions on the same data by putting the old one first on the path:
    git worktree add /tmp/mgst_prefix 6d68d64          # commit before the tCG fix
    PYTHONPATH=/tmp/mgst_prefix/src python 01_benchmark_tr_calls.py --qubits 1q --schedule kraus --outer 20
    python 01_benchmark_tr_calls.py --qubits 1q --schedule kraus --outer 20
    git worktree remove /tmp/mgst_prefix

Options:
    --relative-precision 1e-3      enable the relative-precision criterion (default: disabled -> always --outer iterations)
    --absolute-tolerance 0.1       enable the absolute-tolerance criterion (only exists after commit 81a14cf)
    --use-jit                      use the jitted objective (only exists with src/mGST/jitted_objective.py)
"""
import argparse
import collections
import contextlib
import io
import time

import jax
import mGST.trust_region as tr
import mGST.trust_region_linearized as trl
from mGST.low_level_jit import cost_function_jax_mps
from mGST.typing import OperatorSchedule, TrustRegionOptions

from common import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--qubits", default="1q")
parser.add_argument("--schedule", choices=["kraus", "all"], default="kraus", help="kraus: only Kraus linearized. all: all operators linearized (POVM/state with radius_init=1e-4)")
parser.add_argument("--outer", type=int, default=20)
parser.add_argument("--relative-precision", type=float, default=None)
parser.add_argument("--absolute-tolerance", type=float, default=None)
parser.add_argument("--use-jit", action="store_true")
args = parser.parse_args()
print("code:", tr.__file__)

# ---- count and time the expensive functions (wrappers around the module-level names used by the TR code) ----
counts, times = collections.Counter(), collections.Counter()

def wrap(module, name):
    function = getattr(module, name)
    def wrapper(*a, **k):
        t = time.perf_counter()
        out = function(*a, **k)
        if not callable(out):
            jax.block_until_ready(out)
        counts[name] += 1
        times[name] += time.perf_counter() - t
        return out
    setattr(module, name, wrapper)

for name in ["truncated_conjugate_gradient", "compute_quality_quotient", "riemannian_hessian_vector_fn",
             "retraction_first_order", "linearize_riemannian_gradient_and_hvp", "riemannian_and_euclidean_gradient_fn"]:
    wrap(tr, name)
wrap(trl, "rhessian_vector_product_from_linear_map")

def cost_function(*a, **k):
    counts["cost_fn (incl. traced)"] += 1
    return cost_function_jax_mps(*a, **k)

# ---- problem ----
_, init, kw = load_data(args.qubits)
options = lambda **k: TrustRegionOptions(num_iterations=5, verbose=False, verbose_cg=False, num_iterations_cg=10, tol_grad=1e-8, **k)
schedule = lambda o: OperatorSchedule.default(args.outer, o)
if args.schedule == "kraus":
    optimization_schedule = {"kraus": schedule(options(linearize=True)), "povm": schedule(options(linearize=False)), "state": schedule(options(linearize=False))}
else:
    small = options(linearize=True, radius_init=1e-4)
    optimization_schedule = {"kraus": schedule(options(linearize=True)), "povm": schedule(small), "state": schedule(small)}

extra = {}
if args.absolute_tolerance is not None:
    extra["absolute_tolerance"] = args.absolute_tolerance
if args.use_jit:
    extra["use_jit"] = True

# warm up the per-sequence jit caches, so compilation of the loop-based cost is not timed
cost_function(*init.values(), **kw); cost_function(*init.values(), **{**kw, "use_log_likelihood": False})
counts.clear(); times.clear()

t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    _, history, reason = tr.run_riemannian_optimization(
        *init.values(), cost_function=cost_function, cost_fn_kwargs=kw, verbose=False,
        optimization_schedule=optimization_schedule, num_iterations=args.outer, relative_precision=args.relative_precision,
        global_gradient_norm_tol=1e-10, noise_threshold=None, compute_least_squares=1, carry_over_radius=True, **extra)
total = time.perf_counter() - t0
# compute_least_squares=1 stores one value per operator -> 3 per outer iteration
print(f"TOTAL {total:.1f}s | outer iterations ~{len(history['least_squares']) / 3:.1f} | final f={history['main'][-1]:.6e}")
print(f"  reason: {reason}")
print("  (times of nested calls overlap, e.g. gradients traced inside linearize)")
for name in counts:
    print(f"  {name:42s} n={counts[name]:6d}  t={times[name]:7.1f}s")
