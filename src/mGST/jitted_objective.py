"""
Vectorized, jit-compiled cost function and Riemannian derivatives for the trust-region optimization.

`cost_function_jax_mps` loops over the gate sequences in Python. Under `jax.jit` that loop is unrolled,
so the compiled program contains one MPS contraction per gate of every sequence and compilation time grows with the
number of circuits. Here all sequences are contracted together (batched over sequences), so the compiled program
contains the contraction body once per *position* in a sequence (or once in total, for the padded contraction),
independently of the number of circuits. The number of qubits only changes array shapes.

Two contraction strategies are available:
    * "prefix" (default): sequences are sorted by length. At position t, only the sequences that still have a gate
      at that position are contracted (a prefix of the sorted batch, whose size is known before compilation).
      No identity gates are ever multiplied. The loop over positions is unrolled, so compilation time grows with
      the maximal sequence length (not with the number of sequences).
    * "padded": all sequences are padded to the maximal length with an identity gate and contracted with `lax.scan`.
      Constant compilation time, but the identities are multiplied as well.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from mGST.typing import Tensor, Scalar
from mGST.trust_region_linearized import riemannian_and_euclidean_gradient_fn
from mGST.riemannian import riemannian_connection
from mGST.utility_functions_comparisons import tensors_to_isometries, isometry_to_tensor

CONTRACTIONS = ("prefix", "padded")
_OPERATOR_POSITION = {"kraus": 0, "povm": 1, "state": 2}


def _apply_gate_batched(kraus: Tensor, rho: Tensor) -> Tensor:
    """Apply one Kraus channel per sequence: rho_b -> sum_i K_bi rho_b K_bi^dagger.

    Args:
        kraus: Kraus operators, one gate per sequence. Shape (batch, kraus_rank, dim_out, dim_in).
        rho: Density matrices. Shape (batch, dim_in, dim_in).
    """
    return jnp.einsum("bijk,bkl,biml->bjm", kraus, rho, kraus.conj())


def _povm_probabilities(povm_psd: Tensor, rho: Tensor) -> Tensor:
    """Compute <povm_i|rho_b> for all POVM elements and sequences. Shape (batch, num_povm)."""
    return jnp.einsum("ijk,bkl,ijl->bi", povm_psd, rho, povm_psd.conj())


def _gates_in_application_order(indices_list: list, rows: np.ndarray, max_length: int, fill_value: int) -> np.ndarray:
    """Stack the gate sequences given by `rows` into a (len(rows), max_length) array.

    Column t holds the t-th gate *applied* to the state, i.e. the sequences are reversed
    (as in `reversed(gates_indices)` of `contract_mps_all_povm`). Missing entries are set to fill_value.
    """
    gates = np.full((len(rows), max_length), fill_value, dtype=np.int32)
    for row, sequence_idx in enumerate(rows):
        sequence = np.asarray(indices_list[sequence_idx])[::-1]
        gates[row, :len(sequence)] = sequence
    return gates


def make_vectorized_cost(indices_list: list, prob_matrix: Tensor, use_log_likelihood: bool = False,
                         num_shots: int | None = None, contraction: str = "prefix") -> Callable[[Tensor, Tensor, Tensor], Scalar]:
    """Build a jitted cost(kraus_tensor, povm_psd, state_psd) equivalent to `cost_function_jax_mps`.

    The data (gate sequences, probabilities) are compile-time constants. The operators are arguments,
    so changing any operator never triggers a recompilation.

    Args:
        indices_list: List of gate sequences (without padding), one per column of prob_matrix.
        prob_matrix: Measured probabilities. Shape (num_povm, num_sequences).
        use_log_likelihood: If True, the negative log-likelihood, otherwise the least squares.
        num_shots: Number of shots per sequence. Required for the log-likelihood.
        contraction: Contraction strategy, "prefix" or "padded". See module docstring.

    Returns:
        The jitted cost function.
    """
    if contraction not in CONTRACTIONS:
        raise ValueError(f"Invalid contraction {contraction}. Valid options are: {CONTRACTIONS}")
    if use_log_likelihood and num_shots is None:
        raise ValueError("num_shots must be provided when using log-likelihood cost function.")

    num_povm, num_sequences = prob_matrix.shape
    lengths = np.array([len(indices) for indices in indices_list])
    max_length = int(lengths.max())

    if contraction == "prefix":
        # Sort sequences by decreasing length: the sequences with a gate at position t are the first num_active[t].
        rows = np.argsort(-lengths, kind="stable")
        gates = jnp.asarray(_gates_in_application_order(indices_list, rows, max_length, fill_value=0))
        num_active = [int(np.sum(lengths[rows] > t)) for t in range(max_length)]
    else:
        rows = np.arange(num_sequences)
        # The identity gate is appended at index num_gates (known only at call time), so use -1 and wrap around.
        gates = jnp.asarray(_gates_in_application_order(indices_list, rows, max_length, fill_value=-1))
    # Probabilities in the same (sequence) order as the contraction. Shape (num_sequences, num_povm)
    targets = jnp.asarray(prob_matrix).T[rows]

    @jax.jit
    def cost(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor) -> Scalar:
        rho_init = state_psd @ state_psd.conj().T
        rho = jnp.broadcast_to(rho_init, (num_sequences,) + rho_init.shape)

        if contraction == "prefix":
            # Unrolled over positions; only the sequences that still have a gate are contracted.
            for t, n in enumerate(num_active):
                rho = rho.at[:n].set(_apply_gate_batched(kraus_tensor[gates[:n, t]], rho[:n]))
        else:
            _, kraus_rank, dim, _ = kraus_tensor.shape
            identity_gate = jnp.zeros((1, kraus_rank, dim, dim), kraus_tensor.dtype).at[0, 0].set(jnp.eye(dim))
            kraus_extended = jnp.concatenate([kraus_tensor, identity_gate], axis=0)  # index -1 is the identity
            apply_position = lambda rho, gates_t: (_apply_gate_batched(kraus_extended[gates_t], rho), None)
            rho, _ = jax.lax.scan(apply_position, rho, gates.T) # this is what helps avoiding the unrolling of the for loop over all sequences.

        probabilities = _povm_probabilities(povm_psd, rho)  # (num_sequences, num_povm)
        if use_log_likelihood:
            return -num_shots * jnp.sum(jnp.log(jnp.abs(probabilities)) * targets)
        return jnp.sum(jnp.abs(probabilities - targets) ** 2) / (num_sequences * num_povm)

    return cost


def make_jitted_derivatives(cost: Callable[[Tensor, Tensor, Tensor], Scalar], operator_type: str, metric: str = "euclidean") -> tuple[Callable, Callable]:
    """Build the jitted Riemannian gradient and Hessian-vector product with respect to one operator.

    The other two operators are arguments, not closed-over constants, so one compilation serves the whole optimization.

    Args:
        cost: Cost function cost(kraus_tensor, povm_psd, state_psd), e.g. from `make_vectorized_cost`.
        operator_type: The operator to differentiate with respect to ('kraus', 'povm', 'state').
        metric: The metric used for the Riemannian gradient and Hessian ('euclidean' or 'canonical').

    Returns:
        A tuple containing:
        - rgrad_fn(kraus_tensor, povm_psd, state_psd) -> (Riemannian gradient, Euclidean gradient 2df/dx),
          as `riemannian_and_euclidean_gradient_fn`.
        - rhvp_fn(kraus_tensor, povm_psd, state_psd, z) -> (Riemannian Hessian-vector product, Euclidean Hessian-vector product).
          The Euclidean one is what `compute_quality_quotient` needs.
    """
    position = _OPERATOR_POSITION[operator_type]

    def rgrad_and_euclidean(x: Tensor, operators: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        def cost_x(y):
            arguments = list(operators)
            arguments[position] = y
            return cost(*arguments)
        return riemannian_and_euclidean_gradient_fn(x=x, cost_fn=cost_x, operator_type=operator_type, metric=metric)

    @jax.jit
    def rgrad_fn(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor) -> tuple[Tensor, Tensor]:
        operators = (kraus_tensor, povm_psd, state_psd)
        return rgrad_and_euclidean(operators[position], operators)

    @jax.jit
    def rhvp_fn(kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        operators = (kraus_tensor, povm_psd, state_psd)
        x = operators[position]
        (rgrad, _), (d_rgrad, euclidean_hvp) = jax.jvp(lambda y: rgrad_and_euclidean(y, operators), (x,), (z,))
        x_matrix, z_matrix, rgrad_matrix, d_rgrad_matrix = tensors_to_isometries(x, z, rgrad, d_rgrad, operator_type=operator_type)
        rhvp = riemannian_connection(x=x_matrix, w_x=rgrad_matrix, z=z_matrix, Dw_x_to_z=d_rgrad_matrix, metric=metric)
        return isometry_to_tensor(rhvp, x.shape), euclidean_hvp

    return rgrad_fn, rhvp_fn


@dataclass
class JittedObjective:
    """Jitted cost function and (lazily built) jitted derivatives for all operators.

    Build it once per optimization run: the jit caches are attached to the function objects held here,
    so building a new JittedObjective triggers new compilations.
    """
    cost: Callable[[Tensor, Tensor, Tensor], Scalar]
    _derivatives: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_cost_fn_kwargs(cls, cost_fn_kwargs: dict, contraction: str = "prefix") -> "JittedObjective":
        """Build the objective from the keyword arguments used for `cost_function_jax_mps`."""
        cost = make_vectorized_cost(
            indices_list=cost_fn_kwargs["indices_list"],
            prob_matrix=cost_fn_kwargs["prob_matrix"],
            use_log_likelihood=cost_fn_kwargs.get("use_log_likelihood", False),
            num_shots=cost_fn_kwargs.get("num_shots"),
            contraction=contraction,
        )
        return cls(cost=cost)

    def derivatives(self, operator_type: str, metric: str) -> tuple[Callable, Callable]:
        """Return (rgrad_fn, rhvp_fn) for the operator and metric, building them on first use."""
        key = (operator_type, metric)
        if key not in self._derivatives:
            self._derivatives[key] = make_jitted_derivatives(self.cost, operator_type=operator_type, metric=metric)
        return self._derivatives[key]

    def bind(self, operator_type: str, kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, metric: str) -> tuple[Callable, Callable, Callable]:
        """Fix the two operators that are not optimized and return single-argument functions of the optimized one.

        Args:
            operator_type: The operator being optimized ('kraus', 'povm', 'state').
            kraus_tensor, povm_psd, state_psd: Current operators. The one given by operator_type is ignored.
            metric: The metric used for the Riemannian gradient and Hessian.

        Returns:
            A tuple containing:
            - cost_fn(x) -> Scalar
            - rgrad_and_euclidean_fn(x) -> (Riemannian gradient, Euclidean gradient)
            - rhvp_and_euclidean_fn(x, z) -> (Riemannian HVP, Euclidean HVP)
        """
        rgrad_fn, rhvp_fn = self.derivatives(operator_type, metric)
        position = _OPERATOR_POSITION[operator_type]
        operators = (kraus_tensor, povm_psd, state_psd)

        def with_x(x):
            arguments = list(operators)
            arguments[position] = x
            return arguments

        cost_fn = lambda x: self.cost(*with_x(x))
        rgrad_and_euclidean_fn = lambda x: rgrad_fn(*with_x(x))
        rhvp_and_euclidean_fn = lambda x, z: rhvp_fn(*with_x(x), z)
        return cost_fn, rgrad_and_euclidean_fn, rhvp_and_euclidean_fn

    def validate(self, cost_function: Callable, cost_fn_kwargs: dict, kraus_tensor: Tensor, povm_psd: Tensor, state_psd: Tensor, rtol: float = 1e-8) -> None:
        """Check that the jitted cost agrees with `cost_function` at the given operators.

        The jitted cost re-implements `cost_function_jax_mps` only. This guards against silently optimizing a different
        objective (e.g. when `cost_function` is a regularized version).

        Raises:
            ValueError: If the two cost values differ by more than rtol (relative).
        """
        reference = float(cost_function(kraus_tensor=kraus_tensor, povm_psd=povm_psd, state_psd=state_psd, **cost_fn_kwargs))
        value = float(self.cost(kraus_tensor, povm_psd, state_psd))
        if abs(value - reference) > rtol * max(1.0, abs(reference)):
            raise ValueError(
                f"Jitted cost {value:.12e} differs from cost_function {reference:.12e}. "
                "The jitted objective only reproduces `cost_function_jax_mps`; use `use_jit=False` for other cost functions."
            )
