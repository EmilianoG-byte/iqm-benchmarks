"""Beta module to possibly improve the efficiency of Riemannian gradient and Hessian-vector computations."""

from mGST.typing import Scalar, Matrix, Tensor
from typing import Callable
from mGST.utility_functions_comparisons import isometry_to_tensor, tensors_to_isometries
from mGST.riemannian import riemannian_connection
from mGST.automatic_diff import automatic_gradient
from mGST.utility_functions_comparisons import euclidean_gradients_to_stiefel

import jax

def riemannian_and_euclidean_gradient_fn(x:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type: str, metric: str) -> tuple[Tensor, Tensor]:
    """Compute the Riemannian gradient of a cost function at a point x, along with the raw Euclidean gradient.

    Handles batch dimensions.

    Args:
        x: The point at which to evaluate the gradient.
        cost_fn: The cost function to differentiate.

    Returns:
        A tuple containing:
        - The Riemannian gradient of the cost function at x.
        - The raw Euclidean gradient (2df/dx), returned so callers can reuse it without recomputing it.
    """
    # Compute the Euclidean gradient
    gradient_euclidean_jax = automatic_gradient(cost_fn)(x) # 2df/dx

    # Take the adjoint due to JAX convention
    gradient_euclidean = gradient_euclidean_jax.conj() # 2df/dx*
    # Riemannian gradient based on metric
    gradient_stiefel_matrix, _ = euclidean_gradients_to_stiefel(
        gradient_tensor=gradient_euclidean,
        operator_tensor=x, operator_type=operator_type, metric=metric,
    )
    return isometry_to_tensor(gradient_stiefel_matrix, x.shape), gradient_euclidean_jax

def linearize_riemannian_gradient_and_hvp(x:Tensor, rgrad_and_euclidean_fn:Callable[[Tensor], Tensor], operator_type:str, metric:str)-> tuple[Tensor, Tensor, Callable[[Tensor], tuple[Tensor, Tensor]]]:
    """Linearize the Riemannian-gradient map at x once, returning a linear map for repeated Hessian-vector products.

    Reasoning:
    Since `x` stays fixed for an entire TCG inner loop (and for the quality-quotient
    evaluation that follows), this replaces up to num_iterations_cg + 1 independent
    `hvp`/`jax.grad` evaluations at the same point with a single primal+linearization
    pass, followed by cheap linear applications for each tangent direction.

    Args:
        x: The point at which to linearize.
        cost_fn: The cost function to differentiate.
        operator_type: The type of operator ('kraus', 'state', 'povm').
        metric: The metric to use ('canonical' or 'euclidean').

    Returns:
        A tuple containing:
        - The Riemannian gradient at x.
        - The Euclidean gradient at x (2df/dx).
        - A linear map taking a tangent vector z (same shape as x) to a tuple
          (Riemannian directional derivative, Euclidean directional derivative) at x in direction z.
    """
    (rgradient, euclidean_gradient_conjugated), hvp_map_linearized = jax.linearize(rgrad_and_euclidean_fn, x)
            
    rhessian_vector_fn = lambda x, z: rhessian_vector_product_from_linear_map(x=x, z=z, rgradient=rgradient, operator_type=operator_type, metric=metric, linearized_hvp_map=hvp_map_linearized)
    
    return rgradient, euclidean_gradient_conjugated,hvp_map_linearized, rhessian_vector_fn

def assemble_riemannian_hessian_vector_product(x:Tensor, tangent_vector:Tensor, rgrad_x_tensor:Tensor, Drgrad_x_to_z_tensor:Tensor, operator_type:str, metric:str="canonical", return_tensor:bool=True)-> Tensor|Matrix:
    """Assemble the Riemannian Hessian-vector product from the individual components.

    Args:
        x: The point on the Stiefel manifold where the Hessian is evaluated.
        tangent_vector: The tangent vector to multiply the Hessian with.
        rgrad_x_tensor: The Riemannian gradient at x (shared across all tangent directions at this x).
        Drgrad_x_to_z_tensor: The Euclidean directional derivative of the Riemannian gradient field at x in direction tangent_vector,
            i.e. the output of the linear map returned by `linearize_riemannian_gradient` applied to tangent_vector.
        operator_type: The type of operator that x corresponds to ('povm', 'kraus', 'state').
        metric: The metric to use for the Hessian calculation.
        return_tensor: Whether to return the result as a tensor (True) or as a matrix (False).

    Returns:
        The Riemannian Hessian-vector product, either as a tensor or matrix depending on return_tensor.
    """
    x_matrix, tangent_vector_matrix, rgrad_x_matrix, Drgrad_x_to_z_matrix = tensors_to_isometries(
        x, tangent_vector, rgrad_x_tensor, Drgrad_x_to_z_tensor, operator_type=operator_type)

    rhessian_vector_product_matrix = riemannian_connection(x=x_matrix, w_x=rgrad_x_matrix, z=tangent_vector_matrix, Dw_x_to_z=Drgrad_x_to_z_matrix, metric=metric)

    if return_tensor:
        return isometry_to_tensor(rhessian_vector_product_matrix, x.shape)
    return rhessian_vector_product_matrix

def rhessian_vector_product_from_linear_map(x: Tensor, z: Tensor, rgradient: Tensor, operator_type: str, metric: str, linearized_hvp_map: Callable[[Tensor], Tensor]) -> Tensor:
    """Compute the Riemannian Hessian-vector product using a linearization of the riemannian gradient function."""
    Drgrad_x_to_z_tensor, _  = linearized_hvp_map(z) # I belive the second output is the jvp for the euclidean gradient which we don't need.
    return assemble_riemannian_hessian_vector_product(
        x=x, tangent_vector=z, rgrad_x_tensor=rgradient,
        Drgrad_x_to_z_tensor=Drgrad_x_to_z_tensor, operator_type=operator_type, metric=metric, return_tensor=True,
    )