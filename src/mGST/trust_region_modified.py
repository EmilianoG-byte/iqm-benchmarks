"""Beta module"""

from mGST.typing import Scalar, Matrix, Tensor
from typing import Callable
from mGST.automatic_diff import automatic_gradient
from mGST.utility_functions_comparisons import euclidean_gradients_to_stiefel, isometry_to_tensor, tensors_to_isometries, tensor_to_isometry
from mGST.riemannian import riemannian_connection, riemannian_metric
from mGST.trust_region import compute_step_size, determine_cg_stopping_criteria

import jax
import jax.numpy as jnp

def riemannian_gradient_fn(x:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type: str, metric: str) -> tuple[Tensor, Tensor]:
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


def linearize_riemannian_gradient(x:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type:str, metric:str)-> tuple[Tensor, Tensor, Callable[[Tensor], tuple[Tensor, Tensor]]]:
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
    rgrad_and_euclidean_fn = lambda x: riemannian_gradient_fn(x=x, cost_fn=cost_fn, operator_type=operator_type, metric=metric)
    (rgradient, euclidean_gradient), linearized_map = jax.linearize(rgrad_and_euclidean_fn, x)
    return rgradient, euclidean_gradient, linearized_map

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

def rhessian_vector_product_from_linear_map(x: Tensor, delta_tensor: Tensor, rgradient: Tensor, operator_type: str, metric: str, linearized_hvp_map: Callable[[Tensor], Tensor]) -> Tensor:
    """Compute the Riemannian Hessian-vector product using a linearization of the riemannian gradient function."""
    Drgrad_x_to_z_tensor, _  = linearized_hvp_map(delta_tensor) # I belive the second output is the jvp for the euclidean gradient which we don't need.
    return assemble_riemannian_hessian_vector_product(
        x=x, tangent_vector=delta_tensor, rgrad_x_tensor=rgradient,
        Drgrad_x_to_z_tensor=Drgrad_x_to_z_tensor, operator_type=operator_type, metric=metric, return_tensor=True,
    )

def truncated_conjugate_gradient(x: Tensor, radius: float, num_iterations:int, rgradient:Tensor, operator_type:str, metric:str, n:int, p:int, linearized_hvp_map:Callable[[Tensor], Tensor], verbose:bool=True, theta:float | None = None, kappa: float | None = None) -> tuple[Tensor, bool]:
    """ Truncated Conjugate Gradient (TCG) method to approximately solve the trust region subproblem on the Stiefel manifold.
    
    TODO: determine if we can use the Hessian-vector product output to compute the denominator in the quality coefficient
    (reduce by half computation time).
    
    References:
    [1] Trust-region methods on Riemannian manifolds, P. A. Absil, C. G. Baker, and K. A. Gallivan, 2006.
    [2] https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf (Algorithm 6.4)
    
    Args:
        x: Current point on the manifold.
        radius: Trust region radius.
        num_iterations: Maximum number of iterations to perform.
        rgradient: Riemannian gradient at point x.
        metric: The metric to use to calculate inner products in the tangent space. Can be "canonical" or "euclidean".
        n, p: Dimensions of the Stiefel manifold.
        rhessian_vector_fn: Function to compute the Riemannian Hessian-vector product.
        verbose: Whether to print information during the TCG algorithm.
        theta: The theta parameter for the stopping criteria. If None, stopping criteria is not used. See [2] for an example of parameters.
        kappa: The kappa parameter for the stopping criteria. If None, stopping criteria is not used. See [2] for an example of parameters.
    
    Returns:
        A tuple containing:
        - The approximate solution to the trust region subproblem (tangent vector), i.e. the proposed direction to update x.
        - A boolean indicating whether the solution lies on the boundary of the trust region.
    """
    solution_tensor = jnp.zeros_like(x) # η 
    r_tensor = rgradient # r
    delta_tensor = -r_tensor # δ
    on_boundary = False

    x_matrix = tensor_to_isometry(x, n=n, p=p)
    r_matrix = tensor_to_isometry(r_tensor, n=n, p=p)
    # Save one computation of the norm of r per iteration by initializing like this
    norm_sqrd_r = riemannian_metric(r_matrix, r_matrix, x=x_matrix, metric=metric)
    norm_r0 = jnp.sqrt(norm_sqrd_r)
    check_stopping_criteria = theta is not None and kappa is not None
    reason = "Max iterations reached ⏳."
    
    for iter in range(num_iterations):
        solution_matrix = tensor_to_isometry(solution_tensor, n=n, p=p)
        
        # This is the only part that changes        
        # Compute rHessian-tangent-vector product using the precomputed linearization (no new autodiff call)
        rhessian_delta_tensor = rhessian_vector_product_from_linear_map(
            x=x, delta_tensor=delta_tensor, rgradient=rgradient, operator_type=operator_type, metric=metric, linearized_hvp_map=linearized_hvp_map
        )
        
        rhessian_delta_matrix = tensor_to_isometry(rhessian_delta_tensor, n=n, p=p)
        delta_matrix = tensor_to_isometry(delta_tensor, n=n, p=p)
        # Compute curvature
        curvature = riemannian_metric(delta_matrix, rhessian_delta_matrix, x=x_matrix, metric=metric)        
        # Negative curvature
        if curvature < 0:
            reason = "Negative curvature encountered 📉."
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix, metric=metric)
            solution_tensor += step * delta_tensor
            on_boundary = True
            break
        
        alpha = norm_sqrd_r / curvature
        solution_tensor += alpha * delta_tensor
        
        solution_matrix = tensor_to_isometry(solution_tensor, n=n, p=p) 
        norm_solution = jnp.sqrt(riemannian_metric(solution_matrix, solution_matrix, x_matrix, metric=metric))
        
        # Check if the solution exceeds the trust region boundary
        if norm_solution >= radius:
            reason = "Exceeded trust region boundary 📈."
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix,metric=metric)
            solution_tensor += step * delta_tensor
            on_boundary = True
            break
        
        r_tensor += alpha * rhessian_delta_tensor
        r_matrix = tensor_to_isometry(r_tensor, n=n, p=p)
        norm_sqrd_r_next = riemannian_metric(r_matrix, r_matrix, x=x_matrix, metric=metric)
        
        # Check stopping criteria
        if check_stopping_criteria:
            norm_rk = jnp.sqrt(norm_sqrd_r_next)
            if determine_cg_stopping_criteria(norm_r0, norm_rk, theta=theta, kappa=kappa):
                reason = f"(tCG) Stopping criteria met 🛑. |r_k| = {norm_rk:.2e}"
                on_boundary = False
                break

        beta = norm_sqrd_r_next / norm_sqrd_r
        delta_tensor = -r_tensor + beta * delta_tensor
        norm_sqrd_r = norm_sqrd_r_next

    if verbose:
            print(f"TCG finished after: {iter + 1}/{num_iterations} iters. \n Reason: {reason}")
            print("---------------------------------------")
    return solution_tensor, on_boundary