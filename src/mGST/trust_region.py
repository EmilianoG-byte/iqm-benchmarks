"""
Trust Region module for Optimization on the Stiefel Manifold
"""

import jax.numpy as jnp
import warnings
from mGST.automatic_diff import hvp, automatic_gradient
from mGST.utility_functions_comparisons import tensor_to_isometry, euclidean_gradients_to_stiefel, isometry_to_tensor, GRADIENT_FUNCTIONS
from mGST.riemannian import get_isometry_dimensions_from_tensor, riemannian_connection, riemannian_metric
from mGST.typing import Tensor, Matrix

from typing import Callable

def riemannian_gradient_fn_povm(kraus_tensor:Tensor, povm_psd:Tensor, state_psd:Tensor, indices_list:list[list[int]], prob_matrix:Matrix, operator_type:str="povm", metric:str = "canonical", )->Tensor:
    """Calculate the riemannian gradient of the cost function wrt to the POVM tensor
    
    Args:
        x: The POVM tensor to calculate the gradient at.
        metric: The metric to use for the gradient calculation. Can be "canonical" or "euclidean".
    Returns:
        The riemannian gradient (tensor) of the cost function wrt to the POVM tensor.
    """
    
    operator_tensors = {
        "povm": povm_psd,
        "kraus": kraus_tensor,
        "state": state_psd
    }
    
    if operator_type != "povm":
        raise NotImplementedError("Only POVM operators are tested for now.")

    gradient_euclidean_jax = GRADIENT_FUNCTIONS[operator_type](
        kraus_tensor, povm_psd, state_psd,
        indices_list, prob_matrix) # 2df/dx
    
    gradient_euclidean = gradient_euclidean_jax.conj() # 2df/dx*
    operator_tensor = operator_tensors[operator_type]
    gradient_stiefel_matrix, _ = euclidean_gradients_to_stiefel(
        gradient_tensor=gradient_euclidean,
        operator_tensor=operator_tensor, operator_type=operator_type, metric=metric,
    ) # Riemannian gradient
    return isometry_to_tensor(gradient_stiefel_matrix, operator_tensor.shape)

def riemannian_hessian_vector_povm_jax(tangent_vector:Tensor, kraus_tensor:Tensor, povm_psd:Tensor, state_psd:Tensor, indices_list:list[list[int]], prob_matrix:Matrix, metric:str="canonical", return_tensor:bool=True)-> Tensor|Matrix:
    """Compute the Riemannian Hessian-vector product for the POVM tensor using JAX.
    
    NOTE: For the calculation of the Riemannian Hessian-vector product, we should use df/dx*
    as the euclidean derivative as this is the actual gradient. 
    See:
    - Corollary 4.0.1. of An introduction to complex differentials and complex differentiability - Hunger.
        
    Args:
        povm_psd: POVM square-root-factor of shape (num_povm, povm_rank, dim).
        tangent_vector: Tangent vector determining direction of covariant derivative of shape (num_povm, povm_rank, dim).
        metric: The metric to use for the Hessian-vector product. Can be "canonical" or "euclidean".
    """
    riemannian_gradient_function = lambda x: riemannian_gradient_fn_povm(
        kraus_tensor=kraus_tensor, povm_psd=x, state_psd=state_psd, indices_list=indices_list, prob_matrix=prob_matrix, operator_type="povm",metric=metric) # Riemannian gradient (from df/dx*)

    rgrad_at_x_tensor, Drgrad_to_z_tensor = hvp(function=riemannian_gradient_function, x=povm_psd, z=tangent_vector)

    n, p = get_isometry_dimensions_from_tensor(povm_psd, tensor_type="povm")
    rgrad_at_x_matrix = tensor_to_isometry(rgrad_at_x_tensor, n, p)
    Drgrad_to_z_matrix = tensor_to_isometry(Drgrad_to_z_tensor, n, p)
    povm_psd_matrix = tensor_to_isometry(povm_psd, n, p)
    tangent_vector_matrix = tensor_to_isometry(tangent_vector, n, p)
    
    rhessian_vector_product_matrix = riemannian_connection(x = povm_psd_matrix, w_x=rgrad_at_x_matrix, z=tangent_vector_matrix, Dw_in_z_at_x=Drgrad_to_z_matrix, metric=metric)
    
    if return_tensor:
        return isometry_to_tensor(rhessian_vector_product_matrix, povm_psd.shape)
    return rhessian_vector_product_matrix


def compute_step_size(z:jnp.ndarray, delta:jnp.ndarray, radius:float, x:jnp.ndarray = None, metric:str = "euclidean")-> float:
    """
    Move to the unit ball boundary by solving
    ||z_sol|| = || z + t * delta || == radius
    for t with t > 0.
    """
    metric_delta = riemannian_metric(delta, delta, x=x, metric=metric)
    if jnp.allclose(metric_delta, 0):
        warnings.warn("tangent vector 'delta' has norm zero")
        return 0 # t =0 such that the next iteration is the same z
    p = riemannian_metric(z, delta, x=x, metric=metric) / metric_delta
    q = (riemannian_metric(z, z, x=x, metric=metric) - radius**2) / metric_delta
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn(f"encountered t < 0: {t}", RuntimeWarning)
    return t


def solve_quadratic_equation(p:float, q:float)->tuple[float, float]:
    """
    Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
    
    The solution should be  -p ± sqrt(p**2 - q).
    
    Args:
        p: Coefficient of the linear term (half of the coefficient of x).
        q: Constant term of the quadratic equation.
    
    Returns:
        A tuple containing the two solutions of the quadratic equation, (negative, positive).
        
    Raises:
        ValueError: If the discriminant is negative, i.e., p**2 - q < 0.
    """
    if (p**2 - q) < 0:
        raise ValueError(f"Discriminant p^2 - q < 0: {p**2 - q}")
    # Handle the case when p is close to zero separately.
    if jnp.isclose(p, 0):
        x = jnp.sqrt(-q)
        return (-x, x)
    # Stable solution to avoid cancellation errors
    x1 = -(p + jnp.sign(p)*jnp.sqrt(p**2 - q))
    # Use Vieta's formulas to compute the second root
    x2 = q / x1
    return tuple(sorted((x1, x2)))

def truncated_conjugate_gradient(x: Tensor, radius: float, num_iterations:int, rgradient:Tensor, metric:str, n:int, p:int, rhessian_vector_fn:Callable, verbose:bool=True) -> tuple[Tensor, bool]:
    """ This function returns the direction to update x"""
    solution_tensor = jnp.zeros_like(x) # η 
    r_tensor = rgradient # r
    delta_tensor = -r_tensor # δ
    on_boundary = False

    x_matrix = tensor_to_isometry(x, n=n, p=p)

    for iter in range(num_iterations):
        if verbose:
            print(f"Iteration {iter + 1}/{num_iterations}", end="\n")
        rhessian_delta_tensor = rhessian_vector_fn(x=x, tangent_vector=delta_tensor)
        rhessian_delta_matrix = tensor_to_isometry(rhessian_delta_tensor, n=n, p=p)
        delta_matrix = tensor_to_isometry(delta_tensor, n=n, p=p)
        # Compute curvature
        curvature = riemannian_metric(delta_matrix, rhessian_delta_matrix, x=x_matrix, metric=metric)
        # Negative curvature
        delta_matrix = tensor_to_isometry(delta_tensor, n=n, p=p)
        solution_matrix = tensor_to_isometry(solution_tensor, n=n, p=p) 

        if curvature < 0:
            if verbose:
                print("Negative curvature encountered.")
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix,metric=metric)
            solution_tensor += step * delta_tensor
            on_boundary = True
            break
        r_matrix = tensor_to_isometry(r_tensor, n=n, p=p)
        norm_sqrd_r = riemannian_metric(r_matrix, r_matrix, x=x_matrix, metric=metric)
        alpha = norm_sqrd_r / curvature
        solution_tensor += alpha * delta_tensor
        
        solution_matrix = tensor_to_isometry(solution_tensor, n=n, p=p) 
        norm_solution = jnp.sqrt(riemannian_metric(solution_matrix, solution_matrix, x_matrix, metric=metric))
        
        # Exceeds trust region boundary
        if norm_solution >= radius:
            if verbose:
                print("Exceeded trust region boundary.")
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix,metric=metric)
            solution_tensor += step * delta_tensor
            on_boundary = True
            break
        
        r_tensor += alpha * rhessian_delta_tensor
        r_matrix = tensor_to_isometry(r_tensor, n=n, p=p)
        norm_sqrd_r_next = riemannian_metric(r_matrix, r_matrix, x=x_matrix, metric=metric)
        beta = norm_sqrd_r_next / norm_sqrd_r
        delta_tensor = -r_tensor + beta * delta_tensor

    return solution_tensor, on_boundary

def compute_euclidean_inner_product_tensors(tensor_1, tensor_2, adjoint:bool=False)->float:
    if adjoint:
        tensor_1 = tensor_1.conj()
    return jnp.einsum("...,...->", tensor_1, tensor_2).real

def _compute_first_order_term(z:Tensor, gradient_conjugated:Tensor)->float:
    return compute_euclidean_inner_product_tensors(z, gradient_conjugated, adjoint=False)

def _compute_second_order_term(x:Tensor, z:Tensor, gradient:Tensor, hessian_z:Tensor, n:int, p:int)->float:
    gradient_mtrx = tensor_to_isometry(gradient, n=n, p=p) # Dx = 2df/dx*
    x_mtrx = tensor_to_isometry(x, n=n, p=p)
    z_mtrx = tensor_to_isometry(z, n=n, p=p)
    extra_factor = jnp.trace(z_mtrx.conj().T @ z_mtrx @ x_mtrx.conj().T @ gradient_mtrx).real
    return 0.5 * (compute_euclidean_inner_product_tensors(z, hessian_z, adjoint=False) - extra_factor)

def compute_approx_terms_from_tensors(x:Tensor, z:Tensor, n:int, p:int, cost_function:Callable)->tuple[float, float]:
    # TODO: use grad_and_value to evaluate the cost function when computing the gradient at hvp.
    gradient_function = automatic_gradient(cost_function)
    gradient_conjugated, hessian_z = hvp(function=gradient_function, x=x, z=z) # 2df/dx, 2(Hxx dx + Hx*x dx*)
    first_order_term = _compute_first_order_term(z, gradient_conjugated)
    second_order_term = _compute_second_order_term(x=x, z=z, gradient=gradient_conjugated.conj(), hessian_z=hessian_z, n=n, p=p)
    return first_order_term, second_order_term

def compute_model_approximation(x:Tensor, update_direction:Tensor, cost_function:Callable, n:int, p:int, order:int=2, include_zero:bool=True)->float:
    if order not in [1, 2]:
        raise ValueError("Invalid order value. Supported orders are 1 and 2.")

    approximation_value = 0
    if include_zero:
        approximation_value += cost_function(x)
    # Add higher-order terms if needed
    first_order_term, second_order_term = compute_approx_terms_from_tensors(
        x=x, z=update_direction, n=n, p=p, cost_function=cost_function,
    )
    if order >= 1:
        approximation_value += first_order_term
    if order == 2:
        approximation_value += second_order_term
    return approximation_value

def compute_quality_quotient(x:Tensor, update_direction:Tensor, cost_function:Callable, x_next:Tensor, n:int, p:int) -> float:
    return (cost_function(x_next) - cost_function(x)) / compute_model_approximation(x, update_direction, cost_function, n, p, order=2, include_zero=False)

def run_trust_region_optimization(cost_function:Callable, retraction:Callable, x_init:Tensor, radius_init:float, num_iterations:int, max_radius:float, quotient_trust:float, operator_type:str,  rgradient_fn:Callable, rhessian_vector_fn:Callable, metric_cg:str, num_iterations_cg:int, verbose_cg:bool=True, verbose:bool=True)->tuple[Tensor, list[Tensor]]:
    x_k = x_init
    x_k_array = [x_k]
    radius_k = radius_init
    n, p = get_isometry_dimensions_from_tensor(x_init, operator_type)
    try:
        for idx in range(num_iterations):
            if verbose:
                print("Iteration:", idx)
                
            # Solve the trust region subproblem
            rgradient = rgradient_fn(x_k)
            update_direction, on_boundary = truncated_conjugate_gradient(x=x_k, radius=radius_k, num_iterations=num_iterations_cg, rgradient=rgradient, metric=metric_cg, n=n, p=p, rhessian_vector_fn=rhessian_vector_fn, verbose=verbose_cg)
            
            # Compute the quality quotient
            x_next = retraction(x_k, update_direction)
            quality_quotient = compute_quality_quotient(x=x_k, update_direction=update_direction, cost_function=cost_function, x_next=x_next, n=n, p=p)
            
            if quality_quotient < 0.25:
                # Reduce the trust region radius
                radius_k *= 0.25
            elif quality_quotient > 0.75 and on_boundary:
                # Increase the trust region radius
                radius_k = min(2.0 * radius_k, max_radius)
            else:
                # Keep the trust region radius the same
                radius_k = radius_k
                
            if quality_quotient > quotient_trust:
                # Accept the new point
                x_k = x_next
                # Only store accepted points
                x_k_array.append(x_k)
            else:
                # Reject the new point
                x_k = x_k

    except KeyboardInterrupt:
        print(f"Interrupted by user at iteration {idx}.")
    return x_k, x_k_array