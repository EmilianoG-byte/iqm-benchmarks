"""
Trust Region module for Optimization on the Stiefel Manifold
"""

import jax.numpy as jnp
import warnings
from mGST.automatic_diff import hvp, automatic_gradient
from mGST.utility_functions_comparisons import tensor_to_isometry, euclidean_gradients_to_stiefel, isometry_to_tensor, GRADIENT_FUNCTIONS
from mGST.riemannian import get_isometry_dimensions_from_tensor, riemannian_connection, riemannian_metric, retraction_polar_decomposition
from mGST.typing import Tensor, Matrix, Scalar

from typing import Callable


def retraction_first_order(x:Tensor, z:Tensor, n:int, p:int)-> Tensor:
    """First order retraction using the polar decomposition.
    
    Args:
        x: Tensor representing the point on the Stiefel manifold.
        z: Tensor representing the tangent vector at point x.
        n: The Stiefel n dimension.
        p: The Stiefel p dimension.
    
    Returns:
        The retracted point on the Stiefel manifold.
    """
    x_matrix = tensor_to_isometry(x, n=n, p=p)
    z_matrix = tensor_to_isometry(z, n=n, p=p)
    return isometry_to_tensor(retraction_polar_decomposition(x=x_matrix, z=z_matrix, step_size=-1.0), x.shape)

def riemannian_gradient_fn(x:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type: str, metric: str) -> Tensor:
    """Compute the Riemannian gradient of a cost function at a point x using automatic differentiation.

    Args:
        x: The point at which to evaluate the gradient.
        cost_fn: The cost function to differentiate.

    Returns:
        The Riemannian gradient of the cost function at x.
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
    return isometry_to_tensor(gradient_stiefel_matrix, x.shape)

def riemannian_hessian_vector_fn(x:Tensor, tangent_vector:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type:str, metric:str="canonical", return_tensor:bool=True)-> Tensor|Matrix:
    """Compute the Riemannian Hessian-vector product at a point x in the Stiefel manifold using automatic differentiation.
    
    NOTE: For the calculation of the Riemannian Hessian-vector product, we should use df/dx*
    as the euclidean derivative because this is the actual gradient. 
    See:
    - Corollary 4.0.1. of An introduction to complex differentials and complex differentiability - Hunger.
        
    Args:
        x: The point on the Stiefel manifold where the Hessian is evaluated.
        tangent_vector: The tangent vector to multiply the Hessian with. Should be of the same shape as x.
        cost_fn: The cost function to calculate the Hessian of.
        operator_type: The type of operator that x corresponds to ('povm', 'kraus', 'state').
        metric: The metric to use for the Hessian calculation. Can be "canonical" or "euclidean".
        return_tensor: Whether to return the result as a tensor (True) or as a matrix (False).
        
    Returns:
        The Riemannian Hessian-vector product, either as a tensor or matrix depending on return_tensor.
    """
    rgrad_fn = lambda x: riemannian_gradient_fn(x=x, cost_fn=cost_fn, operator_type=operator_type, metric=metric) # Riemannian gradient (from df/dx*)

    rgrad_x_tensor, Drgrad_x_to_z_tensor = hvp(function=rgrad_fn, x=x, z=tangent_vector)

    n, p = get_isometry_dimensions_from_tensor(x, tensor_type=operator_type)
    
    rgrad_x_matrix = tensor_to_isometry(rgrad_x_tensor, n, p)
    Drgrad_to_z_matrix = tensor_to_isometry(Drgrad_x_to_z_tensor, n, p)
    x_matrix = tensor_to_isometry(x, n, p)
    tangent_vector_matrix = tensor_to_isometry(tangent_vector, n, p)
    
    rhessian_vector_product_matrix = riemannian_connection(x=x_matrix, w_x=rgrad_x_matrix, z=tangent_vector_matrix, Dw_x_to_z=Drgrad_to_z_matrix, metric=metric)
    
    if return_tensor:
        return isometry_to_tensor(rhessian_vector_product_matrix, x.shape)
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

def truncated_conjugate_gradient(x: Tensor, radius: float, num_iterations:int, rgradient:Tensor, metric:str, n:int, p:int, rhessian_vector_fn:Callable[[Tensor, Tensor], Tensor], verbose:bool=True, theta:float | None = None, kappa: float | None = None) -> tuple[Tensor, bool]:
    """ This function returns the direction to update x"""
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
        # Compute rHessian-tangent-vector product
        rhessian_delta_tensor = rhessian_vector_fn(x=x, z=delta_tensor)
        rhessian_delta_matrix = tensor_to_isometry(rhessian_delta_tensor, n=n, p=p)
        delta_matrix = tensor_to_isometry(delta_tensor, n=n, p=p)
        # Compute curvature
        curvature = riemannian_metric(delta_matrix, rhessian_delta_matrix, x=x_matrix, metric=metric)        
        # Negative curvature
        if curvature < 0:
            reason = "Negative curvature encountered 📉."
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix,metric=metric)
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
                reason = f"Stopping criteria met 🛑. |r_k| = {norm_rk:.2e}"
                on_boundary = False
                break

        beta = norm_sqrd_r_next / norm_sqrd_r
        delta_tensor = -r_tensor + beta * delta_tensor
        norm_sqrd_r = norm_sqrd_r_next

    if verbose:
            print(f"TCG finished after: {iter + 1}/{num_iterations} iters. \n Reason: {reason}")
            print("---------------------------------------")
    return solution_tensor, on_boundary

def determine_cg_stopping_criteria(norm_r0:Tensor, norm_rk:Tensor, theta:float = 1, kappa:float = 0.1)->bool:
    """Determine whether to stop the conjugate gradient inner iteration based on the stopping criteria.
    
    References:
    [1] Absil, P.-A., Mahony, R., & Sepulchre, R. (2004). Optimization Algorithms on Matrix Manifolds. Princeton University Press.

    Stopping criteria from Eq. (7.10) in [1]: |r_{k}| < |r0| min(|r0|**theta, kappa)
    
    Args:
        r0: The initial Riemannian gradient at the start of the trust region iteration.
        r_k: The current Riemannian r vector at iteration k. 
        theta: Exponent for the stopping criteria. Typically in (0, 1). Localte rate is min(theta + 1, 2).
        kappa: Threshold for the stopping criteria. Typically a small positive value.
    """
    return norm_rk <= norm_r0 * min(norm_r0**theta, kappa)

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

def compute_quality_quotient(x:Tensor, update_direction:Tensor, cost_function:Callable, x_next:Tensor, n:int, p:int) -> tuple[Scalar, Scalar]:
    cost_fx_next = cost_function(x_next)
    return (cost_fx_next - cost_function(x)) / compute_model_approximation(x, update_direction, cost_function, n, p, order=2, include_zero=False), cost_fx_next

def run_trust_region_optimization(
    x_init:Tensor, cost_function:Callable[[Tensor], Scalar], 
    radius_init:float = 0.1, num_iterations:int = 20, max_radius:float = 2.0, quotient_trust:float = 0.125, tol_grad:float = 1e-6, operator_type:str = "povm",
    metric_cg:str = "euclidean", num_iterations_cg:int = 10, theta_cg:float = None, kappa_cg:float = None, verbose_cg:bool=True,
    verbose:bool=True)->tuple[Tensor, list[Tensor], list[Scalar]]:
    """
    Run the trust region optimization algorithm.

    Args:
        cost_function: The cost function to minimize. Should take a tensor as input and return a scalar.
        x_init: The initial point on the manifold.
        radius_init: The initial trust region radius.
        num_iterations: The maximum number of trust region iterations to perform.
        max_radius: The maximum trust region radius to allow.
        quotient_trust: The lower threshold for accepting a step based on the quality quotient.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        metric_cg: The metric to use to calculate inner products in the truncated conjugate gradient algorithm. Can be "canonical" or "euclidean".
        num_iterations_cg: The maximum number of iterations to perform in the truncated conjugate gradient algorithm.
        theta_cg: The theta parameter for the stopping criteria of the truncated conjugate gradient algorithm.
        kappa_cg: The kappa parameter for the stopping criteria of the truncated conjugate gradient algorithm.
        tol_grad: The tolerance for the norm of the Riemannian gradient to determine convergence.
        verbose_cg: Whether to print information during the truncated conjugate gradient algorithm.
        verbose: Whether to print information during the trust region outer loop optimization.
        
    Returns:
        A tuple containing:
        - The optimized point on the manifold.
        - A list of all accepted points during the optimization.
        - A list of the cost function values at each accepted point.
    """
    x_k = x_init
    x_k_array = [x_k]
    cost_fx_array = [cost_function(x_k)]
    radius_k = radius_init
    num_rejections = 0
    n, p = get_isometry_dimensions_from_tensor(x_init, operator_type)

    # Define the riemannian gradient, riemannian hessian-vector product, and retraction functions
    rgradient_fn = lambda x: riemannian_gradient_fn(x=x, cost_fn=cost_function, operator_type=operator_type, metric=metric_cg)
    rhessian_vector_fn = lambda x, z: riemannian_hessian_vector_fn(x=x, tangent_vector=z, cost_fn=cost_function, operator_type=operator_type, metric=metric_cg, return_tensor=True)
    retraction = lambda x, z: retraction_first_order(x, z, n=n, p=p)

    # Compute the initial Riemannian gradient and its norm
    rgradient = rgradient_fn(x_k)    
    norm_grad_init = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_init, metric=metric_cg))
    norm_grad = norm_grad_init

    print(f"TR started for operator: {operator_type} 🚀.")
    print("=======================================")
    try:
        for idx in range(num_iterations):
            if verbose:
                print(f"TR Iteration: {idx}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}")

            # Solve the trust region subproblem
            update_direction, on_boundary = truncated_conjugate_gradient(x=x_k, radius=radius_k, num_iterations=num_iterations_cg, rgradient=rgradient, metric=metric_cg, n=n, p=p, rhessian_vector_fn=rhessian_vector_fn, verbose=verbose_cg, theta=theta_cg, kappa=kappa_cg)
            
            # Compute the quality quotient
            x_next = retraction(x_k, update_direction)
            quality_quotient, cost_fx_next = compute_quality_quotient(x=x_k, update_direction=update_direction, cost_function=cost_function, x_next=x_next, n=n, p=p)
            
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
                cost_fx_array.append(cost_fx_next)
            else:
                # Reject the new point
                num_rejections += 1
                if verbose:
                    print("Update rejected ❌.")
                x_k = x_k
            # Compute the Riemannian gradient at the new point
            rgradient = rgradient_fn(x_k)
            
            # Determine stopping criteria based on gradient norm
            norm_grad = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_k, metric=metric_cg))
            if determine_tr_stopping_criteria(norm_grad, norm_grad_init, tol_grad=tol_grad):
                if verbose:
                    print(f"Stopping criteria met. 🛑")
                break
                
    except KeyboardInterrupt:
        print(f"Optimized interrupted by user {idx}.")
    print("=======================================")
    print(f"Optimization finished ✅. \n Iters: {idx+1}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}. Rejections: {num_rejections}")
    return x_k, x_k_array, cost_fx_array

def riemannian_metric_from_tensors(n:int, p: int, z1:Tensor, z2:Tensor, x:Tensor = None, metric:str = "euclidean")-> float:
    """Compute the Riemannian metric at point x between two tangent vectors represented as tensors.

    Args:
        z1: The first tensor.
        z2: The second tensor.
        x: The point at which to evaluate the metric (optional).
        metric: The type of metric to use (default is "euclidean").

    Returns:
        The value of the Riemannian metric.
    """
    z1_matrix = tensor_to_isometry(z1, n=n, p=p)
    z2_matrix = tensor_to_isometry(z2, n=n, p=p)
    if x is not None:
        x_matrix = tensor_to_isometry(x, n=n, p=p)
    return riemannian_metric(z1_matrix, z2_matrix, x=x_matrix, metric=metric)

def determine_tr_stopping_criteria(norm_grad:float, norm_grad_init:float, tol_grad:float = 1e-8)->bool:
    """Determine whether to stop the trust region optimization based on the gradient norm.
    
    References:
        - ChatGPT
    
    Args:
        norm_grad: The norm of the Riemannian gradient at the current point.
        tol_grad: The tolerance for the gradient norm. If None, no stopping criteria is applied.
    """
    return norm_grad <= tol_grad * max(1.0, norm_grad_init)