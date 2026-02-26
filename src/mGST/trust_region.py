"""
Trust Region module for Optimization on the Stiefel Manifold
"""

import jax.numpy as jnp
import warnings
from mGST.automatic_diff import hvp, automatic_gradient
from mGST.utility_functions_comparisons import tensor_to_isometry, euclidean_gradients_to_stiefel, isometry_to_tensor, tensors_to_isometries, get_isometry_dimensions_from_tensor, _update_tensor_via_gradient
from mGST.riemannian import riemannian_connection, riemannian_metric, update_isometry_tensors
from mGST.linear_algebra import transpose
from mGST.typing import Tensor, Matrix, Scalar, TrustRegionOptions, GradientDescentOptions, OptimizationOptions, OperatorSchedule, OptimizationScheduleItem


from typing import Callable, Any

def retraction_first_order(x:Tensor, z:Tensor, operator_type:str)-> Tensor:
    """First order retraction using the polar decomposition.
    
    Args:
        x: Tensor representing the point on the Stiefel manifold.
        z: Tensor representing the tangent vector at point x.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
    
    Returns:
        The retracted point on the Stiefel manifold.
    """

    x_matrices, z_matrices = tensors_to_isometries(x, z, operator_type=operator_type)
    # Rx(z)
    new_isometries = update_isometry_tensors(
        isometries=x_matrices, update_directions=z_matrices, step_size=-1.0, operator_type=operator_type, use_geodesic=False
    ) 
    return isometry_to_tensor(new_isometries, x.shape)

def riemannian_gradient_fn(x:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type: str, metric: str) -> Tensor:
    """Compute the Riemannian gradient of a cost function at a point x using automatic differentiation.
    
    Handles batch dimensions.

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
        
    Handles batch dimensions.
        
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

    x_matrix, tangent_vector_matrix, rgrad_x_matrix, Drgrad_x_to_z_matrix = tensors_to_isometries(
        x, tangent_vector, rgrad_x_tensor, Drgrad_x_to_z_tensor, operator_type=operator_type)
    
    rhessian_vector_product_matrix = riemannian_connection(x=x_matrix, w_x=rgrad_x_matrix, z=tangent_vector_matrix, Dw_x_to_z=Drgrad_x_to_z_matrix, metric=metric)
    
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
        warnings.warn("tangent vector 'delta' has norm zero", UserWarning)
        return 0 # t =0 such that the next iteration is the same z
    p = riemannian_metric(z, delta, x=x, metric=metric) / metric_delta
    q = (riemannian_metric(z, z, x=x, metric=metric) - radius**2) / metric_delta
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn(f"encountered t < 0: {t}", UserWarning)
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
        # Compute rHessian-tangent-vector product
        rhessian_delta_tensor = rhessian_vector_fn(x=x, z=delta_tensor)
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
    """Compute the Euclidean inner product between two tensors.
    
    By default this computes the inner product: <tensor_1, tensor_2> = Re(Tr(tensor_1^T tensor_2)) (without adjoint)
    
    Handles batch dimensions as all dimensions are summed over.
    
    Args:
        tensor_1: First tensor.
        tensor_2: Second tensor.
        adjoint: Whether to take the adjoint of the first tensor before computing the inner product. Defaults to False.
    """
    if adjoint:
        tensor_1 = tensor_1.conj()
    return jnp.einsum("...,...->", tensor_1, tensor_2).real

def _compute_first_order_term(z:Tensor, gradient_conjugated:Tensor)->float:
    return compute_euclidean_inner_product_tensors(z, gradient_conjugated, adjoint=False)

def _compute_second_order_term(x:Tensor, z:Tensor, gradient_conjugated:Tensor, hessian_z:Tensor, n:int, p:int)->float:
    """Compute the second-order term approximation term for the local cost function in the tangent spac

    Handles batch dimensions since all dimensions are summed over using `compute_euclidean_inner_product_tensors`.

    Args:
        x: Current point on the manifold.
        z: Update direction (tangent vector).
        gradient_conjugated: Conjugate gradient at point x. (2df/dx)
        hessian_z: Hessian-vector product at point x in the direction z. (2(Hxx dx + Hx*x dx*))
        n,p : Dimensions of the Stiefel manifold.

    Returns:
        The second-order term of the model approximation.
    """
    gradient_mtrx, x_mtrx, z_mtrx = tensors_to_isometries(gradient_conjugated, x, z, n=n, p=p)  # Dx = 2df/dx
    extra_factor = z_mtrx.conj() @ transpose(x_mtrx) @ gradient_mtrx
    extra_factor_tensor = isometry_to_tensor(extra_factor, x.shape)
    # (1/2)<z, Hx[z] - z* x.T 2df/dx>e with <z1, z2>e = Re(Tr(z1.T z2))
    return 0.5 * (compute_euclidean_inner_product_tensors(z, hessian_z - extra_factor_tensor, adjoint=False))
     
def compute_approx_terms_from_tensors(x:Tensor, z:Tensor, n:int, p:int, cost_function:Callable)->tuple[float, float]:
    """Helper function to compute the first and second order terms of the model approximation for the local cost function in the tangent space.
    
    Handles batch dimensions.
    
    Args:
        x: Current point on the manifold.
        z: Update direction (tangent vector).
        n: Dimension of the Stiefel manifold.
        p: Dimension of the Stiefel manifold.
        cost_function: The cost function to approximate.
    """
    # TODO: use grad_and_value to evaluate the cost function when computing the gradient at hvp.
    gradient_function = automatic_gradient(cost_function)
    gradient_conjugated, hessian_z = hvp(function=gradient_function, x=x, z=z) # 2df/dx, 2(Hxx dx + Hx*x dx*)
    first_order_term = _compute_first_order_term(z, gradient_conjugated)
    second_order_term = _compute_second_order_term(x=x, z=z, gradient_conjugated=gradient_conjugated, hessian_z=hessian_z, n=n, p=p)
    return first_order_term, second_order_term

def compute_model_approximation(x:Tensor, update_direction:Tensor, cost_function:Callable, n:int, p:int, order:int=2, include_zero:bool=True)->float:
    """Compute the model approximation up to second order of the local cost function in the tangent space.
    
    The implementation here is inspired by the metric-free second-order approximation described in Eq. (31 - 33) of [1]
    
    References:
    [1] Optimization algorithms exploiting unitary constraints, Manton, 2002.
    
    Args:
        x: Current point on the manifold.
        update_direction: Proposed update direction (tangent vector).
        cost_function: The cost function to approximate.
        n: Dimension of the Stiefel manifold.
        p: Dimension of the Stiefel manifold.
        order: Order of the approximation (1 or 2).
        include_zero: Whether to include the cost function value at x in the approximation (zeroth order).
        
    Returns:
        The model approximation value up to the specified order.
        
    Raises:
        ValueError: If the specified order is not 1 or 2.
    """
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
    """
    Compute the quality quotient for a proposed update direction in the trust region method.
    
    Handles batch dimensions.
    
    Args:
        x: Current point on the manifold.
        update_direction: Proposed update direction (tangent vector).
        cost_function: The cost function to approximate.
        x_next: The proposed next point on the manifold after applying the update direction using the retraction.
        n,p : Dimensions of the Stiefel manifold.
        
    Returns:
        A tuple containing:
        - The quality quotient (Scalar).
        - The cost function value at the proposed next point (Scalar).
    """
    cost_fx_next = cost_function(x_next)
    return (cost_fx_next - cost_function(x)) / compute_model_approximation(x, update_direction, cost_function, n, p, order=2, include_zero=False), cost_fx_next

def run_trust_region_optimization(
    x_init:Tensor, cost_function:Callable[[Tensor], Scalar], operator_type:str,
    radius_init:float = 0.1, num_iterations:int = 20, max_radius:float = 2.0, quotient_trust:float = 0.125, tol_grad:float = 1e-6, 
    metric:str = "euclidean", num_iterations_cg:int = 10, theta_cg:float = None, kappa_cg:float = None, verbose_cg:bool=True,
    verbose:bool=True)->tuple[Tensor, list[Tensor], list[Scalar], bool]:
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
        metric: The metric to use to calculate inner products in the truncated conjugate gradient algorithm. Can be "canonical" or "euclidean".
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

    See references for details on the choice of the parameters such as the trust region radius [5], tolerances [1, 2, 4], kappa and theta [1, 2, 3].

    References:
        * [1] https://chatgpt.com/share/68cd51cc-73b8-8009-be12-a7ae8624c73e
        * [2] https://pymanopt.org/docs/stable/_modules/pymanopt/optimizers/trust_regions.html#TrustRegions
        * [3] Absil, P.-A., Mahony, R., & Sepulchre, R. (2004). Optimization Algorithms on Matrix Manifolds. Princeton University Press.
        * [4] https://github.com/pymanopt/pymanopt/blob/a1f52e74092535cd416ba3ec68252eb74ed53178/src/pymanopt/optimizers/optimizer.py#L45
        * [5] https://github.com/qc-tum/rqcopt/blob/master/rqcopt/trust_region.py
    """
    x_k = x_init
    x_k_array = [x_k]
    cost_fx_array = [cost_function(x_k)]
    radius_k = radius_init
    num_rejections = 0
    n, p = get_isometry_dimensions_from_tensor(x_init, operator_type)

    # Define the riemannian gradient, riemannian hessian-vector product, and retraction functions
    rgradient_fn = lambda x: riemannian_gradient_fn(x=x, cost_fn=cost_function, operator_type=operator_type, metric=metric)
    rhessian_vector_fn = lambda x, z: riemannian_hessian_vector_fn(x=x, tangent_vector=z, cost_fn=cost_function, operator_type=operator_type, metric=metric, return_tensor=True)
    retraction = lambda x, z: retraction_first_order(x, z, operator_type=operator_type)

    # Compute the initial Riemannian gradient and its norm
    rgradient = rgradient_fn(x_k)    
    norm_grad_init = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_init, metric=metric))
    norm_grad = norm_grad_init

    finished_early = False
    print(f"TR started for operator: {operator_type} 🚀.")
    if verbose:
        print("=======================================")
    try:
        for idx in range(num_iterations):
            if verbose:
                print(f"TR Iteration: {idx}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}")

            # Solve the trust region subproblem
            update_direction, on_boundary = truncated_conjugate_gradient(x=x_k, radius=radius_k, num_iterations=num_iterations_cg, rgradient=rgradient, metric=metric, n=n, p=p, rhessian_vector_fn=rhessian_vector_fn, verbose=verbose_cg, theta=theta_cg, kappa=kappa_cg)
            
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
            norm_grad = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_k, metric=metric))
            if determine_tr_stopping_criteria(norm_grad, norm_grad_init, tol_grad=tol_grad):
                if verbose:
                    print(f"(RTR) Stopping criteria met. 🚨 |r∇f(x)|/|r∇f(x_0)| = {norm_grad/norm_grad_init:.6e}")
                finished_early = True
                break
                
    except KeyboardInterrupt:
        print(f"Optimized interrupted by user {idx}.")
    print("=======================================")
    print(f"Optimization finished ✅. \n Iters: {idx+1}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}. Rejections: {num_rejections}")
    return x_k, x_k_array, cost_fx_array, finished_early

def riemannian_metric_from_tensors(n:int, p: int, z1:Tensor, z2:Tensor, x:Tensor, metric:str = "euclidean")-> float:
    """Compute the Riemannian metric at point x between two tangent vectors represented as tensors.

    Args:
        z1: The first tensor.
        z2: The second tensor.
        x: The point at which to evaluate the metric (optional).
        metric: The type of metric to use (default is "euclidean").

    Returns:
        The value of the Riemannian metric.
    """
    z1_matrix, z2_matrix, x_matrix = tensors_to_isometries(z1, z2, x, n=n, p=p)
    
    return riemannian_metric(z1_matrix, z2_matrix, x=x_matrix, metric=metric)

def determine_tr_stopping_criteria(norm_grad:float, norm_grad_init:float, tol_grad:float = 1e-8)->bool:
    """Determine whether to stop the trust region optimization based on the gradient norm.
    
    References:
        - ChatGPT
        - https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf (Section 6.4.6)
    
    Args:
        norm_grad: The norm of the Riemannian gradient at the current point.
        tol_grad: The tolerance for the gradient norm. If None, no stopping criteria is applied.
    """
    # Should the norm_grad_init be here the one for the current iteration of the outer loop
    # Or should it be overall the initial gradient.
    # Perhaps the latter 🤔
    return norm_grad <= tol_grad * norm_grad_init


def validate_optimization_options(optimization_options:dict[str, OptimizationOptions])->dict[str, str]:
    """Validate the optimization options dictionary.
    
    Args:
        optimization_options: A map between operator types and their corresponding optimization options.
            The keys should be "kraus", "povm", and "state".
    Returns:
        A validated dictionary with default options if none are provided.
    Raises:
        ValueError: If the input is not a dictionary or contains invalid keys or values.
    """
    if optimization_options is None:
        return {
            "kraus": TrustRegionOptions(),
            "povm": TrustRegionOptions(),
            "state": TrustRegionOptions(),
        }

    if not isinstance(optimization_options, dict):
        raise ValueError("Optimization schedule must be a dictionary with string keys and OptimizationOptions values.")

    valid_operators = {"kraus", "povm", "state"}
    provided_operators = set(optimization_options.keys())
    if provided_operators != valid_operators:
        raise ValueError(f"Invalid optimization schedule keys: {provided_operators}. Valid keys are: {valid_operators}")

    for key, value in optimization_options.items():
        if not isinstance(value, OptimizationOptions):
            raise ValueError(f"Invalid optimization options for {key}: {value}. Valid options are: {OptimizationOptions}")

    return optimization_options

def validate_optimization_schedule(
    optimization_schedule: dict[str, OperatorSchedule], 
    num_iterations: int
) -> dict[str, OperatorSchedule]:
    """Validate the optimization schedule dictionary."""
    if optimization_schedule is None:
        default_schedule = OperatorSchedule.default(num_iterations)
        return {op: default_schedule for op in ["kraus", "povm", "state"]}

    valid_operators = {"kraus", "povm", "state"}
    if set(optimization_schedule.keys()) != valid_operators:
        raise ValueError(f"Invalid operators in schedule. Expected {valid_operators}")

    # Validate each operator's schedule
    for schedule in optimization_schedule.values():
        schedule.validate(num_iterations)

    return optimization_schedule

def run_riemannian_optimization(
    kraus_tensor_init:jnp.ndarray,
    povm_psd_init:jnp.ndarray,
    state_psd_init:jnp.ndarray,
    cost_function:Callable,
    cost_fn_kwargs:dict[str, Any],
    num_iterations:int,
    optimization_schedule:dict[str, OperatorSchedule] = None,
    save_intermediate_cost_values:bool=False,
    noise_threshold:float|None = None,
    relative_precision: float|None = 1e-5,
    verbose:bool=True
    )-> tuple[dict[str, jnp.ndarray], list[float]]:
    """
    Run the Riemannian optimization for each operator (Kraus, POVM, State) in an alternating fashion.
    
    NOTE: This function assumes the optimization order is povm -> kraus -> state.
    TODO: allow user to specify order.
    
    Args:
        kraus_tensor_init: Initial Kraus operator tensor. Shape: (num_gates, kraus_rank, dim_out, dim_in).
        povm_psd_init: Initial POVM operator tensor. Shape: (num_povm, rank_povm, dim_in).
        state_psd_init: Initial State operator tensor. Shape: (rank_state, dim_in).
        cost_function: The cost function to minimize. Should take kraus_tensor, povm_psd, state_psd as keyword arguments and return a scalar.
        cost_fn_kwargs: Additional keyword arguments to pass to the cost function.
        num_iterations: Number of outer iterations to perform (each iteration optimizes all operators once).
        optimization_options: A dictionary specifying the optimization options for each operator type.
            The keys should be "kraus", "povm", and "state", and the values should be instances of TrustRegionOptions or GradientDescentOptions.
            If None, default TrustRegionOptions will be used for all operators.
        save_intermediate_cost_values: Whether to save intermediate cost function values during the optimization of each operator. For now, we can only save all values if using TrustRegionOptions, since the GDS does not return intermediate values.
        noise_threshold: The noise threshold to determine if the optimization has converged. If the cost function value is below this threshold, the optimization will stop early. If None, this stopping criterion is disabled. See `estimate_noise_floor_threshold`.
        relative_precision: The relative precision to determine if the optimization has converged. If the relative change in the cost function value between the start and end of an iteration is below this threshold, the optimization will stop early. If None, this stopping criterion is disabled.
        verbose: Whether to print information during the optimization
        
    Returns:
        A tuple containing:
        - A dictionary with the optimized operators: {"kraus": kraus_tensor, "povm": povm_psd, "state": state_psd}.
        - A list of the cost function values at each optimization step.
    """
    # Validate optimization options
    optimization_schedule = validate_optimization_schedule(optimization_schedule=optimization_schedule, num_iterations=num_iterations)
    
    cost_fn_history = []
    kraus_tensor_k = kraus_tensor_init
    povm_psd_k = povm_psd_init
    state_psd_k = state_psd_init
    
    # If not given, set threshold to zero to disable early stopping based on progress.
    convergence_criteria = "* Convergence Criteria: \n"
    if noise_threshold is None:
        noise_threshold = 0.0
    else:
        convergence_criteria += f"+ Noise Threshold 📶 {noise_threshold:.2e} \n" 
    if relative_precision is None:
        relative_precision = 0.0 # Set to zero to disable early stopping based on relative precision.
    else:
        convergence_criteria += f"+ Relative Precision 📏 {relative_precision:.2e} \n"
        
    # Convergence based on early stopp using norms is always present as a last resource
    convergence_criteria += "+ Gradient(s) Norm(s) 📐"
    # Default convergence reason
    convergence_reason = "Max iterations reached ⏳."
    
    optimization_schedule_print = "\n".join(
    f"{operator}: {str(option)}"
    for operator, option in optimization_schedule.items()
)
    
    print("🔰 Starting Riemannian Optimization 🔰 \n * optimization schedule: \n"
          f"{optimization_schedule_print} \n"
          f"{convergence_criteria}")
    try:
        for idx in range(num_iterations):
            # Optimize POVM
            povm_options = optimization_schedule["povm"].get_options_for_iteration(idx)
            povm_psd_k, cost_values_povm, finished_early_povm = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=povm_options, operator_type="povm", cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values
            )
            # Save the initial cost value from this iteration for convergence check
            cost_fn_init_k = cost_values_povm[0]
            cost_fn_history.extend(cost_values_povm)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator="POVM")
            if cost_below_threshold:
                break
                
            # Optimize Kraus
            kraus_options = optimization_schedule["kraus"].get_options_for_iteration(idx)
            kraus_tensor_k, cost_values_kraus, finished_early_kraus = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=kraus_options, operator_type="kraus", cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values
            )

            cost_fn_history.extend(cost_values_kraus)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator="KRAUS")
            if cost_below_threshold:
                break
            # Optimize State
            state_options = optimization_schedule["state"].get_options_for_iteration(idx)
            state_psd_k, cost_values_state, finished_early_state = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=state_options, operator_type="state", cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values
            )
            # Saved the final cost value from this iteration for convergence check
            cost_fn_final_k = cost_values_state[-1]
            cost_fn_history.extend(cost_values_state)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator="STATE")
            if cost_below_threshold:
                break    
            
            if verbose:
                print(f"🏁 Iteration: {idx + 1}/{num_iterations}. f(x): {cost_fn_history[-1]:.6e}.")
            
            relative_cost_below_precision, convergence_reason = convergence_criteria_from_relative_precision(cost_fn_previous=cost_fn_init_k, cost_fn_current=cost_fn_final_k, relative_precision=relative_precision)
            if relative_cost_below_precision:
                break
            if verbose:
                print(f"\n 🎯 Relative change in cost function value: {abs(cost_fn_final_k - cost_fn_init_k)/abs(cost_fn_init_k):.2e}. Relative precision target: {relative_precision:.2e}.")
                
            # This would only happen if we use RTR for all 3 operators, since GDS does not stop early.
            # TODO: implement stopping criteria for GDS based on gradient norm to allow early stopping.
            all_gradients_converged = finished_early_povm and finished_early_kraus and finished_early_state
            if all_gradients_converged:
                convergence_reason = "All gradients converged 💥." 
                break

    except KeyboardInterrupt:
        print(f"Optimized interrupted by user at outer iteration {idx}.")
        
    
    final_message = f"‼️ Full optimization finished at iteration {idx + 1}/{num_iterations} ‼️ \n Reason: {convergence_reason}"
    print(final_message) 
        
    optimized_operators = {
        "kraus": kraus_tensor_k,
        "povm": povm_psd_k,
        "state": state_psd_k,
    }
    return optimized_operators, cost_fn_history

def convergence_criteria_from_noise_threshold(cost_fn_value:float, noise_threshold:float, operator:str)-> tuple[bool, str]:
    """Determine convergence based on noise threshold.
    
    Args:
        cost_fn_value: The current value of the cost function.
        noise_threshold: The noise threshold to determine if the optimization has converged. If the cost function value is below this threshold, the optimization will stop early.
        operator: The operator type for which to check convergence ('kraus', 'povm', or 'state'). This is used for the convergence message.
        
    Returns:
        A tuple containing:
        - A boolean indicating whether convergence criteria are met.
        - A string describing the reason for convergence.
    """
    message = "No convergence criteria met yet ⏳."
    converged = False
    if cost_fn_value < noise_threshold:
        converged = True
        message = f"Cost function value for {operator.capitalize()} below noise threshold {noise_threshold:.2e} 📶."
    return converged, message

def convergence_criteria_from_relative_precision(cost_fn_previous:float, cost_fn_current:float, relative_precision:float)-> tuple[bool, str]:
    """Determine convergence based on relative precision of cost function values.
    
    Args:
        cost_fn_previous: The previous value of the cost function.
        cost_fn_current: The current value of the cost function.
        relative_precision: The target relative precision to determine convergence. If the relative change in the cost function value is below this threshold, the optimization is considered converged.
        
    Returns:
        A tuple containing:
        - A boolean indicating whether convergence criteria are met.
        - A string describing the reason for convergence.
    """
    message = "No convergence criteria met yet ⏳."
    converged = False
    relative_change = abs(cost_fn_current - cost_fn_previous) / abs(cost_fn_previous)
    if relative_change < relative_precision:
        converged = True
        message = f"Relative change in cost function value {relative_change:.2e} below target relative precision {relative_precision:.2e} 🎯."
    return converged, message

def _optimize_single_operator(kraus_tensor:jnp.ndarray, povm_psd:jnp.ndarray, state_psd:jnp.ndarray, optimization_options:OptimizationOptions, operator_type:str, cost_fn_kwargs:dict[str, Any], cost_function:Callable = None, save_intermediate_cost_values:bool=False)-> tuple[jnp.ndarray, list[float]]:
    """
    Optimize a single operator (Kraus, POVM, or State) using the specified optimization options.
    
    Args:
        kraus_tensor: The current Kraus operator tensor.
        povm_psd: The current POVM operator tensor.
        state_psd: The current State operator tensor.
        optimization_options: The optimization options to use (TrustRegionOptions or GradientDescentOptions).
        operator_type: The type of operator to optimize ('kraus', 'povm', or 'state').
        cost_function: The cost function to minimize. Needed if using TrustRegionOptions.
        cost_fn_kwargs: Additional keyword arguments to pass to the cost function. Needed both for TrustRegionOptions and GradientDescentOptions.
        save_intermediate_cost_values: Whether to save all cost function values during the optimization of the operator. For now, we can only save intermediate values if using TrustRegionOptions, since the GDS does not return intermediate values.
    """
    options_dict = optimization_options.to_dict()
    if isinstance(optimization_options, TrustRegionOptions):
        if operator_type == "povm":
            x_init = povm_psd
            cost_fn_x = lambda x: cost_function(
                kraus_tensor=kraus_tensor, povm_psd=x, state_psd=state_psd, **cost_fn_kwargs)
        elif operator_type == "kraus":
            x_init = kraus_tensor
            cost_fn_x = lambda x: cost_function(
                kraus_tensor=x, povm_psd=povm_psd, state_psd=state_psd, **cost_fn_kwargs)
        else:
            x_init = state_psd
            cost_fn_x = lambda x: cost_function(
                kraus_tensor=kraus_tensor, povm_psd=povm_psd, state_psd=x, **cost_fn_kwargs)
            
        # Purposedly supressing warnings since the TR method raises warnings often, e.g. when the gradient is close to zero or t < 0.
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                optimized_operator, _, cost_values, finished_early = run_trust_region_optimization(x_init=x_init, cost_function=cost_fn_x, operator_type=operator_type, **options_dict)
        if save_intermediate_cost_values:
            saved_cost_values = cost_values
        else:
            saved_cost_values = [cost_values[-1]]
        
        
    elif isinstance(optimization_options, GradientDescentOptions):
        gds_options = options_dict | cost_fn_kwargs
        gds_options.pop("jit", None)  # Remove jit option if present, as it's not used in _update_tensor_via_gradient
        print(f"GDS started for operator: {operator_type} 🚀.")
        optimized_operator, _, cost_value = _update_tensor_via_gradient(operator_type=operator_type, kraus_tensor=kraus_tensor, povm_psd=povm_psd, state_psd=state_psd, return_cost_fn_value=True, **gds_options)
        finished_early = False # GDS does not have a built-in stopping criteria as a single step. TODO: implement stopping criteria based on gradient norm.
        saved_cost_values = [cost_value]
    else:
        raise ValueError(f"Invalid optimization options: {optimization_options}. Must be TrustRegionOptions or GradientDescentOptions.")

    return optimized_operator, saved_cost_values, finished_early


def estimate_noise_floor_threshold(probability_matrix:jnp.ndarray, num_shots:int, multiplier:float = 5.0)->float:
    """Estimate the noise floor threshold for the least-squares cost from empirical probabilities.

    Estimates the expected mean squared error per (sequence, POVM element) due to
    finite-sampling (shot) noise. 
    
    When the cost function drops below this threshold,
    further optimization would mostly fit statistical fluctuations rather than
    improve the physical model.

    Each empirical frequency y_ij is a binomial estimate of the true probability
    p_ij with variance Var(y_ij) = p_ij(1 - p_ij) / num_shots, approximated by
    y_ij(1 - y_ij) / num_shots. The threshold is the average of this variance
    over all sequences and POVM elements, scaled by ``multiplier``::

        threshold = multiplier / (num_sequences * num_povm * num_shots)
                    * sum_{i,j} y_ij (1 - y_ij)

    Args:
        probability_matrix: Empirical outcome probabilities, shape ``(num_povm, num_sequences)``.
        num_shots: Number of measurement shots per sequence.
        multiplier: Safety factor applied to the noise floor. Adjust based on empirical observations of typical noise levels. Default is 5.0.

    Returns:
        The noise floor threshold.
    """
    num_povm, num_sequences = probability_matrix.shape
    threshold = jnp.sum(probability_matrix * (1 - probability_matrix))
    # Expected square error from binomial noise
    threshold /= num_shots 
    # average over all POVM and Sequences
    threshold /= (num_sequences * num_povm)
    # Apply multiplier to adjust the threshold based on empirical observations (typical noise level). This can be tuned.
    threshold *= multiplier
    return threshold