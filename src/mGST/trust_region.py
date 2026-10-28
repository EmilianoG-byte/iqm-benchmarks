"""
Trust Region module for Optimization on the Stiefel Manifold
"""

import jax.numpy as jnp
import warnings
from mGST.automatic_diff import hvp, automatic_gradient
from mGST.utility_functions_comparisons import tensor_to_isometry, isometry_to_tensor, tensors_to_isometries, get_isometry_dimensions_from_tensor, _update_tensor_via_gradient
from mGST.riemannian import riemannian_connection, riemannian_metric, update_isometry_tensors, project_onto_tangent_space, is_isometry
from mGST.linear_algebra import transpose
from mGST.typing import Tensor, Matrix, Scalar, TrustRegionOptions, GradientDescentOptions, OptimizationOptions, OperatorSchedule
from mGST.trust_region_linearized import riemannian_and_euclidean_gradient_fn, linearize_riemannian_gradient_and_hvp
from mGST.jitted_objective import JittedObjective

import jax

from typing import Callable, Any

RADIUS_HEURISTICS = frozenset({"auto", "typical_distance", "gradient"})
# Smallest initial radius the heuristics may return, as a fraction of the maximum radius.
RADIUS_FLOOR_FRACTION = 1e-8

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
    rgrad, _ = riemannian_and_euclidean_gradient_fn(x=x, cost_fn=cost_fn, operator_type=operator_type, metric=metric)
    return rgrad

def riemannian_hessian_vector_fn(x:Tensor, tangent_vector:Tensor, cost_fn:Callable[[Tensor], Scalar], operator_type:str, metric:str="canonical", return_tensor:bool=True, return_euclidean:bool=False)-> Tensor|Matrix|tuple[Tensor|Matrix, Tensor]:
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
        return_euclidean: Whether to also return the Euclidean Hessian-vector product (directional derivative of 2df/dx along
            tangent_vector), as needed by `compute_quality_quotient`. It comes out of the same jvp, so it is free.

    Returns:
        The Riemannian Hessian-vector product, either as a tensor or matrix depending on return_tensor.
        If return_euclidean, a tuple (Riemannian Hessian-vector product, Euclidean Hessian-vector product as a tensor).
    """
    # Riemannian gradient (from df/dx*) together with the Euclidean gradient 2df/dx
    rgrad_and_euclidean_fn = lambda x: riemannian_and_euclidean_gradient_fn(x=x, cost_fn=cost_fn, operator_type=operator_type, metric=metric)

    (rgrad_x_tensor, _), (Drgrad_x_to_z_tensor, euclidean_hvp_tensor) = hvp(function=rgrad_and_euclidean_fn, x=x, z=tangent_vector)

    x_matrix, tangent_vector_matrix, rgrad_x_matrix, Drgrad_x_to_z_matrix = tensors_to_isometries(
        x, tangent_vector, rgrad_x_tensor, Drgrad_x_to_z_tensor, operator_type=operator_type)
    
    rhessian_vector_product_matrix = riemannian_connection(x=x_matrix, w_x=rgrad_x_matrix, z=tangent_vector_matrix, Dw_x_to_z=Drgrad_x_to_z_matrix, metric=metric)
    
    if return_tensor:
        rhessian_vector_product_matrix = isometry_to_tensor(rhessian_vector_product_matrix, x.shape)
    if return_euclidean:
        return rhessian_vector_product_matrix, euclidean_hvp_tensor
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
        
    if jnp.isnan(t):
        raise ValueError(f"Step size 't' is NaN 👵. z: {z}, delta: {delta}, radius: {radius}")
        
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

def truncated_conjugate_gradient(x: Tensor, radius: float, num_iterations:int, rgradient:Tensor, metric:str, n:int, p:int, rhessian_vector_fn:Callable[[Tensor, Tensor], Tensor], verbose:bool=True, theta:float | None = None, kappa: float | None = None, return_euclidean_hvp:bool = False) -> tuple[Tensor, bool] | tuple[Tensor, bool, Tensor]:
    """ Truncated Conjugate Gradient (TCG) method to approximately solve the trust region subproblem on the Stiefel manifold.

    With return_euclidean_hvp=True, the Euclidean Hessian-vector product along the solution η is accumulated for free:
    η = Σ_j α_j δ_j (plus a final τ δ on the boundary), and the Euclidean Hessian-vector product is linear in its direction,
    so H_e[η] = Σ_j α_j H_e[δ_j] (+ τ H_e[δ]), where every H_e[δ_j] comes out of the same jvp as the Riemannian H[δ_j].
    This saves the extra Hessian-vector product needed by the model in `compute_quality_quotient`.

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
        return_euclidean_hvp: If True, rhessian_vector_fn(x, z) must return a tuple (Riemannian HVP, Euclidean HVP),
            and the Euclidean HVP along the solution is returned as third output.

    Returns:
        A tuple containing:
        - The approximate solution to the trust region subproblem (tangent vector), i.e. the proposed direction to update x.
        - A boolean indicating whether the solution lies on the boundary of the trust region.
        - Only if return_euclidean_hvp: the Euclidean Hessian-vector product along the solution.
    """
    solution_tensor = jnp.zeros_like(x) # η
    euclidean_hvp_solution = jnp.zeros_like(x) if return_euclidean_hvp else None # H_e[η]
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
        if return_euclidean_hvp:
            rhessian_delta_tensor, euclidean_hvp_delta = rhessian_vector_fn(x=x, z=delta_tensor)
        else:
            rhessian_delta_tensor = rhessian_vector_fn(x=x, z=delta_tensor)
        # Remove the numerically small normal component of Hδ, so that r and δ stay in the tangent space at x.
        # Otherwise it gets amplified in r as tCG converges (|r| << |α Hδ|) and breaks the CG recurrences.
        rhessian_delta_tensor = project_tensor_onto_tangent_space(x=x, z=rhessian_delta_tensor, n=n, p=p)
        
        rhessian_delta_matrix = tensor_to_isometry(rhessian_delta_tensor, n=n, p=p)
        delta_matrix = tensor_to_isometry(delta_tensor, n=n, p=p)
        # Compute curvature
        curvature = riemannian_metric(delta_matrix, rhessian_delta_matrix, x=x_matrix, metric=metric)        
        # Negative curvature
        if curvature < 0:
            reason = "Negative curvature encountered 📉."
            step = compute_step_size(z=solution_matrix, delta=delta_matrix, radius=radius, x=x_matrix, metric=metric)
            solution_tensor += step * delta_tensor
            if return_euclidean_hvp:
                euclidean_hvp_solution += step * euclidean_hvp_delta
            on_boundary = True
            break
        
        solution_matrix_old = solution_matrix
        alpha = norm_sqrd_r / curvature
        solution_tensor_new  = solution_tensor + alpha * delta_tensor
        
        solution_matrix = tensor_to_isometry(solution_tensor_new, n=n, p=p) 
        norm_solution = jnp.sqrt(riemannian_metric(solution_matrix, solution_matrix, x_matrix, metric=metric))
        
        # Check if the solution exceeds the trust region boundary
        if norm_solution >= radius:
            # this should be evaluated at iteration k not k+1
            reason = "Exceeded trust region boundary 📈."
            step = compute_step_size(z=solution_matrix_old, delta=delta_matrix, radius=radius, x=x_matrix,metric=metric)
            solution_tensor += step * delta_tensor
            if return_euclidean_hvp:
                euclidean_hvp_solution += step * euclidean_hvp_delta
            on_boundary = True
            break

        solution_tensor = solution_tensor_new
        if return_euclidean_hvp:
            euclidean_hvp_solution += alpha * euclidean_hvp_delta

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
            
    if jnp.isnan(solution_tensor).any():
        raise ValueError("Solution tensor contains NaN values. 👵.")

    if return_euclidean_hvp:
        return solution_tensor, on_boundary, euclidean_hvp_solution
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
     
def compute_approx_terms_from_cost_fn(x:Tensor, z:Tensor, n:int, p:int, cost_function:Callable)->tuple[float, float]:
    """Helper function to compute the first and second order terms of the model approximation for the local cost function in the tangent space.
    
    Handles batch dimensions.
    
    Args:
        x: Current point on the manifold.
        z: Update direction (tangent vector).
        n: Dimension of the Stiefel manifold.
        p: Dimension of the Stiefel manifold.
        cost_function: The cost function to approximate.

    Returns:
        A tuple containing:
        - The first-order term of the model approximation.
        - The second-order term of the model approximation.
    """
    # TODO: use grad_and_value to evaluate the cost function when computing the gradient at hvp.
    gradient_function = automatic_gradient(cost_function)
    gradient_conjugated, euclidean_hvp = hvp(function=gradient_function, x=x, z=z) # 2df/dx, 2(Hxx dx + Hx*x dx*)
    return compute_first_and_second_order_terms(
        x=x, z=z, n=n, p=p, euclidean_gradient_conjugated=gradient_conjugated, euclidean_hvp=euclidean_hvp
    )

def compute_first_and_second_order_terms(x:Tensor, z:Tensor, n:int, p:int, euclidean_gradient_conjugated:Tensor, euclidean_hvp:Tensor)->tuple[float, float]:
    """Compute first and second order terms of the model approximation, reusing an already-computed gradient/HVP.

    Args:
        x: Current point on the manifold.
        z: Update direction (tangent vector).
        n, p: Dimensions of the Stiefel manifold.
        euclidean_gradient_conjugated: Euclidean gradient at x (2df/dx), reused from the outer loop's linearization.
        euclidean_hvp: Euclidean Hessian-vector product at x in direction z, from the same linearization's linear map applied to z.
    Returns:
        A tuple containing:
        - The first-order term of the model approximation.
        - The second-order term of the model approximation.
    """
    first_order_term = _compute_first_order_term(z, euclidean_gradient_conjugated)
    second_order_term = _compute_second_order_term(x=x, z=z, gradient_conjugated=euclidean_gradient_conjugated, hessian_z=euclidean_hvp, n=n, p=p)
    return first_order_term, second_order_term

def compute_model_approximation(
    x:Tensor,
    update_direction:Tensor,
    cost_function:Callable,
    n:int,
    p:int,
    euclidean_gradient_conjugated:Tensor = None,
    euclidean_hvp:Tensor = None,
    order:int=2,
    include_zero:bool=True)->float:
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
    # if the gradient and hvp are provided, use them.
    if euclidean_gradient_conjugated is not None and euclidean_hvp is not None:
        first_order_term, second_order_term = compute_first_and_second_order_terms(x=x, z=update_direction, n=n, p=p, euclidean_gradient_conjugated=euclidean_gradient_conjugated, euclidean_hvp=euclidean_hvp)
    else:
        first_order_term, second_order_term = compute_approx_terms_from_cost_fn(
        x=x, z=update_direction, n=n, p=p, cost_function=cost_function,
    )
    # Add higher-order terms if needed
    if order >= 1:
        approximation_value += first_order_term
    if order == 2:
        approximation_value += second_order_term
    return approximation_value

def compute_quality_quotient(
    x:Tensor,
    update_direction:Tensor,
    cost_function:Callable,
    x_next:Tensor,
    n:int,
    p:int,
    euclidean_gradient_conjugated:Tensor = None,
    euclidean_hvp:Tensor = None,
    cost_fx:Scalar|None = None,
    ) -> tuple[Scalar, Scalar]:
    """
    Compute the quality quotient for a proposed update direction in the trust region method.

    Handles batch dimensions.

    Args:
        x: Current point on the manifold.
        update_direction: Proposed update direction (tangent vector).
        cost_function: The cost function to approximate.
        x_next: The proposed next point on the manifold after applying the update direction using the retraction.
        n,p : Dimensions of the Stiefel manifold.
        cost_fx: The cost function value at x, if already known. If None, it is recomputed.

    Returns:
        A tuple containing:
        - The quality quotient (Scalar).
        - The cost function value at the proposed next point (Scalar).
    """
    if cost_fx is None:
        cost_fx = cost_function(x)
    cost_fx_next = cost_function(x_next)
    # f(x_next) - f(x)
    cost_decrease = cost_fx_next - cost_fx
    # m(η) - m(0), negative for a descent step
    model_decrease = compute_model_approximation(x, update_direction, cost_function, n, p, euclidean_gradient_conjugated=euclidean_gradient_conjugated, euclidean_hvp=euclidean_hvp, order=2, include_zero=False)
    # regularization for qualitient quotient.
    # see manopt implementation: https://manoptjl.org/v0.2/solvers/trust_regions.html
    # and "regularized ratio from Conn, Gould, and Toint, Trust-Region Methods (2000) §17.4.2." for details
    rho_reg = max(1.0, abs(cost_fx)) * jnp.finfo(float).eps * 1e3
    quality_quotient = (cost_decrease - rho_reg) / (model_decrease - rho_reg)
    if jnp.isnan(quality_quotient):
        raise ValueError(f"Quotient is NaN. 👵. Δf = {cost_decrease}, Δm = {model_decrease}")
    return quality_quotient, cost_fx_next

def run_trust_region_optimization(
    x_init:Tensor, cost_function:Callable[[Tensor], Scalar], operator_type:str,
    radius_init:float|None = None, num_iterations:int = 20, max_radius:float|None = None, quotient_trust:float = 0.125, tol_grad:float = 1e-6, 
    metric:str = "euclidean", num_iterations_cg:int = 10, theta_cg:float = None, kappa_cg:float = None, verbose_cg:bool=True,
    global_norm_grad_init: float| None = None,
    verbose:bool=False, linearize:bool=False, radius_heuristic:str = "auto",
    rgrad_and_euclidean_fn:Callable[[Tensor], tuple[Tensor, Tensor]]|None = None,
    rhvp_and_euclidean_fn:Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]|None = None,
    )->tuple[Tensor, list[Tensor], list[Scalar], Scalar, Scalar]:
    """
    Run the trust region optimization algorithm.

    Args:
        cost_function: The cost function to minimize. Should take a tensor as input and return a scalar.
        x_init: The initial point on the manifold.
        radius_init: The initial trust region radius. If None, it is chosen by `radius_heuristic`.
        num_iterations: The maximum number of trust region iterations to perform.
        max_radius: The maximum trust region radius to allow. If None, the typical distance of the manifold is used.
        quotient_trust: The lower threshold for accepting a step based on the quality quotient.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        metric: The metric to use to calculate inner products in the truncated conjugate gradient algorithm. Can be "canonical" or "euclidean".
        num_iterations_cg: The maximum number of iterations to perform in the truncated conjugate gradient algorithm.
        theta_cg: The theta parameter for the stopping criteria of the truncated conjugate gradient algorithm.
        kappa_cg: The kappa parameter for the stopping criteria of the truncated conjugate gradient algorithm.
        tol_grad: The tolerance for the norm of the Riemannian gradient to determine convergence.
        verbose_cg: Whether to print information during the truncated conjugate gradient algorithm.
        global_norm_grad_init: A gradient norm value to used as the initial gradient norm for the stopping criteria. 
            Within an optimization outer loop, this can be the gradient before any optimization step is taken.
            If not provided, the initial gradient norm will be computed from the initial point within this RTR sweep.
        verbose: Whether to print information during the trust region outer loop optimization.
        linearize: Whether to linearize the Riemannian gradient once per TR iteration and reuse the linear map for the HVPs.
            Ignored if rhvp_and_euclidean_fn is given.
        radius_heuristic: Which heuristic resolves `radius_init` and `max_radius` when they are None:
            "auto", "typical_distance" or "gradient". See `resolve_trust_region_radii`. Explicit radii always win.
        rgrad_and_euclidean_fn: Precompiled (e.g. jitted) function x -> (Riemannian gradient, Euclidean gradient).
            Must be given together with rhvp_and_euclidean_fn. If None, it is built from cost_function with automatic differentiation.
        rhvp_and_euclidean_fn: Precompiled (e.g. jitted) function (x, z) -> (Riemannian HVP, Euclidean HVP).
            Must be given together with rgrad_and_euclidean_fn. See `mGST.jitted_objective.JittedObjective.bind`.

    Returns:
        A tuple containing:
        - The optimized point on the manifold.
        - A list of all accepted points during the optimization.
        - A list of the cost function values at each accepted point.
        - The final gradient norm.
        - The final trust region radius. To be used in the next iteration.

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
    num_rejections = 0
    n, p, k = get_isometry_dimensions_from_tensor(x_init, operator_type)

    # Define the riemannian gradient, riemannian hessian-vector product, and retraction functions
    retraction = lambda x, z: retraction_first_order(x, z, operator_type=operator_type)

    precompiled = rhvp_and_euclidean_fn is not None
    if precompiled != (rgrad_and_euclidean_fn is not None):
        raise ValueError("rgrad_and_euclidean_fn and rhvp_and_euclidean_fn must be given together.")
    if not precompiled:
        rgrad_and_euclidean_fn = lambda x: riemannian_and_euclidean_gradient_fn(x=x, cost_fn=cost_function, operator_type=operator_type, metric=metric)

    # In all three cases rhessian_vector_fn(x, z) returns (Riemannian HVP, Euclidean HVP). tCG accumulates the Euclidean HVP
    # along its solution, which is what the model in the quality quotient needs, so no extra HVP is evaluated for it.
    if precompiled:
        # The precompiled HVP replaces the linearization.
        linearize = False
        rgradient, euclidean_gradient_conjugated = rgrad_and_euclidean_fn(x_k)
        rhessian_vector_fn = rhvp_and_euclidean_fn

    elif linearize:
        rgradient, euclidean_gradient_conjugated, _, rhessian_vector_fn = linearize_riemannian_gradient_and_hvp(x=x_k, rgrad_and_euclidean_fn=rgrad_and_euclidean_fn, operator_type=operator_type, metric=metric, return_euclidean=True)

    else:
        # Compute the initial Riemannian gradient and its norm
        rgradient, euclidean_gradient_conjugated = rgrad_and_euclidean_fn(x_k)

        rhessian_vector_fn = lambda x, z: riemannian_hessian_vector_fn(x=x, tangent_vector=z, cost_fn=cost_function, operator_type=operator_type, metric=metric, return_tensor=True, return_euclidean=True)
    
    # TODO: Think whether we need to compute the initial gradient norm all the time.
    norm_grad_init = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_init, metric=metric))
    
    norm_grad = norm_grad_init

    # Resolved before `global_norm_grad_init` replaces norm_grad_init: the heuristics need the gradient at x_init.
    radius_k, max_radius = resolve_trust_region_radii(
        p=p, k=k, norm_grad=norm_grad_init, metric=metric, radius_init=radius_init, max_radius=max_radius, heuristic=radius_heuristic)
    
    if global_norm_grad_init is not None:
        norm_grad_init = global_norm_grad_init
    if verbose:
        print(f"🚀 TR started for operator: {operator_type} 🚀.")
        print(f"Radii ({radius_heuristic} heuristic): Δ0 = {radius_k:.2e}. Δmax = {max_radius:.2e}")
        print("=======================================")
    try:
        for idx in range(num_iterations):
            if verbose:
                print(f"TR Iteration: {idx}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}")

            # Solve the trust region subproblem
            # euclidean_hvp: Euclidean Hessian-vector product along update_direction, accumulated inside tCG.
            update_direction, on_boundary, euclidean_hvp = truncated_conjugate_gradient(x=x_k, radius=radius_k,num_iterations=num_iterations_cg, rgradient=rgradient, metric=metric, n=n, p=p, rhessian_vector_fn=rhessian_vector_fn, verbose=verbose_cg, theta=theta_cg, kappa=kappa_cg, return_euclidean_hvp=True)

            # Retract the update direction to obtain the next iterate
            x_next = retraction(x_k, update_direction)
            # Compute the quality quotient
            quality_quotient, cost_fx_next = compute_quality_quotient(
                x=x_k,
                update_direction=update_direction,
                cost_function=cost_function,
                x_next=x_next,
                n=n,
                p=p,
                euclidean_gradient_conjugated=euclidean_gradient_conjugated,
                euclidean_hvp=euclidean_hvp,
                # only accepted points are stored, so this is f(x_k)
                cost_fx=cost_fx_array[-1],
                )

            radius_k, x_k, x_k_array, cost_fx_array, num_rejections, accepted = _update_trust_region_radius_and_xk(
                quality_quotient=quality_quotient,
                radius_k=radius_k,
                max_radius=max_radius,
                on_boundary=on_boundary,
                quotient_trust=quotient_trust,
                x_k=x_k,
                x_next=x_next,
                x_k_array=x_k_array,
                cost_fx_array=cost_fx_array,
                cost_fx_next=cost_fx_next,
                num_rejections=num_rejections,
                verbose=verbose
            )
            # If the step was rejected, x_k did not change: the gradient, linearization and gradient norm are still valid.
            if not accepted:
                continue
            # Compute the Riemannian gradient at the new point
            if linearize:
                # Linearize at the new point for the next iteration
                # this creates a new rhessian_vector_fn using the new linear map.
                rgradient, euclidean_gradient_conjugated, _, rhessian_vector_fn = linearize_riemannian_gradient_and_hvp(x=x_k, rgrad_and_euclidean_fn=rgrad_and_euclidean_fn, operator_type=operator_type, metric=metric, return_euclidean=True)
            else:
                rgradient, euclidean_gradient_conjugated = rgrad_and_euclidean_fn(x_k)
            # Determine stopping criteria based on gradient norm
            norm_grad = jnp.sqrt(riemannian_metric_from_tensors(n=n, p=p, z1=rgradient, z2=rgradient, x=x_k, metric=metric))
            if determine_tr_stopping_criteria(norm_grad, norm_grad_init, tol_grad=tol_grad):
                if verbose:
                    print(f"(RTR) Stopping criteria met. 🚨 |r∇f(x)|/|r∇f(x_0)| = {norm_grad/norm_grad_init:.6e}")
                break
                
    except KeyboardInterrupt:
        print(f"Optimized interrupted by user {idx}.")
    if verbose:
        print("=======================================")
        print(f"Optimization finished ✅. \n Iters: {idx+1}. f(x): {cost_fx_array[-1]:.6e}. |r∇f(x)|: {norm_grad:.6e}. Radius: {radius_k:.2e}. Rejections: {num_rejections}")
    return x_k, x_k_array, cost_fx_array, norm_grad, radius_k

def _update_trust_region_radius_and_xk(quality_quotient:float, radius_k:float, max_radius:float, on_boundary:bool, quotient_trust:float, x_k:Tensor, x_next:Tensor, x_k_array:list, cost_fx_array:list, cost_fx_next:float, num_rejections:int, verbose:bool):
    "Auxiliary function to update the trust region radius and accept/reject the new point based on the quality quotient."
    # adding this extra check to not silently have a nan' value
    if jnp.isnan(quality_quotient):
        raise ValueError("Quality quotient is NaN. 👵.")
    
    if quality_quotient < 0.25:
        # Reduce the trust region radius
        radius_k *= 0.25
    elif quality_quotient > 0.75 and on_boundary:
        # Increase the trust region radius
        radius_k = min(2.0 * radius_k, max_radius)
    else:
        # Keep the trust region radius the same
        radius_k = radius_k
        
    accepted = bool(quality_quotient > quotient_trust)
    if accepted:
        # Accept the new point
        x_k = x_next
        # Only store accepted points
        x_k_array.append(x_k)
        cost_fx_array.append(cost_fx_next)
    else:
        # Reject the new point
        num_rejections += 1
        if verbose:
            print(f"Update rejected ❌. Quality quotient: {quality_quotient:.6e} < {quotient_trust:.6e}")

    return radius_k, x_k, x_k_array, cost_fx_array, num_rejections, accepted

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
    
    return riemannian_metric(z1=z1_matrix, z2=z2_matrix, x=x_matrix, metric=metric)

def compute_typical_distance(p:int, k:int, metric:str = "euclidean")->float:
    """Order of magnitude of the distance one can travel on a product of k Stiefel manifolds St(n, p).

    Every X in St(n, p) satisfies ||X||_F^2 = tr(X^H X) = p, so the manifold sits on a sphere of radius
    sqrt(p) and its diameter is O(sqrt(p)); for k independent factors the squared norms add up.
    This is pymanopt's `typical_dist`, used as the natural scale of the trust region radius.

    Args:
        p: The Stiefel p dimension.
        k: The number of independent Stiefel factors (see `get_isometry_dimensions_from_tensor`).
        metric: The metric in which the distance is measured, "euclidean" or "canonical".

    Returns:
        The typical distance of the manifold, in the units the trust region radius is measured in.

    Raises:
        ValueError: If the metric is not recognized.
    """
    if metric == "euclidean":
        scale = 1.0
    elif metric == "canonical":
        # <z,z>_c = ||A||^2/2 + ||B||^2 for z = xA + x_perp B, so the canonical norm of the rotational
        # directions is smaller than the euclidean one by sqrt(2).
        scale = 0.5
    else:
        raise ValueError(f"Metric: {metric} is not recognized. Please use one of the following: 'euclidean', 'canonical'.")
    return float(jnp.sqrt(scale * k * p))

def resolve_trust_region_radii(
    p:int, k:int, norm_grad:Scalar, metric:str = "euclidean",
    radius_init:float|None = None, max_radius:float|None = None,
    heuristic:str = "typical_distance", gradient_fraction:float = 0.1)->tuple[float, float]:
    """Choose the initial and maximum trust region radius when they are not given explicitly.

    `max_radius` always comes from the typical distance of the manifold: the gradient carries no
    information about how far one can travel on it. `radius_init` depends on the heuristic:

    - "typical_distance": Δ0 = typical_dist / 8, the pymanopt/Manopt default. Purely geometric,
      it knows nothing about the cost function.
    - "gradient": Δ0 = gradient_fraction * |r∇f(x_0)|, one of the classical choices surveyed in
      Conn, Gould & Toint, Trust-Region Methods (2000), §17.2. Function-aware, but it blows up
      where the log-likelihood is steep (model probabilities close to zero).
    - "auto": the minimum of the two, i.e. the function-aware choice tamed by the geometric cap.

    References:
        * [1] Absil, Baker & Gallivan, Trust-region methods on Riemannian manifolds, FoCM 7(3), 2007.
          Global convergence holds for any Δ0 in (0, Δ_max], so Δ0 is only a matter of efficiency.
        * [2] Boumal, An Introduction to Optimization on Smooth Manifolds (2023), Ch. 6.
        * [3] Conn, Gould & Toint, Trust-Region Methods, SIAM (2000), §17.2.

    Args:
        p: The Stiefel p dimension.
        k: The number of independent Stiefel factors (see `get_isometry_dimensions_from_tensor`).
        norm_grad: Norm of the Riemannian gradient at the initial point, in the same `metric`.
        metric: The metric the trust region radius is measured in, "euclidean" or "canonical".
        radius_init: The initial radius. Returned unchanged if not None.
        max_radius: The maximum radius. Returned unchanged if not None.
        heuristic: Which heuristic to use for `radius_init`: "auto", "typical_distance" or "gradient".
        gradient_fraction: Fraction of the initial gradient norm used by the "gradient" and "auto" heuristics.

    Returns:
        A tuple containing the resolved initial and maximum trust region radii.

    Raises:
        ValueError: If the heuristic is not recognized.
    """
    if heuristic not in RADIUS_HEURISTICS:
        raise ValueError(f"Radius heuristic '{heuristic}' is not recognized. Please use one of the following: {sorted(RADIUS_HEURISTICS)}.")
    if radius_init is not None and max_radius is not None:
        return radius_init, max_radius

    typical_distance = compute_typical_distance(p=p, k=k, metric=metric)
    if max_radius is None:
        max_radius = typical_distance

    if radius_init is None:
        radius_from_gradient = gradient_fraction * float(norm_grad)
        if heuristic == "typical_distance":
            radius_init = typical_distance / 8
        elif heuristic == "gradient":
            radius_init = radius_from_gradient
        else:
            radius_init = min(radius_from_gradient, typical_distance / 8)
        # The gradient vanishes at a saddle of the factorization (e.g. zero POVM/state columns),
        # where an unclamped radius would stall the optimization.
        radius_init = min(max(radius_init, RADIUS_FLOOR_FRACTION * max_radius), max_radius)

    return radius_init, max_radius

def determine_tr_stopping_criteria(norm_grad:float, norm_grad_init:float, tol_grad:float = 1e-8)->bool:
    """Determine whether to stop the trust region optimization based on the gradient norm.
    
    References:
        - ChatGPT
        - https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf (Section 6.4.6)
            They suggest to use 1e-8 as a default value for tol_grad, but this can be tuned based on the problem and desired precision.
    
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
    global_gradients_norm_init: dict[str, float] | None = None,
    global_gradient_norm_tol: float = 1e-6,
    verbose:bool=False,
    compute_least_squares:bool| int = False,
    carry_over_radius: bool = True,
    cost_floor: float|None = None,
    absolute_tolerance: float|None = None,
    use_jit: bool = True, # new default
    jit_contraction: str = "prefix",
    jitted_objective: JittedObjective|None = None,
    )-> tuple[dict[str, jnp.ndarray], list[float], str]:
    """
    Run the Riemannian optimization for each operator (Kraus, POVM, State) in an alternating fashion.
    
    NOTE: This function assumes the optimization order is povm -> kraus -> state.
    TODO: allow user to specify order.
    
    Args:
        kraus_tensor_init: Initial Kraus operator tensor. Shape: (num_gates, kraus_rank, dim_out, dim_in).
        povm_psd_init: Initial POVM operator tensor. Shape: (num_povm, rank_povm, dim_in).
        state_psd_init: Initial State operator tensor. Shape: (dim_in, rank_state).
        cost_function: The cost function to minimize. Should take kraus_tensor, povm_psd, state_psd as keyword arguments and return a scalar.
        cost_fn_kwargs: Additional keyword arguments to pass to the cost function.
        num_iterations: Number of outer iterations to perform (each iteration optimizes all operators once).
        optimization_options: A dictionary specifying the optimization options for each operator type.
            The keys should be "kraus", "povm", and "state", and the values should be instances of TrustRegionOptions or GradientDescentOptions.
            If None, default TrustRegionOptions will be used for all operators.
        save_intermediate_cost_values: Whether to save intermediate cost function values during the optimization of each operator. For now, we can only save all values if using TrustRegionOptions, since the GDS does not return intermediate values.
        noise_threshold: The noise threshold to determine if the optimization has converged. If the cost function value is below this threshold, the optimization will stop early. If None, this stopping criterion is disabled. See `estimate_noise_floor_threshold`.
        relative_precision: The relative precision to determine if the optimization has converged. If the relative change in the cost function value between the start and end of an iteration is below this threshold, the optimization will stop early. If None, this stopping criterion is disabled.
        global_gradients_norm_init: A dictionary containing the initial gradient norms for each operator type to be used in the stopping criteria based on gradient norms. The keys should be "kraus", "povm", and "state". If None, the initial gradient norms will be computed from the initial points within each optimization step.
        global_gradient_norm_tol: The tolerance for the gradient norm stopping criterion. If the norm of the gradient falls below this value, the optimization will stop.
        verbose: Whether to print information during the optimization
        compute_least_squares: Whether to compute the least squares value during the optimization in addition to the cost function. This can be useful for debugging and analysis, but may add additional computational overhead.
        cost_floor: The minimum cost function value considered for the relative precision stopping criterion. If None, the cost floor is computed from the data and cost function type.
            If this is any integer, we assume that we want to compute after every operator.
        carry_over_radius: Whether to carry over the trust region radius from the previous outer iteration.
            Only affects operators currently using TrustRegionOptions;
            ignored for operators using GradientDescentOptions.
        absolute_tolerance: Tolerance, in units of the cost function (nats for the log-likelihood), on the estimated
            remaining decrease of the cost function. See `convergence_criteria_from_absolute_tolerance`. If None, this stopping criterion is disabled.
        use_jit: Whether to use the vectorized, jit-compiled cost function, Riemannian gradient and HVP (see `mGST.jitted_objective`)
            for the operators optimized with TrustRegionOptions. The `linearize` option is then ignored.
            Only reproduces `cost_function_jax_mps`: this is checked against `cost_function` at the initial point.
        jit_contraction: Contraction strategy of the jitted cost function, "prefix" or "padded". Only used if use_jit=True.
        jitted_objective: An already built `JittedObjective` to use instead of building one from `cost_fn_kwargs`.
            `use_jit` and `jit_contraction` are then ignored, and the caller is responsible for having validated it
            against `cost_function` (see `JittedObjective.validate`). Useful to reuse one compilation across several calls.

    Returns:
        A tuple containing:
        - A dictionary with the optimized operators: {"kraus": kraus_tensor, "povm": povm_psd, "state": state_psd}.
        - A list of the cost function values at each optimization step.
        - The reason for convergence (if any).
    """
    # Validate optimization options
    optimization_schedule = validate_optimization_schedule(optimization_schedule=optimization_schedule, num_iterations=num_iterations)
    
    if compute_least_squares:
        if verbose:
            print("⚠️ Computing least squares values during optimization. This may add additional computational overhead. ⚠️")
        cost_fn_least_squares = []
        cost_fn_kwargs_least_squares = cost_fn_kwargs.copy()
        cost_fn_kwargs_least_squares["use_log_likelihood"] = False  # Ensure we compute least squares, not log-likelihood
        cost_fn_kwargs_least_squares.pop("num_shots", None)  # Remove num_shots if present
        
        compute_ls_every_step = False
        if isinstance(compute_least_squares, int):
            compute_ls_every_step = True
        
    kraus_tensor_k = kraus_tensor_init
    povm_psd_k = povm_psd_init
    state_psd_k = state_psd_init
    # Initially, we don't have any gradient norm
    if global_gradients_norm_init is not None:
        global_gradient_norm_init_povm = global_gradients_norm_init.get("povm", None)
        global_gradient_norm_init_kraus = global_gradients_norm_init.get("kraus", None)
        global_gradient_norm_init_state = global_gradients_norm_init.get("state", None)
    else:
        global_gradient_norm_init_povm = None
        global_gradient_norm_init_kraus = None
        global_gradient_norm_init_state = None
    
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
    if absolute_tolerance is not None:
        convergence_criteria += f"+ Absolute Tolerance 📉 {absolute_tolerance:.2e} \n"

    # Convergence based on early stopp using norms is always present as a last resource
    convergence_criteria += "+ Gradient(s) Norm(s) 📐"
    # Default convergence reason
    convergence_reason = "Max iterations reached ⏳."
    
    
    optimization_schedule_print = "\n".join(
    f"{operator}: {str(option)}"
    for operator, option in optimization_schedule.items()
    )
    
    # Per-operator trust-region radius carried over.
    # None on the first call for each operator, so the schedule's own radius_init is used.
    # Will continue to be None unless `carry_over_radius=True`.
    carried_radius = {"kraus": None, "povm": None, "state": None}
    
    if cost_floor is None:    
            cost_floor = compute_cost_floor(
                prob_matrix=cost_fn_kwargs["prob_matrix"],
                use_log_likelihood=cost_fn_kwargs.get("use_log_likelihood", False),
                num_shots=cost_fn_kwargs.get("num_shots"),
            )
    
    # Built once per run: compilation happens on the first call of each operator and is reused afterwards.
    if jitted_objective is None and use_jit:
        jitted_objective = JittedObjective.from_cost_fn_kwargs(cost_fn_kwargs, contraction=jit_contraction)
        jitted_objective.validate(cost_function, cost_fn_kwargs, kraus_tensor=kraus_tensor_init, povm_psd=povm_psd_init, state_psd=state_psd_init)

    if verbose:
        print("🔰 Starting Riemannian Optimization 🔰")
        print(f" * optimization schedule: \n{optimization_schedule_print} \n{convergence_criteria}")
        print(f" * cost floor: {cost_floor:.6e}")
        print(f" * jitted objective: {f'yes ({jit_contraction} contraction)' if jitted_objective is not None else 'no'}")
        
    # Cost at the start of the current outer iteration.
    cost_fn_previous_iteration = cost_function(kraus_tensor=kraus_tensor_init, povm_psd=povm_psd_init, state_psd=state_psd_init, **cost_fn_kwargs)
    cost_fn_history = [cost_fn_previous_iteration]
    # Decrease of the cost function over the previous outer iteration (used by the absolute tolerance criterion).
    cost_decrease_previous_iteration = None

    try:
        for idx in range(num_iterations):
            # TODO: I think I should be able to convert each optimization on each operator into a function to call it 3 times.
            # Optimize POVM
            current_operator = "povm"
            povm_options = optimization_schedule[current_operator].get_options_for_iteration(idx)
            povm_psd_k, cost_values_povm, gradient_norm_k_povm, final_radius_povm = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=povm_options, operator_type=current_operator, cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values, global_norm_grad_init=global_gradient_norm_init_povm, radius_init_override=carried_radius[current_operator], jitted_objective=jitted_objective
            )
            
            # only compute the least squares at the end.
            if compute_least_squares and compute_ls_every_step:
                least_squares_value = cost_function(kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, **cost_fn_kwargs_least_squares)
                cost_fn_least_squares.append(least_squares_value)
                if verbose:
                    print(f"👾 Least squares value after POVM: {least_squares_value:.6e} 👾")
            
            # Save the initial cost value from this iteration for convergence check
            cost_fn_history.extend(cost_values_povm)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator=current_operator)
            if cost_below_threshold:
                break
                            
            # Optimize Kraus
            current_operator = "kraus"
            kraus_options = optimization_schedule[current_operator].get_options_for_iteration(idx)
            kraus_tensor_k, cost_values_kraus, gradient_norm_k_kraus, final_radius_kraus = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=kraus_options, operator_type=current_operator, cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values, global_norm_grad_init=global_gradient_norm_init_kraus, radius_init_override=carried_radius[current_operator], jitted_objective=jitted_objective,
            )
            
            # only compute the least squares at the end.
            if compute_least_squares and compute_ls_every_step:
                least_squares_value = cost_function(kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, **cost_fn_kwargs_least_squares)
                cost_fn_least_squares.append(least_squares_value)
                if verbose:
                    print(f"👾 Least squares value after Kraus: {least_squares_value:.6e} 👾")
            
            cost_fn_history.extend(cost_values_kraus)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator=current_operator)
            if cost_below_threshold:
                break
            
            # Optimize State
            current_operator = "state"
            state_options = optimization_schedule[current_operator].get_options_for_iteration(idx)
            state_psd_k, cost_values_state, gradient_norm_k_state, final_radius_state = _optimize_single_operator(
                kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, optimization_options=state_options, operator_type=current_operator, cost_function=cost_function, cost_fn_kwargs=cost_fn_kwargs, save_intermediate_cost_values=save_intermediate_cost_values, global_norm_grad_init=global_gradient_norm_init_state, radius_init_override=carried_radius[current_operator], jitted_objective=jitted_objective
            )
            # only compute the least squares at the end.
            if compute_least_squares:
                least_squares_value = cost_function(kraus_tensor=kraus_tensor_k, povm_psd=povm_psd_k, state_psd=state_psd_k, **cost_fn_kwargs_least_squares)
                cost_fn_least_squares.append(least_squares_value)
                if verbose:
                    print(f"👾 Least squares value after State at iteration {idx + 1}: {least_squares_value:.6e} 👾")
            
            
            # Saved the final cost value from this iteration for convergence check
            cost_fn_current_k = cost_values_state[-1]
            cost_fn_history.extend(cost_values_state)
            # Check cost fn convergence
            cost_below_threshold, convergence_reason = convergence_criteria_from_noise_threshold(cost_fn_history[-1], noise_threshold, operator=current_operator)
            if cost_below_threshold:
                break    
            
            if verbose:
                print(f"🏁 Iteration: {idx + 1}/{num_iterations}. f(x): {cost_fn_history[-1]:.6e}.")
            
            relative_cost_below_precision, convergence_reason, relative_change = convergence_criteria_from_relative_precision(cost_fn_previous=cost_fn_previous_iteration, cost_fn_current=cost_fn_current_k, relative_precision=relative_precision, cost_floor=cost_floor)
            
            if verbose:
                print(f"\n 🎯 Relative change in reducible cost over iteration {idx + 1}: {relative_change:.2e}. Target: {relative_precision:.2e}.")
            
            if relative_cost_below_precision:
                break

            if absolute_tolerance is not None:
                cost_decrease = cost_fn_previous_iteration - cost_fn_current_k
                remaining_below_tolerance, convergence_reason, remaining_decrease = convergence_criteria_from_absolute_tolerance(cost_decrease=cost_decrease, cost_decrease_previous=cost_decrease_previous_iteration, absolute_tolerance=absolute_tolerance)
                if verbose:
                    print(f" 📉 Decrease over iteration {idx + 1}: {cost_decrease:.2e}. Estimated remaining decrease: {remaining_decrease:.2e}. Target: {absolute_tolerance:.2e}.")
                if remaining_below_tolerance:
                    break
                cost_decrease_previous_iteration = cost_decrease

            cost_fn_previous_iteration = cost_fn_current_k

            # This would only happen if we use RTR for all 3 operators, since GDS does not stop early.
            # TODO: implement stopping criteria for GDS based on gradient norm to allow early stopping.
            all_gradients_small, convergence_reason = convergence_criteria_from_gradients_norms(gradient_norm_k_povm, gradient_norm_k_kraus, gradient_norm_k_state, threshold=global_gradient_norm_tol)
            if all_gradients_small:
                break
            
            # If False, they remain None.
            if carry_over_radius:
                carried_radius["kraus"] = final_radius_kraus
                carried_radius["povm"] = final_radius_povm
                carried_radius["state"] = final_radius_state

    except KeyboardInterrupt:
        print(f"Optimized interrupted by user at outer iteration {idx}.")
        
    
    final_message = f"‼️ Full optimization finished at iteration {idx + 1}/{num_iterations} ‼️ \n Reason: {convergence_reason}. \n Trust region radii carried over: {carried_radius}."
    if verbose:
        print(final_message)

    optimized_operators = {
        "kraus": kraus_tensor_k,
        "povm": povm_psd_k,
        "state": state_psd_k,
    }
    
    cost_dict = {"main": cost_fn_history}
    if compute_least_squares:
        cost_dict["least_squares"] = cost_fn_least_squares
    
    return optimized_operators, cost_dict, convergence_reason


def convergence_criteria_from_gradients_norms(*gradients_norms, threshold:float = 1e-6)-> tuple[bool, str]:
    """Determine convergence based on the norms of the gradients.
    
    Args:
        *gradients_norms: The norms of the gradients for each operator.
        threshold: The threshold for the norm of the gradients to determine convergence. If the norm of all gradients is below this threshold, the optimization is considered converged.
        
    Returns:
        A tuple containing:
        - A boolean indicating whether convergence criteria are met.
        - A string describing the reason for convergence.
    """
    message = "No convergence criteria met yet ⏳."
    converged = False
    if max(gradients_norms) < threshold:
        converged = True
        message = f"Norms of all gradients below threshold {threshold:.2e} 📐."
    return converged, message


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


def compute_cost_floor(prob_matrix:jnp.ndarray, use_log_likelihood:bool=False, num_shots:int|None=None)->float:
    """Lower bound of the cost function, attained when the model probabilities equal the empirical frequencies.

    - Least squares: 0.
    - Negative log-likelihood: saturated-model value -N * sum_{n,o} y log y (with 0 log 0 = 0).
      The difference f - f_floor is N * sum_n KL(y_n || p_n), the part of the cost the optimizer can reduce.
    """
    if not use_log_likelihood:
        return 0.0
    y = prob_matrix
    y_log_y = jnp.where(y > 0, y * jnp.log(jnp.where(y > 0, y, 1.0)), 0.0)
    return float(-num_shots * jnp.sum(y_log_y))

def convergence_criteria_from_relative_precision(cost_fn_previous:float, cost_fn_current:float, relative_precision:float, cost_floor:float = 0.0)-> tuple[bool, str, float]:
    """Determine convergence based on the relative decrease of the reducible part of the cost function.

    relative_change = |f_prev - f_curr| / (f_prev - f_floor)

    i.e. the fraction of the remaining removable misfit eliminated during the last outer iteration.
    For the log-likelihood, f_prev - f_floor is N * sum_n KL(y_n || p_n) (in nats), which is O(degrees of freedom)
    at a statistically good fit, so relative_precision * (f_prev - f_floor) is the per-iteration improvement in nats.
    
    Args:
        cost_fn_previous: The previous value of the cost function.
        cost_fn_current: The current value of the cost function.
        relative_precision: The target relative precision to determine convergence. If the relative change in the cost function value is below this threshold, the optimization is considered converged.
        cost_floor: The lower bound of the cost function, used to compute the reducible part of the cost. Defaults to 0.0.
    Returns:
        A tuple containing:
        - A boolean indicating whether convergence criteria are met.
        - A string describing the reason for convergence.
        - The relative change in the cost function value.
    """
    message = "No convergence criteria met yet ⏳."
    converged = False
    excess_previous = max(cost_fn_previous - cost_floor, jnp.finfo(jnp.float64).tiny)
    relative_change = abs(cost_fn_current - cost_fn_previous) / excess_previous
    if relative_change < relative_precision:
        converged = True
        message = f"Relative change in reducible cost {relative_change:.2e} below target relative precision {relative_precision:.2e} 🎯."
    return converged, message, relative_change

def convergence_criteria_from_absolute_tolerance(cost_decrease:float, cost_decrease_previous:float|None, absolute_tolerance:float)-> tuple[bool, str, float]:
    """Determine convergence from an estimate of how much the cost function can still decrease.

    In the tail of the optimization the outer iterations converge linearly: the decrease per outer iteration
    shrinks by a roughly constant factor q = Δf_k / Δf_{k-1} < 1. Summing the geometric series, the remaining decrease is

        f_k - f_∞ ≈ Δf_k * q / (1 - q).

    We stop when max(Δf_k, Δf_k * q / (1 - q)) < absolute_tolerance. Taking the max with Δf_k makes the estimate
    conservative when q < 1/2 (e.g. noisy q early on). If q >= 1 (no linear convergence yet) or q is undefined
    (first iteration, or Δf_{k-1} <= 0), the remaining decrease is considered infinite and we never stop.

    For the negative log-likelihood the tolerance is in nats: 2 Δf is a likelihood-ratio statistic, so a change in f
    of 0.5 nats corresponds to moving ~1 standard deviation along one parameter direction. A tolerance of ~0.1 nats
    therefore means the optimization error is well below the statistical uncertainty of the estimate.

    Args:
        cost_decrease: Δf_k = f_{k-1} - f_k, the decrease of the cost function over the last outer iteration.
        cost_decrease_previous: Δf_{k-1}, the decrease over the outer iteration before. None on the first iteration.
        absolute_tolerance: Tolerance on the estimated remaining decrease, in units of the cost function.

    Returns:
        A tuple containing:
        - A boolean indicating whether convergence criteria are met.
        - A string describing the reason for convergence.
        - The estimated remaining decrease of the cost function.
    """
    message = "No convergence criteria met yet ⏳."
    converged = False
    remaining_decrease = jnp.inf
    if cost_decrease_previous is not None and cost_decrease_previous > 0:
        rate = cost_decrease / cost_decrease_previous
        if 0 <= rate < 1:
            remaining_decrease = max(cost_decrease, cost_decrease * rate / (1 - rate))
    if remaining_decrease < absolute_tolerance:
        converged = True
        message = f"Estimated remaining decrease of the cost function {remaining_decrease:.2e} below absolute tolerance {absolute_tolerance:.2e} 📉."
    return converged, message, remaining_decrease

def _optimize_single_operator(kraus_tensor:Tensor, povm_psd:Tensor, state_psd:Tensor, optimization_options:OptimizationOptions, operator_type:str, cost_fn_kwargs:dict[str, Any], cost_function:Callable = None, save_intermediate_cost_values:bool=False, global_norm_grad_init:Scalar|None = None, radius_init_override:Scalar|None = None, jitted_objective:JittedObjective|None = None)-> tuple[Tensor, list[Scalar], Scalar, Scalar]:
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
        global_gradient_norm_init: A gradient norm value to used as the initial gradient norm for the stopping criteria.
        radius_init_override: If provided overrides the schedule's radius_init for this
        call.
        jitted_objective: If provided (and using TrustRegionOptions), the jitted cost function, gradient and HVP are used
            instead of `cost_function` and automatic differentiation. Ignored for GradientDescentOptions.

    Returns:
        A tuple containing:
        - The optimized operator tensor.
        - A list of cost function values during the optimization of this operator (if save_intermediate_cost_values is True and using TrustRegionOptions, otherwise a list with only the final cost function value).
        - The gradient norm at the end of this step: for TrustRegionOptions, the Riemannian gradient norm at the returned
          (optimized) point; for GradientDescentOptions, the norm of the gradient at the input point (the one used for the step).
        - The final trust-region radius (TrustRegionOptions) or None (GradientDescentOptions, which has no radius concept).
    """
    options_dict = optimization_options.to_dict()
    if isinstance(optimization_options, TrustRegionOptions):
        # override the initial radius if provided
        if radius_init_override is not None:
            options_dict["radius_init"] = radius_init_override

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

        # Replace the cost function and its derivatives by the jitted ones, with the other two operators fixed.
        precompiled_fns = {}
        if jitted_objective is not None:
            cost_fn_x, precompiled_fns["rgrad_and_euclidean_fn"], precompiled_fns["rhvp_and_euclidean_fn"] = jitted_objective.bind(
                operator_type, kraus_tensor=kraus_tensor, povm_psd=povm_psd, state_psd=state_psd, metric=options_dict["metric"])

        # Purposedly supressing warnings since the TR method raises warnings often, e.g. when the gradient is close to zero or t < 0.
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                optimized_operator, _, cost_values, norm_grad_k, final_radius = run_trust_region_optimization(x_init=x_init, cost_function=cost_fn_x, operator_type=operator_type, global_norm_grad_init=global_norm_grad_init, **options_dict, **precompiled_fns)
        if save_intermediate_cost_values:
            saved_cost_values = cost_values
        else:
            saved_cost_values = [cost_values[-1]]
        
        
    elif isinstance(optimization_options, GradientDescentOptions):
        gds_options = options_dict | cost_fn_kwargs
        gds_options.pop("jit", None)  # Remove jit option if present, as it's not used in _update_tensor_via_gradient
        print(f"GDS started for operator: {operator_type} 🚀.")
        optimized_operator, _, norm_grad_k ,cost_value = _update_tensor_via_gradient(operator_type=operator_type, kraus_tensor=kraus_tensor, povm_psd=povm_psd, state_psd=state_psd, return_cost_fn_value=True, **gds_options)
        saved_cost_values = [cost_value]
        final_radius = None
    else:
        raise ValueError(f"Invalid optimization options of type: {type(optimization_options)}. Must be TrustRegionOptions or GradientDescentOptions.")

    return optimized_operator, saved_cost_values, norm_grad_k, final_radius


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

def project_tensor_onto_tangent_space(x: Tensor, z: Tensor, operator_type: str = None, n: int = None, p: int = None) -> Tensor:
    """Project a tensor z onto the tangent space of the Stiefel manifold at the point represented by the tensor x.

    The tensors are internally reshaped into isometries of dimensions (n, p), projected, and reshaped back to the shape of z.

    Handles batch dimensions.

    Args:
        x: Tensor representing the base point on the Stiefel manifold.
        z: Tensor to be projected onto the tangent space at x. Must have the same shape as x.
        operator_type: The type of the input operator used to infer the dimension of
            the isometry manifold (n, p). Can be: ('kraus', 'state', or 'povm')
        n: First dimension of isometry if known
        p: Second dimension of isometry if known

    Returns:
        The projected tensor, with the same shape as z.
    """
    x_matrix, z_matrix = tensors_to_isometries(x, z, operator_type=operator_type, n=n, p=p)
    return isometry_to_tensor(project_onto_tangent_space(x=x_matrix, z=z_matrix), tensor_shape=z.shape)


def is_tensor_isometry(x: Tensor, operator_type: str)->bool:
    """Check if a tensor represents an isometry on the Stiefel manifold.

    Args:
        x: Tensor to be checked.
        operator_type: The type of the input operator used to infer the dimension of
            the isometry manifold (n, p). Can be: ('kraus', 'state', or 'povm')

    Returns:
        True if the tensor represents an isometry, False otherwise.
    """
    x_matrix = tensors_to_isometries(x, operator_type=operator_type)[0] # since this returns a tuple
    return is_isometry(x_matrix)

def are_all_tensors_in_gate_set_isometries(gate_set: dict[str, Tensor])->bool:
    """Check if all tensors in the gate set represent isometries on the Stiefel manifold.

    Args:
        gate_set: Mapping from gate names to their corresponding tensors.
            The names can be 'kraus', 'state', or 'povm'.
        

    Returns:
        True if all tensors in the gate set represent isometries, False otherwise.
    """
    return all(is_tensor_isometry(tensor, operator_type=name) for name, tensor in gate_set.items())