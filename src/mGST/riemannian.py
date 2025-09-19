"""
Riemannian functions for the Stiefel Manifold, its tangent space and related operations.
"""

import jax.numpy as jnp
import jax
from typing import Sequence

from mGST.typing import Tensor, Matrix, Scalar
from mGST.utility_functions_comparisons import transpose

def riemannian_connection(x:Matrix, w_x:Matrix, z:Matrix, Dw_x_to_z:Matrix, metric:str = "euclidean", alphas:tuple|None = None)-> Matrix:
    """
    Riemannian connection ∇w(x)[z] parametrized for different metrics on the stiefel manifold.
    
    Handles batch dimensions.

    From equation 5.4 of https://arxiv.org/abs/2009.10159
    
    Args:
        x: Base point isometry of the tangent space. Shape (..., n, p)
        w_x: Riemannian vector field evaluated at x. Shape (..., n, p)
        z: Riemannian tangent vector equivalent to the "direction" of the derivative. Shape (..., n, p)
        Dw_x_to_z: Euclidean directional derivative of the vector field w in the direction of z evaluated at x. Shape (..., n, p)
        metric: The type of the Riemannian metric to use ('euclidean' or 'canonical'). Defaults to 'euclidean'.
        alphas: Optional tuple of alpha0 and alpha1 parameters to define a custom metric. If provided, overrides the metric parameter.
    Returns:
        The riemannian connection of the vector field w_x in the direction of z at x. The last two dimensions are (n, p)
    Raises:
        ValueError: If the metric is not recognized.
    """
    if metric == "euclidean":
        alpha0, alpha1 = 1, 1
    elif metric == "canonical":
        alpha0, alpha1 = 1, 0.5
    else:
        raise ValueError("Metric must be either 'euclidean' or 'canonical'")
    
    if alphas is not None:
        alpha0, alpha1 = alphas
    
    # Identity with batch broadcast
    *batch_shape, n, p = x.shape
    In = jnp.eye(n, dtype=x.dtype)
    In = jnp.broadcast_to(In, tuple(batch_shape) + (n, n))

    # NOTE @ already handles batched matrix multiplication.
    # --- term1: 0.5 * x @ (zᴴ w_x + w_xᴴ z)
    term1 = 0.5 * x @ (transpose(z.conj()) @ w_x + transpose(w_x.conj()) @ z)

    # --- term2: (In - x xᴴ)(z w_xᴴ + w_x zᴴ)x
    proj = In - x @ transpose(x.conj())
    middle = z @ transpose(w_x.conj()) + w_x @ transpose(z.conj())
    term2 = proj @ middle @ x

    return Dw_x_to_z + term1 + ((alpha0 - alpha1) / alpha0) * term2

def riemannian_metric(z1:Matrix, z2:Matrix, x:Matrix = None, metric:str = "euclidean")-> Scalar:
    """
    Compute the riemannian metric on the stiefel manifold at the point x for two tangent vectors z1 and z2.
    
    Handles batch dimensions. To compute the metric of the isometries in the product manifold, we make use of the following relations:

    * The tangent space of the cartesian product of two manifolds (which is also a manifold) M1 x M2 decomposes as the direct sum of the individual tangent spaces at a point (p1, p2): T_(p1,p2)(M1 x M2) ≅ T_p1(M1) ⊕ T_p2(M2) (natural identification) [1]
    * Thus, the product metric g = g1 ⊕ g2 is defined as g((u1, u2), (v1, v2)) = g1(u1, v1) + g2(u2, v2) for tangent vectors (u1, u2), (v1, v2) in T_(p1,p2)(M1 x M2) [1, 2]

    References:
    [1] https://en.wikipedia.org/wiki/Product_metric
    [2] https://math.stackexchange.com/questions/173159/product-of-riemannian-manifolds

    Args:
        z1: First tangent vector of dimensions (n, p)
        z2: Second tangent vector of dimensions (n, p)
        x: Point on the stiefel manifold of dimensions (n, p)
        metric: The type of the metric to use ('euclidean' or 'canonical'). Defaults to 'euclidean'.
    
    Returns:
        The inner product of the two tangent vectors at the point x
    """
    *batch_shape, n, p = z1.shape
    In = jnp.eye(n, dtype=z1.dtype)
    In = jnp.broadcast_to(In, tuple(batch_shape) + (n, n))
    if metric == "euclidean":
        gamma = In
    elif metric == "canonical":
        if x is None:
            raise ValueError("To use the canonical metric, the point x on the stiefel manifold must be provided.")
        gamma = In - 0.5 * (x @ transpose(x.conj()))
    else:
        raise ValueError(f"Metric: {metric} is not recognized. Please use one of the following: 'euclidean', 'canonical'")
    # Take the trace of the last two dimensions and sum over batch dimensions
    return jnp.einsum("...ii->", transpose(z1.conj()) @ gamma @ z2).real

def update_isometry_tensors(isometries:Sequence[Matrix] | Matrix, update_directions:Sequence[Matrix] | Matrix, step_size:float, operator_type:str, use_geodesic:bool =  True) ->  jnp.ndarray:
    """Update a tensor of isometries in the direction of the tensor of tangent vectors scaled by step size
    
    In order to retract back to the stiefel manifold we either follow the geodesic or use a first order retraction.

    Handles batch dimension explicitly for kraus operators.

    Args:
        isometries: The isometries to be updated. Dimensions are (n, p) or (num_gates, n, p) for kraus operators
        tangent_vectors: The tangent vectors at the point x on the stiefel manifold. 
            Must have the same dimensions as isometries.
        step_size: The step size of the update.
        operator_type: The type of the input operator ('kraus', 'state', 'povm')
        use_geodesic: Whether to use the geodesic to compute to the updated Kraus tensor. Defaults to True.
    """
    
    if operator_type!= "kraus":
        isometries = [isometries] # for povm and state isometries we only have one isometry
        update_directions = [update_directions]
        
    new_isometries = []
    for isometry, vector in zip(isometries, update_directions):
        new_isometry = update_isometry_via_retraction(x=isometry, z=vector, step_size=step_size, use_geodesic=use_geodesic)
                    
        new_isometries.append(new_isometry)
        
    if operator_type != "kraus":
        return new_isometries[0]
    
    # For Kraus operators
    return jnp.array(new_isometries)

def update_isometry_via_retraction(x:Matrix, z:Matrix, step_size:float = 1, use_geodesic:bool = True)->Matrix:
    """Update a single isometry in the direction of the tangent vector scaled by step size
    
    Args:
        x: The isometry to be updated. Shape: (n, p)
        z: The tangent vector at the point x on the stiefel manifold. Shape: (n, p)
        step_size: The step size of the update.
        use_geodesic: Whether to use the geodesic as retraction. Defaults to True.
            Else, uses a first order retraction based on polar decomposition.
    Returns:
        The updated isometry of shape (n, p)
    """
    if use_geodesic:
        return retraction_geodesic(x=x, z=z, step_size=step_size)
    return retraction_polar_decomposition(x=x, z=z, step_size=step_size)

def retraction_polar_decomposition(x:Matrix, z:Matrix, step_size:float = 1)->Matrix:
    """
    Retraction based on canonical polar decomposition of scipy. Uses the SVD decomposition to obtain the isometry corresponding to z.
    
    Args:
        x: The base point of the retraction
        z: Tangent vector at x, corresponding to the update direction. 
        step_size: The step size of the retraction
    Returns:
        The retracted matrix

    References:
        [1] https://page.math.tu-berlin.de/~mehl/papers/hmt1.pdf
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html
    """
    return jax.scipy.linalg.polar(x - step_size * z)[0]

def retraction_geodesic(x: Matrix, z: Matrix, step_size: float = 1) -> Matrix:
    """Compute a new point following the geodesic for a single isometry
    
    Source: Eq. 27 of https://arxiv.org/pdf/2112.05176

    Args:
        x: Current isometry of dimension (n, p)
        z: Element of the tangent space at x corresponding to the update direction.
            For instance, the riemannian gradient of the cost function. Dimensions are (n, p)
        step_size: Geodesic curve parameter
    Returns:
        x_new: New position given by x_new = g(a) with g(a) being a geodesic with g(0) = x, [dg/dt](0) = z
    """
    
    n, p = x.shape
    dim = p
    
    Q, R = jnp.linalg.qr((jnp.eye(n) - x @ x.T.conj()) @ z)
    
    # Construct AR_mat directly using jnp.block
    AR_mat = jnp.block([
        [x.T.conj() @ z, -R.T.conj()],
        [R, jnp.zeros((dim, dim), dtype=jnp.complex128)]
    ])
    
    MN = eigy_expm_jax(-step_size * AR_mat) @ jnp.eye(2 * dim, dim)
    
    return x @ MN[:dim, :] + Q @ MN[dim:, :]
    
def eigy_expm_jax(A:jnp.ndarray):
    """Custom Matrix exponential using the eigendecomposition of jax.linalg

    Args:
        A: Matrix to be exponentiated

    Returns:
        Matrix exponential of A
    """
    eigvals, eigvects = jnp.linalg.eig(A)
    return jnp.einsum("...ik, ...k, ...kj -> ...ij", eigvects, jnp.exp(eigvals), jnp.linalg.inv(eigvects))

def is_isometry(x:jnp.ndarray)->bool:
    "check if `x` belongs to the stiefel manifold"
    return jnp.allclose(x.conj().T @ x, jnp.eye(x.shape[1]))

def is_in_tangent_space(x:jnp.ndarray, z:jnp.ndarray)->bool:
    """
    Checks if the matrix z is in the tangent space of isometry x 
    
    Checks tangent space condition x^H z + z^H x = 0
    Args:
        x: Stiefel matrix of dimensions (n, p)
        z: Any matrix of dimensions (n, p)
    """
    return jnp.allclose(x.conj().T @ z, - z.conj().T @ x)