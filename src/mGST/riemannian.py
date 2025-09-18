"""
Riemannian functions for the Stiefel Manifold, its tangent space and related operations.
"""

import jax.numpy as jnp
import jax
from typing import Sequence

from mGST.typing import Tensor, Matrix, Scalar

def get_isometry_dimensions_from_tensor(tensor:Tensor, tensor_type:str)->tuple[int, int]:
    """
    Get the Stiefel dimensions n and p from the given tensor.
    
    Args:
        tensor: The tensor from where dimensions will be inferred.
        tensor_type: The type of the tensor. Can be one of the following: 'state', 'povm', 'kraus'.
            Shape according to the type should be:
            * POVM: (num_povm, povm_rank, dim)
            * State: (dim, rank_state)
            * (single) Kraus: (kraus_rank, dim, dim)
    Returns:
        n: The Stiefel n dimension
        p: The Stiefel p dimension
        
    Raises:
        ValueError: If the tensor_type is not recognized.
    """
    if tensor_type == "state":
        dim, rank_state = tensor.shape # dim, rank_state
        n = dim * rank_state
        p = 1

    elif tensor_type == "povm":
        num_povm, rank_povm, dim = tensor.shape # num_povm, rank_povm, dim
        n = num_povm * rank_povm
        p = dim
    elif tensor_type == "kraus":
        rank_kraus, dim, _ = tensor.shape # kraus_rank, dim_out, dim_in
        n = rank_kraus * dim
        p = dim
    else:
        raise ValueError(f"Tensor name '{tensor_type}' is not recognized. Please use one of the following: 'state', 'povm' or 'kraus'.")
    return n, p

def riemannian_connection(x:Matrix, w_x:Matrix, z:Matrix, Dw_x_to_z:Matrix, metric:str = "euclidean", alphas:tuple|None = None)-> Matrix:
    """
    Riemannian connection ∇w(x)[z] parametrized for different metrics on the stiefel manifold. 

    From equation 5.4 of https://arxiv.org/abs/2009.10159
    
    Args:
        x: Base point isometry of the tangent space
        w_x: Riemannian vector field evaluated at x.
        z: Riemannian tangent vector equivalent to the "direction" of the derivative
        Dw_x_to_z: Euclidean directional derivative of the vector field w in the direction of z evaluated at x
        metric: The type of the Riemannian metric to use ('euclidean' or 'canonical'). Defaults to 'euclidean'.
        alphas: Optional tuple of alpha0 and alpha1 parameters to define a custom metric. If provided, overrides the metric parameter.
    Returns:
        The riemannian connection of the vector field w_x in the direction of z at x.
    """
    if metric == "euclidean":
        alpha0, alpha1 = 1, 1
    elif metric == "canonical":
        alpha0, alpha1 = 1, 0.5
    else:
        raise ValueError("Metric must be either 'euclidean' or 'canonical'")
    
    if alphas is not None:
        alpha0, alpha1 = alphas
    
    In = jnp.eye(x.shape[0])
    return Dw_x_to_z + 0.5 * x @ (z.conj().T @ w_x + w_x.conj().T @ z) + ((alpha0-alpha1)/alpha0)*(In - x @ x.conj().T) @ (z @ w_x.conj().T + w_x @ z.conj().T) @ x

def riemannian_metric(z1:Matrix, z2:Matrix, x:Matrix = None, metric:str = "euclidean")-> Scalar:
    """
    Compute the riemannian metric on the stiefel manifold at the point x for two tangent vectors z1 and z2.
    
    Args:
        z1: First tangent vector of dimensions (n, p)
        z2: Second tangent vector of dimensions (n, p)
        x: Point on the stiefel manifold of dimensions (n, p)
        metric: The type of the metric to use ('euclidean' or 'canonical'). Defaults to 'euclidean'.
    
    Returns:
        The inner product of the two tangent vectors at the point x
    """
    n, p = z1.shape
    if metric == "euclidean":
        gamma = jnp.eye(n)
    elif metric == "canonical":
        if x is None:
            raise ValueError("To use the canonical metric, the point x on the stiefel manifold must be provided.")
        gamma = jnp.eye(n) - 0.5 * (x@x.conj().T)
    else:
        raise ValueError(f"Metric: {metric} is not recognized. Please use one of the following: 'euclidean', 'canonical'")
    return jnp.trace(z1.conj().T @ gamma @ z2).real

def update_isometry_tensors(isometries:Sequence[Matrix] | Matrix, update_directions:Sequence[Matrix] | Matrix, step_size:float, operator_type:str = "kraus", use_geodesic:bool =  True) ->  jnp.ndarray:
    """Update a tensor of isometries in the direction of the tensor of tangent vectors scaled by step size
    
    In order to retract back to the stiefel manifold we either follow the geodesic or use a first order retraction.
    
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
