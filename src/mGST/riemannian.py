"""
Riemannian functions for the Stiefel Manifold, its tangent space and related operations.
"""

import jax.numpy as jnp

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

def riemannian_connection(x:Matrix, w_x:Matrix, z:Matrix, Dw_in_z_at_x:Matrix, metric:str = "euclidean", alphas:tuple|None = None)-> Matrix:
    """
    Riemannian connection ∇w(x)[z] parametrized for different metrics on the stiefel manifold. 

    From equation 5.4 of https://arxiv.org/abs/2009.10159
    
    Args:
        x: Base point isometry of the tangent space
        w_x: Riemannian vector field evaluated at x.
        z: Riemannian tangent vector equivalent to the "direction" of the derivative
        Dw_in_z_at_x: Euclidean directional derivative of the vector field w in the direction of z evaluated at x
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
    return Dw_in_z_at_x + 0.5 * x @ (z.conj().T @ w_x + w_x.conj().T @ z) + ((alpha0-alpha1)/alpha0)*(In - x @ x.conj().T) @ (z @ w_x.conj().T + w_x @ z.conj().T) @ x

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
