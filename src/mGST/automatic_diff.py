"""
This module contains functions for automatic differentiation using JAX.
"""
import jax
import jax.numpy as jnp
from typing import Callable

def vhp(function:callable, x:jnp.ndarray, z:jnp.ndarray)-> tuple[jnp.ndarray, jnp.ndarray]:
    function_at_x, vjp_function = jax.vjp(function, x)
    (vjp_vector, ) = vjp_function(z)
    return function_at_x, vjp_vector

def hvp(function:callable, x:jnp.ndarray, z:jnp.ndarray)-> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Hessian-vector product using JAX's jvp function.
    
    NOTE: primals and tangents must be of the same shape.
    
    Args:
        function: The function for which to compute the Hessian-vector product.
        x: The point at which to evaluate the function and its gradient.
        z: The vector with which to compute the Hessian-vector product.
    Returns:
        A tuple containing the function value at x and the Hessian-vector product.
    """
    return jax.jvp(function, (x,), (z,))

def automatic_gradient(function:Callable)->Callable:
    return jax.grad(function)