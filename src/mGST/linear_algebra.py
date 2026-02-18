"""Module for linear algebra operations on tensors"""

from jax import config
import jax
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from mGST.typing import Matrix


def transpose(A:Matrix)->Matrix:
    """
    Transpose a matrix, swapping its last two dimensions.

    Handles batch dimensions.

    Args:
        A: Matrix to be transposed. Shape (..., n, p)
        
    Returns:
        Transposed matrix
    """
    return A.swapaxes(-1, -2)

def symmetrize(A:Matrix)->Matrix:
    """
    Symmetrize a matrix by projecting it onto the symmetric subspace.
    
    Handles batch dimensions.
    
    Args:
        A: square matrix to be symmetrized. Shape (..., n, n)
    Returns:
        Symmetrized matrix. Shape (..., n, n)
    """
    return 0.5 * (A + transpose(A).conj())

def skew_symmetrize(A:Matrix)->Matrix:
    """
    Skew-symmetrize a matrix by projecting it onto the skew-symmetric subspace.
    
    Handles batch dimensions.
    
    Args:
        A: square matrix to be skew-symmetrized. Shape (..., n, n)
    Returns:
        Skew-symmetrized matrix. Shape (..., n, n)
    """
    return 0.5 * (A - transpose(A).conj())

def random_hermitian_matrix(n:int, seed:int)-> Matrix:
    """
    Generate a random Hermitian matrix of size n x n.

    Args:
        n: Size of the matrix
        seed: Random seed for reproducibility

    Returns:
        Random Hermitian matrix of shape (n, n)
    """
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    return symmetrize(A)

def random_anti_hermitian_matrix(n:int, seed:int)-> Matrix:
    """
    Generate a random anti-Hermitian matrix of size n x n.

    Args:
        n: Size of the matrix
        seed: Random seed for reproducibility
    Returns:
        Random anti-Hermitian matrix of shape (n, n)
    """
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    return skew_symmetrize(A)

def is_hermitian(A:Matrix, tol:float=1e-8)->bool:
    """
    Check if a matrix is symmetric.

    Args:
        A: Matrix to be checked. Shape (..., n, n)
        tol: Tolerance for numerical precision
    Returns:
        True if A is symmetric, False otherwise
    """
    return jnp.allclose(A, transpose(A).conj(), atol=tol)

def is_anti_hermitian(A:Matrix, tol:float=1e-8)->bool:
    """
    Check if a matrix is anti-symmetric.

    Args:
        A: Matrix to be checked. Shape (..., n, n)
        tol: Tolerance for numerical precision
    Returns:
        True if A is anti-symmetric, False otherwise
    """
    return jnp.allclose(A, -transpose(A).conj(), atol=tol)