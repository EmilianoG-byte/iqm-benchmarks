"""Module for linear algebra operations on tensors"""

from jax import config
import jax
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from mGST.typing import Matrix, Tensor


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

def is_square_matrix(A:Matrix)->bool:
    """
    Check if a matrix is square.

    Args:
        A: Matrix to be checked. Shape (..., n, m)
    Returns:
        True if A is square, False otherwise
    """
    return A.shape[-2] == A.shape[-1]


def generate_anti_hermitian_basis(dim:int)->Tensor:
    """
    Generate a *Real* basis for the vector space of anti-Hermitian matrices of size dim x dim.

    The anti-Hermitian *Real* vector space is of dimension dim^2.
     
    Using the elementary directions `E_ij` defined as the matrix with a 1 in the (i, j) position and 0 elsewhere, we construct the following basis elements:
    
    1. For (i, j) with i < j, we include the matrices `E_ij - E_ji` and `i(E_ij + E_ji)`
        * These correspond to the off-diagonal anti-Hermitian matrices.
        * `2 * dim * (dim - 1)` basis elements.
    2. For i = j, we include the matrices `i * E_ii`.
        * These correspond to the diagonal anti-Hermitian matrices
        * `dim` basis elements.

    Args:
        dim: Size of the matrices in the basis
    Returns:
        Basis of anti-Hermitian matrices. Shape (dim^2, dim, dim)
    """
    basis = []
    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                # Diagonal elements are purely imaginary
                mat = jnp.zeros((dim, dim), dtype=jnp.complex128)
                mat = mat.at[i, i].set(1j)
                basis.append(mat)
            else:
                # 1) E_ij - E_ji
                mat = jnp.zeros((dim, dim), dtype=jnp.complex128)
                mat = mat.at[i, j].set(1)
                mat = mat.at[j, i].set(-1)
                basis.append(mat)
                
                # 2) i(E_ij + E_ji)
                mat = jnp.zeros((dim, dim), dtype=jnp.complex128)
                mat = mat.at[i, j].set(1j)
                mat = mat.at[j, i].set(1j)
                basis.append(mat)
    if len(basis) != dim**2:
        raise ValueError(f"Ups🫣! Expected {dim**2} basis elements, got {len(basis)}")
    return jnp.array(basis)


def compute_manual_rank(matrix:Matrix, tol:float=1e-8)->tuple[int, Tensor]:
    """
    Compute the rank of a matrix using its singular values and a specified tolerance.

    Args:
        matrix: Input matrix. Shape (m, n)
        tol: Tolerance for numerical precision
    Returns:
        rank: Rank of the matrix
        singular_values: Singular values of the matrix
    """
    S = jnp.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    if tol is None:
        tol = max(matrix.shape) * jnp.finfo(S.dtype).eps * S[0]
        print(f"Using default tolerance: {tol}")
    rank = jnp.sum(S > tol)
    return rank, S