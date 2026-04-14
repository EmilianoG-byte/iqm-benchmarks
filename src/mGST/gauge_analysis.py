"""Module containing the tools to study then Gauge present in our model"""

import jax.numpy as jnp
from mGST.typing import Matrix
from mGST.linear_algebra import generate_anti_hermitian_basis, is_anti_hermitian, is_square_matrix, random_anti_hermitian_matrix

def velocity_at_x(x:Matrix, A: Matrix)->Matrix:
    "Compute the tangent velocity vector at x"
    # check that A's dimensions are commpatible with x
    if not is_square_matrix(A):
        raise ValueError("A must be a square matrix")
    n, p = x.shape
    dim_A = A.shape[0]
    if dim_A > n:
        raise ValueError(f"A must have dimensions compatible with x. Got {A.shape}, expected at most ({n}, {n})")
    # check this is an isometry
    if not is_anti_hermitian(A):
        raise ValueError("A must be anti-hermitian")
    
    dim_id = int(n / dim_A)
    identity = jnp.eye(dim_id)
    # here we are assuming that the isometry n dimension is ordered as (kraus_rank, dim_id).
    return jnp.kron(A, identity) @ x 


def random_vertical_velocity(x:Matrix, seed:int, dim_A: int)->Matrix:
    "Generate a random vertical velocity vector at x"
    A = random_anti_hermitian_matrix(dim_A, seed)
    return velocity_at_x(x=x, A=A)


def generate_vertical_fundamental_vectors(x: Matrix, dim_A: int)->Matrix:
    "Generate the vertical fundamental vectors at x"
    basis = generate_anti_hermitian_basis(dim_A)
    return jnp.array([velocity_at_x(x=x, A=A) for A in basis])