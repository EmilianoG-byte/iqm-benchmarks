""""
SLQ method for approximating the spectral density of the Riemannian Hessian.
"""

import jax.numpy as jnp
from jax import random
import jax

jax.config.update("jax_enable_x64", True)

from typing import Callable

def complex_normalized_vector(key: jax.Array, dim: int) -> jnp.ndarray:
    """
    Complex Gaussian random unit vector.
    """
    key_r, key_i = random.split(key)
    v = (random.normal(key_r, (dim,))
         + 1j * random.normal(key_i, (dim,))) / jnp.sqrt(2.0)
    v = v / jnp.linalg.norm(v)
    return v

def lanczos_from_vector(
    hvp: Callable[[jnp.ndarray], jnp.ndarray],
    v0: jnp.ndarray,
    order_m: int,
    reorth: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    m-step Lanczos algorithm for a Hermitian operator accessed via HVP.

    Args:
        hvp: function that computes Hessian-vector product
        v0: initial vector of shape (dim,)
        order_m: number of Lanczos steps
        reorth: whether to use reorthogonalization

    Returns:
        alphas: shape (m,)
        betas:  shape (m-1,)
    """

    q = v0 # (dim, )
    q_prev = jnp.zeros_like(q) # (dim, )

    alphas = []
    betas = []

    if reorth:
        lanczos_vectors = [q]

    beta = 0.0

    for k in range(order_m):
      # print(f"Lanczos step {k+1}/{order_m}")
      w = hvp(q) # (dim, )
      alpha = jnp.real(jnp.vdot(q, w)) # (1,)
      alphas.append(alpha)

      r = w - alpha * q # (dim,)
      if k > 0:
          r = r - beta * q_prev # (dim,)

      if reorth:
          for qi in lanczos_vectors:
              r = r - jnp.vdot(qi, r) * qi # (dim,)

      beta = jnp.linalg.norm(r) # (1,)
      
      if beta <= 1e-6:
        raise ValueError("Beta < 1e-6 was found. This means the lanczos vectors are linearly dependent.")

      if k < order_m - 1:
          betas.append(beta)
          q_prev = q # (dim,)
          q = r / beta # (dim,)
          if reorth:
              lanczos_vectors.append(q)

    alphas = jnp.array(alphas) # (order_m,)
    betas = jnp.array(betas) # (order_m - 1,)
    if len(alphas) != len(betas) + 1 != order_m:
      raise ValueError(f"Wrong shape for alphas and/or betas. Expected: ({order_m},). Got: {alphas.shape} and {betas.shape}")
    
    return alphas, betas

def nodes_and_weights_from_lanczos(alphas:jnp.ndarray, betas:jnp.ndarray)->tuple[jnp.ndarray, jnp.ndarray]:
  """Compute the nodes li and the weights wi for the quadrature approximation using the alphas and betas returned by Lanczos.

  Args:
    alphas: 1D array containing the alphas (main diagonal) from the Lanczos algorithm. Must have shape lanczos_steps.
    betas: 1D array containing the betas (k=1 and k=-1 main diagonals) from the Lanczos algorithm. Must have shape lanczos_steps.
  Returns:
    A tuple containing the nodes and weights for the quadrature approximation.
  """
  k = alphas.shape[0]

  # Build tridiagonal T
  T = jnp.diag(alphas)
  if k > 1:
      T = T + jnp.diag(betas, 1) + jnp.diag(betas, -1)

  # Eigen-decomposition of T
  eigvals, eigvecs = jnp.linalg.eigh(T)
  # Quadrature weights
  weights = jnp.real(eigvecs[0, :] ** 2)
  
  # Do we need this?
  # weights = jnp.maximum(weights, 0.0)
  # weights = weights / jnp.sum(weights)
  
  # Sort them
  indices_sort = jnp.argsort(eigvals)
  nodes = eigvals[indices_sort]
  weights = weights[indices_sort]
  
  return nodes, weights

def slq_spectral_density(
    hvp: Callable[[jnp.ndarray], jnp.ndarray],
    dim: int,
    key: jax.Array,
    num_probes_k: int = 20,
    lanczos_order_m: int = 80,
    num_points_grid: int = 400,
    sigma: float = 1e-2,
    min_eigval: float | None = None,
    max_eigval: float | None = None,
    reorth: bool = True,
    normalize_spectral_density: bool = True,
    verbose: bool = True,
):
    """
    Stochastic Lanczos Quadrature for spectral density estimation.

    Args:
        hvp: function that computes Hessian-vector product. It is expected to receive a vector of dimension
          (dim,) and also output a vector of dimension (dim,). Namely: hvp: v -> w : C^dim -> C^dim.
        dim: This determines the dimension of the random vector to use as probe.
          In theory this is also the dimension of the Hessian (dim x dim), but since we are using wirtinger 
          formalis, we actually have a hessian of (2dim x 2dim) and the input vector for the hvp becomes (z, z*). 
        key: jax PRNG key
        num_probes_k: number of random probe vectors
        lanczos_order_m: number of Lanczos steps per probe
        num_points: number of points in the eigenvalue grid
        sigma: Gaussian smoothing parameter
        min_eigval: minimum eigenvalue for the grid (if None, determined from data)
        max_eigval: maximum eigenvalue for the grid (if None, determined from data)
        reorth: whether to use reorthogonalization in Lanczos steps
        normalize_spectral_density: Whether to normalize the spectral density to integrate over all grid to be 1.
          Namely:  \int rho(x) dx = 1

    Returns:
        grid: eigenvalue grid
        spectral_density: estimated density
    """
    nodes_all_probes, weights_all_probes = generate_nodes_and_weights_for_all_probes(hvp=hvp, dim=dim, key=key, num_probes_k=num_probes_k, lanczos_order_m=lanczos_order_m, reorth=reorth, verbose=verbose)
    
    return smoothened_density_from_nodes_and_weights(
      nodes_all_probes=nodes_all_probes,
      weights_all_probes=weights_all_probes,
      num_points_grid=num_points_grid,
      sigma=sigma,
      min_eigval=min_eigval,
      max_eigval=max_eigval,
      normalize=normalize_spectral_density,
    )
  
  
def slq_rank(
  hvp: Callable[[jnp.ndarray], jnp.ndarray],
  dim: int,
  key: jax.Array,
  eps: float,
  num_probes_k: int = 20,
  lanczos_order_m: int = 80,
  reorth: bool = True,
  verbose: bool = True,
  use_nullity: bool = True,
):
  
  nodes_all_probes, weights_all_probes = generate_nodes_and_weights_for_all_probes(hvp=hvp, dim=dim, key=key, num_probes_k=num_probes_k, lanczos_order_m=lanczos_order_m, reorth=reorth, verbose=verbose)
  
  if use_nullity:
    nullity_mean, nullity_std, mean_frac, std_frac = nullity_from_ritz(nodes_all_probes=nodes_all_probes, weights_all_probes=weights_all_probes, eps=eps, dim=dim)
    
    rank_mean = dim - nullity_mean
    rank_std = nullity_std
  
  else:
    rank_mean, rank_std, mean_frac, std_frac = rank_from_ritz(nodes_all_probes=nodes_all_probes, weights_all_probes=weights_all_probes, eps=eps, dim=dim)
    
    nullity_mean = dim - rank_mean
    nullity_std = rank_std
  
  return {
    "rank": (rank_mean, rank_std),
    "nullity": (nullity_mean, nullity_std)
  }
  
def nullity_from_ritz(nodes_all_probes:jnp.ndarray, weights_all_probes:jnp.ndarray, eps: float, dim: int):
  per_probe_fractions = []
  for nodes, weights in zip(nodes_all_probes, weights_all_probes):
    # Determine the nodes (eigvals) that are within eps of zero: |λi| <= ε
    mask = (nodes >= -eps) & (nodes <= eps)
    # sum the weights for these nodes: ∑ω
    per_probe_fractions.append(jnp.sum(weights[mask]))
  per_probe_fractions = jnp.stack(per_probe_fractions)
  # Compute the mean and standard deviation across all probes
  mean_frac = jnp.mean(per_probe_fractions)
  std_frac  = jnp.std(per_probe_fractions)
  nullity_mean = dim * mean_frac
  nullity_std  = dim * std_frac
  return nullity_mean, nullity_std, mean_frac, std_frac

def rank_from_ritz(nodes_all_probes:jnp.ndarray, weights_all_probes:jnp.ndarray, eps: float, dim: int):
  per_probe_fractions = []
  for nodes, weights in zip(nodes_all_probes, weights_all_probes):
    # Determine the nodes (eigvals) that are within eps of zero: |λi| <= ε
    mask = jnp.abs(nodes) >= eps
    # sum the weights for these nodes: ∑ω
    per_probe_fractions.append(jnp.sum(weights[mask]))
  per_probe_fractions = jnp.stack(per_probe_fractions)
  # Compute the mean and standard deviation across all probes
  mean_frac = jnp.mean(per_probe_fractions)
  std_frac  = jnp.std(per_probe_fractions)
  rank_mean = dim * mean_frac
  rank_std  = dim * std_frac
  return rank_mean, rank_std, mean_frac, std_frac

def generate_nodes_and_weights_for_all_probes(
  hvp: Callable[[jnp.ndarray], jnp.ndarray],
  dim: int,
  key: jax.Array,
  num_probes_k: int = 20,
  lanczos_order_m: int = 80,
  reorth: bool = True,
  verbose: bool = True
  ):
  
  nodes_all_samples = []
  weights_all_samples = []

  for i in range(num_probes_k):
    if verbose:
      print(f"SLQ probe {i+1}/{num_probes_k}")
    key, subkey = random.split(key)
    v0 = complex_normalized_vector(subkey, dim) # (dim,)

    alphas, betas = lanczos_from_vector(
        hvp, v0, lanczos_order_m, reorth=reorth
    )

    nodes, weights = nodes_and_weights_from_lanczos(alphas=alphas, betas=betas) # (lanczos_steps_m, ), (lanczos_steps_m, )
    
    nodes_all_samples.append(nodes)
    weights_all_samples.append(weights)
    
  return jnp.array(nodes_all_samples), jnp.array(weights_all_samples) # (num_probes_k, lanczos_steps_m)
  
def smoothened_density_from_nodes_and_weights(nodes_all_probes:jnp.ndarray, weights_all_probes:jnp.ndarray, num_points_grid:int, sigma: float = 1e-2, min_eigval: float | None = None, max_eigval: float | None = None, normalize:bool=True)->tuple[jnp.ndarray, jnp.ndarray, float]:
  """Generate the smoothened spectral density convoluted with a Gaussian function using the nodes and weights from the Gaussian Quadrature using Lanczos.

  Args:
      nodes_all_probes: (num_probes_k, lanczos_steps_m)
      weights_all_probes: (num_probes_k, lanczos_steps_m)
      num_points_grid: _description_
      sigma: _description_. Defaults to 1e-2.
      min_eigval: _description_. Defaults to None.
      max_eigval: _description_. Defaults to None.

  Returns:
      The grid used to evaluate the spectral density. These are the t's in the f(t) formula.
      The spectral density evaluated at every t in the grid.
      The actual sigma used in the Gaussian formula.
  """
  # Determine spectral window
  if min_eigval is None:
      min_eigval = jnp.min(nodes_all_probes) # do we need to take average min/max over probes?
  if max_eigval is None:
      max_eigval = jnp.max(nodes_all_probes)

  grid = jnp.linspace(min_eigval, max_eigval, num_points_grid) # (num_points_grid)
  spectral_density = jnp.zeros_like(grid) # (num_points_grid)

  # Rescaling. Suggested in https://github.com/google/spectral-density/blob/master/jax/density.py#L81
  if sigma is None:
    sigma = 10 ** -5 * max(1, (max_eigval - min_eigval))
  else:
    sigma = sigma**2 * max(1, (max_eigval - min_eigval))
  
  print(f"σ used: {sigma:.2e}")
  # Gaussian convolution
  norm_const = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)

  # Put all the nodes and weights in a single 1D array. Is this correct?
  nodes_all_probes = jnp.concatenate(nodes_all_probes) # (num_probes_k x lanczos_steps_m)
  weights_all_probes = jnp.concatenate(weights_all_probes) # (num_probes_k x lanczos_steps_m)
  
  for nodes_m, weights_m in zip(nodes_all_probes, weights_all_probes):
    spectral_density = spectral_density + weights_m * norm_const * jnp.exp(
        -0.5 * ((grid - nodes_m) / sigma) ** 2
    )

  # Is this needed based on how we are doing the computations?
  num_probes_k = nodes_all_probes.shape[0]
  # Divide by number of probes (since the values were just added before)
  spectral_density /= num_probes_k

  # Normalize to integrate up to 1 \int rho(x) dx = 1
  if normalize:
    dx = grid[1] - grid[0]
    spectral_density /= (dx * jnp.sum(spectral_density))
  return grid, spectral_density, sigma

def nullity_from_density(grid: jnp.ndarray, spectral_density: jnp.ndarray, eps: float, dim: int)->tuple[float, float]:
    dx = grid[1] - grid[0]
    mask = (grid >= -eps) & (grid <= eps)
    null_frac = jnp.sum(spectral_density[mask]) * dx   # fraction of eigenvalues inside [-eps,eps]
    nullity_est = dim * null_frac
    return nullity_est, null_frac

def rank_from_density(grid: jnp.ndarray, spectral_density: jnp.ndarray, eps: float, dim: int)->tuple[float, float]:
    nullity, _ = nullity_from_density(grid=grid, spectral_density=spectral_density, eps=eps, dim=dim)
    return dim - nullity, nullity

def compute_spectral_resolution(min_eigval:float, max_eigval:float, lanczos_num_steps:int)->float:
    return (max_eigval - min_eigval)/lanczos_num_steps

def gaussian_density_single_t_single_probe(t:float, sigma:float, nodes:jnp.ndarray, weights:jnp.ndarray)->float:
  """Compute the spectral density convoluted a Gaussian function for a single probe vector and single point t.

  Args:
      t: Point in the grid where to compute the spectral density / f(t)
      sigma: Square root of variance in Gaussian model
      nodes: All (lanczos_steps_m,) eigenvalues where to compute the spectral density / Gaussian convolution on.
      weights All (lanczos_steps_m,) weights where to compute the spectral density / Gaussian convolution on.

  Returns:
      Phi_k_t: the estimation for a single probe vector of the spectral density convoluted with Gaussian around a point t.
  """
  norm_const = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
  f_t = norm_const * jnp.exp(-(t - nodes) ** 2 / (2 * sigma**2)) # (lanczos_steps_m, )
  return jnp.sum(weights * f_t) # phi_k_t

def exact_gaussian_spectral_density(
    eigenvals: jnp.ndarray,
    grid_xs: jnp.ndarray,
    sigma: float
):
    """
    Exact Gaussian-smoothed spectral density from full eigenvalues.
    
    Args:
      eigenvals: The exact eigenvalues from the Hermitian operator
      grid_xs: The values of t where to evaluate the spectral density.
    """
    dim = eigenvals.shape[0]
    norm = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)

    diffs = grid_xs[None, :] - eigenvals[:, None]   # (dim, num_points_grid)
    rho = norm * jnp.exp(-0.5 * (diffs / sigma) ** 2) # (dim, num_points_grid)
    rho = jnp.sum(rho, axis=0) / dim # (num_points_grid)

    return rho