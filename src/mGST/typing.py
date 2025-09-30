"""
This module provides type aliases for commonly used tensor shapes and structures in the mGST library.
"""
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Literal

Tensor = jnp.ndarray
Matrix = jnp.ndarray
Scalar = jnp.ndarray

from dataclasses import dataclass, asdict
from typing import Optional, Any, Literal

@dataclass
class BaseOptimizationOptions:
    """Base class for optimization options."""
    num_iterations: int = 20
    metric: str = "euclidean"
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert options object to dictionary"""
        return asdict(self)

@dataclass
class TrustRegionOptions(BaseOptimizationOptions):
    """Options for trust region optimization method."""
    radius_init: float = 0.1
    max_radius: float = 2.0
    quotient_trust: float = 0.125
    tol_grad: float = 1e-6
    num_iterations_cg: int = 10
    theta_cg: float | None = None
    kappa_cg: float | None = None
    verbose_cg: bool = True

@dataclass
class GradientDescentOptions(BaseOptimizationOptions):
    """Options for gradient descent optimization method."""
    init_step: float = 1.0
    optimize_step: bool = True
    use_geodesic: bool = True
    ls_max_iter: int = 200
    ls_method: str = "COBYLA"

# Type alias for optimization options
OptimizationOptions = TrustRegionOptions | GradientDescentOptions

@dataclass
class OptimizationConfig:
    """Configuration for optimization of an operator."""
    method: Literal["trust_region", "gds"]
    options: OptimizationOptions