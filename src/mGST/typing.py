"""
This module provides type aliases for commonly used tensor shapes and structures in the mGST library.
"""
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Literal, Any

Tensor = jnp.ndarray
Scalar = jnp.ndarray
Vector = jnp.ndarray
Matrix = jnp.ndarray

OptimizationResult = dict[str, Any]

from dataclasses import dataclass, asdict
from typing import Optional, Any, Literal

@dataclass
class BaseOptimizationOptions:
    """Base class for optimization options."""
    metric: str = "euclidean"
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert options object to dictionary"""
        return asdict(self)

@dataclass
class TrustRegionOptions(BaseOptimizationOptions):
    """Options for trust region optimization method."""
    num_iterations: int = 20
    radius_init: float = 0.1
    max_radius: float = 2.0
    quotient_trust: float = 0.125
    tol_grad: float = 1e-6
    num_iterations_cg: int = 10
    theta_cg: float | None = 1 # before by mistake we set it to 0.5 (WRONG). Based on book suggestions
    kappa_cg: float | None = 1/10 # decided to set it as default based on experimentation and book suggestions.
    verbose_cg: bool = False

@dataclass
class GradientDescentOptions(BaseOptimizationOptions):
    """Options for gradient descent optimization method."""
    initial_step: float = 1.0
    optimize_step: bool = True
    use_geodesic: bool = True
    ls_max_iter: int = 200
    ls_method: str = "COBYLA"
    
# Type alias for optimization options
OptimizationOptions = TrustRegionOptions | GradientDescentOptions

@dataclass
class OptimizationScheduleItem:
    """Represents a single item in an optimization schedule."""
    start: int  # Starting iteration (inclusive)
    end: int    # Ending iteration (inclusive)
    options: OptimizationOptions
    
    def __str__(self) -> str:
        return f"[{self.start}-{self.end}]: {type(self.options).__name__}"
    
@dataclass
class OperatorSchedule:
    """Schedule for a single operator."""
    items: list[OptimizationScheduleItem]

    def __str__(self) -> str:
        items_str = "\n  ".join(str(item) for item in sorted(self.items, key=lambda x: x.start))
        return f"Schedule:\n  {items_str}"

    @classmethod
    def default(cls, num_iterations: int, options: OptimizationOptions | None = None) -> "OperatorSchedule":
        """Create a default schedule that uses the same options for all iterations.
        
        Args:
            num_iterations: Total number of iterations.
            options: Optimization options to use. If None, uses default TrustRegionOptions.
        
        Returns:
            An OperatorSchedule with a single item covering all iterations.
        """
        default_options = options if options is not None else TrustRegionOptions()
        return cls([
            OptimizationScheduleItem(0, num_iterations-1, default_options)
        ])

    def get_options_for_iteration(self, iteration: int) -> OptimizationOptions:
        """Get the optimization options for a given iteration."""
        for item in self.items:
            if item.start <= iteration <= item.end:
                return item.options
        raise ValueError(f"No optimization options defined for iteration {iteration}")

    def validate(self, num_iterations: int):
        """Validate the schedule covers all iterations and has no gaps/overlaps."""
        if not self.items:
            raise ValueError("Schedule cannot be empty")
        
        # Sort items by start time
        sorted_items = sorted(self.items, key=lambda x: x.start)
        
        # Check first item starts at 0
        if sorted_items[0].start != 0:
            raise ValueError("Schedule must start at iteration 0")
            
        # Check last item ends at num_iterations-1
        if sorted_items[-1].end != num_iterations-1:
            raise ValueError(f"Schedule must end at iteration {num_iterations-1}. Instead got {sorted_items[-1].end}")
            
        # Check for gaps and overlaps
        for i in range(len(sorted_items)-1):
            if sorted_items[i].end + 1 != sorted_items[i+1].start:
                raise ValueError(f"Gap or overlap detected between iterations {sorted_items[i].end} and {sorted_items[i+1].start}")
            
    def shorten_to(self, num_iterations: int) -> "OperatorSchedule":
        """Return a shortened schedule covering iterations [0, num_iterations-1].

        This method only allows shortening. Trying to extend beyond current coverage
        raises ValueError.
        """
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

        sorted_items = sorted(self.items, key=lambda x: x.start)
        if not sorted_items:
            raise ValueError("Schedule cannot be empty")

        # Ensure the current schedule is internally consistent first.
        current_num_iterations = sorted_items[-1].end + 1
        self.validate(current_num_iterations)

        new_last_iteration = num_iterations - 1
        current_last_iteration = sorted_items[-1].end

        if new_last_iteration > current_last_iteration:
            raise ValueError(
                f"Cannot extend schedule from last iteration {current_last_iteration} "
                f"to {new_last_iteration}. Only shortening is allowed."
            )

        if new_last_iteration == current_last_iteration:
            return OperatorSchedule(
                [OptimizationScheduleItem(it.start, it.end, it.options) for it in sorted_items]
            )

        new_items: list[OptimizationScheduleItem] = []
        for it in sorted_items:
            if it.start > new_last_iteration:
                break
            clipped_end = min(it.end, new_last_iteration)
            new_items.append(OptimizationScheduleItem(it.start, clipped_end, it.options))

        shortened = OperatorSchedule(new_items)
        shortened.validate(num_iterations)
        return shortened