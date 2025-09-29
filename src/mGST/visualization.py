"""
Utility functions for visualization purposes.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp

COLOUR_PALETTE = [
    "#294C60",
    "#2EC4B6",
    "#E71D36",
    "#88D498",
    "#FF8C42",
    "#B9C7DF",
    "#4A4E69",
    "#F4D35E",
    "#8EC0E4",
]
MARKERS = [
    "o",
    "s",
    "D",
    "^",
    "v",
    "<",
    ">",
    "p",
    "*",
    "h",
    "x",
    "+",
]  # Extend as needed

def plot_cost_function(
    cost_array: list[float] | list[list[float]],
    title: str = "Cost Function Convergence",
    use_semilogy: bool = True,
    labels: list[str] = None,
    comparison_yline: int | None = None,
    comparison_label: str | None = None,
):
    """Plot the cost function values over training iterations.

    Args:
        cost_array (list[float]): List of cost function values at each iteration.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6), dpi=250)
    plot_function = plt.semilogy if use_semilogy else plt.plot

    if not isinstance(cost_array[0], list):
        labels = [labels] if labels else ["Cost"]
        cost_array = [cost_array]
        
    if not len(labels) == len(cost_array):
        raise ValueError("Length of labels must match number of cost arrays.")

    for i, cost in enumerate(cost_array):
        marker = MARKERS[
            i % len(MARKERS)
        ]  # Cycle through markers if more curves than markers
        color = COLOUR_PALETTE[i]
        plot_function(
            cost,
            linewidth=2,
            color=color,
            label=labels[i] if labels else f"Run {i+1}",
            marker=marker,
        )

    if comparison_yline is not None:
        label = comparison_label if comparison_label else "Comparison Point"
        plt.axhline(y=comparison_yline, color="red", linestyle="--", label=label)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost Value", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
def plot_bars_with_error(
    data: dict[str, tuple[float, float]],
    title: str = None,
    ylabel: str = None,
    xlabel: str = None,
    use_log_scale: bool = False,
    bar_spacing: float = 1.0, 
) -> None:
    """
    Plot bar chart of expectation values with error bars.

    Args:
        data: Map of x labels to (mean, std) tuples.
        title: Title of the plot.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        use_log_scale: Whether to use a logarithmic scale for the y-axis.
        bar_spacing: Relative spacing between bars.
    """
    x_labels = list(data.keys())
    x_positions = jnp.arange(len(x_labels)) * bar_spacing

    means = [data[x][0] for x in x_labels]
    stds = [data[x][1] for x in x_labels]
    colours = COLOUR_PALETTE[: len(x_labels)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=250)

    ax.bar(x_positions, means, yerr=stds, capsize=4, color=colours)

    if use_log_scale:
        ax.set_yscale('log')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
        
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    plt.show()
    
def average_and_error_of_sum(*args):
    """Compute the average and error of the sum of independent measurements.

    Args:
        *args: Tuples of (mean, std) for each measurement.

    Returns:
        Tuple of (total_mean, total_std).
    """
    total_mean = sum(mean for mean, _ in args)
    total_variance = sum(std**2 for _, std in args)
    total_std = jnp.sqrt(total_variance)
    return total_mean, total_std

def average_and_error_times_scalar(data: tuple[float, float], scalar: float) -> tuple[float, float]:
    """Scale a measurement with its error by a scalar.

    Args:
        data: Tuple of (mean, std).
        scalar: Scalar to multiply the mean and std.

    Returns:
        Tuple of (scaled_mean, scaled_std).
    """
    mean, std = data
    scaled_mean = mean * scalar
    scaled_std = std * abs(scalar)
    return scaled_mean, scaled_std