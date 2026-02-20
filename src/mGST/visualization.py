"""
Utility functions for visualization purposes.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

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


def default_cost_function_formatting(title:str):
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost Value", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

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
        cost_array: List of cost function values at each iteration.
        title: Title of the plot.
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

    default_cost_function_formatting(title=title)
    
    
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

def plot_alternating_optimization(
    cost_array: list[float],
    title: str = "Alternating Optimization Convergence",
    use_semilogy: bool = True,
    data_operator_map: list[tuple[str, int]] = None,
    operator_colors: dict[str, str] = None,
):
    """
    Plot cost function values with different colors/markers for each operator optimization.

    Args:
        cost_array: List of cost function values at each iteration.
        title: Title of the plot.
        use_semilogy: Whether to use semilogy scale.
        data_operator_map: List of (operator, n_i) tuples, where n_i is the number of consecutive cost values for that operator (excluding the first "start" value).
        operator_colors: Dictionary mapping operators to colors. If None, uses default colors.
    """
    default_colors = {
        "start": COLOUR_PALETTE[-1],
        "povm": COLOUR_PALETTE[0],
        "kraus": COLOUR_PALETTE[1],
        "state": COLOUR_PALETTE[2]
    }
    colors = operator_colors if operator_colors else default_colors

    default_markers = {
        "start": "X",
        "povm": MARKERS[0],
        "kraus": MARKERS[1],
        "state": MARKERS[2]
    }

    plt.figure(figsize=(10, 6), dpi=250)
    plot_function = plt.semilogy if use_semilogy else plt.plot

    # Always plot the first point as "start"
    plot_function(
        [0],
        [cost_array[0]],
        color=colors["start"],
        marker=default_markers["start"],
        linestyle='',
        label="Start",
        markersize=13,
        markeredgewidth=0.25,
    )
    shown_labels = {"start"}

    # Now plot the rest according to data_operator_map
    if data_operator_map is None:
        # fallback to round-robin if not provided
        operator_order = ["povm", "kraus", "state"]
        data_operator_map = [(operator_order[i % len(operator_order)], 1) for i in range(len(cost_array) - 1)]

    # Verify the length of the data_operator_map matches cost_array length minus 1
    total_length = sum(n for _, n in data_operator_map)
    if total_length != len(cost_array) - 1:
        raise ValueError("Sum of n_i in data_operator_map must equal len(cost_array) - 1.")

    idx = 1
    for i, (operator, n_i) in enumerate(data_operator_map):
        indices = list(range(idx, idx + n_i))
        values = cost_array[idx:idx + n_i]
        label = f"{operator.capitalize()}" if operator not in shown_labels else None
        plot_function(
            indices,
            values,
            color=colors.get(operator, COLOUR_PALETTE[i % len(COLOUR_PALETTE)]),
            marker=default_markers.get(operator, MARKERS[i % len(MARKERS)]),
            linestyle='',  # Only markers
            label=label,
            markersize=8,
        )
        shown_labels.add(operator)
        idx += n_i

    # Plot connecting line for all points
    plot_function(
        range(len(cost_array)),
        cost_array,
        color=COLOUR_PALETTE[0],
        alpha=0.5,
        linewidth=2,
        zorder=0,  # Put line behind markers
    )

    default_cost_function_formatting(title=title)
        
    
from matplotlib.colors import to_rgba, to_hex
    
def idx_and_color_to_face_color(idx: int, n_sets: int, color: str) -> str:
    """Return a marker face color for the idx-th dataset out of n_sets.

    The last set (idx == n_sets - 1) gets the full colour.
    Earlier sets are progressively blended toward white (more pastel).

    Args:
        idx: Index of the current dataset (0-based).
        n_sets: Total number of datasets.
        color: Hex or named colour string for this method.
    """
    rgba = np.array(to_rgba(color), dtype=float)
    # How far are we from the last set? 0 for the last, up to 1 for the first.
    distance_from_last = (n_sets - 1 - idx) / max(n_sets - 1, 1)
    # Scale up to a maximum pastel factor (0.75 keeps the colour recognisable)
    factor = distance_from_last * 0.6
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * factor
    return to_hex(np.clip(rgba, 0.0, 1.0))        
        
def plot_many_sets_wall_time(
    time_data_sets: list[dict[str, list[tuple[float, float]]]],
    x_values: list[int | float],
    title: str = "Wall Time Comparison",
    xlabel: str = "Number of qubits",
    ylabel: str = "Wall time [s]",
    logscale: str | None = None,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
) -> None:
    """Plot wall time for multiple sets of timing data on a single figure.

    Methods share the same color across datasets (determined by their order in COLOUR_PALETTE).
    Datasets are visually separated by varying opacity: earlier datasets are more transparent,
    the last dataset is fully opaque.

    Args:
        time_data_sets: List of dicts, each mapping method names to lists of (mean, std) tuples.
            All dicts are assumed to share the same keys and each list should have the same
            length as x_values.
        x_values: Values for the x-axis (e.g. number of qubits).
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        logscale: 'x', 'y', or 'xy' for logarithmic scale on the respective axes, or None for linear.
        label_fontsize: Font size for axis labels and title.
        legend_fontsize: Font size for the legend.
        legend_loc: Location string for the legend (passed directly to matplotlib).
    """
    # Lightest alpha for the first set, fully opaque for the last
    # alphas = [1.0] if n_sets == 1 else [0.35 + 0.65 * j / (n_sets - 1) for j in range(n_sets)]

    plt.figure(figsize=(9, 5), dpi=250)
    if logscale == "y":
        plot_function = plt.semilogy
    elif logscale == "x":
        plot_function = plt.semilogx
    elif logscale == "xy":
        plot_function = plt.loglog
    else:
        plot_function = plt.plot

    n_sets = len(time_data_sets)
    
    alphas = [1.0] if n_sets == 1 else [0.35 + 0.65 * j / (n_sets - 1) for j in range(n_sets)]
    
    for j, time_data in enumerate(time_data_sets):
        for i, (method, results) in enumerate(time_data.items()):
            color = COLOUR_PALETTE[i % len(COLOUR_PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            means = jnp.array([r[0] for r in results])
            stds = jnp.array([r[1] for r in results])

            facecolor = idx_and_color_to_face_color(j, n_sets, color)

            plot_function(
                x_values,
                means,
                label=f"{method}",
                color=facecolor,
                marker=marker,
                linestyle="-",
                linewidth=2,
                markersize=6,
                # alpha=alphas[j],
                # markerfacecolor=facecolor,
                markeredgecolor="black",
            )
            plt.fill_between(
                x_values,
                means - stds,
                means + stds,
                color=color,
                alpha= 0.2,
            )

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=label_fontsize)
    plt.grid(alpha=0.3)
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=legend_fontsize
    )
    plt.tight_layout()
    plt.show()
    
        
def plot_wall_time(
    time_data: dict[str, list[tuple[float, float]]],
    x_values: list[int | float],
    title: str = "Wall Time Comparison",
    xlabel: str = "Number of qubits",
    ylabel: str = "Wall time [s]",
    comparison_data: dict[str, tuple[list, list]] = None,
    logscale: str | None = None,
    different_lengths_allowed: bool = False,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    legend_loc: str = "best",
) -> None:
    """Plot wall time against a parameter for different methods/configurations.

    Args:
        time_data: Dictionary mapping method names to lists of (mean, std) tuples.
            Each list should have the same length as x_values.
        x_values: Values for x-axis (e.g. number of qubits).
        title: Title of the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        logscale: 'x', 'y', 'xy' for logarithmic scale on respective axes, or None for linear scale.
        comparison_data: Optional dictionary mapping method names to (x, y) data for additional comparison curves.
        different_lengths_allowed: If True, allows time_data entries to have different lengths than x_values (plots only available points). If False, raises an error if lengths don't match.
        label_fontsize: Font size for axis labels and title.
        legend_fontsize: Font size for legend.
    """
    plt.figure(figsize=(8, 5), dpi=250)
    if logscale == "y":
        plot_function = plt.semilogy
    elif logscale == "x":
        plot_function = plt.semilogx
    elif logscale == "xy":
        plot_function = plt.loglog
    else:
        plot_function = plt.plot

    for i, (method, results) in enumerate(time_data.items()):
        if len(results) != len(x_values):
            if not different_lengths_allowed:
                raise ValueError(f"Number of timing results for '{method}' doesn't match number of x values")
            else:
                # Truncate or pad results to match x_values length
                x_values_method = x_values[:len(results)]
        else:
            x_values_method = x_values
        
        marker = MARKERS[i % len(MARKERS)]
        color = COLOUR_PALETTE[i]
        # Extract mean times and std devs
        means = jnp.array([result[0] for result in results])
        stds = jnp.array([result[1] for result in results])

        # Plot main line with markers
        plot_function(
            x_values_method,
            means,
            label=method,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=6,
            markeredgecolor="black"
        )
        
        plt.fill_between(
            x_values_method,
            means - stds,
            means + stds,
            color=color,
            alpha=0.2
        )
        
    if comparison_data is not None:
        for label, (comp_x, comp_y) in comparison_data.items():
            plot_function(
                comp_x,
                comp_y,
                linestyle="--",
                label=label,
                color="#E71D36",
                linewidth=2,
            )

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=label_fontsize)
    plt.grid(alpha=0.3)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.tight_layout()
    plt.show()