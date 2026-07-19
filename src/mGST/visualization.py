"""
Utility functions for visualization purposes.
"""

from collections.abc import Sequence

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
    "+",
    "x",
]  # Extend as needed


def default_cost_function_formatting(title:str, ylabel:str = "Cost Value", xlabel:str = "Iteration"):
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, wrap=True)
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
    ylabel: str = "Cost Value",
    xlabel: str = "Iteration",
):
    """Plot the cost function values over training iterations.

    Args:
        cost_array: List of cost function values at each iteration.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 6), dpi=250)
    plot_function = plt.semilogy if use_semilogy else plt.plot

    if not isinstance(cost_array[0], (Sequence, jnp.ndarray, np.ndarray)):
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

    default_cost_function_formatting(title=title, ylabel=ylabel, xlabel=xlabel)
    
    
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
    comparison_data: dict[str, tuple[list, list]] = None,
    logscale: str | None = None,
    different_lengths_allowed: bool = False,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    legend_loc: str = "best",
) -> plt.Figure:
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
        comparison_data: Optional dictionary mapping method names to (x, y) data for additional comparison curves.
        logscale: 'x', 'y', or 'xy' for logarithmic scale on the respective axes, or None for linear.
        different_lengths_allowed: If True, allows time_data_sets entries to have different lengths than x_values.
        label_fontsize: Font size for axis labels and title.
        legend_fontsize: Font size for the legend.
        legend_loc: Location string for the legend (passed directly to matplotlib).

    Returns:
        The figure object containing the plot.
    """
    figure = plt.figure(figsize=(8, 5), dpi=250)
    if logscale == "y":
        plot_function = plt.semilogy
    elif logscale == "x":
        plot_function = plt.semilogx
    elif logscale == "xy":
        plot_function = plt.loglog
    else:
        plot_function = plt.plot

    n_sets = len(time_data_sets)
    
    for j, time_data in enumerate(time_data_sets):
        for i, (method, results) in enumerate(time_data.items()):
            if len(results) != len(x_values):
                if not different_lengths_allowed:
                    raise ValueError(f"Number of timing results for '{method}' doesn't match number of x values")
                else:
                    x_values_method = x_values[:len(results)]
            else:
                x_values_method = x_values
                
            color = COLOUR_PALETTE[i % len(COLOUR_PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            means = jnp.array([r[0] for r in results])
            stds = jnp.array([r[1] for r in results])

            facecolor = idx_and_color_to_face_color(j, n_sets, color)

            plot_function(
                x_values_method,
                means,
                label=f"{method}",
                color=facecolor,
                marker=marker,
                linestyle="-",
                linewidth=2,
                markersize=6,
                markeredgecolor="black",
            )
            plt.fill_between(
                x_values_method,
                means - stds,
                means + stds,
                color=color,
                alpha=0.2,
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
    figure.tight_layout()
    plt.show()
    return figure
    
        
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
    figure = plt.figure(figsize=(8, 5), dpi=250)
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
    return figure  # Return the figure object for further manipulation if needed
    
    
def visualize_data_sets_as_bars(
    data: dict[str, dict[str, float]],
    bar_labels: list[str] = None,
    title: str = None,
    ylabel: str = "Time [s]",
    xlabel: str = None,
    horizontal_lines: dict[str, float] = None,
    logscale: bool = False,
    dpi: int = 300,
) -> None:
    """
    Plot grouped bar chart of expectation values with error bars.

    Args:
        data: Map of x labels to dicts of {bar label: value}.
        bar_labels: Optional list of legend labels for the bars (e.g. qubit names).
        title: Title of the plot.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        horizontal_lines: Optional dict of {line label: y value} to add horizontal lines for reference.
    """
    x_labels = list(data.keys())
    x_positions = np.arange(len(x_labels))

    # Get bar labels from first item
    bar_keys = list(next(iter(data.values())).keys())
    num_bars = len(bar_keys)

    if bar_labels is None:
        bar_labels = [f"Qubit {i}" for i in range(num_bars)]

    colours = COLOUR_PALETTE[:num_bars]  # use your custom palette

    total_width = 0.8
    bar_width = total_width / num_bars

    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

    for i, (bar_key, color, label) in enumerate(zip(bar_keys, colours, bar_labels)):
        means = [data[x][bar_key] for x in x_labels]
        bar_pos = x_positions + i * bar_width - total_width / 2 + bar_width / 2
        ax.bar(
            bar_pos,
            means,
            width=bar_width,
            label=label,
            color=color,
        )

    # Add horizontal lines if provided
    if horizontal_lines:
        # Create a list of colors for the lines that differ from bar colors
        line_colors = ["red", "black", "blue", "green", "purple"][
            : len(horizontal_lines)
        ]

        line_handles = []
        for (label, y_value), color in zip(horizontal_lines.items(), line_colors):
            line = ax.axhline(y=y_value, color=color, linestyle="--", linewidth=1.5)
            line_handles.append((line, label))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    if logscale:
        ax.set_yscale("log")
        
    if title:
        ax.set_title(title)
        
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    # Legend 1: for bar groups (qubits)
    legend1 = ax.legend(loc="best", fontsize=10, title_fontsize=11)
    ax.add_artist(legend1)  # ✅ Add legend1 explicitly

    # Legend 3: for horizontal lines (if provided)
    if horizontal_lines:
        line_handles_list, line_labels = zip(*line_handles)
        legend3 = ax.legend(
            handles=line_handles_list,
            labels=line_labels,
            loc="upper left",
            fontsize=10,
            title_fontsize=11,
        )
        ax.add_artist(legend3)  # Add legend3 explicitly

    fig.tight_layout()
    plt.show()
    return fig


def plot_pauli_probabilities_indexed(probabilities: dict[str, float], title:str = None, sort: bool = False, ylabel: str = "Pauli Probability") -> plt.Figure:
    """Plot the probabilities of Pauli operators as a line chart with indexed x-axis.

    Args:
        probabilities: Dictionary mapping Pauli string labels to their corresponding probabilities.
        title: Title of the plot. Defaults to None.
        sort: Whether to sort the probabilities in descending order. Defaults to False.
        ylabel: Label for the y-axis. Defaults to "Pauli Probability".

    Returns:
        The figure object containing the plot.
    """

    figure = plt.figure(figsize=(8,4), dpi=200)
    
    if sort:
        print("Sorting")
        probabilites_sorted = [jnp.sort(probs, descending=True) for probs in probabilities.values()]
        probabilities = dict(zip(probabilities.keys(), probabilites_sorted))

    for i, (label, probs) in enumerate(probabilities.items()):
        plt.semilogy(probs, MARKERS[i % len(MARKERS)] + "-", label=label, color=COLOUR_PALETTE[i % len(COLOUR_PALETTE)], markersize=5, linewidth=1.5)
    plt.xlabel(f"Pauli Index (sorted = {sort})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    return figure

def plot_pauli_probabilities_labeled(probabilities: dict[str, float], max_num_labels: int = None, safe: bool = True , log_scale: bool = False, title: str = None) -> plt.Figure:
    """
    Plot the probabilities of Pauli operators as a bar chart.

    Args:
        probabilities: Dictionary mapping Pauli string labels to their corresponding probabilities.
        max_num_labels: Maximum number of labels to display on the x-axis.
        safe: If True, will not display more than 20 labels to avoid clutter. Set to False to disable this check.
        log_scale: Whether to use a logarithmic scale for the y-axis. Defaults to False.
        title: Title of the plot. Defaults to None.
    Returns:
        The figure object containing the plot.
    """
    if max_num_labels is not None and max_num_labels > len(probabilities):
        raise ValueError(f"max_num_labels ({max_num_labels}) exceeds the number of probabilities ({len(probabilities)}).")
    if max_num_labels is None:
        max_num_labels = len(probabilities)
        if max_num_labels > 20 and safe:
            print(f"Warning: Displaying all {max_num_labels} labels may result in a cluttered plot. Defaulting to 20 labels. To forcefully display all labels, set safe=False to disable this warning.")
            max_num_labels = 20
    
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    # Sort probabilities in descending order and keep only the top max_num_labels
    sorted_probs = sorted_probs[:max_num_labels]
    probabilities = dict(sorted_probs)
    labels = list(probabilities.keys())
    probs = list(probabilities.values())

    figure = plt.figure(figsize=(10, 6), dpi=200)
    plt.bar(labels, probs, color=COLOUR_PALETTE)
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Pauli Operators')
    plt.ylabel('Probability')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return figure

def plot_weight_histogram(histogram: dict[int, float], ascending: bool = True, title: str = None, log_scale: bool = True, ylabel:str = None, xlabel:str = None) -> plt.Figure:
    """
    Plot the histogram of probabilities by Pauli weight.

    Args:
        histogram: Dictionary mapping Pauli weight to total probability.
        ascending: Whether to sort the histogram by weight in ascending order. Defaults to True.
        title: Title of the plot.
        log_scale: Whether to use a logarithmic scale for the y-axis. Defaults to True.

    
    Returns:
        The figure object containing the plot.
    """
    if ylabel is None:
        ylabel = "Probability"
    if xlabel is None:
        xlabel = "Pauli Weight"

    weights = sorted(histogram.keys(), reverse=not ascending)
    probabilities = [histogram[w] for w in weights]

    figure = plt.figure(figsize=(6, 4), dpi=250)
    plt.bar(weights, probabilities, color=COLOUR_PALETTE)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title)
    plt.xticks(weights)
    if log_scale:
        plt.yscale('log')
    plt.grid(axis='y')
    plt.show()
    return figure

def plot_mean_std_vs_x(
    data: dict[str, dict[str, float]],
    title: str = "Mean and Std vs Number of Sequences",
    xlabel: str = "Number of sequences",
    ylabel: str = "Distance",
    use_log_scale: bool = False,
    dpi: int = 250,
    comparison_value: tuple[str, float] = None,
) -> plt.Figure:
    """
    Plot a dictionary of the form:
    {
        "100": {"mean": ..., "std": ...},
        "200": {"mean": ..., "std": ...},
        ...
    }

    Uses the project's COLOUR_PALETTE and MARKERS.
    """
    if not data:
        raise ValueError("Input data is empty.")

    # Sort numerically by x key even if keys are strings
    x_labels = sorted(data.keys(), key=lambda k: float(k))
    x_vals = np.array([float(k) for k in x_labels], dtype=float)

    # Convert possible JAX scalars to Python floats
    means = np.array([float(data[k]["mean"]) for k in x_labels], dtype=float)
    stds = np.array([float(data[k]["std"]) for k in x_labels], dtype=float)

    figure, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    color = COLOUR_PALETTE[0]
    marker = MARKERS[0]

    ax.plot(
        x_vals,
        means,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=7,
        markeredgecolor="black",
        label="Mean distance",
    )

    ax.fill_between(
        x_vals,
        means - stds,
        means + stds,
        color=color,
        alpha=0.2,
        label="±1 std",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)

    if use_log_scale:
        ax.set_yscale("log")

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=10)
    
    if comparison_value is not None:
        comp_label, comp_y = comparison_value
        ax.axhline(y=comp_y, color="red", linestyle="--", label=comp_label)
        ax.legend(fontsize=10)
    
    figure.tight_layout()
    plt.show()
    return figure

from mGST.analysis import generate_weighted_fitted_values, generate_fitted_values

def plot_mve_multiple_seq_lengths_with_error_bars(mve_dict: dict[int, list[tuple]], shots_list: jnp.ndarray, title: str | None = None):
    """Plot the MVE against number of shots for different sequence lengths, with error bars and weighted fit."""
    parameters_dict = {}
    handles, labels = [], []

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    colours = plt.cm.viridis(np.linspace(0, 1, len(mve_dict.keys())))

    for seq_length, colour in zip(mve_dict.keys(), colours):
        mve_vs_shots = mve_dict[seq_length] # list of tuples (avg, std)

        y_fit, slope, intercept, slope_std, *_ = generate_weighted_fitted_values(
            x=shots_list, y_avg_std=mve_vs_shots, base=10
        )

        parameters_dict[seq_length] = (slope, intercept, slope_std)

        mve_avg = jnp.array([val for val, _ in mve_vs_shots])
        mve_std = jnp.array([std for _, std in mve_vs_shots])

        error_bar = ax.errorbar(
            shots_list, mve_avg, yerr=mve_std,
            fmt='o', markersize=8, capsize=4, linewidth=1.5,
            color=colour
        )
        line, = ax.loglog(shots_list, y_fit, '--', linewidth=2, color=colour)

        handles += [error_bar, line]
        labels += [
            f'seq_len={seq_length} (data ± std)',
            f'seq_len={seq_length} (weighted fit, slope={slope:.3f}±{slope_std:.3f})',
        ]

    
    ax.set_xlabel('Number of Shots', fontsize=14)
    ax.set_ylabel('Mean Variation Error', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(handles, labels, fontsize=10, loc='best')
    ax.set_xlim(shots_list[0] * 0.5, shots_list[-1] * 2)  # <-- add this

    plt.tight_layout()
    plt.show()

    return fig, parameters_dict

def plot_mve_multiple_seq_lengths(mve_dict:dict[int, list[tuple]], shots_list:jnp.ndarray, title:str|None=None):
    """Plot the MVE against number of shots for different number of sequences."""
    parameters_dict = {}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    # Color palette for different sequence lengths
    colours = plt.cm.viridis(np.linspace(0, 1, len(mve_dict.keys())))
    
    for seq_length, colour in zip(mve_dict.keys(), colours):
        
        mve_vs_shots = mve_dict[seq_length]
        if isinstance(mve_vs_shots[0], tuple):
            # we are taking only the avg out of a tuple (avg, std)
            mve_vs_shots = [result[0] for result in mve_vs_shots]
        mve_vs_shots = jnp.array(mve_vs_shots)
        
        y_fit, slope, intercept = generate_fitted_values(x=shots_list, y=mve_vs_shots, base=10)
        
        parameters_dict[seq_length] = (slope, intercept)
        
        
        # Plot data and fit for this sequence length
        ax.loglog(shots_list, mve_vs_shots, 'o', linewidth=2, markersize=8,
                 color=colour, label=f'seq_len={seq_length} (data)')
        ax.loglog(shots_list, y_fit, '--', linewidth=2, color=colour,
                 label=f'seq_len={seq_length} (fit, slope={slope:.3f})')
        
    ax.set_xlabel('Number of Shots', fontsize=14)
    ax.set_ylabel('Mean Variation Error', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()
    
    return fig, parameters_dict