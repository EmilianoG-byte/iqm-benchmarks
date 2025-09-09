"""
Utility functions for visualization purposes.
"""

import matplotlib.pyplot as plt

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