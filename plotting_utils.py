import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set_theme()


def plot_with_regimes(y, regimes, regime_colors=None, line_color='black', title='Regimes over time', ax: np.ndarray = None):
    """
    Plots a line chart with background colors highlighting different regimes.
    
    Args:
        x (np.ndarray): The x values (e.g., time).
        y (np.ndarray): The y values (e.g., the data to plot).
        regimes (np.ndarray): A vector indicating the regime for each x value.
        regime_colors (dict, optional): A dictionary mapping regime numbers to colors. If None, random colors will be used.
        line_color (str, optional): The color of the line plot. Default is 'black'.
        title (str, optional): The title of the plot. Default is 'Line Plot with Regime Backgrounds'.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(y))
    
    unique_regimes = np.unique(regimes)
    
    # Set default colors for regimes if no colors are provided
    if regime_colors is None:
        regime_colors = {regime: plt.cm.Set3(i / len(unique_regimes)) for i, regime in enumerate(unique_regimes)}
    
    # Highlight background for each regime
    start_idx = 0
    for i in range(1, len(x)):
        # If the regime changes or we reach the end, highlight the region
        if regimes[i] != regimes[start_idx] or i == len(x) - 1:
            end_idx = i if regimes[i] != regimes[start_idx] else i + 1  # Include the last point
            ax.axvspan(
                x[start_idx],
                x[end_idx - 1],
                color=regime_colors[regimes[start_idx]],
                alpha=0.3
            )
            start_idx = i
    
    # Plot the line chart
    ax.plot(x, y, color=line_color, lw=2)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    # Create a legend for the regimes
    # handles = [plt.Line2D([0], [0], color=regime_colors[regime], lw=6, label=f'Regime {regime}') for regime in unique_regimes]
    # ax.legend(handles=handles, loc='upper left')
    
    return ax


def plot_multiple_with_regimes(Xs: list, Zs: list, regime_colors: dict = None):
    n_rows = len(Xs)
    assert len(Xs) == len(Zs)
    fig, ax = plt.subplots(nrows=n_rows, sharex=True)
    plot_with_regimes(Xs[0], Zs[0], ax=ax[0], regime_colors=regime_colors)
    for a, X, Z in zip(ax, Xs, Zs):
        plot_with_regimes(X, Z, ax=a, regime_colors=regime_colors, title='')
    # for a in ax[:-1]:
        # a.get_legend().remove()
    return ax


def plot_parallel_multiple_with_regimes(
        Xs: list, Zs_a: list, Zs_b: list, regime_colors: dict = None,
    ):
    n_rows = len(Xs)
    n_cols = 2
    assert len(Xs) == len(Zs_a)
    assert len(Xs) == len(Zs_b)

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)
    plot_with_regimes(Xs[0], Zs_a[0], ax=ax[0,0], regime_colors=regime_colors, title="Regimes as is")
    plot_with_regimes(Xs[0], Zs_b[0], ax=ax[0,1], regime_colors=regime_colors, title="Regimes +1")

    for a, X, Z in zip(ax[1:,0], Xs[1:], Zs_a[1:], strict=True):
        plot_with_regimes(X, Z, ax=a, regime_colors=regime_colors, title='')
    
    for a, X, Z in zip(ax[1:,1], Xs[1:], Zs_b[1:]):
        plot_with_regimes(X, Z, ax=a, regime_colors=regime_colors, title='')
    
    # for a in ax.flatten()[:-1]:
        # a.get_legend().remove()
    plt.tight_layout()
    return ax


def plot_transition_cost(tcm):
    seaborn.heatmap(tcm, annot=True, fmt=".1f", vmin=0)
def plot_parallel_transition_cost(tcm_a, tcm_b):
    vmax = max(
        np.max(tcm_a[np.isfinite(tcm_a)]),
        np.max(tcm_b[np.isfinite(tcm_b)]),
    )
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    seaborn.heatmap(tcm_a, annot=True, fmt=".1f", ax=ax[0], vmin=0, vmax=vmax, cbar=False)
    seaborn.heatmap(tcm_b, annot=True, fmt=".1f", ax=ax[1], vmin=0, vmax=vmax)