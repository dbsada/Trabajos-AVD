'''
Script con funciones para dibujar gráficos no incluidos en librerias estandar.
Autor: Diego Besada Rodríguez
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from typing import Callable, Optional    
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import cm

def draw_piano(ax: plt.Axes,  
               note_values: dict[str, float],
               title: str, 
               cmap: Colormap,
               norm_func: Callable[[float], float],
               label_func: Optional[Callable[[float], str]] = None,
               gap: float = 0.06,
               rounding: float = 0.08,
               annotation: Optional[str] = None
) -> None:
    '''
    Draws a piano-style heatmap.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes where the piano will be drawn.
    note_values : Mapping[str, float]
        Dictionary mapping note names (e.g., 'C', 'C#') to numeric values.
    title : str
        Title displayed on the left side of the piano.
    cmap : Colormap
        Matplotlib colormap used to color keys.
    norm_func : Callable[[float], float]
        Function that normalizes note values to [0, 1] for colormap intensity.
    label_func : Optional[Callable[[float], str]]
        Optional function to generate label text from a note value.
    gap : float
        Horizontal gap between keys.
    rounding : float
        Rounding radius for key corners.
    annotation : Optional[str]
        How to annotation values: 'sum' or 'mean'. Shows result on the right.
    '''

    white_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    black_notes = ['C#', 'D#', None, 'F#', 'G#', 'A#', None]

    white_width, white_height = 1, 1.5
    black_width, black_height = 0.6, 1

    text_effects = [pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]
    piano_width = len(white_notes) * white_width

    def draw_key(note, x, y, width, height):
        value = note_values.get(note, 0)
        intensity = norm_func(value)

        rect = FancyBboxPatch(
            (x + gap / 2, y),
            width - gap,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=cmap(intensity),
            edgecolor="black",
            linewidth=2
        )
        ax.add_patch(rect)

        if label_func and value is not None:
            label = label_func(value)
            if label:
                ax.text(x + 0.07, y + 0.05,
                        label,
                        ha='left', va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color="white",
                        path_effects=text_effects)

    # White keys
    for i, note in enumerate(white_notes):
        draw_key(note, i * white_width, 0, white_width, white_height)
        ax.text(i + 0.5, 0.4, note,
                ha='center', fontsize=14,
                fontweight='bold', alpha=0.3)

    # Black keys
    for i, note in enumerate(black_notes):
        if note:
            draw_key(note,
                     i * white_width + 0.7,
                     white_height - black_height,
                     black_width,
                     black_height)

    ax.set_xlim(-0.1, piano_width + 0.1)
    ax.set_ylim(-0.1, white_height + 0.15)
    ax.axis('off')

    ax.text(-0.15, white_height / 2,
            title,
            rotation=90,
            va='center',
            ha='center',
            fontsize=16,
            fontweight='bold')

    # Acumulado
    if annotation:
        ax.text(piano_width + 0.15, white_height / 2,
                annotation,
                rotation=-90,
                va='center',
                ha='center',
                fontsize=12,
                fontweight='bold')

def joyplot(ax, data, group_col, value_col,
            offset=1.2,
            bins=40,
            scale=0.9,
            cmap="viridis"):
    """
    Professional histogram-only ridgeplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where the plot will be drawn.
    data : pandas.DataFrame
        DataFrame in long format.
    group_col : str or list[str]
        Column(s) defining groups. If list, groups are created by combining columns.
    value_col : str
        Numeric variable.
    offset : float, default=1.2
        Vertical separation between ridges.
    bins : int, default=40
        Number of histogram bins.
    scale : float, default=0.9
        Vertical scaling factor (controls ridge height).
    cmap : str, default="viridis"
        Matplotlib colormap name.
    """

    # Handle multiple group columns
    if isinstance(group_col, list):
        group_key = "_".join(group_col)
        data = data.copy()
        data[group_key] = data[group_col].astype(str).agg("_".join, axis=1)
        group_col = group_key

    groups = list(data[group_col].dropna().unique())
    n_groups = len(groups)

    # Global x-range
    all_values = data[value_col].dropna()
    x_min, x_max = all_values.min(), all_values.max()

    # Create colormap with evenly spaced colors
    colormap = cm.get_cmap(cmap)
    colors = [colormap(i / max(n_groups - 1, 1)) for i in range(n_groups)]

    # Compute maximum density across groups for consistent scaling
    global_max_density = 0
    histograms = {}

    for group in groups:
        values = data.loc[data[group_col] == group, value_col].dropna()
        counts, edges = np.histogram(
            values, bins=bins, range=(x_min, x_max), density=True
        )
        histograms[group] = (counts, edges)
        global_max_density = max(global_max_density, counts.max())

    # Plot
    for i, group in enumerate(groups):
        counts, edges = histograms[group]
        bin_width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Normalize by global max to keep relative structure
        scaled_counts = (counts / global_max_density) * scale

        ax.bar(
            centers,
            scaled_counts,
            width=bin_width,
            bottom=i * offset,
            color=colors[i],
            edgecolor=colors[i],
            linewidth=1.5,
            alpha=0.85,
            align="center",
        )

    # Axis formatting
    ax.set_yticks([i * offset for i in range(n_groups)])
    ax.set_yticklabels(groups)
    ax.set_ylabel(group_col if isinstance(group_col, str) else ", ".join(group_col))
    ax.set_xlabel(value_col)

    # Minimalist styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(False)

    ax.set_xlim(x_min, x_max)


