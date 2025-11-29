# src/jax_gen/visualizer.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from jax_gen import data

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# 1. Visualization Configuration
# --------------------------------------------------------


@dataclass(frozen=True)
class VisConfig:
    """Configuration for visualization settings."""

    dpi: int = 150
    """Resolution (dots per inch) for saved static images."""

    marker_size: float = 10.0
    """Size of the scatter plot markers."""

    color: str = "#005AFF"
    """Hex color code for markers."""

    alpha: float = 0.6
    """Transparency level of markers (0.0 to 1.0)."""

    figsize: tuple[int, int] = (6, 6)
    """Figure dimensions (width, height) in inches."""

    xlim: tuple[float, float] | None = (-3.0, 3.0)
    """Tuple determining the X-axis range (min, max), or None for auto."""

    ylim: tuple[float, float] | None = (-3.0, 3.0)
    """Tuple determining the Y-axis range (min, max), or None for auto."""

    enable_rerun: bool = True
    """Flag to enable/disable Rerun logging."""

    headless: bool = False
    """Flag to run Rerun in headless mode (no GUI)."""

    rerun_log_save_path: Path | None = Path("outputs/rerun_vis_log.rrd")
    """Path to save the Rerun recording (.rrd)."""

    sample_log_interval: int = 500
    """Training step interval for logging visualizations."""

    num_vis_samples: int = 1000
    """Number of samples to generate for visualization."""


# --------------------------------------------------------
# 2. Static Plotting (Matplotlib)
# --------------------------------------------------------


def setup_2d_plot(
    vis_config: VisConfig,
    initial_data: np.ndarray | None = None,
) -> tuple[
    matplotlib.figure.Figure,
    matplotlib.axes.Axes,
    matplotlib.collections.PathCollection,
]:
    """Initializes a Matplotlib figure and scatter plot with consistent styling.

    This helper function is designed to be shared between static visualization
    and animation routines to ensure visual consistency.

    Args:
        vis_config: Visualization style settings.
        initial_data: Optional initial data (N, 2) to plot.
                      If None, initializes an empty scatter plot.

    Returns:
        A tuple containing:
            - fig: The Matplotlib Figure object.
            - ax: The Matplotlib Axes object.
            - scatter: The PathCollection object (scatter plot artist) for updates.
    """
    fig, ax = plt.subplots(figsize=vis_config.figsize)

    # If no data is provided, initialize with empty array for animation setup
    if initial_data is None:
        initial_data = np.empty((0, 2))

    scatter = ax.scatter(
        initial_data[:, 0],
        initial_data[:, 1],
        s=vis_config.marker_size,
        alpha=vis_config.alpha,
        edgecolors="none",
        c=vis_config.color,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    if vis_config.xlim is not None:
        ax.set_xlim(vis_config.xlim)
    if vis_config.ylim is not None:
        ax.set_ylim(vis_config.ylim)

    return fig, ax, scatter


def visualize_dataset_batch(
    batch: Array,
    dataconfig: data.DatasetConfig,
    vis_config: VisConfig,
    save_path: Path,
) -> None:
    """Visualizes and saves a batch of samples to a file using Matplotlib.

    Args:
        batch: A JAX array of samples with shape (B, D).
        dataconfig: Configuration object describing the dataset structure.
        vis_config: Visualization style configuration.
        save_path: Filesystem path to save the resulting image.

    Raises:
        ValueError: If the batch shape is not 2D.
    """
    batch_np = np.array(batch)

    if batch_np.ndim != 2:
        raise ValueError(f"Expected batch shape (B, D), got {batch_np.shape}")

    if dataconfig.data_dim != 2:
        logger.warning(
            f"Visualization for data_dim={dataconfig.data_dim} is not implemented. "
            f"Skipping plot for {dataconfig.name}."
        )
        return

    # Use the shared setup function
    fig, _, _ = setup_2d_plot(vis_config, initial_data=batch_np)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=vis_config.dpi, bbox_inches="tight", pad_inches=0)
    logger.info(f"Saved visualization to {save_path}")

    plt.close(fig)
