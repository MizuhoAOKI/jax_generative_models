from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.image
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
# 2. Helper Functions
# --------------------------------------------------------


def reshape_to_image_grid(batch: np.ndarray, data_dim: int) -> np.ndarray:
    """Reshapes a batch of flattened vectors into a single square image grid.

    This function handles padding if the batch size is not a perfect square,
    stitching individual samples into a large mosaic.

    Args:
        batch: Array of shape (B, D).
        data_dim: Dimensionality D of a single sample (must be a perfect square, e.g. 784).

    Returns:
        A 2D numpy array representing the stitched grid of images.

    Raises:
        ValueError: If data_dim is not a perfect square.
    """
    B, D = batch.shape
    side = int(np.sqrt(D))

    if side * side != D:
        raise ValueError(f"Data dim {D} is not a perfect square (cannot be reshaped to image).")

    # Determine grid size (e.g., sqrt(64) -> 8x8 grid)
    grid_side = int(np.ceil(np.sqrt(B)))

    # Pad batch with zeros if needed to fill the square grid
    pad_size = grid_side**2 - B
    if pad_size > 0:
        batch = np.pad(batch, ((0, pad_size), (0, 0)), constant_values=0)

    # Reshape to (Grid_H, Grid_W, H, W)
    images = batch.reshape(grid_side, grid_side, side, side)

    # Transpose to (Grid_H, H, Grid_W, W) for row-major stitching
    # Then flatten the spatial dimensions to create the final 2D image
    images = images.transpose(0, 2, 1, 3)
    grid_image = images.reshape(grid_side * side, grid_side * side)

    return grid_image


# --------------------------------------------------------
# 3. Plotting Setup (Shared with Animation)
# --------------------------------------------------------


def setup_point_scatter(
    vis_config: VisConfig,
    initial_data: np.ndarray | None = None,
) -> tuple[
    matplotlib.figure.Figure,
    matplotlib.axes.Axes,
    matplotlib.collections.PathCollection,
]:
    """Initializes a Matplotlib figure and scatter plot for Point Cloud data.

    This helper is designed to be shared between static visualization and
    animation routines to ensure visual consistency.

    Args:
        vis_config: Visualization style settings.
        initial_data: Optional initial data (N, 2) to plot.
                      If None, initializes an empty scatter plot.

    Returns:
        A tuple containing:
            - fig: The Matplotlib Figure object.
            - ax: The Matplotlib Axes object.
            - scatter: The PathCollection object (scatter plot artist).
    """
    fig, ax = plt.subplots(figsize=vis_config.figsize)

    # Remove all margins and padding from the figure to eliminate whitespace
    # in animations.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

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


def setup_image_grid(
    vis_config: VisConfig,
    initial_data: np.ndarray | None = None,
    data_dim: int = 784,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> tuple[
    matplotlib.figure.Figure,
    matplotlib.axes.Axes,
    matplotlib.image.AxesImage,
]:
    """Initializes a Matplotlib figure and imshow artist for Image data.

    Args:
        vis_config: Visualization style settings.
        initial_data: Optional initial batch data (B, D).
        data_dim: The dimensionality of a single sample (D).
        vmin: Minimum value for colormap normalization.
        vmax: Maximum value for colormap normalization.

    Returns:
        A tuple containing:
            - fig: The Matplotlib Figure object.
            - ax: The Matplotlib Axes object.
            - im: The AxesImage object (image artist).
    """
    fig, ax = plt.subplots(figsize=vis_config.figsize)

    # Remove all margins and padding from the figure to eliminate whitespace
    # in animations.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if initial_data is None:
        # Create a dummy blank grid if no initial data provided (for animation init)
        side = int(np.sqrt(data_dim))
        dummy_grid = np.zeros((side, side))
        im_data = dummy_grid
    else:
        im_data = reshape_to_image_grid(initial_data, data_dim)

    # 'animated=True' is an optimization for FuncAnimation blitting
    im = ax.imshow(
        im_data,
        cmap="gray",
        interpolation="nearest",
        animated=True,
        vmin=vmin,
        vmax=vmax,
    )

    ax.axis("off")
    # Aspect ratio is automatically handled by imshow (default 'equal')

    return fig, ax, im


# --------------------------------------------------------
# 4. Main Visualization Entry Point
# --------------------------------------------------------


def visualize_dataset_batch(
    batch: Array,
    dataconfig: data.DatasetConfig,
    vis_config: VisConfig,
    save_path: Path,
) -> None:
    """Visualizes and saves a batch of samples to a file using Matplotlib.

    Dispatches to the correct plotting routine based on the dataset type.

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

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Dispatch based on Data Type
    match dataconfig.data_type:
        case data.DataType.IMAGE:
            # For images, we use the dataset scale to set vmax, ensuring consistent brightness.
            fig, _, _ = setup_image_grid(
                vis_config,
                initial_data=batch_np,
                data_dim=dataconfig.data_dim,
                vmax=dataconfig.scale,
            )

        case data.DataType.POINT_2D:
            fig, _, _ = setup_point_scatter(vis_config, initial_data=batch_np)

        case _:
            logger.warning(
                f"Visualization for data_type={dataconfig.data_type} is not implemented. "
                f"Skipping plot for {dataconfig.name}."
            )
            return

    fig.savefig(save_path, dpi=vis_config.dpi, bbox_inches="tight", pad_inches=0)
    logger.info(f"Saved visualization to {save_path}")

    plt.close(fig)
