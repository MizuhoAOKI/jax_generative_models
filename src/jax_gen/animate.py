from __future__ import annotations

import logging
from typing import Union

import equinox as eqx
import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from jax_gen import models, visualizer
from jax_gen.config import AnimateConfig
from jax_gen.strategies import create_strategy

logger = logging.getLogger(__name__)


def animate(cfg: AnimateConfig, key: jax.Array) -> None:
    """Executes the animation pipeline to visualize the generation process.

    This function leverages the shared visualization logic from `visualizer.py`
    to ensure the animation style matches the static generated images.

    Args:
        cfg: Animation configuration object.
        key: JAX PRNGKey for stochastic sampling.

    Raises:
        FileNotFoundError: If the model file is missing.
        ValueError: If the data dimensionality is not supported (currently only 2D).
    """
    logger.info(f"Starting animation task. Mode: {cfg.mode}")

    # -------------------------------------------------------------------------
    # 1. Validation
    # -------------------------------------------------------------------------
    if cfg.dataset.data_dim != 2:
        raise ValueError(
            f"Animation is currently only supported for 2D datasets. "
            f"Got dimensions: {cfg.dataset.data_dim}"
        )

    # -------------------------------------------------------------------------
    # 2. Load Model & Strategy
    # -------------------------------------------------------------------------
    if not cfg.model_path.exists():
        logger.error(f"Model file not found at: {cfg.model_path}")
        raise FileNotFoundError(f"Model file not found at: {cfg.model_path}")

    logger.info(f"Loading model from {cfg.model_path}...")
    model_init = models.create_model(cfg.model, key, data_dim=cfg.dataset.data_dim)
    model = eqx.tree_deserialise_leaves(cfg.model_path, model_init)

    logger.info(f"Initializing strategy: {cfg.strategy.name}")
    strategy = create_strategy(cfg.strategy)

    # -------------------------------------------------------------------------
    # 3. Generate Trajectory
    # -------------------------------------------------------------------------
    logger.info(f"Generating trajectory for {cfg.num_samples} samples...")
    key, subkey = jax.random.split(key)

    # Retrieve the full trajectory.
    # Shape of x_traj: (num_time_steps, num_samples, data_dim)
    _, x_traj = strategy.sample_from_target_distribution(
        model=model,
        key=subkey,
        num_samples=cfg.num_samples,
        data_dim=cfg.dataset.data_dim,
    )

    # Convert to NumPy for Matplotlib
    trajectory_np = np.array(x_traj)
    num_frames = trajectory_np.shape[0]

    logger.info(f"Trajectory captured: {num_frames} frames.")

    # -------------------------------------------------------------------------
    # 4. Create Animation
    # -------------------------------------------------------------------------
    save_path = cfg.output_video_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering frames...")

    # Reuse the logic from visualizer to ensure consistent styling
    # We initialize with empty data; the update function will handle the rest.
    fig, ax, scatter = visualizer.setup_2d_plot(cfg.vis, initial_data=None)

    def init() -> list:
        """Initializes the animation frame (required for blit=True)."""
        scatter.set_offsets(np.empty((0, 2)))
        return [scatter]

    def update(frame_idx: int) -> list:
        """Updates the scatter plot for a given frame index."""
        # Get data for the current time step: (Batch, 2)
        data_t = trajectory_np[frame_idx]
        scatter.set_offsets(data_t)
        return [scatter]

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=int(1000 / cfg.fps)
    )

    # -------------------------------------------------------------------------
    # 5. Save Video
    # -------------------------------------------------------------------------
    logger.info(f"Saving animation to {save_path} (FPS={cfg.fps})...")

    writer: Union[animation.FFMpegWriter, animation.PillowWriter]

    if save_path.suffix == ".mp4":
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=cfg.fps)
        else:
            logger.warning("FFMpeg is not available. Falling back to Pillow (GIF).")
            save_path = save_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=cfg.fps)
    elif save_path.suffix == ".gif":
        writer = animation.PillowWriter(fps=cfg.fps)
    else:
        logger.warning(f"Unknown extension {save_path.suffix}. Defaulting to Pillow (GIF).")
        writer = animation.PillowWriter(fps=cfg.fps)

    anim.save(save_path, writer=writer, dpi=cfg.vis.dpi)

    # Close the figure to free memory
    plt.close(fig)

    logger.info("Animation completed successfully.")
