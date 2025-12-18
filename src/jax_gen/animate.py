from __future__ import annotations

import logging
from typing import Callable, Union

import equinox as eqx
import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from jax_gen import data, generate, models, visualizer
from jax_gen.config import AnimateConfig
from jax_gen.strategies import create_strategy

logger = logging.getLogger(__name__)


def animate(cfg: AnimateConfig, key: jax.Array) -> None:
    """Executes the animation pipeline to visualize the generation process.

    This function leverages shared visualization logic from `visualizer.py`
    to ensure the animation style matches static generated images.
    It supports both Point Cloud (scatter) and Image (grid) datasets via
    DataType branching.

    Args:
        cfg: Animation configuration object.
        key: JAX PRNGKey for stochastic sampling.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    logger.info(f"Starting animation task. Mode: {cfg.mode}")

    # -------------------------------------------------------------------------
    # 1. Load Model & Strategy
    # -------------------------------------------------------------------------
    if not cfg.model_path.exists():
        logger.error(f"Model file not found at: {cfg.model_path}")
        raise FileNotFoundError(f"Model file not found at: {cfg.model_path}")

    logger.info(f"Loading model from {cfg.model_path}...")
    model_init = models.create_model(cfg.model, cfg.dataset, key)
    model = eqx.tree_deserialise_leaves(cfg.model_path, model_init)

    logger.info(f"Initializing strategy: {cfg.strategy.name}")
    strategy = create_strategy(cfg.strategy)

    # -------------------------------------------------------------------------
    # 2. Generate Trajectory
    # -------------------------------------------------------------------------
    logger.info(f"Generating trajectory for {cfg.num_samples} samples...")
    key, cond_key, subkey = jax.random.split(key, 3)
    cond_batch = generate.get_condition_batch(cfg, cfg.dataset, cond_key)

    # Retrieve the full trajectory.
    # Shape of x_traj: (num_time_steps, num_samples, data_dim)
    _, x_traj = strategy.sample_from_target_distribution(
        model=model,
        key=subkey,
        num_samples=cfg.num_samples,
        data_dim=cfg.dataset.data_dim,
        cond=cond_batch,
    )

    # Convert to NumPy for Matplotlib
    trajectory_np = np.array(x_traj)
    num_frames = trajectory_np.shape[0]

    logger.info(f"Trajectory captured: {num_frames} frames.")

    # -------------------------------------------------------------------------
    # 3. Configure Animation (Dispatcher)
    # -------------------------------------------------------------------------
    logger.info("Setting up animation renderer...")

    # Define init/update functions based on the data type
    init_fn: Callable[[], list]
    update_fn: Callable[[int], list]

    match cfg.dataset.data_type:
        case data.DataType.IMAGE:
            # Setup for Image Grid
            fig, _, im = visualizer.setup_image_grid(
                cfg.vis,
                initial_data=None,
                data_dim=cfg.dataset.data_dim,
                vmax=cfg.dataset.scale,
            )

            def init_image() -> list:
                """Initializes with a blank grid."""
                dummy_batch = np.zeros_like(trajectory_np[0])
                grid = visualizer.reshape_to_image_grid(dummy_batch, cfg.dataset.data_dim)
                im.set_data(grid)
                return [im]

            def update_image(frame_idx: int) -> list:
                """Updates the grid pixels."""
                batch_t = trajectory_np[frame_idx]
                grid = visualizer.reshape_to_image_grid(batch_t, cfg.dataset.data_dim)
                im.set_data(grid)
                return [im]

            init_fn = init_image
            update_fn = update_image

        case data.DataType.POINT_2D:
            # Setup for Scatter Plot
            fig, _, scatter = visualizer.setup_point_scatter(cfg.vis, initial_data=None)

            def init_scatter() -> list:
                """Initializes with an empty scatter plot."""
                scatter.set_offsets(np.empty((0, 2)))
                return [scatter]

            def update_scatter(frame_idx: int) -> list:
                """Updates the scatter points coordinates."""
                data_t = trajectory_np[frame_idx]
                scatter.set_offsets(data_t)
                return [scatter]

            init_fn = init_scatter
            update_fn = update_scatter

        case _:
            raise NotImplementedError(
                f"Animation not supported for data type: {cfg.dataset.data_type}"
            )

    # -------------------------------------------------------------------------
    # 4. Render & Save Video
    # -------------------------------------------------------------------------
    save_path = cfg.output_video_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    anim = animation.FuncAnimation(
        fig,
        update_fn,
        frames=num_frames,
        init_func=init_fn,
        blit=True,
        interval=int(1000 / cfg.fps),
    )

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
