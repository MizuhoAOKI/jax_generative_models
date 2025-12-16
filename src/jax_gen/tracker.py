from __future__ import annotations

import logging
import math

import equinox as eqx
import jax
import numpy as np
import rerun as rr

from jax_gen import data
from jax_gen.config import TrainConfig
from jax_gen.data import DataType
from jax_gen.strategies import Strategy

logger = logging.getLogger(__name__)


class RerunTracker:
    """Manages experiment tracking and visualization using Rerun.

    This class encapsulates all interactions with the Rerun SDK, including
    initialization, metric logging, and converting JAX arrays into Rerun-compatible
    visualizations (e.g., point clouds or images).

    Attributes:
        cfg: The training configuration object.
        _enabled: Boolean flag indicating if Rerun logging is active.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        """Initializes the Rerun tracker.

        Sets up the Rerun application, configures the output file (if specified),
        and defines static visualization properties for the logs.

        Args:
            cfg: The training configuration containing visualization settings.
        """
        self.cfg = cfg
        self._enabled = cfg.vis.enable_rerun

        if not self._enabled:
            return

        # 1. Initialize Rerun Application
        # Naming convention: {dataset}_{model}_{strategy}
        exp_name = f"{cfg.dataset.name}_{cfg.model.type}_{cfg.strategy.name}"
        rr.init(application_id=exp_name)

        # 2. Setup Log File (for Headless/Server-side logging)
        # Rerun streams data to this file automatically (rrd format).
        save_path = self.cfg.vis.rerun_log_save_path
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))
            logger.info(f"Rerun logging enabled. Saving .rrd to: {save_path}")

        # 3. Configure Static Visualization Properties
        # Sets the color and legend name for the loss curve ahead of time.
        rr.log(
            "train/loss",
            rr.SeriesLines(
                colors=[3, 175, 122],  # Green
                names="loss",
            ),
            static=True,
        )

    def log_ground_truth(self, key: jax.Array) -> None:
        """Logs a batch of ground truth data samples for reference.

        Typically called once at the beginning of training to establish a baseline
        visual comparison for generated samples.

        Args:
            key: JAX PRNGKey for data sampling.
        """
        if not self._enabled:
            return

        gt_batch = data.get_batch(key, self.cfg.dataset, self.cfg.batch_size)

        self._log_batch(
            name="ground_truth/data_samples",
            batch=gt_batch,
            step=0,
            colors=[255, 75, 0],  # Red
        )

    def log_step(
        self,
        step: int,
        loss: float,
        model: eqx.Module,
        strategy: Strategy,
        key: jax.Array,
    ) -> None:
        """Logs training metrics and conditionally logs generated samples.

        Args:
            step: The current training iteration.
            loss: The scalar loss value.
            model: The current model state.
            strategy: The generation strategy (for sampling).
            key: JAX PRNGKey for sampling.
        """
        if not self._enabled:
            return

        # Log Scalar Metrics
        rr.set_time("train_step", sequence=step)
        rr.log("train/loss", rr.Scalars(loss))

        # Log Visual Samples (at specified intervals)
        if step % self.cfg.vis.sample_log_interval == 0:
            self._log_samples(step, model, strategy, key)

    def _log_samples(
        self,
        step: int,
        model: eqx.Module,
        strategy: Strategy,
        key: jax.Array,
    ) -> None:
        """Internal helper to generate and log samples from the model.

        Args:
            step: Current training step.
            model: Current model state.
            strategy: Strategy used to sample from the target distribution.
            key: JAX PRNGKey.
        """
        samples, _ = strategy.sample_from_target_distribution(
            model=model,
            key=key,
            num_samples=self.cfg.vis.num_vis_samples,
            data_dim=self.cfg.dataset.data_dim,
        )

        self._log_batch(
            name="model_inference/data_samples",
            batch=samples,
            step=step,
            colors=[0, 90, 255],  # Blue
        )

    def _log_batch(
        self,
        name: str,
        batch: jax.Array,
        step: int,
        colors: list[int] | None = None,
    ) -> None:
        """Helper method to log a data batch to Rerun.

        Handles data conversion (JAX -> Numpy) and switches visualization method
        based on the dataset type (Point Cloud vs Image).

        Args:
            name: The entity path in Rerun (e.g. "train/samples").
            batch: The data batch to log. Shape (B, D).
            step: The current training step.
            colors: RGB list [r, g, b]. Used for point clouds.
        """
        if colors is None:
            colors = [100, 100, 100]

        rr.set_time("train_step", sequence=step)

        batch_np = np.array(batch)
        data_type = self.cfg.dataset.data_type

        # --- Case A: 2D Point Cloud ---
        if data_type == DataType.POINT_2D:
            # Create a copy to avoid mutating the original batch
            points = batch_np.copy()

            # Flip Y-axis to match Rerun's coordinate system expectations
            # (Standard Cartesian vs Image coordinate differences)
            points[:, 1] *= -1

            rr.log(
                name,
                rr.Points2D(
                    points,
                    radii=0.015,
                    colors=colors,
                ),
            )

        # --- Case B: Image (e.g. MNIST) ---
        elif data_type == DataType.IMAGE:
            # Retrieve shape from config, e.g. (1, 28, 28)
            # Flattened batch (B, 784) -> Reshaped (B, C, H, W)
            shape = self.cfg.dataset.shape
            batch_reshaped = batch_np.reshape(-1, *shape)

            # Create a grid of images for visualization
            grid_image = self._make_image_grid(batch_reshaped)

            rr.log(
                name,
                rr.Image(grid_image),
            )

        else:
            logger.warning(
                f"No Rerun implementation for data type: {data_type}. Skipping log for {name}."
            )

    def _make_image_grid(self, images: np.ndarray) -> np.ndarray:
        """Combines a batch of images into a single grid image.

        Args:
            images: Batch of images with shape (B, C, H, W).
                    Assumes pixel values are in [0, 1] or roughly in that range.

        Returns:
            A single numpy array (GridH, GridW, C) or (GridH, GridW).
        """
        b, c, h, w = images.shape

        # Calculate grid dimensions (approx square)
        grid_cols = int(math.ceil(math.sqrt(b)))
        grid_rows = int(math.ceil(b / grid_cols))

        # Create empty canvas
        grid = np.zeros((grid_rows * h, grid_cols * w, c), dtype=images.dtype)

        for i in range(b):
            row = i // grid_cols
            col = i % grid_cols

            # Transpose (C, H, W) -> (H, W, C) for assignment
            img = images[i].transpose(1, 2, 0)

            grid[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img

        # If 1 channel (Grayscale), squeeze to (H, W) for Rerun compatibility
        if c == 1:
            grid = grid.squeeze(axis=-1)

        return grid
