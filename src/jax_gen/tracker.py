# src/jax_gen/tracker.py

from __future__ import annotations

import logging

import equinox as eqx
import jax
import numpy as np
import rerun as rr

from jax_gen import data
from jax_gen.config import TrainConfig
from jax_gen.strategies import Strategy

logger = logging.getLogger(__name__)


class RerunTracker:
    """Manages experiment tracking and visualization using Rerun.

    This class encapsulates all interactions with the Rerun SDK, including
    initialization, metric logging, and converting JAX arrays into Rerun-compatible
    visualizations (e.g., point clouds).

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
        rr.set_time_sequence("train_step", step)
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

        Handles data conversion (JAX -> Numpy) and coordinate system adjustments
        (e.g., flipping Y-axis for correct visual orientation).

        Args:
            name: The entity path in Rerun (e.g. "train/samples").
            batch: The data batch to log.
            step: The current training step.
            colors: RGB list [r, g, b]. Defaults to grey if None.
        """
        if colors is None:
            colors = [100, 100, 100]

        rr.set_time_sequence("train_step", step)

        batch_np = np.array(batch)
        data_dim = self.cfg.dataset.data_dim

        # --- Case A: 2D Point Cloud ---
        if data_dim == 2:
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
        else:
            # Fallback for unsupported dimensions
            logger.warning(
                f"No Rerun implementation for data dimension: {data_dim}. Skipping log for {name}."
            )
