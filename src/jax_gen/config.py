from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jax_gen import data, optimizers, visualizer
from jax_gen.models import MLPConfig, ModelConfig
from jax_gen.strategies import StrategyConfig
from jax_gen.strategies.ddpm import DDPMStrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class CommonConfig:
    """Common configuration shared by all modes."""

    seed: int = 123
    """Random seed for reproducibility."""

    strategy: StrategyConfig = field(default_factory=DDPMStrategyConfig)
    """Generative modeling strategy configuration."""

    model: ModelConfig = field(default_factory=MLPConfig)
    """Model configuration."""

    model_path: Path = Path("outputs/model.eqx")
    """Path to save or load the model."""

    dataset: data.DatasetConfig = field(default_factory=data.CatConfig)
    """Dataset configuration."""

    vis: visualizer.VisConfig = field(default_factory=visualizer.VisConfig)
    """Visualization configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""


@dataclass
class TrainConfig(CommonConfig):
    """Configuration for model training mode."""

    mode: Literal["train"] = "train"
    """This is the training mode."""

    batch_size: int = 1024
    """Batch size for training the model."""

    train_steps: int = 30000
    """Number of training steps."""

    optimizer: optimizers.OptimizerConfig = field(default_factory=optimizers.AdamConfig)
    """Optimizer configuration."""

    log_interval: int = 100
    """Interval (in steps) for logging training progress."""


@dataclass
class GenerateConfig(CommonConfig):
    """Configuration for data generation mode."""

    mode: Literal["generate"] = "generate"
    """This is the generation mode."""

    num_samples: int = 2000
    """Number of samples to generate."""

    output_image_path: Path = Path("outputs/generated.png")
    """Path to save the generated output image."""


@dataclass
class AnimateConfig(CommonConfig):
    """Configuration for generating a video of the generative process."""

    mode: Literal["animate"] = "animate"
    """Execution mode literal (fixed to 'animate')."""

    num_samples: int = 1000
    """Number of samples to visualize in the animation."""

    output_video_path: Path = Path("outputs/animation.mp4")
    """Filesystem path to save the resulting video (.mp4 or .gif)."""

    fps: int = 30
    """Frames per second for the video (determines playback speed)."""
