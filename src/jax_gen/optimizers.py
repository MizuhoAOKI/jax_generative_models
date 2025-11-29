from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Literal, Union

import optax
import tyro

# Initialize module-level logger
logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Configuration Definitions
# --------------------------------------------------------


@dataclass(frozen=True)
class AdamConfig:
    """Configuration for the Adam optimizer.

    Adam is an adaptive learning rate optimization algorithm designed specifically
    for training deep neural networks.
    """

    type: Literal["adam"] = "adam"
    """Optimizer identifier (fixed to 'adam')."""

    lr: float = 1e-4
    """The learning rate."""


@dataclass(frozen=True)
class SGDConfig:
    """Configuration for the Stochastic Gradient Descent (SGD) optimizer."""

    type: Literal["sgd"] = "sgd"
    """Optimizer identifier (fixed to 'sgd')."""

    lr: float = 1e-3
    """The learning rate."""


# Union type for Tyro CLI parsing.
# This allows the user to select the optimizer via command line subcommands.
OptimizerConfig = Union[
    Annotated[AdamConfig, tyro.conf.subcommand("adam")],
    Annotated[SGDConfig, tyro.conf.subcommand("sgd")],
]


# --------------------------------------------------------
# Factory Function
# --------------------------------------------------------


def create_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    """Creates an Optax optimizer instance based on the provided configuration.

    Args:
        config: The optimizer configuration object (AdamConfig or SGDConfig).

    Returns:
        An `optax.GradientTransformation` object (a tuple of init and update functions).

    Raises:
        ValueError: If the provided configuration type is unknown.
    """
    logger.debug(f"Creating optimizer: {config.type} with lr={config.lr}")

    match config:
        case AdamConfig():
            return optax.adam(learning_rate=config.lr)
        case SGDConfig():
            return optax.sgd(learning_rate=config.lr)
        case _:
            raise ValueError(f"Unknown optimizer config: {type(config)}")
