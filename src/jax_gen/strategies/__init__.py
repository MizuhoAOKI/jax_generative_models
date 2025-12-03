"""
Generative Modeling Strategies Package.

This package provides implementations for various generative modeling strategies,
such as Denoising Diffusion Probabilistic Models (DDPM) and Flow Matching.
It includes a factory function `create_strategy` to instantiate strategies
based on configuration objects.
"""

import logging
from typing import Annotated, Union

import tyro

from jax_gen.strategies.base import (
    Strategy,
)
from jax_gen.strategies.ddpm import DDPMStrategy, DDPMStrategyConfig
from jax_gen.strategies.flow_matching import FlowMatchingStrategy, FlowMatchingStrategyConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DDPMStrategy",
    "DDPMStrategyConfig",
    "FlowMatchingStrategy",
    "FlowMatchingStrategyConfig",
    "Strategy",
    "StrategyConfig",
    "create_strategy",
]

# Configuration Union for Strategies.
# This allows internal functions to accept any strategy configuration.
StrategyConfig = Union[
    Annotated[DDPMStrategyConfig, tyro.conf.subcommand("ddpm")],
    Annotated[FlowMatchingStrategyConfig, tyro.conf.subcommand("flow-matching")],
]


def create_strategy(cfg: StrategyConfig) -> Strategy:
    """Factory function to create a Strategy instance from a configuration object.

    Dispatches the configuration to the appropriate concrete strategy class
    (e.g., DDPM or Flow Matching) based on the config type.

    Args:
        cfg: The strategy configuration object (polymorphic).

    Returns:
        An initialized instance of a class implementing the `Strategy` protocol.

    Raises:
        ValueError: If the provided configuration type is not recognized.
    """
    logger.debug(f"Initializing strategy: {cfg.name}")

    match cfg:
        case DDPMStrategyConfig():
            return DDPMStrategy.from_config(cfg)
        case FlowMatchingStrategyConfig():
            return FlowMatchingStrategy.from_config(cfg)
        case _:
            raise ValueError(f"Unknown strategy config type: {type(cfg)}")
