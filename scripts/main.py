import logging
from typing import Annotated, Union, cast

import jax
import tyro

from jax_gen import utils
from jax_gen.animate import animate
from jax_gen.config import AnimateConfig, GenerateConfig, TrainConfig
from jax_gen.generate import generate
from jax_gen.train import train

logger = logging.getLogger(__name__)

# Union type for internal functions (Type Checking)
Config = Union[TrainConfig, GenerateConfig, AnimateConfig]

# Union type for Tyro CLI parsing (Subcommand Definition)
CliConfig = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
    Annotated[GenerateConfig, tyro.conf.subcommand("generate")],
    Annotated[AnimateConfig, tyro.conf.subcommand("animate")],
]


def main(cfg: Config) -> None:
    """Entry point from the command line interface.

    Dispatches the execution to the appropriate module (training, generation,
    or animation) based on the provided configuration.

    Args:
        cfg: The configuration object parsed from CLI.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    # Initialize random key
    key = jax.random.PRNGKey(cfg.seed)

    # Setup logging
    utils.setup_logging(level=cfg.log_level)

    match cfg:
        case TrainConfig():
            train(cfg, key)
        case GenerateConfig():
            generate(cfg, key)
        case AnimateConfig():
            animate(cfg, key)
        case _:
            raise ValueError(f"Unknown config type: {type(cfg)}")


if __name__ == "__main__":
    # Parse command line arguments
    config = cast(
        Config,
        tyro.cli(CliConfig),  # type: ignore[call-overload]
    )
    main(config)
