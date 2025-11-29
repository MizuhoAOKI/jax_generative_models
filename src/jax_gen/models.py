from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Protocol, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optuna
import tyro

# Initialize module-level logger
logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Config Definitions & Hyperparameter Tuning
# --------------------------------------------------------


class TunableConfig(Protocol):
    """Protocol for configuration classes that support Optuna hyperparameter tuning."""

    def suggest_params(self, trial: optuna.Trial) -> TunableConfig:
        """Suggests hyperparameters for the current trial.

        Args:
            trial: The current Optuna trial object.

        Returns:
            A new instance of the configuration class with suggested parameters.
        """
        ...


@dataclass(frozen=True)
class MLPConfig(TunableConfig):
    """Configuration for a Multi-Layer Perceptron (MLP) model."""

    type: Literal["mlp"] = "mlp"
    """Model type identifier."""

    hidden_dim: int = 512
    """Dimensionality of hidden layers."""

    depth: int = 10
    """Number of layers."""

    activation: Literal["gelu", "relu", "swish"] = "gelu"
    """Activation function name ('gelu', 'relu', 'swish')."""

    def suggest_params(self, trial: optuna.Trial) -> MLPConfig:
        """Suggests MLP hyperparameters using Optuna."""
        return dataclasses.replace(
            self,
            hidden_dim=trial.suggest_categorical("mlp.hidden_dim", [128, 256, 512, 1024]),
            depth=trial.suggest_int("mlp.depth", 2, 8),
            activation=cast(
                Literal["gelu", "relu", "swish"],
                trial.suggest_categorical("mlp.activation", ["gelu", "relu", "swish"]),
            ),
        )


@dataclass(frozen=True)
class ResNetConfig(TunableConfig):
    """Configuration for a Residual Network (ResNet) model."""

    type: Literal["resnet"] = "resnet"
    """Model type identifier."""

    hidden_dim: int = 512
    """Dimensionality of hidden layers."""

    num_blocks: int = 4
    """Number of residual blocks."""

    use_layer_norm: bool = True
    """Whether to apply Layer Normalization."""

    dropout_rate: float = 0.0
    """Dropout probability (currently unused in logic, reserved)."""

    def suggest_params(self, trial: optuna.Trial) -> ResNetConfig:
        """Suggests ResNet hyperparameters using Optuna."""
        return dataclasses.replace(
            self,
            hidden_dim=trial.suggest_categorical("resnet.hidden_dim", [128, 256, 512, 1024]),
            num_blocks=trial.suggest_int("resnet.num_blocks", 2, 8),
            use_layer_norm=trial.suggest_categorical("resnet.use_layer_norm", [True, False]),
        )


# Union type for Tyro CLI parsing (Subcommands)
ModelConfig = Union[
    Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
    Annotated[ResNetConfig, tyro.conf.subcommand("resnet")],
]


# --------------------------------------------------------
# Shared Utilities (Time Embedding & Activation)
# --------------------------------------------------------


def get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Retrieves a JAX activation function by name.

    Args:
        name: The name of the activation function (e.g., 'relu', 'gelu').

    Returns:
        The corresponding function from `jax.nn`.

    Raises:
        ValueError: If the activation name is not found in `jax.nn`.
    """
    if hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    raise ValueError(f"Unknown activation: {name}")


class SinusoidalTimeEmbed(eqx.Module):
    """Standard sinusoidal time embedding for Diffusion/Flow models.

    Projects a scalar time `t` into a high-dimensional vector. This allows the
    network to be conditioned on the continuous time variable, providing high-frequency
    features that help distinguish close time steps.

    Formula:
        PE(t, 2i)   = sin(t / 10000^(2i / d))
        PE(t, 2i+1) = cos(t / 10000^(2i / d))
    """

    dim: int
    half_dim: int
    freqs: jax.Array

    def __init__(self, dim: int):
        """Initializes the embedding module.

        Args:
            dim: The output dimensionality of the embedding.
        """
        self.dim = dim
        self.half_dim = dim // 2
        # Precompute frequencies
        # exp(-log(10000) * i / half_dim) creates a geometric decay of frequencies
        self.freqs = jnp.exp(-jnp.log(10000) * jnp.arange(0, self.half_dim) / self.half_dim)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Computes the embedding for a given time `t`.

        Args:
            t: Scalar time input or shape (1,).

        Returns:
            Embedding vector of shape (dim,).
        """
        args = t * self.freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        # Handle odd dimensions by padding (edge case)
        if self.dim % 2 == 1:
            embedding = jnp.pad(embedding, (0, 1))
        return embedding


# --------------------------------------------------------
# Model Implementations (Equinox Modules)
# --------------------------------------------------------


class MLP(eqx.Module):
    """Vanilla Multi-Layer Perceptron with Time Embedding.

    Concatenates the time embedding to the input coordinates before passing
    through a sequence of linear layers.
    """

    time_embed: SinusoidalTimeEmbed
    layers: list[eqx.nn.Linear]
    activation: Callable[[jax.Array], jax.Array]

    def __init__(self, config: MLPConfig, data_dim: int, key: jax.Array):
        """Initializes the MLP.

        Args:
            config: Configuration object.
            data_dim: Dimensionality of the input data.
            key: JAX PRNGKey.
        """
        self.activation = get_activation(config.activation)

        # Time embedding dimension (heuristic: 1/4 of hidden dim)
        time_dim = config.hidden_dim // 4
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        # Split keys for layer initialization
        keys = jax.random.split(key, config.depth + 1)

        self.layers = []

        # Input Layer: Concatenates data and time embedding
        self.layers.append(eqx.nn.Linear(data_dim + time_dim, config.hidden_dim, key=keys[0]))

        # Hidden Layers
        for i in range(config.depth - 1):
            self.layers.append(eqx.nn.Linear(config.hidden_dim, config.hidden_dim, key=keys[i + 1]))

        # Output Layer: Projects back to data_dim
        self.layers.append(eqx.nn.Linear(config.hidden_dim, data_dim, key=keys[-1]))

    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            t: Scalar time input.
            x: Input data vector of shape (data_dim,).

        Returns:
            Output vector of shape (data_dim,).
        """
        # 1. Embed time
        t_emb = self.time_embed(t)  # Shape: (time_dim,)

        # 2. Concatenate input and time
        h = jnp.concatenate([x, t_emb], axis=-1)

        # 3. Forward pass through hidden layers
        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)

        # 4. Final projection
        return self.layers[-1](h)


class ResNetBlock(eqx.Module):
    """Residual Block: x + Linear(Activation(Linear(Norm(x))))."""

    norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    act: Callable[[jax.Array], jax.Array]
    use_norm: bool

    def __init__(self, dim: int, use_norm: bool, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.use_norm = use_norm
        self.norm = eqx.nn.LayerNorm(dim)
        self.linear1 = eqx.nn.Linear(dim, dim, key=k1)
        self.linear2 = eqx.nn.Linear(dim, dim, key=k2)
        self.act = jax.nn.gelu

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.norm(x) if self.use_norm else x
        h = self.act(self.linear1(h))
        h = self.linear2(h)
        return x + h  # Skip connection


class ResNet(eqx.Module):
    """ResNet-MLP Architecture for Flow Matching / Diffusion.

    Embeds time globally and adds it to the hidden state before processing
    through residual blocks.
    """

    input_proj: eqx.nn.Linear
    time_proj: eqx.nn.Linear
    blocks: list[ResNetBlock]
    output_proj: eqx.nn.Linear
    time_embed: SinusoidalTimeEmbed

    def __init__(self, config: ResNetConfig, data_dim: int, key: jax.Array):
        """Initializes the ResNet.

        Args:
            config: Configuration object.
            data_dim: Dimensionality of the input data.
            key: JAX PRNGKey.
        """
        # Calculate keys required: Input + Time + Blocks + Output
        keys = jax.random.split(key, config.num_blocks + 3)

        # 1. Input Projections
        self.input_proj = eqx.nn.Linear(data_dim, config.hidden_dim, key=keys[0])

        # 2. Time Embedding & Projection
        # Embed time to 64 dimensions first, then project to hidden_dim
        self.time_embed = SinusoidalTimeEmbed(64)
        self.time_proj = eqx.nn.Linear(64, config.hidden_dim, key=keys[1])

        # 3. Residual Blocks
        self.blocks = [
            ResNetBlock(config.hidden_dim, config.use_layer_norm, key=k) for k in keys[2:-1]
        ]

        # 4. Output Projection
        self.output_proj = eqx.nn.Linear(config.hidden_dim, data_dim, key=keys[-1])

    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            t: Scalar time input.
            x: Input data vector of shape (data_dim,).

        Returns:
            Output vector of shape (data_dim,).
        """
        # Process inputs
        h = self.input_proj(x)
        t_emb = self.time_proj(self.time_embed(t))

        # Add global time information to the hidden state
        # Broadcasting handles the addition if necessary
        h = h + t_emb

        # Pass through residual blocks
        for block in self.blocks:
            h = block(h)

        return self.output_proj(h)


# --------------------------------------------------------
# Factory Function
# --------------------------------------------------------


def create_model(config: ModelConfig, key: jax.Array, data_dim: int) -> eqx.Module:
    """Factory function to instantiate the correct model based on configuration.

    Args:
        config: The model configuration object (MLPConfig or ResNetConfig).
        key: JAX PRNGKey for initialization.
        data_dim: The dimensionality of the input data.

    Returns:
        An initialized Equinox Module.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    logger.info(f"Creating model: {config.type}")
    logger.debug(f"Model config: {config}")

    match config:
        case MLPConfig():
            return MLP(config, data_dim, key)
        case ResNetConfig():
            return ResNet(config, data_dim, key)
        case _:
            raise ValueError(f"Unknown model config: {type(config)}")
