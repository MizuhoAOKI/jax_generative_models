from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Protocol, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optuna
import tyro

from jax_gen import data  # dataset_cfg の型参照のために追加

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


@dataclass(frozen=True)
class UNetConfig(TunableConfig):
    """Configuration for a U-Net model (optimized for images)."""

    type: Literal["unet"] = "unet"
    """Model type identifier."""

    hidden_dim: int = 64
    """Base number of channels in the first layer."""

    dim_mults: tuple[int, ...] = (1, 2, 4)
    """Channel multipliers for each level of the U-Net."""

    image_channels: int = 1
    """Number of input image channels (1 for MNIST, 3 for RGB)."""

    def suggest_params(self, trial: optuna.Trial) -> UNetConfig:
        """Suggests U-Net hyperparameters using Optuna."""
        return dataclasses.replace(
            self,
            hidden_dim=trial.suggest_categorical("unet.hidden_dim", [16, 32, 48, 64]),
        )


# Union type for Tyro CLI parsing (Subcommands)
ModelConfig = Union[
    Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
    Annotated[ResNetConfig, tyro.conf.subcommand("resnet")],
    Annotated[UNetConfig, tyro.conf.subcommand("unet")],
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
        self.freqs = jnp.exp(
            -math.log(10000) * jnp.arange(0, self.half_dim, dtype=jnp.float32) / self.half_dim
        )

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
# CNN Components for U-Net
# --------------------------------------------------------


class ConvBlock(eqx.Module):
    """Basic Convolutional Block with GroupNorm and SiLU."""

    norm: eqx.nn.GroupNorm
    conv: eqx.nn.Conv2d
    act: Callable = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, key: jax.Array):
        # Dynamic GroupNorm Configuration for stability with small channels
        default_groups = 32
        if in_channels < default_groups or in_channels % default_groups != 0:
            groups = 4 if in_channels % 4 == 0 else 1
        else:
            groups = default_groups

        self.norm = eqx.nn.GroupNorm(groups, in_channels)
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=key)
        # Assign here to prevent method binding issues with dataclass defaults
        self.act = jax.nn.silu

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (C, H, W)
        return self.conv(self.act(self.norm(x)))


class ResnetBlockConv(eqx.Module):
    """Residual Block for U-Net with Time Embedding injection."""

    block1: ConvBlock
    block2: ConvBlock
    time_proj: eqx.nn.Linear
    residual_conv: eqx.nn.Conv2d | None

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.block1 = ConvBlock(in_channels, out_channels, key=k1)
        self.block2 = ConvBlock(out_channels, out_channels, key=k2)

        # Projection for time embedding to match output channels
        self.time_proj = eqx.nn.Linear(time_emb_dim, out_channels, key=k3)

        # 1x1 conv if channel counts change, otherwise Identity
        if in_channels != out_channels:
            self.residual_conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=k4)
        else:
            self.residual_conv = None

    def __call__(self, x: jax.Array, t_emb: jax.Array) -> jax.Array:
        # 1. First convolution
        h = self.block1(x)

        # 2. Add time embedding (Project -> Reshape for broadcasting)
        # t_emb: (time_dim,) -> (out_channels,) -> (out_channels, 1, 1)
        t_proj = jax.nn.silu(self.time_proj(t_emb))[..., None, None]
        h = h + t_proj

        # 3. Second convolution
        h = self.block2(h)

        # 4. Residual connection
        res = x if self.residual_conv is None else self.residual_conv(x)
        return h + res


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
    # Mark as static to prevent Equinox from treating it as a trainable leaf
    activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

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

    def __call__(self, t: jax.Array, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
        """Forward pass.

        Args:
            t: Scalar time input.
            x: Input data vector of shape (data_dim,).
            cond: Optional conditioning (unused in this model).

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
    # Mark as static to prevent Equinox from treating it as a trainable leaf
    act: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
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

    def __call__(self, t: jax.Array, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
        """Forward pass.

        Args:
            t: Scalar time input.
            x: Input data vector of shape (data_dim,).
            cond: Optional conditioning (unused in this model).

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


class UNet(eqx.Module):
    """U-Net Architecture adapted for flat input vectors (wraps CNN).

    This model automatically reshapes the input vector (D,) -> (C, H, W) based on
    the assumed image size, processes it with a U-Net, and flattens it back.
    """

    input_conv: eqx.nn.Conv2d
    time_embed: SinusoidalTimeEmbed
    time_proj: eqx.nn.Linear
    label_embed: eqx.nn.Embedding | None

    downs: list[tuple[ResnetBlockConv, eqx.nn.Conv2d]]
    mid_block: ResnetBlockConv
    ups: list[tuple[ResnetBlockConv, eqx.nn.ConvTranspose2d]]

    output_conv: eqx.nn.Conv2d

    # Fields marked as static are not saved in the Pytree leaves.
    img_size: int = eqx.field(static=True)
    img_channels: int = eqx.field(static=True)

    def __init__(self, config: UNetConfig, dataset_cfg: data.DatasetConfig, key: jax.Array):
        self.img_channels = config.image_channels
        data_dim = dataset_cfg.data_dim

        # Infer image size from data_dim (assuming square: H*W*C = data_dim)
        self.img_size = int(math.sqrt(data_dim / self.img_channels))
        if self.img_size * self.img_size * self.img_channels != data_dim:
            logger.warning(
                f"Data dim {data_dim} does not match square image "
                f"({self.img_channels}x{self.img_size}x{self.img_size})."
            )

        key, k_in, k_time, k_mid, k_out, k_embed = jax.random.split(key, 6)

        dims = [config.hidden_dim * m for m in config.dim_mults]

        time_dim = config.hidden_dim * 4
        self.time_embed = SinusoidalTimeEmbed(config.hidden_dim)
        self.time_proj = eqx.nn.Linear(config.hidden_dim, time_dim, key=k_time)

        # Conditional Embedding derived from dataset metadata
        if dataset_cfg.condition_type == data.ConditionType.DISCRETE:
            num_classes = data.get_num_classes(dataset_cfg)
            if num_classes is not None:
                # +1 for the Null Token used in CFG
                self.label_embed = eqx.nn.Embedding(
                    num_embeddings=num_classes + 1,
                    embedding_size=time_dim,
                    key=k_embed,
                )
            else:
                self.label_embed = None
        else:
            self.label_embed = None

        # Input Projection
        self.input_conv = eqx.nn.Conv2d(
            self.img_channels, dims[0], kernel_size=3, padding=1, key=k_in
        )

        # Downsample Path
        self.downs = []
        curr_dim = dims[0]
        keys = jax.random.split(key, len(dims))

        for i, dim in enumerate(dims[:-1]):
            # ResBlock + Downsample
            k_res, k_down = jax.random.split(keys[i], 2)
            block = ResnetBlockConv(curr_dim, curr_dim, time_dim, key=k_res)
            # Stride 2 conv for downsampling
            downsample = eqx.nn.Conv2d(
                curr_dim, dims[i + 1], kernel_size=4, stride=2, padding=1, key=k_down
            )
            self.downs.append((block, downsample))
            curr_dim = dims[i + 1]

        # Middle Block
        self.mid_block = ResnetBlockConv(curr_dim, curr_dim, time_dim, key=k_mid)

        # Upsample Path
        self.ups = []
        keys = jax.random.split(keys[-1], len(dims) - 1)

        for i, dim in enumerate(reversed(dims[:-1])):
            skip_dim = dim
            k_res, k_up = jax.random.split(keys[i], 2)

            # Upsample
            upsample = eqx.nn.ConvTranspose2d(
                curr_dim, dim, kernel_size=4, stride=2, padding=1, key=k_up
            )

            # ResBlock takes concatenated input (curr_dim_after_upsample + skip_dim)
            block = ResnetBlockConv(dim + skip_dim, dim, time_dim, key=k_res)

            self.ups.append((block, upsample))
            curr_dim = dim

        # Output Projection
        self.output_conv = eqx.nn.Conv2d(
            curr_dim, self.img_channels, kernel_size=3, padding=1, key=k_out
        )

    def __call__(self, t: jax.Array, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
        # 1. Reshape Flat Vector -> Image (C, H, W)
        x_img = x.reshape(self.img_channels, self.img_size, self.img_size)

        # 2. Time Embedding
        t_emb = jax.nn.silu(self.time_proj(self.time_embed(t)))

        # 3. Add Condition Embedding
        if self.label_embed is not None and cond is not None:
            # cond is expected to be scalar (int) here (inside vmap)
            c_emb = self.label_embed(cond)
            t_emb = t_emb + c_emb

        # 4. Encoder
        h = self.input_conv(x_img)
        skips = []

        for block, downsample in self.downs:
            h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # 5. Bottleneck
        h = self.mid_block(h, t_emb)

        # 6. Decoder
        for block, upsample in self.ups:
            h = upsample(h)
            skip = skips.pop()
            # Concatenate skip connection along channel axis (axis 0)
            h = jnp.concatenate([h, skip], axis=0)
            h = block(h, t_emb)

        # 7. Output & Flatten -> (D,)
        out = self.output_conv(h)
        return out.flatten()


# --------------------------------------------------------
# Factory Function
# --------------------------------------------------------


def create_model(
    config: ModelConfig,
    dataset_cfg: data.DatasetConfig,
    key: jax.Array,
) -> eqx.Module:
    """Factory function to instantiate the correct model based on configuration.

    Args:
        config: The model configuration object.
        dataset_cfg: The dataset configuration, defining dimensionality and conditioning.
        key: JAX PRNGKey for initialization.

    Returns:
        An initialized Equinox Module.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    logger.info(f"Creating model: {config.type} for dataset: {dataset_cfg.name}")

    match config:
        case MLPConfig():
            return MLP(config, dataset_cfg.data_dim, key)

        case ResNetConfig():
            return ResNet(config, dataset_cfg.data_dim, key)

        case UNetConfig():
            return UNet(config, dataset_cfg, key)

        case _:
            raise ValueError(f"Unknown model config: {type(config)}")
