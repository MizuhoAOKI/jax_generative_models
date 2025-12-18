from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from PIL import Image
from sklearn.datasets import fetch_openml, make_moons, make_swiss_roll

# Initialize module-level logger
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Type Definitions
# --------------------------------------------------------


class DataType(str, Enum):
    """Defines the semantic type of the dataset."""

    POINT_2D = "point_2d"
    """2D Point cloud data (2,)."""

    IMAGE = "image"
    """Image data (C, H, W)."""


class ConditionType(str, Enum):
    """Defines the type of conditioning used by the dataset."""

    NONE = "none"
    """No conditioning."""

    DISCRETE = "discrete"
    """Discrete class labels (e.g., MNIST digits)."""

    CONTINUOUS = "continuous"
    """Continuous vectors (e.g., robotics goal states)."""


class Batch(eqx.Module):
    """Standard data container for generative tasks.

    Inheriting from eqx.Module automatically registers it as a JAX Pytree,
    allowing it to be passed into JIT-compiled functions.
    It functions as a frozen dataclass by default.
    """

    x: jax.Array
    """Main data tensor (e.g., images or points)."""

    cond: jax.Array | None = None
    """Conditioning data (e.g., class labels or context vectors)."""


# --------------------------------------------------------
# Configuration Definitions
# --------------------------------------------------------


@dataclass(frozen=True)
class GaussianMixtureConfig:
    """Configuration for the Gaussian Mixture dataset."""

    name: Literal["gaussian_mixture"] = "gaussian_mixture"
    """The dataset identifier."""

    data_dim: int = 2
    """Dimensionality of the data."""

    scale: float = 1.0
    """Scaling factor for the distance of cluster centers from the origin."""

    noise_std: float = 0.5
    """Standard deviation of the Gaussian noise around each center."""

    @property
    def data_type(self) -> DataType:
        return DataType.POINT_2D

    @property
    def condition_type(self) -> ConditionType:
        return ConditionType.NONE

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)


@dataclass(frozen=True)
class CatConfig:
    """Configuration for the Cat Image Point Cloud dataset."""

    name: Literal["cat"] = "cat"
    """The dataset identifier."""

    data_dim: int = 2
    """Dimensionality of the data."""

    scale: float = 2.0
    """Global scaling factor for the point cloud coordinates."""

    noise_scale: float = 0.005
    """Standard deviation of noise added to the points (jitter)."""

    @property
    def data_type(self) -> DataType:
        return DataType.POINT_2D

    @property
    def condition_type(self) -> ConditionType:
        return ConditionType.NONE

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)


@dataclass(frozen=True)
class MoonConfig:
    """Configuration for the Sklearn 'Make Moons' dataset."""

    name: Literal["moon"] = "moon"
    """The dataset identifier."""

    data_dim: int = 2
    """Dimensionality of the data."""

    noise: float = 0.05
    """Standard deviation of Gaussian noise added to the data (sklearn param)."""

    scale: float = 1.0
    """Rescaling factor to fit the data within approximately [-scale, scale]."""

    @property
    def data_type(self) -> DataType:
        return DataType.POINT_2D

    @property
    def condition_type(self) -> ConditionType:
        return ConditionType.NONE

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)


@dataclass(frozen=True)
class SwissRollConfig:
    """Configuration for the Sklearn 'Make Swiss Roll' dataset."""

    name: Literal["swiss_roll"] = "swiss_roll"
    """The dataset identifier."""

    data_dim: int = 2
    """Dimensionality of the data."""

    noise: float = 0.05
    """Standard deviation of Gaussian noise added to the data (sklearn param)."""

    scale: float = 0.15
    """Rescaling factor to fit the data within approximately [-scale, scale]."""

    @property
    def data_type(self) -> DataType:
        return DataType.POINT_2D

    @property
    def condition_type(self) -> ConditionType:
        return ConditionType.NONE

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)


@dataclass(frozen=True)
class MnistConfig:
    """Configuration for the MNIST dataset."""

    name: Literal["mnist"] = "mnist"
    """The dataset identifier."""

    data_dim: int = 784
    """Dimensionality of the data (28x28 flattened)."""

    scale: float = 1.0
    """Scaling factor applied after normalizing pixel values to [0, 1]."""

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE

    @property
    def condition_type(self) -> ConditionType:
        return ConditionType.DISCRETE

    @property
    def num_classes(self) -> int:
        """Number of classes (digits 0-9)."""
        return 10

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, 28, 28)


# Union type for Tyro CLI parsing.
# This enables automatic subcommand generation (e.g., `python main.py --dataset:cat ...`).
DatasetConfig = Union[
    Annotated[GaussianMixtureConfig, tyro.conf.subcommand("gaussian-mixture")],
    Annotated[CatConfig, tyro.conf.subcommand("cat")],
    Annotated[MoonConfig, tyro.conf.subcommand("moon")],
    Annotated[SwissRollConfig, tyro.conf.subcommand("swiss-roll")],
    Annotated[MnistConfig, tyro.conf.subcommand("mnist")],
]


# --------------------------------------------------------
# Asset Loading & Preprocessing
# --------------------------------------------------------


def get_repo_root() -> Path:
    """Locates the repository root directory by searching for 'pyproject.toml'.

    Returns:
        The absolute path to the repository root.

    Raises:
        FileNotFoundError: If 'pyproject.toml' is not found in any parent directory.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find repository root (pyproject.toml).")


_REPO_ROOT = get_repo_root()
_CAT_ASSET_PATH = _REPO_ROOT / "assets" / "cat.png"


@lru_cache(maxsize=1)
def _load_cat_geometry_cached(scale: float) -> jax.Array:
    """Loads the cat image, converts it to a 2D point cloud, and caches the result.

    This function performs I/O and preprocessing (black pixel extraction,
    normalization, and scaling). It uses `lru_cache` to ensure this expensive
    operation happens only once per execution.

    Args:
        scale: The scaling factor to apply to the normalized coordinates.

    Returns:
        A JAX array of shape (N, 2) containing the point cloud coordinates.

    Raises:
        FileNotFoundError: If the cat image asset is missing.
    """
    if not _CAT_ASSET_PATH.exists():
        logger.error(f"Cat asset missing at: {_CAT_ASSET_PATH}")
        raise FileNotFoundError(f"Cat asset not found at {_CAT_ASSET_PATH}")

    logger.debug(f"Loading cat geometry from: {_CAT_ASSET_PATH}")
    img = Image.open(_CAT_ASSET_PATH).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Extract coordinates of dark pixels (threshold < 128)
    mask = arr < 128
    ys, xs = np.nonzero(mask)

    h, w = arr.shape
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    # Normalize coordinates to [-1, 1]
    # Invert Y to match Cartesian coordinates (image origin is top-left)
    xs = (xs / (w - 1.0)) * 2.0 - 1.0
    ys = (h - 1.0 - ys) / (h - 1.0) * 2.0 - 1.0

    coords = np.stack([xs, ys], axis=-1)

    # Apply user scale
    coords = coords * scale

    return jnp.array(coords, dtype=jnp.float32)


@lru_cache(maxsize=1)
def _load_mnist_cached() -> tuple[jax.Array, jax.Array]:
    """Loads the MNIST dataset using sklearn and caches the result in memory.

    Note:
        On the first run, this will download the dataset to ~/scikit_learn_data.
        This uses `fetch_openml` to avoid adding heavy dependencies like torch/tf.

    Returns:
        A tuple of (data, targets).
        data: shape (70000, 784), dtype float32
        targets: shape (70000,), dtype int32
    """
    logger.debug("Loading MNIST dataset (this may take a while on first run)...")
    # as_frame=False ensures we get a numpy array instead of pandas DataFrame
    # fetch_openml returns targets as object (string) array by default.
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")

    # Explicitly convert the object array to int32 numpy array BEFORE creating JAX array.
    y_int = np.asarray(y, dtype=np.int32)

    return jnp.array(X, dtype=jnp.float32), jnp.array(y_int)


# --------------------------------------------------------
# Batch Generators
# --------------------------------------------------------


def _get_gaussian_mixture_batch(
    key: jax.Array, config: GaussianMixtureConfig, batch_size: int
) -> Batch:
    """Generates a batch from a 4-component Gaussian Mixture Model.

    Args:
        key: JAX PRNGKey.
        config: Gaussian mixture configuration.
        batch_size: Number of samples to generate.

    Returns:
        Sampled batch of shape (batch_size, 2).
    """
    centers = jnp.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * config.scale

    k1, k2 = jax.random.split(key)

    # Randomly select a center for each sample
    idx = jax.random.randint(k1, (batch_size,), 0, centers.shape[0])
    base = centers[idx]

    # Add Gaussian noise
    noise = jax.random.normal(k2, (batch_size, 2)) * config.noise_std
    return Batch(x=base + noise)


def _get_cat_batch(key: jax.Array, config: CatConfig, batch_size: int) -> Batch:
    """Generates a batch of points sampled from the cat image geometry.

    Args:
        key: JAX PRNGKey.
        config: Cat dataset configuration.
        batch_size: Number of samples to generate.

    Returns:
        Sampled batch of shape (batch_size, 2).
    """
    # Retrieve cached geometry (creates it on first call)
    pts = _load_cat_geometry_cached(config.scale)

    k1, k2 = jax.random.split(key)

    # Uniformly sample points from the fixed geometry
    idx = jax.random.randint(k1, (batch_size,), 0, pts.shape[0])
    base = pts[idx]

    # Add small Gaussian noise (jitter) to avoid discrete grid artifacts
    noise = jax.random.normal(k2, base.shape) * config.noise_scale
    return Batch(x=base + noise)


def _get_moon_batch(key: jax.Array, config: MoonConfig, batch_size: int) -> Batch:
    """Generates a batch from the 'Two Moons' dataset using scikit-learn.

    Note:
        `sklearn.datasets.make_moons` is CPU-bound and Numpy-based.
        This function handles the necessary synchronization between JAX and CPU.

    Args:
        key: JAX PRNGKey.
        config: Moon dataset configuration.
        batch_size: Number of samples to generate.

    Returns:
        Sampled batch of shape (batch_size, 2).
    """
    # Convert JAX key to a standard integer seed for sklearn (CPU)
    seed = jax.random.randint(key, (), 0, 2**30).item()

    # Generate data on CPU
    x_np, _ = make_moons(n_samples=batch_size, noise=config.noise, random_state=seed)

    # Center and scale
    # make_moons usually ranges roughly [0, 2] on x-axis; shift to center.
    x_np = x_np - 0.5
    x_np = x_np * config.scale

    return Batch(x=jnp.array(x_np, dtype=jnp.float32))


def _get_swiss_roll_batch(key: jax.Array, config: SwissRollConfig, batch_size: int) -> Batch:
    """Generates a batch from the 'Swiss Roll' dataset using scikit-learn.

    Note:
        `sklearn.datasets.make_swiss_roll` is CPU-bound and Numpy-based.
        This function handles the necessary synchronization between JAX and CPU.

    Args:
        key: JAX PRNGKey.
        config: Swiss Roll dataset configuration.
        batch_size: Number of samples to generate.

    Returns:
        Sampled batch of shape (batch_size, 2).
    """
    # Convert JAX key to a standard integer seed for sklearn (CPU)
    seed = jax.random.randint(key, (), 0, 2**30).item()

    # Generate data on CPU
    x_np, _ = make_swiss_roll(n_samples=batch_size, noise=config.noise, random_state=seed)

    # We only take the first two dimensions to get a 2D dataset
    x_np = x_np[:, [0, 2]]

    # Center and scale
    x_np = x_np - jnp.mean(x_np, axis=0)
    x_np = x_np * config.scale

    return Batch(x=jnp.array(x_np, dtype=jnp.float32))


def _get_mnist_batch(key: jax.Array, config: MnistConfig, batch_size: int) -> Batch:
    """Generates a batch from the MNIST dataset.

    Args:
        key: JAX PRNGKey.
        config: MNIST dataset configuration.
        batch_size: Number of samples to generate.

    Returns:
        Sampled batch of shape (batch_size, 784).
    """
    # Retrieve cached full dataset
    data, targets = _load_mnist_cached()

    # Sample random indices
    idx = jax.random.randint(key, (batch_size,), 0, data.shape[0])
    batch_x = data[idx]
    batch_y = targets[idx]

    # Normalize to [0, 1] and apply scale
    batch_x = (batch_x / 255.0) * config.scale

    return Batch(x=batch_x, cond=batch_y)


# --------------------------------------------------------
# Public Dispatcher & Utilities
# --------------------------------------------------------


def get_batch(key: jax.Array, config: DatasetConfig, batch_size: int) -> Batch:
    """Dispatches generation to the correct specific function based on config type.

    Args:
        key: JAX PRNGKey for reproducibility.
        config: The dataset configuration object (Polymorphic).
        batch_size: Number of samples to generate.

    Returns:
        A Batch object containing the data and optional conditions.

    Raises:
        ValueError: If the provided configuration type is unknown.
    """
    match config:
        case GaussianMixtureConfig():
            return _get_gaussian_mixture_batch(key, config, batch_size)
        case CatConfig():
            return _get_cat_batch(key, config, batch_size)
        case MoonConfig():
            return _get_moon_batch(key, config, batch_size)
        case SwissRollConfig():
            return _get_swiss_roll_batch(key, config, batch_size)
        case MnistConfig():
            return _get_mnist_batch(key, config, batch_size)
        case _:
            raise ValueError(f"Unknown config type: {type(config)}")


def get_num_classes(config: DatasetConfig) -> int | None:
    """Retrieves the number of classes based on the dataset conditioning type.

    This explicitly uses the config type (match/case) to provide type-safe access
    to `num_classes`, satisfying static analysis tools like Mypy.

    Args:
        config: The dataset configuration.

    Returns:
        The number of classes if the dataset is DISCRETE, otherwise None.
    """
    match config:
        case MnistConfig():
            # MnistConfig is explicitly defined to have num_classes
            return config.num_classes
        case _:
            # Other configs do not have num_classes
            return None


def apply_cfg_dropout(
    batch: Batch, key: jax.Array, dropout_prob: float, config: DatasetConfig
) -> Batch:
    """Applies Classifier-Free Guidance (CFG) dropout logic based on dataset type.

    Encapsulates the logic for 'dropping' a condition. For discrete data, this
    usually means replacing the label with a 'null token' index. For continuous data,
    this usually means masking the vector with zeros.

    Args:
        batch: The input data batch.
        key: JAX PRNGKey for mask generation.
        dropout_prob: Probability of dropping the condition (0.0 to 1.0).
        config: The dataset configuration (determines the condition type).

    Returns:
        A new Batch object with conditions potentially masked.
    """
    # 1. Early exit if no conditions or no dropout
    if config.condition_type == ConditionType.NONE or batch.cond is None or dropout_prob <= 0.0:
        return batch

    # 2. Generate dropout mask
    # True = Drop (Replace with Null), False = Keep
    B = batch.x.shape[0]
    mask = jax.random.bernoulli(key, p=dropout_prob, shape=(B,))

    new_cond = batch.cond

    # 3. Apply mask based on condition type (Explicit Usage of Enum)
    match config.condition_type:
        case ConditionType.DISCRETE:
            num_classes = get_num_classes(config)
            if num_classes is not None:
                new_cond = jnp.where(mask, num_classes, batch.cond)
            else:
                logger.warning(
                    f"Dataset {config.name} is DISCRETE but missing 'num_classes'. CFG skipped."
                )

        case ConditionType.CONTINUOUS:
            # For continuous vectors, replace with zeros (Zero Masking)
            # Reshape mask to broadcast: (B,) -> (B, 1, ...)
            mask_reshaped = mask.reshape((B,) + (1,) * (batch.cond.ndim - 1))
            new_cond = jnp.where(mask_reshaped, jnp.zeros_like(batch.cond), batch.cond)

        case _:
            pass

    return dataclasses.replace(batch, cond=new_cond)
