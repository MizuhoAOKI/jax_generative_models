from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from PIL import Image
from sklearn.datasets import make_moons

# Initialize module-level logger
logger = logging.getLogger(__name__)

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


# Union type for Tyro CLI parsing.
# This enables automatic subcommand generation (e.g., `python main.py --dataset:cat ...`).
DatasetConfig = Union[
    Annotated[GaussianMixtureConfig, tyro.conf.subcommand("gaussian-mixture")],
    Annotated[CatConfig, tyro.conf.subcommand("cat")],
    Annotated[MoonConfig, tyro.conf.subcommand("moon")],
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


# --------------------------------------------------------
# Batch Generators
# --------------------------------------------------------


def _get_gaussian_mixture_batch(
    key: jax.Array, config: GaussianMixtureConfig, batch_size: int
) -> jax.Array:
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
    return base + noise


def _get_cat_batch(key: jax.Array, config: CatConfig, batch_size: int) -> jax.Array:
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
    return base + noise


def _get_moon_batch(key: jax.Array, config: MoonConfig, batch_size: int) -> jax.Array:
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

    return jnp.array(x_np, dtype=jnp.float32)


# --------------------------------------------------------
# Public Dispatcher
# --------------------------------------------------------


def get_batch(key: jax.Array, config: DatasetConfig, batch_size: int) -> jax.Array:
    """Dispatches generation to the correct specific function based on config type.

    Args:
        key: JAX PRNGKey for reproducibility.
        config: The dataset configuration object (Polymorphic).
        batch_size: Number of samples to generate.

    Returns:
        A JAX array containing the data batch with shape (batch_size, data_dim).

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
        case _:
            raise ValueError(f"Unknown config type: {type(config)}")
