import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_gen import data, models, visualizer
from jax_gen.config import GenerateConfig
from jax_gen.strategies import create_strategy

logger = logging.getLogger(__name__)


def get_condition_batch(
    cfg: GenerateConfig, dataset_cfg: data.DatasetConfig, key: jax.Array
) -> jax.Array | None:
    """Creates a batch of conditions based on the user configuration.

    Handles three priority levels:
    1. fixed_condition: Direct value (e.g., class label).
    2. condition_from_idx: Extract condition from dataset index.
    3. condition_from_file: Load condition from an external file.

    Args:
        cfg: Generation configuration.
        dataset_cfg: Dataset configuration.
        key: JAX PRNGKey.

    Returns:
        A JAX array of shape (num_samples, ...) or None if unconditional.
    """
    if cfg.condition is not None:
        logger.info(f"Using fixed condition: {cfg.condition}")
        cond = jnp.array([cfg.condition], dtype=jnp.int32)
        cond = jnp.repeat(cond, cfg.num_samples, axis=0)
        return cond

    elif cfg.condition_from_idx is not None:
        logger.info(f"Using condition from dataset index: {cfg.condition_from_idx}")
        raise NotImplementedError("Loading condition from index is not yet implemented.")

    elif cfg.condition_from_file is not None:
        logger.info(f"Loading condition from file: {cfg.condition_from_file}")
        raise NotImplementedError("Loading condition from file is not yet implemented.")

    return None


def generate(cfg: GenerateConfig, key: jax.Array) -> jax.Array:
    """Executes the generation pipeline using a trained model.

    This function performs the following steps:
    1.  Reconstructs the model architecture based on the configuration.
    2.  Deserializes and loads trained weights from the specified path.
    3.  Initializes the generative strategy (e.g., DDPM or Flow Matching).
    4.  Samples new data points from the target distribution.
    5.  Visualizes and saves the generated results.

    Args:
        cfg: Configuration object for generation.
        key: JAX PRNGKey for stochastic sampling operations.

    Returns:
        A JAX array containing the generated samples.

    Raises:
        FileNotFoundError: If the model file specified in `cfg.model_path` does not exist.
    """
    logger.info(f"Starting generation task. Mode: {cfg.mode}")

    # -------------------------------------------------------------------------
    # 1. Load Trained Model
    # -------------------------------------------------------------------------
    if not cfg.model_path.exists():
        logger.error(f"Model file not found at: {cfg.model_path}")
        raise FileNotFoundError(f"Model file not found at: {cfg.model_path}")

    logger.info(f"Loading model architecture and weights from {cfg.model_path}...")

    # Initialize the model structure (skeleton)
    # Using the same factory call as train.py to ensure PyTree structure matches
    model_init = models.create_model(cfg.model, cfg.dataset, key)

    # Load weights into the skeleton
    model = eqx.tree_deserialise_leaves(cfg.model_path, model_init)

    # -------------------------------------------------------------------------
    # 2. Prepare Transport Strategy & Condition
    # -------------------------------------------------------------------------
    logger.info(f"Initializing generation strategy: {cfg.strategy.name}")
    strategy = create_strategy(cfg.strategy)

    key, cond_key, subkey = jax.random.split(key, 3)

    # Resolve conditioning based on user configuration
    cond_batch = get_condition_batch(cfg, cfg.dataset, cond_key)

    # -------------------------------------------------------------------------
    # 3. Sample from Target Distribution
    # -------------------------------------------------------------------------
    logger.info(f"Generating {cfg.num_samples} samples...")

    x_target, _ = strategy.sample_from_target_distribution(
        model=model,
        key=subkey,
        num_samples=cfg.num_samples,
        data_dim=cfg.dataset.data_dim,
        cond=cond_batch,
    )

    # -------------------------------------------------------------------------
    # 4. Visualization & Post-processing
    # -------------------------------------------------------------------------
    save_path = Path(cfg.output_image_path)
    logger.info(f"Saving generated visualization to {save_path}...")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    visualizer.visualize_dataset_batch(x_target, cfg.dataset, cfg.vis, save_path=save_path)

    logger.info("Generation completed successfully.")
    return x_target
