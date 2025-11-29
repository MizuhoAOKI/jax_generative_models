import logging
from pathlib import Path

import equinox as eqx
import jax

from jax_gen import models, visualizer
from jax_gen.config import GenerateConfig
from jax_gen.strategies import create_strategy

logger = logging.getLogger(__name__)


def generate(cfg: GenerateConfig, key: jax.Array) -> jax.Array:
    """Executes the generation pipeline using a trained model.

    This function performs the following steps:
    1.  Reconstructs the model architecture based on the configuration.
    2.  Deserializes and loads trained weights from the specified path.
    3.  Initializes the generative strategy (e.g., DDPM or Flow Matching).
    4.  Samples new data points from the target distribution.
    5.  Visualizes and saves the generated results.

    Args:
        cfg: Configuration object for generation (includes model path, sample count, etc.).
        key: JAX PRNGKey for stochastic sampling operations.

    Returns:
        A JAX array containing the generated samples with shape `(num_samples, data_dim)`.

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
    model_init = models.create_model(cfg.model, key, data_dim=cfg.dataset.data_dim)

    # Load weights into the skeleton
    model = eqx.tree_deserialise_leaves(cfg.model_path, model_init)

    # -------------------------------------------------------------------------
    # 2. Prepare Transport Strategy
    # -------------------------------------------------------------------------
    logger.info(f"Initializing generation strategy: {cfg.strategy.name}")
    strategy = create_strategy(cfg.strategy)

    # -------------------------------------------------------------------------
    # 3. Sample from Target Distribution
    # -------------------------------------------------------------------------
    logger.info(f"Generating {cfg.num_samples} samples...")
    key, subkey = jax.random.split(key)

    # Execute the reverse process (e.g., denoising or flow integration)
    x_target, _ = strategy.sample_from_target_distribution(
        model=model,
        key=subkey,
        num_samples=cfg.num_samples,
        data_dim=cfg.dataset.data_dim,
    )

    # -------------------------------------------------------------------------
    # 4. Visualization & Post-processing
    # -------------------------------------------------------------------------
    save_path = Path(cfg.output_image_path)
    logger.info(f"Saving generated visualization to {save_path}...")

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    visualizer.visualize_dataset_batch(x_target, cfg.dataset, cfg.vis, save_path=save_path)

    logger.info("Generation completed successfully.")
    return x_target
