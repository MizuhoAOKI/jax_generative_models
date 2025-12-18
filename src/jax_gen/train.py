import logging
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jax_gen import data, models, optimizers, tracker
from jax_gen.config import TrainConfig
from jax_gen.strategies import create_strategy

logger = logging.getLogger(__name__)


def train(cfg: TrainConfig, key: jax.Array) -> None:
    """Executes the training loop for the generative model.

    This function handles:
    1.  Initialization of the model, optimizer, and generation strategy.
    2.  Setup of experiment tracking (Rerun).
    3.  Definition of the loss function and JIT-compiled training step.
    4.  Execution of the main training loop with logging.
    5.  Serialization of the final trained model.

    Args:
        cfg: Configuration object containing training hyperparameters.
        key: JAX PRNGKey for random number generation.
    """

    # -------------------------------------------------------------------------
    # 1. Setup (Model, Optimizer, Strategy)
    # -------------------------------------------------------------------------
    key, init_key = jax.random.split(key)

    # Initialize model and strategy based on config
    # Now passing the full dataset config to ensure interface consistency
    model = models.create_model(cfg.model, cfg.dataset, init_key)
    strategy = create_strategy(cfg.strategy)

    # Partition model parameters (learnable vs. static)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    # Initialize optimizer
    optimizer = optimizers.create_optimizer(cfg.optimizer)
    opt_state = optimizer.init(params)

    # -------------------------------------------------------------------------
    # 2. Tracker Initialization
    # -------------------------------------------------------------------------
    # Initialize the experiment tracker (e.g., Rerun)
    exp_tracker = tracker.RerunTracker(cfg)

    # Log Ground Truth samples once at the beginning for reference
    key, gt_key = jax.random.split(key)
    exp_tracker.log_ground_truth(gt_key)

    # -------------------------------------------------------------------------
    # 3. Define Loss & Step Functions
    # -------------------------------------------------------------------------
    def batch_loss(
        params: eqx.Module, static: eqx.Module, batch: data.Batch, keys: jax.Array
    ) -> jax.Array:
        """Computes the mean loss over a batch of data, handling CFG dropout."""
        model = eqx.combine(params, static)

        # Apply Classifier-Free Guidance (CFG) dropout
        batch_masked = data.apply_cfg_dropout(batch, keys[0], cfg.cond_dropout_rate, cfg.dataset)

        # Vectorize the strategy's loss function over the batch
        loss_per_sample = jax.vmap(strategy.loss_fn, in_axes=(None, 0, 0, 0))(
            model, batch_masked.x, batch_masked.cond, keys
        )
        return jnp.mean(loss_per_sample)

    @eqx.filter_jit
    def train_step(
        params: eqx.Module,
        static: eqx.Module,
        opt_state: optax.OptState,
        batch: data.Batch,
        key: jax.Array,
    ) -> tuple[eqx.Module, eqx.Module, optax.OptState, jax.Array, jax.Array]:
        """Performs a single optimization step."""
        # Split keys for the batch dimension
        key, subkey = jax.random.split(key)
        keys_batch = jax.random.split(subkey, batch.x.shape[0])

        # Compute gradients
        (loss, grads) = eqx.filter_value_and_grad(batch_loss)(params, static, batch, keys_batch)

        # Apply updates using Optax
        updates, opt_state = optimizer.update(grads, opt_state, cast(optax.Params, params))
        params = eqx.apply_updates(params, updates)

        return params, static, opt_state, loss, key

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info(f"Starting training for {cfg.train_steps} steps...")

    for step in range(cfg.train_steps):
        # Manage PRNG keys: separate keys for data sampling, training noise, and visualization
        key, data_key, train_key, vis_key = jax.random.split(key, 4)

        # Load data batch (Returns a data.Batch object)
        batch = data.get_batch(data_key, cfg.dataset, cfg.batch_size)

        # Execute training step
        params, static, opt_state, loss, _ = train_step(params, static, opt_state, batch, train_key)

        # Logging (Console)
        if (step + 1) % cfg.log_interval == 0:
            logger.info(f"step={step + 1:6d} | loss={float(loss):.6f}")

        # Logging (Experiment Tracker)
        exp_tracker.log_step(
            step=step,
            loss=float(loss),
            model=eqx.combine(params, static),
            strategy=strategy,
            key=vis_key,
        )

    # -------------------------------------------------------------------------
    # 5. Save Model
    # -------------------------------------------------------------------------
    final_model = eqx.combine(params, static)

    # Ensure the output directory exists
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize the model
    eqx.tree_serialise_leaves(cfg.model_path, final_model)
    logger.info(f"Model saved successfully to {cfg.model_path}")
