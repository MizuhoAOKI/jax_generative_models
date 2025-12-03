from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jax_gen.strategies import (
    DDPMStrategy,
    DDPMStrategyConfig,
    FlowMatchingStrategy,
    FlowMatchingStrategyConfig,
)
from jax_gen.strategies.base import Strategy

# -----------------------------------------------------------------------------
# Test Fixtures & Utilities
# -----------------------------------------------------------------------------


# 1. Define test configurations explicitly
# List of tuples: (StrategyClass, ConfigInstance)
# This allows adding new strategies without touching the fixture logic.
STRATEGY_TEST_CASES = [
    (
        DDPMStrategy,
        DDPMStrategyConfig(num_transport_steps=10),
    ),
    (
        FlowMatchingStrategy,
        FlowMatchingStrategyConfig(num_transport_steps=10),
    ),
]


@pytest.fixture(
    params=STRATEGY_TEST_CASES,
    ids=lambda x: x[1].name,  # Use config name for readable test IDs (e.g., "ddpm")
)
def strategy(request) -> Strategy:
    """Parametrized fixture that instantiates strategies dynamically.

    No if-statements required. It receives the class and the config directly.
    """
    strategy_cls, config = request.param
    return strategy_cls.from_config(config)


class DummyModel(eqx.Module):
    """Minimal Equinox model for testing strategy interfaces.

    Preserves input shape and allows checking gradient flow.
    """

    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        # Return output with same shape as input to maintain graph connectivity
        # t is scalar, x is (data_dim,)
        return x * (1.0 - t) + 0.1


@pytest.fixture
def data_dim() -> int:
    return 4


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


def test_sample_source_shape(strategy: Strategy, key: jax.Array, batch_size: int, data_dim: int):
    """Verifies the shape and dtype of samples from the source distribution."""
    # Sampling from t=0 (Source)
    samples = strategy.sample_from_source_distribution(key, batch_size, data_dim)

    assert samples.shape == (batch_size, data_dim)
    assert samples.dtype == jnp.float32


def test_forward_boundary_consistency(strategy: Strategy, key: jax.Array, data_dim: int):
    """Checks boundary conditions for the forward process (t=0 vs t=1).

    Under the unified API:
    - t=0.0: Source distribution (Noise)
    - t=1.0: Target distribution (Data)
    """
    x_data = jax.random.normal(key, (data_dim,))

    # Case A: t=0.0 (Source/Noise)
    # The result should be noise-like, distinct from input data
    t_start = jnp.array(0.0)
    x_t0, _ = strategy.forward(t_start, x_data, key)
    assert x_t0.shape == x_data.shape

    # Case B: t=1.0 (Target/Data)
    # This is the generation goal, so it should match the input data roughly
    t_end = jnp.array(1.0)
    x_t1, _ = strategy.forward(t_end, x_data, key)

    assert x_t1.shape == x_data.shape
    # Check consistency within tolerance (DDPM alpha schedule may cause slight deviation)
    assert jnp.allclose(x_t1, x_data, atol=1e-1)


def test_loss_fn_execution(
    strategy: Strategy, dummy_model: DummyModel, key: jax.Array, data_dim: int
):
    """Ensures the loss function executes without errors and returns a scalar."""
    x_data = jax.random.normal(key, (data_dim,))

    # Check JIT compatibility
    loss_fn_jit = jax.jit(strategy.loss_fn)
    loss = loss_fn_jit(dummy_model, x_data, key)

    # Loss must be a 0-dim scalar
    assert loss.shape == ()
    assert not jnp.isnan(loss)
    assert not jnp.isinf(loss)


def test_generation_loop_trajectory(
    strategy: Strategy,
    dummy_model: DummyModel,
    key: jax.Array,
    batch_size: int,
    data_dim: int,
):
    """Verifies the generation loop and trajectory shape.

    Checks if `sample_from_target_distribution` correctly integrates/denoises
    from t=0 to t=1.
    """
    # Execute with JIT
    sample_fn = jax.jit(
        lambda k: strategy.sample_from_target_distribution(dummy_model, k, batch_size, data_dim)
    )
    x_final, x_traj = sample_fn(key)

    # 1. Check final output shape
    assert x_final.shape == (batch_size, data_dim)

    # 2. Check trajectory shape
    # Trajectory includes initial state (t=0), so length is steps + 1
    if hasattr(strategy, "num_transport_steps"):
        expected_steps = strategy.num_transport_steps  # type: ignore
    elif hasattr(strategy, "num_steps"):  # Flow Matching specific
        expected_steps = strategy.num_steps  # type: ignore
    else:
        # Fallback for generic strategy
        expected_steps = x_traj.shape[0] - 1

    assert x_traj.shape == (expected_steps + 1, batch_size, data_dim)

    # 3. Check consistency
    # The last step (t=1) should match x_final
    assert jnp.allclose(x_traj[-1], x_final, atol=1e-5)
    # The first step (t=0) should not contain NaNs
    assert not jnp.isnan(x_traj[0]).any()


def test_vmap_compatibility(
    strategy: Strategy,
    dummy_model: DummyModel,
    key: jax.Array,
    batch_size: int,
    data_dim: int,
):
    """Checks if the loss function is compatible with `jax.vmap` for batched training."""
    x_batch = jax.random.normal(key, (batch_size, data_dim))
    keys = jax.random.split(key, batch_size)

    # loss_fn is designed for single samples, so we vmap it for batches
    batch_loss_fn = jax.vmap(strategy.loss_fn, in_axes=(None, 0, 0))

    losses = batch_loss_fn(dummy_model, x_batch, keys)

    assert losses.shape == (batch_size,)
    assert not jnp.isnan(losses).any()
