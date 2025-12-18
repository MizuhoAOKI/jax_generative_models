from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_gen.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowMatchingStrategyConfig:
    """Configuration for Flow Matching strategy."""

    name: Literal["flow_matching"] = "flow_matching"
    """Strategy identifier."""

    num_transport_steps: int = 50
    """Number of integration steps for the ODE solver."""

    base_std: float = 1.0
    """Standard deviation of the source Gaussian distribution."""


@dataclass
class FlowMatchingStrategy(Strategy):
    """Flow Matching strategy in R^d using Optimal Transport Conditional Flow Matching.

    Unified Time Convention:
        t = 0.0 : Source Distribution (Prior / Noise)
        t = 1.0 : Target Distribution (Data)

    Dynamics:
        ODE integration from t=0 to t=1.
        x_t = (1 - t) * x_source + t * x_target

    Attributes:
        num_transport_steps: Number of integration steps.
        base_std: Std of the base Gaussian.
    """

    num_transport_steps: int
    base_std: float

    @classmethod
    def from_config(cls, cfg: FlowMatchingStrategyConfig) -> FlowMatchingStrategy:
        """Initializes the strategy from a configuration object."""
        return cls(
            num_transport_steps=cfg.num_transport_steps,
            base_std=cfg.base_std,
        )

    # --------------------------------------------------------
    # Strategy Interface Implementation
    # --------------------------------------------------------

    def loss_fn(
        self,
        model: eqx.Module,
        x: jax.Array,
        cond: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """Computes the Flow Matching loss for a single sample.

        Args:
            model: The Equinox model.
            x: A single clean data sample (t=1).
            cond: Optional conditioning information.
            key: PRNGKey.

        Returns:
            The scalar MSE loss.
        """
        key_t, key_base = jax.random.split(key, 2)

        # 1. Sample continuous time t ~ Uniform(0, 1)
        t = jax.random.uniform(
            key_t,
            shape=(),
            minval=0.0,
            maxval=1.0,
        )

        # 2. Compute intermediate state x_t and target velocity
        x_t, target_v = self.forward(t, x, key_base)

        # 3. Predict velocity using the model
        v_pred = model(t, x_t, cond)  # type: ignore # model is Callable

        # 4. Calculate Mean Squared Error
        loss = jnp.mean((v_pred - target_v) ** 2)
        return loss

    def forward(
        self,
        t: jax.Array,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Calculates x_t and target vector field (Forward Process).

        Args:
            t: Continuous time scalar.
            x: Target data sample (Data, t=1).
            key: PRNGKey for sampling source noise.

        Returns:
            (x_t, target_v)
        """
        # Sample x_source (Noise, t=0)
        x_source = jax.random.normal(key, shape=x.shape) * self.base_std

        # Linear Interpolation: x_t = (1 - t) * x_source + t * x_target
        x_t = (1.0 - t) * x_source + t * x

        # Target velocity: d/dt(x_t) = x_target - x_source
        target_v = x - x_source
        return x_t, target_v

    def reverse(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
        cond: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """Performs a single ODE integration step: Moves t towards 1 (Data).

        Args:
            model: The trained Equinox model.
            t: Current time.
            x_t: Current state.
            cond: Optional conditioning information.
            key: PRNGKey (unused for deterministic ODE).

        Returns:
            The state at the next time step.
        """
        del key  # Flow Matching reverse step is deterministic

        # Predict velocity
        v_pred = model(t, x_t, cond)  # type: ignore # model is Callable

        # Forward Euler Integration
        dt = 1.0 / float(self.num_transport_steps)
        x_next = x_t + dt * v_pred

        return x_next

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        r"""Samples initial latent data (Source at t=0)."""
        return jax.random.normal(key, (num_samples, data_dim)) * self.base_std

    def sample_from_target_distribution(
        self,
        model: eqx.Module,
        key: jax.Array,
        num_samples: int,
        data_dim: int,
        cond: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Generates samples by solving the probability flow ODE from t=0 to t=1.

        Args:
            model: The trained Equinox model.
            key: PRNGKey.
            num_samples: Number of samples.
            data_dim: Dimensionality.
            cond: Optional conditioning batch.

        Returns:
            (x_final, x_traj)
        """
        # Start from the source distribution x_source
        x_src = self.sample_from_source_distribution(key, num_samples, data_dim)

        # Define time steps range [0, ..., 1]
        dt = 1.0 / self.num_transport_steps
        ts = jnp.linspace(0.0, 1.0 - dt, self.num_transport_steps)

        # Keys are technically not needed for ODE, but kept for interface consistency
        keys = jax.random.split(key, self.num_transport_steps)

        def scan_body(
            x_t: jax.Array, step_inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            """
            JAX scan expects: (carry, xs) -> (new_carry, y)
            In this context: carry = x_t (state), xs = (t, key)
            The returned value x_next is used as both new_carry and y.
            """
            t, current_key = step_inputs

            # Split keys for batch (though unused in FM reverse)
            B = x_t.shape[0]
            batch_keys = jax.random.split(current_key, B)

            # Vectorize the integration step
            # Note: We close over `cond` which is constant for the trajectory
            x_next = jax.vmap(
                self.reverse,
                in_axes=(None, None, 0, 0, 0),
            )(model, t, x_t, cond, batch_keys)

            return x_next, x_next

        # Run the ODE solver loop
        x_final, x_traj = jax.lax.scan(scan_body, x_src, (ts, keys))

        # Prepend initial state to trajectory
        x_traj_full = jnp.concatenate([x_src[None, ...], x_traj], axis=0)

        return x_final, x_traj_full
