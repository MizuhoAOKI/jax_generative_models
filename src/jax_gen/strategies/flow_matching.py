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
    """Strategy identifier (fixed to 'flow_matching')."""

    num_transport_steps: int = 100
    """Number of integration steps for the ODE solver."""

    base_std: float = 1.0
    """Standard deviation of the source Gaussian distribution."""


@dataclass
class FlowMatchingStrategy(Strategy):
    """Flow Matching strategy in R^d using Optimal Transport Conditional Flow Matching.

    This class implements a Flow Matching strategy with a linear interpolation path
    between a base Gaussian distribution and the data distribution.

    Comparison with DDPMStrategy:
        DDPM:
            - State (x_t): Corrupted data via diffusion process (adding noise).
            - Model: Predicts noise (epsilon) or score.
            - Dynamics: Stochastic reverse diffusion process ($x_T \to x_0$).
            - Loss: MSE between predicted noise and added noise.

        Flow Matching (this class):
            - State (x_t): Linear interpolation between source ($x_0$) and target ($x_1$).
            - Model: Predicts the velocity field $v(t, x_t)$.
            - Dynamics: Deterministic ODE integration ($x_0 \to x_1$).
            - Loss: MSE between predicted velocity and the vector field generating the path.

    Attributes:
        num_transport_steps: Number of integration steps for the ODE solver.
        base_std: Standard deviation of the base Gaussian distribution ($x_0$).
    """

    num_transport_steps: int
    base_std: float

    @classmethod
    def from_config(cls, cfg: FlowMatchingStrategyConfig) -> FlowMatchingStrategy:
        """Initializes the strategy from a configuration object.

        Args:
            cfg: The Flow Matching configuration object.

        Returns:
            An initialized FlowMatchingStrategy instance.
        """
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
        x_1: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Computes the Flow Matching loss for a single sample.

        The loss is defined based on the Conditional Flow Matching objective:
        $$ L_{CFM} = \\mathbb{E}_{t, x_1, x_0} [ \\| v_\theta(t, x_t) - (x_1 - x_0) \\|^2 ] $$

        Process:
            1. Sample time $t \\sim U(0, 1)$.
            2. Sample source noise $x_0 \\sim \\mathcal{N}(0, \\sigma^2 I)$.
            3. Compute interpolated state $x_t = (1 - t) x_0 + t x_1$.
            4. Compute target velocity $u_t(x_1 | x_0) = x_1 - x_0$.
            5. Predict velocity $v_\theta(t, x_t)$ using the model.
            6. Compute MSE loss.

        Args:
            model: The Equinox model to train.
            x_1: A single data sample from the target distribution (batch dim 1).
            key: PRNGKey for sampling time and source noise.

        Returns:
            The scalar MSE loss.
        """
        key_t, key_base = jax.random.split(key, 2)

        # 1. Sample continuous time t ~ Uniform(0, 1)
        t = jax.random.uniform(
            key_t,
            shape=(),  # scalar
            minval=0.0,
            maxval=1.0,
        )

        # 2. Sample x_0 from the base distribution
        x_0 = jax.random.normal(key_base, shape=x_1.shape) * self.base_std

        # 3. Compute intermediate state x_t via linear interpolation
        x_t = (1.0 - t) * x_0 + t * x_1

        # 4. Compute true velocity field (constant derivative for linear path)
        target_v = x_1 - x_0

        # 5. Predict velocity using the model
        v_pred = self.predict_velocity(model, t, x_t)

        # 6. Calculate Mean Squared Error
        loss = jnp.mean((v_pred - target_v) ** 2)
        return loss

    def predict_velocity(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
    ) -> jax.Array:
        """Wrapper to call the model to predict velocity.

        Semantically equivalent to `predict_noise` in DDPM but represents the
        time derivative of the flow.

        Args:
            model: The Equinox model.
            t: Continuous time variable.
            x_t: State at time t.

        Returns:
            Predicted velocity vector $v_\theta(t, x_t)$.
        """
        return model(t, x_t)  # type: ignore # model is Callable

    def forward(
        self,
        t: jax.Array,
        x_1: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Generates an interpolated sample and its target velocity.

        While `loss_fn` is sufficient for training, this utility exposes the
        underlying linear interpolation logic:
        $$ x_t = (1 - t) x_0 + t x_1 $$
        $$ v^* = x_1 - x_0 $$

        Args:
            t: Continuous time scalar.
            x_1: Target data sample.
            key: PRNGKey for sampling source noise $x_0$.

        Returns:
            A tuple containing:
                - x_t: The interpolated state.
                - target_v: The target velocity vector.
        """
        x_0 = jax.random.normal(key, shape=x_1.shape) * self.base_std
        x_t = (1.0 - t) * x_0 + t * x_1
        target_v = x_1 - x_0
        return x_t, target_v

    def backward(
        self,
        model: eqx.Module,
        t_idx: jax.Array,
        x_t: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Performs a single ODE integration step for generation.

        Solves the ODE $dx/dt = v_\theta(t, x)$ using the Forward Euler method.

        Contrast with DDPM:
            - DDPM: $x_t \to x_{t-1}$ (Stochastic denoising).
            - Flow Matching: $x_t \to x_{t+1}$ (Deterministic flow integration).

        Note: To maintain interface consistency with `Strategy`, this method accepts
        `t_idx` (integer step) but converts it to continuous time for the model.

        Args:
            model: The trained Equinox model.
            t_idx: Current integer time step index.
            x_t: Current state at time corresponding to `t_idx`.
            key: PRNGKey (unused here as Flow Matching sampling is deterministic).

        Returns:
            The state at the next time step $x_{t+1}$.
        """
        del key  # Flow Matching backward step is deterministic

        # Convert discrete index t_idx to continuous time t in range (0, 1]
        # Using midpoint or start point depending on integration scheme preference.
        # Here we use midpoint for the velocity query.
        t = (t_idx.astype(jnp.float32) + 0.5) / float(self.num_transport_steps)

        # Predict velocity
        v_pred = self.predict_velocity(model, t, x_t)

        # Forward Euler Integration
        dt = 1.0 / float(self.num_transport_steps)
        x_next = x_t + dt * v_pred

        return x_next

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        r"""Samples initial latent data $x_0 \sim \mathcal{N}(0, \sigma^2 I)$."""
        return jax.random.normal(key, (num_samples, data_dim)) * self.base_std

    def sample_from_target_distribution(
        self, model: eqx.Module, key: jax.Array, num_samples: int, data_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        """Generates samples by solving the probability flow ODE.

        Integrates from $t=0$ (source) to $t=1$ (target).

        Args:
            model: The trained Equinox model.
            key: PRNGKey for initial sampling.
            num_samples: Number of samples to generate.
            data_dim: Dimensionality of the data.

        Returns:
            A tuple containing:
                - x_target: The final generated samples ($x_1$).
                - x_traj: The full trajectory of samples (from $t=0$ to $t=1$).
        """
        # Start from the source distribution x_0
        x_src = self.sample_from_source_distribution(key, num_samples, data_dim)

        def scan_body(
            x_t: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            t, current_key = carry
            # key handling is kept for interface compatibility, though unused in step
            next_key, subkey = jax.random.split(current_key)

            B = x_t.shape[0]
            batch_keys = jax.random.split(subkey, B)

            # Vectorize the integration step
            x_next = jax.vmap(
                self.backward,
                in_axes=(None, None, 0, 0),  # model: None, t: None, x_t: batch, keys: batch
            )(model, t, x_t, batch_keys)

            return x_next, x_next

        # Define time steps range [0, ..., T-1] for integration
        ts = jnp.arange(0, self.num_transport_steps)
        keys = jax.random.split(key, self.num_transport_steps)

        # Run the ODE solver loop
        x_target, x_traj = jax.lax.scan(scan_body, x_src, (ts, keys))
        return x_target, x_traj
