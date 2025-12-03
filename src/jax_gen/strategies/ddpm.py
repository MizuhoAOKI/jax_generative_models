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
class DDPMStrategyConfig:
    """Configuration for Denoising Diffusion Probabilistic Models (DDPM)."""

    name: Literal["ddpm"] = "ddpm"
    """Strategy identifier."""

    num_transport_steps: int = 50
    """Number of discrete steps in the diffusion process."""

    beta_min: float = 1e-4
    """The lower bound of the noise variance schedule."""

    beta_max: float = 0.02
    """The upper bound of the noise variance schedule."""


@dataclass
class DDPMStrategy(Strategy):
    """A Denoising Diffusion Probabilistic Model (DDPM) strategy.

    Unified Time Convention:
        t = 0.0 : Source Distribution (Prior / Noise)
        t = 1.0 : Target Distribution (Data)

    Internal Logic:
        Standard DDPM defines t=0 as Data and t=T as Noise.
        This class maps the unified input t in [0, 1] to the internal DDPM steps:
        - Input t=0.0 (Noise) -> DDPM step T-1
        - Input t=1.0 (Data)  -> DDPM step 0

    Attributes:
        num_transport_steps: The total number of diffusion steps (T).
        betas: The linear variance schedule beta_t.
        alphas_cumprod: The cumulative product of alphas.
    """

    num_transport_steps: int
    betas: jax.Array
    alphas_cumprod: jax.Array

    @classmethod
    def from_config(cls, cfg: DDPMStrategyConfig) -> DDPMStrategy:
        """Initializes the strategy from a configuration object."""
        betas = jnp.linspace(cfg.beta_min, cfg.beta_max, cfg.num_transport_steps)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas)
        return cls(
            num_transport_steps=cfg.num_transport_steps,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
        )

    # --------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------

    def _get_ddpm_index(self, t: jax.Array) -> jax.Array:
        """Maps unified time t in [0, 1] (0=Noise, 1=Data)
        to DDPM index in [T-1, 0] (T-1=Noise, 0=Data)."""
        # t=0.0 -> idx=T-1 (Pure Noise)
        # t=1.0 -> idx=0   (Pure Data)
        # We clip to ensure indices are valid even if t is slightly out of bounds.
        idx = jnp.floor((1.0 - t) * (self.num_transport_steps - 1)).astype(jnp.int32)
        return jnp.clip(idx, 0, self.num_transport_steps - 1)

    # --------------------------------------------------------
    # Strategy Interface Implementation
    # --------------------------------------------------------

    def loss_fn(
        self,
        model: eqx.Module,
        x: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Computes the DDPM loss for a single sample.

        Args:
            model: The Equinox model to train.
            x: A single clean data sample (t=1).
            key: PRNGKey.

        Returns:
            The scalar MSE loss.
        """
        key_t, key_noise = jax.random.split(key, 2)

        # Sample t uniform [0, 1] (Unified time: 0=Noise, 1=Data)
        t = jax.random.uniform(
            key_t,
            shape=(),
            minval=0.0,
            maxval=1.0,
        )

        # Get x_t (noisy state) and the noise added (target)
        x_t, eps_true = self.forward(t, x, key_noise)

        # Predict noise. The model receives normalized time t.
        # Note: If the model expects specific embeddings, ensure t is consistent.
        eps_pred = model(t, x_t)  # type: ignore # model is Callable

        # Calculate Mean Squared Error
        loss = jnp.mean((eps_pred - eps_true) ** 2)
        return loss

    def forward(
        self,
        t: jax.Array,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Corrupts data to state at time t (Forward Process).

        Maps t (0->1) to DDPM schedule (Noise->Data) inversely.

        Args:
            t: Continuous time in [0, 1]. 0=Noise, 1=Data.
            x: Clean data sample (Data).
            key: PRNGKey for noise injection.

        Returns:
            (x_t, eps)
        """
        # Map t to DDPM index (high index = high noise)
        ddpm_idx = self._get_ddpm_index(t)

        # Retrieve alpha_bar for the corresponding DDPM step
        alpha_bar_t = self.alphas_cumprod[ddpm_idx]

        eps = jax.random.normal(key, shape=x.shape)

        # Reparameterization trick
        # t=0 (Noise) -> alpha_bar ~ 0 -> x_t ~ eps
        # t=1 (Data)  -> alpha_bar ~ 1 -> x_t ~ x
        x_t = jnp.sqrt(alpha_bar_t) * x + jnp.sqrt(1.0 - alpha_bar_t) * eps
        return x_t, eps

    def reverse(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Performs a single generation step: Moves t towards 1 (Data).

        Corresponds to the standard DDPM reverse step x_t -> x_{t-1}.

        Args:
            model: The trained Equinox model.
            t: Current time in [0, 1].
            x_t: Current state.
            key: PRNGKey for stochastic sampling.

        Returns:
            The state at the next time step (closer to Data).
        """
        # Current DDPM index (e.g., T-1). We want to move to T-2.
        curr_idx = self._get_ddpm_index(t)

        # Standard DDPM parameters for this index
        beta_t = self.betas[curr_idx]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = self.alphas_cumprod[curr_idx]

        # Get alpha_bar for the previous DDPM step (closer to data)
        # Note: 'previous' in DDPM means index - 1
        alpha_bar_prev = jnp.where(
            curr_idx > 0,
            self.alphas_cumprod[curr_idx - 1],
            jnp.array(1.0, dtype=alpha_bar_t.dtype),
        )

        # 1. Predict noise
        eps_pred = model(t, x_t)  # type: ignore # model is Callable

        # 2. Calculate mean (mu_theta)
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * eps_pred)

        # 3. Sample next state (add noise if not at the very end)
        # In unified time, we are moving t -> 1.0.
        # The final step is when DDPM index reaches 0.
        def no_noise(_: jax.Array) -> jax.Array:
            return mean

        def add_noise(k: jax.Array) -> jax.Array:
            noise = jax.random.normal(k, shape=x_t.shape)
            sigma_t = jnp.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t)
            return mean + sigma_t * noise

        return jax.lax.cond(curr_idx == 0, no_noise, add_noise, key)

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        r"""Samples initial noise (Source at t=0)."""
        return jax.random.normal(key, (num_samples, data_dim))

    def sample_from_target_distribution(
        self, model: eqx.Module, key: jax.Array, num_samples: int, data_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        """Generates samples by solving the reverse chain from t=0 to t=1.

        Args:
            model: The trained Equinox model.
            key: PRNGKey.
            num_samples: Number of samples.
            data_dim: Dimensionality.

        Returns:
            (x_final, x_traj)
        """
        # Start from Source (Noise) at t=0
        x_src = self.sample_from_source_distribution(key, num_samples, data_dim)

        # Define time steps range [0, ..., 1]
        # In unified time, we step forward.
        dt = 1.0 / self.num_transport_steps
        ts = jnp.linspace(0.0, 1.0 - dt, self.num_transport_steps)
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

            # Split keys for batch
            B = x_t.shape[0]
            batch_keys = jax.random.split(current_key, B)

            # Vectorize the reverse step
            x_next = jax.vmap(
                self.reverse,
                in_axes=(None, None, 0, 0),
            )(model, t, x_t, batch_keys)

            return x_next, x_next

        # Run the loop
        x_final, x_traj = jax.lax.scan(scan_body, x_src, (ts, keys))

        # Prepend initial state to trajectory for visualization
        x_traj_full = jnp.concatenate([x_src[None, ...], x_traj], axis=0)

        return x_final, x_traj_full
