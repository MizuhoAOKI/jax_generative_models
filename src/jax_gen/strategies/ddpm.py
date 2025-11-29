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
    """Strategy identifier (fixed to 'ddpm')."""

    num_transport_steps: int = 100
    """Number of discrete steps in the diffusion process."""

    beta_min: float = 1e-4
    """The lower bound of the noise variance schedule."""

    beta_max: float = 1e-2
    """The upper bound of the noise variance schedule."""


@dataclass
class DDPMStrategy(Strategy):
    """A Denoising Diffusion Probabilistic Model (DDPM) strategy.

    This class implements the standard DDPM framework (Ho et al., 2020).
    It manages the discrete-time noise schedule, the forward diffusion process
    (adding noise), and the reverse denoising process (removing noise).

    Assumes the model signature is:
        `eps_pred = model(t_normalized, x_t)`
    where `t_normalized` is a float in range [0, 1].

    Attributes:
        num_transport_steps: The total number of diffusion steps (T).
        betas: The linear variance schedule $\beta_t$.
        alphas_cumprod: The cumulative product of alphas $\bar{\alpha}_t$.
    """

    num_transport_steps: int
    betas: jax.Array
    alphas_cumprod: jax.Array

    @classmethod
    def from_config(cls, cfg: DDPMStrategyConfig) -> DDPMStrategy:
        """Initializes the strategy from a configuration object.

        Constructs the linear beta schedule and pre-computes alpha cumulatives.

        Args:
            cfg: The DDPM configuration object.

        Returns:
            An initialized DDPMStrategy instance.
        """
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

    def _alpha_bar(self, t_idx: jax.Array) -> jax.Array:
        """Retrieves $\bar{\alpha}_t$ for a given integer time index.

        Args:
            t_idx: Integer time index (scalar).

        Returns:
            The cumulative alpha value at index `t_idx`.
        """
        return self.alphas_cumprod[t_idx]

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

        The loss is defined as:
        $$ L = \\mathbb{E}_{t, x_0, \\epsilon}
        [ \\| \\epsilon - \\epsilon_\theta(x_t, t) \\|^2] $$

        Args:
            model: The Equinox model to train.
            x: A single data sample.
            key: PRNGKey for sampling time `t` and noise `eps`.

        Returns:
            The scalar MSE loss.
        """
        key_t, key_noise = jax.random.split(key, 2)

        # Sample a random time step `t` uniformly from [0, T)
        t = jax.random.randint(
            key_t,
            shape=(),  # scalar
            minval=0,
            maxval=self.num_transport_steps,
        )

        x_t, eps_true = self.forward(t, x, key_noise)
        eps_pred = self.predict_noise(model, t, x_t)

        # Calculate Mean Squared Error over all dimensions
        loss = jnp.mean((eps_pred - eps_true) ** 2)
        return loss

    def forward(
        self,
        t: jax.Array,
        x_0: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Simulates the forward diffusion process $q(x_t | x_0)$.

        Computes $x_t$ using the reparameterization trick:
        $$ x_t = \\sqrt{\bar{\alpha}_t} x_0 + \\sqrt{1 - \bar{\alpha}_t} \\epsilon $$

        Args:
            t: Integer time step index.
            x_0: The clean data sample.
            key: PRNGKey for sampling the Gaussian noise.

        Returns:
            A tuple containing:
                - x_t: The noisy sample at time `t`.
                - eps: The noise injected (target for the model).
        """
        eps = jax.random.normal(key, shape=x_0.shape)
        alpha_bar_t = self._alpha_bar(t)

        # Reparameterization trick
        # shape broadcasting: alpha_bar_t is scalar
        x_t = jnp.sqrt(alpha_bar_t) * x_0 + jnp.sqrt(1.0 - alpha_bar_t) * eps
        return x_t, eps

    def predict_noise(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
    ) -> jax.Array:
        """Wrapper to call the model with normalized time.

        Args:
            model: The Equinox model.
            t: Integer time step in [0, num_transport_steps).
            x_t: Noisy input data.

        Returns:
            The predicted noise epsilon.
        """
        # Normalize time to [0, 1] with a half-step offset
        t_norm = (t.astype(jnp.float32) + 0.5) / float(self.num_transport_steps)
        eps_pred = model(t_norm, x_t)  # type: ignore # model is Callable
        return eps_pred

    def backward(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Performs a single reverse diffusion step: $x_t \to x_{t-1}$.

        Follows Equation 11 from Ho et al. (2020) to approximate the posterior:
        $$ p_\theta(x_{t-1} | x_t) =
        \\mathcal{N}(x_{t-1}; \\mu_\theta(x_t, t), \\Sigma_\theta(x_t, t)) $$

        Process:
        1. Predict noise $\\epsilon_\theta$ using the model.
        2. Calculate the posterior mean $\\mu_\theta$.
        3. Sample $x_{t-1}$ by adding noise (only if $t > 0$).

        Args:
            model: The trained Equinox model.
            t: Current integer time step.
            x_t: Sample at current time step `t`.
            key: PRNGKey for stochastic sampling.

        Returns:
            The sample at the previous time step $x_{t-1}$.
        """
        # 1. Predict noise using the model
        eps_pred = self.predict_noise(model, t, x_t)

        # 2. Calculate distribution parameters
        beta_t = self.betas[t]  # scalar
        alpha_t = 1.0 - beta_t  # scalar
        alpha_bar_t = self.alphas_cumprod[t]  # scalar

        # Get alpha_bar for t-1 (handle t=0 case where prev is 1.0)
        alpha_bar_prev = jnp.where(
            t > 0,
            self.alphas_cumprod[t - 1],
            jnp.array(1.0, dtype=alpha_bar_t.dtype),
        )

        # Calculate mean mu_theta (Ho et al. 2020, Eq. 11)
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / jnp.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * eps_pred)

        # 3. Sample x_{t-1}
        # If t=0, return the mean directly (no noise added at the final step).
        def no_noise(_: jax.Array) -> jax.Array:
            return mean

        # If t>0, add scaled Gaussian noise.
        def add_noise(k: jax.Array) -> jax.Array:
            noise = jax.random.normal(k, shape=x_t.shape)
            # Calculate standard DDPM posterior variance (sigma_t)
            # This corresponds to the variance of q(x_{t-1} | x_t, x_0)
            sigma_t = jnp.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t)
            return mean + sigma_t * noise

        return jax.lax.cond(t == 0, no_noise, add_noise, key)

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        r"""Samples initial noise $x_T \sim \mathcal{N}(0, I)$."""
        return jax.random.normal(key, (num_samples, data_dim))

    def sample_from_target_distribution(
        self, model: eqx.Module, key: jax.Array, num_samples: int, data_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        """Generates samples from the target distribution by solving the reverse chain.

        Iterates backwards from $T-1$ to $0$ using `jax.lax.scan`.

        Args:
            model: The trained Equinox model.
            key: PRNGKey for the sampling process.
            num_samples: Number of samples to generate.
            data_dim: Dimensionality of the data.

        Returns:
            A tuple containing:
                - x_target: The final generated samples ($x_0$).
                - x_traj: The full trajectory of samples (from $T-1$ down to $0$).
        """
        # Start from the source distribution x_T ~ N(0, I)
        x_src = self.sample_from_source_distribution(key, num_samples, data_dim)

        def scan_body(
            x_t: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            t, current_key = carry
            next_key, subkey = jax.random.split(current_key)

            B = x_t.shape[0]
            # Split keys for the batch dimension
            batch_keys = jax.random.split(subkey, B)

            # Vectorize the backward step over the batch
            x_prev = jax.vmap(
                self.backward,
                in_axes=(None, None, 0, 0),  # model: None, t: None, x_t: batch, keys: batch
            )(model, t, x_t, batch_keys)

            return x_prev, x_prev

        # Define time steps range [T-1, ..., 0]
        ts = jnp.arange(self.num_transport_steps - 1, -1, -1)
        keys = jax.random.split(key, self.num_transport_steps)

        # Run the reverse process loop
        x_target, x_traj = jax.lax.scan(scan_body, x_src, (ts, keys))
        return x_target, x_traj
