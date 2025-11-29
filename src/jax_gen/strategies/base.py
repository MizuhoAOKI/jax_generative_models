from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax

# --------------------------------------------------------
# Strategy interface
# --------------------------------------------------------


class Strategy(Protocol):
    """Generative modeling strategy interface (DDPM, flow matching, etc.)."""

    def loss_fn(
        self,
        model: eqx.Module,
        x: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """
        Loss for a *single sample*.

        Args:
            model: Equinox model.
            x: One data sample, shape (data_dim,) or (..., data_dim).
            key: PRNGKey for this sample.

        Returns:
            Scalar loss (0-dim JAX jax.Array).
        """
        ...

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        """
        Sample initial data for generation.
        """
        ...

    def sample_from_target_distribution(
        self, model: eqx.Module, key: jax.Array, num_samples: int, data_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        """
        Sample data from the target distribution (data distribution).
        """
        ...

    def forward(
        self,
        t: jax.Array,
        x_0: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Physical forward process: x0 -> x_t.

        For DDPM this means: add noise according to a schedule.

        Args:
            x0: Clean data sample.
            t: Time step / continuous time (implementation-dependent).
            key: PRNGKey for sampling noise, etc.

        Returns:
            (x_t)
              x_t: noised sample.
        """
        ...

    def backward(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """
        Physical backward process: model prediction from x_t, t.

        For DDPM this is typically epsilon prediction or x0 prediction.

        Args:
            model: Equinox model.
            xt: Noised sample x_t.
            t: Time step / continuous time.

        Returns:
            Model prediction (e.g. epsilon_pred), same shape as x0.
        """
        ...
