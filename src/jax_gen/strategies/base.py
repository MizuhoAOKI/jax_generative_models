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
        Compute the intermediate state x_t and its training target.

        This is used by each strategy's loss function to obtain:
          - x_t: the state at time t
          - target: the supervision signal (e.g. noise for DDPM, velocity for Flow Matching)

        Args:
            x:   Data sample used as anchor (x_0 or x_1 depending on strategy).
            t:   Time step or continuous time.
            key: PRNGKey for any stochastic sampling.

        Returns:
            (x_t, target)
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
        Transport step in the generative process.

        Given x_t and t, return the next state (e.g. x_{t-1} or x_{t+1}),
        using the provided model.

        Args:
            model: Equinox model.
            x_t:  State at time t (e.g. noised sample).
            t:    Time step or continuous time.
            key:  PRNGKey for any stochastic sampling.

        Returns:
            The next state in the transport process.
        """
        ...
