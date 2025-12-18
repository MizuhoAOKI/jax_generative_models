from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax

# --------------------------------------------------------
# Strategy interface
# --------------------------------------------------------


class Strategy(Protocol):
    """Generative modeling strategy interface (DDPM, Flow Matching, etc.).

    Unified Time Convention for implementation:
        t = 0.0 : Source Distribution (Prior / Noise)
        t = 1.0 : Target Distribution (Data)

    The generative process (sampling) always runs from t=0 to t=1.
    """

    def loss_fn(
        self,
        model: eqx.Module,
        x: jax.Array,
        cond: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """
        Computes the training loss for a single sample.

        Args:
            model: Equinox model.
            x: One clean data sample, corresponds to data at t=1.
            cond: Optional conditioning information (e.g., class labels).
            key: PRNGKey for stochastic sampling.

        Returns:
            Scalar loss (0-dim JAX jax.Array).
        """
        ...

    def sample_from_source_distribution(
        self, key: jax.Array, num_samples: int, data_dim: int
    ) -> jax.Array:
        """
        Sample initial data for generation (Source Distribution at t=0).
        Typically Gaussian noise N(0, I).

        Args:
            key: PRNGKey.
            num_samples: Number of samples to generate.
            data_dim: Dimensionality of the data.

        Returns:
            Samples from the source distribution.
        """
        ...

    def sample_from_target_distribution(
        self,
        model: eqx.Module,
        key: jax.Array,
        num_samples: int,
        data_dim: int,
        cond: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Sample data from the target distribution (Data Distribution at t=1).
        This executes the full generative process from t=0 to t=1.

        Args:
            model: Trained Equinox model.
            key: PRNGKey.
            num_samples: Number of samples.
            data_dim: Dimensionality.
            cond: Optional conditioning batch of shape (num_samples, ...).

        Returns:
            tuple containing:
            - x_final: The generated samples at t=1.
            - x_traj: The full trajectory (history) from t=0 to t=1.
        """
        ...

    def forward(
        self,
        t: jax.Array,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Compute the intermediate state x_t and its training target.
        (Forward process in terms of training / perturbation).

        Args:
            t:   Time step or continuous time (in [0, 1]).
            x:   Clean data sample (Target data).
            key: PRNGKey for any stochastic sampling.

        Returns:
            (x_t, target)
            - x_t: The state at time t.
            - target: The supervision signal (e.g. noise, velocity).
        """
        ...

    def reverse(
        self,
        model: eqx.Module,
        t: jax.Array,
        x_t: jax.Array,
        cond: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        """
        Transport step in the generative process (Solver step).
        Moves the state towards the target distribution (t -> t + dt).

        Args:
            model: Equinox model.
            t:    Current time step or continuous time.
            x_t:  State at time t.
            cond: Optional conditioning information.
            key:  PRNGKey for any stochastic sampling.

        Returns:
            The next state in the transport process (closer to Data).
        """
        ...
