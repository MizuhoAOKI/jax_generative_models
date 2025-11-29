# JAX Generative Models

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-5d97f6.svg)](./LICENSE.txt) -->
[![Python](https://img.shields.io/badge/Python-3.12-26a69a.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/Powered%20by-JAX-ea80fc.svg)](https://github.com/google/jax)
[![Tyro](https://img.shields.io/badge/Config-Tyro-e78444)](https://github.com/brentyi/tyro)
<br/>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![ruff](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/lint.yaml/badge.svg)](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/lint.yaml)
[![mypy](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/type_check.yaml/badge.svg)](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/type_check.yaml)
<!-- [![test](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/test.yaml/badge.svg)](https://github.com/MizuhoAOKI/jax_generative_models/actions/workflows/test.yaml) -->

**jax_generative_models** implements Diffusion Models (DDPM) and Flow Matching within a unified JAX framework. By abstracting these algorithms behind a shared strategy interface, the project highlights their structural similarities and differences. Utilizing Tyro for configuration and Rerun for visualization, it serves as an extensible base for exploring generative modeling.

## Getting Started

1. Prerequisites
    - [git](https://git-scm.com/)<br/>
        Version control system to clone the repository.
    - [uv](https://docs.astral.sh/uv/getting-started/installation/)<br/>
        A simple and fast Python package manager.
        Refer to the official documentation for one-command installation.
    - [make](https://www.gnu.org/software/make/)<br/>
      Used to run shortcuts such as `make setup`.
      It is optional, so you can also run the commands in the `Makefile` manually.
    - Cuda 12 and compatible GPU (optional)<br/>
      Required for GPU acceleration with JAX.

2. **Clone the repository**
    ```bash
    cd <path-to-your-workspace>
    git clone https://github.com/MizuhoAOKI/jax_generative_models.git
    ```

3. **Set up the virtual environment and install dependencies**
    ```bash
    cd jax_generative_models
    ```
    - For CPU only:
        ```bash
        make setup_cpu
        ```
    - For GPU with CUDA 12 support:
        ```bash
        make setup_gpu_cuda12
        ```
        If you are using a GPU, run the following command before executing the scripts:
        ```bash
        source setup_gpu.sh
        ```

4. **Train a generative model**
    ```bash
    uv run scripts/main.py train strategy:<STRATEGY_NAME> model:<MODEL_NAME> dataset:<DATASET_NAME>
    ```

5. **Generate samples from the trained model**
    ```bash
    uv run scripts/main.py generate strategy:<STRATEGY_NAME> model:<MODEL_NAME> dataset:<DATASET_NAME>
    ```

6. **Make an animation of the transport process from the trained model**
    ```bash
    uv run scripts/main.py animate strategy:<STRATEGY_NAME> model:<MODEL_NAME> dataset:<DATASET_NAME>
    ```

7. **Visualize training progress**
    Run Rerun from another terminal:
    ```bash
    make rerun
    ```

### Command Arguments

Replace the placeholders in the commands above with the following options:

| Placeholder | Options | Description |
| :--- | :--- | :--- |
| `<STRATEGY_NAME>` | `ddpm`, `flow-matching` | Generative modeling strategy. |
| `<MODEL_NAME>` | `mlp`, `resnet` | Model architecture to use. |
| `<DATASET_NAME>` | `cat`, `gaussian-mixture`, `moon` | Target dataset for training/generation. |

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2023)
