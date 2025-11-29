SHELL := /bin/bash
.PHONY: setup_cpu setup_gpu_cuda12 test clean clean_cache clean_outputs

# Install dependencies and set up the development environment
setup_cpu:
	uv sync --dev --extra cpu && \
	uv pip install -e . && \
	uv run pre-commit install

setup_gpu_cuda12:
	uv sync --dev --extra cuda12 && \
	uv pip install -e . && \
	uv run pre-commit install

# Run tests using pytest
test:
	uv run pytest

# Run the rerun visualizer
rerun:
	uv run rerun outputs/rerun_vis_log.rrd configs/rerun_vis_config.rbl

# Clean only caches
clean_cache:
	@echo "Cleaning caches..."
	rm -rf *_cache .*_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Clean outputs
clean_outputs:
	@echo "Cleaning outputs..."
	rm -rf outputs

# Clean everything (caches and outputs)
clean: clean_cache clean_outputs
	@echo "All caches and outputs have been cleaned."
