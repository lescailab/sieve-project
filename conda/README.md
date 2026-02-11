# Conda Packaging Files

This folder contains files for building and documenting the SIEVE conda package.

## Contents

- `meta.yaml`: conda recipe (dependencies + CLI tests)
- `USAGE.md`: user-facing guide for running SIEVE via installed `sieve-*` commands

## Build locally

```bash
conda create -n sieve-build -c conda-forge python=3.11 conda>=26 conda-build>=26
conda activate sieve-build
CONDA_SOLVER=libmamba conda build conda --no-anaconda-upload --croot /tmp/sieve-conda-bld
```

The generated package can then be installed from the local channel:

```bash
conda create -n sieve python=3.10
conda install -n sieve -c file:///tmp/sieve-conda-bld sieve
```

## Maintenance checklist

1. Keep `requirements: run` in `meta.yaml` aligned with `[project.dependencies]` in `pyproject.toml` where possible. If conda-forge/bioconda compatibility requires divergence (for example NumPy/Cython ABI constraints), document the reason in the recipe history.
2. Bump `version` in `meta.yaml` when the project version changes.
3. Keep command tests in `meta.yaml` aligned with `[project.scripts]` entry points.
