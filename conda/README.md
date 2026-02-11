# Conda Packaging Files

This folder contains files for building and documenting the SIEVE conda package.

## Contents

- `meta.yaml`: conda recipe (dependencies + CLI tests)
- `USAGE.md`: user-facing guide for running SIEVE via installed `sieve-*` commands

## Build locally

```bash
conda create -n sieve-build python=3.10 conda-build
conda activate sieve-build
conda build conda
```

The generated package can then be installed from the local channel:

```bash
conda create -n sieve python=3.10
conda install -n sieve -c local sieve
```

## Maintenance checklist

1. Keep `requirements: run` in `meta.yaml` aligned with `[project.dependencies]` in `pyproject.toml`.
2. Bump `version` in `meta.yaml` when the project version changes.
3. Keep command tests in `meta.yaml` aligned with `[project.scripts]` entry points.
