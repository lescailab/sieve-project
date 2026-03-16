# Conda Packaging Files

This folder contains files for building and documenting the SIEVE conda package.

## Contents

- `meta.yaml`: conda recipe (dependencies + CLI tests)
- `USAGE.md`: user-facing guide for running SIEVE via installed `sieve-*` commands

## Build locally

The package is built per-architecture (no `noarch`) to ensure platform-specific
PyTorch variants (including CUDA-enabled builds) are resolved correctly at install
time. Run the build on the target machine for each architecture.

### amd64 / linux-64

```bash
conda create -n sieve-build -c conda-forge python=3.11 conda>=26 conda-build>=26 anaconda-client
conda activate sieve-build
conda build conda \
    -c pytorch -c nvidia -c bioconda -c conda-forge \
    --no-anaconda-upload \
    --croot /tmp/sieve-conda-bld
```

### arm64 / linux-aarch64

Run the identical commands above on a Linux arm64 machine. The resulting package
will be tagged `linux-aarch64` and can be uploaded to the same channel.

### osx-arm64 (macOS Apple Silicon)

On a Mac with Apple Silicon, omit the `pytorch` and `nvidia` channels. The
conda-forge channel provides PyTorch builds compiled against NumPy 2.x with
Metal/MPS support included automatically.

```bash
conda create -n sieve-build -c conda-forge python=3.11 conda>=26 conda-build>=26
conda activate sieve-build
conda build conda \
    -c conda-forge \
    --no-anaconda-upload \
    --croot /tmp/sieve-conda-bld
```

## Upload to Anaconda

```bash
# Requires ANACONDA_API_TOKEN to be set in the environment (e.g. via ~/.bashrc)
anaconda upload \
    /tmp/sieve-conda-bld/linux-64/sieve-1.0.0-*.conda \
    --user lescailab \
    --label main
```

**Important:** use only `--label main`. Do **not** pass `--channel` — on
anaconda.org `--channel` is an alias for `--label` and will create unwanted
extra labels.

Replace `linux-64` with `linux-aarch64` or `osx-arm64` when uploading other
architecture builds.

## Install from the channel

The recommended way to install is via the `environment.yml` at the repository
root, which pins PyTorch to the pytorch channel (for CUDA support) while
letting all other dependencies resolve from conda-forge/bioconda:

### Linux (amd64 / aarch64)

```bash
micromamba env create -f environment.yml
# or
conda env create -f environment.yml
```

The explicit `pytorch::` channel pins in the environment file are required
because the pytorch channel carries old tqdm builds that, under strict channel
priority, block conda-forge's working versions and prevent the solver from
finding a solution.

### macOS (Apple Silicon)

```bash
conda create -n sieve -c lescailab -c conda-forge python=3.12
conda install -n sieve -c lescailab -c conda-forge sieve
```

Metal/MPS GPU acceleration is available out of the box via PyTorch's MPS backend.

## Maintenance checklist

1. Keep `requirements: run` in `meta.yaml` aligned with `[project.dependencies]` in `pyproject.toml` where possible. If conda-forge/pytorch channel compatibility requires divergence (for example NumPy/Cython ABI constraints), document the reason in the recipe history.
2. Bump `version` in `meta.yaml` **and** `pyproject.toml` when the project version changes.
3. Keep command tests in `meta.yaml` aligned with `[project.scripts]` entry points.
4. Build and upload a package for each supported architecture (linux-64, linux-aarch64, osx-arm64) before announcing a release.
