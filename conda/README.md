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

Use the dedicated script, which builds, verifies CUDA availability, runs a
training smoke-test, and only uploads if all checks pass:

```bash
bash conda/build_test_upload.sh
```

### osx-arm64 (macOS Apple Silicon)

Use the dedicated macOS script, which builds, verifies Metal/MPS availability,
runs a training smoke-test, and only uploads if all checks pass:

```bash
bash conda/build_test_upload_macos.sh
```

The pytorch and nvidia channels are intentionally excluded on osx-arm64: their
stale ARM builds (compiled against NumPy 1.x) conflict with the recipe's
`numpy >=2.0` pin and cause `import torch` to fail at runtime. The conda-forge
channel provides PyTorch with Metal/MPS support compiled against NumPy 2.x.

### arm64 / linux-aarch64

No dedicated script exists for this architecture. Build manually from inside the
`sieve-build` environment:

```bash
conda run -n sieve-build \
    env CONDA_SOLVER=libmamba \
    conda build conda \
    -c pytorch -c nvidia -c bioconda -c conda-forge \
    --no-anaconda-upload \
    --croot /tmp/sieve-conda-bld
```

After verifying the build, upload manually (see section below).

## Upload to Anaconda

The build scripts above handle upload automatically after a successful test run.
For manual uploads (e.g. linux-aarch64), use:

```bash
# Requires ANACONDA_API_TOKEN to be set in the environment (e.g. via ~/.bashrc)
anaconda upload \
    /tmp/sieve-conda-bld/linux-64/sieve-1.3.0-*.conda \
    --user lescailab \
    --label main
```

**Important:** use only `--label main`. Do **not** pass `--channel`, on
anaconda.org `--channel` is an alias for `--label` and will create unwanted
extra labels.

Replace `linux-64` with `linux-aarch64` or `osx-arm64` when uploading other
architecture builds.

## Install from the channel

### Linux (amd64 / aarch64)

Create an environment using an `environment.yml` file. The channel order and
priority setting are critical: without `channel_priority: strict` and `pytorch`
listed before `conda-forge`, the solver may pick conda-forge's CPU-only PyTorch
build, producing an environment where `torch.cuda.is_available()` returns `False`.

```yaml
name: sieve
channel_priority: strict
channels:
  - pytorch
  - nvidia
  - bioconda
  - conda-forge
  - lescailab
dependencies:
  - lescailab::sieve
  - pytorch::pytorch>=2.0.0
  - pytorch::pytorch-cuda>=11.8
```

```bash
conda env create -f environment.yml
# or
micromamba env create -f environment.yml
```

### macOS (Apple Silicon)

Do **not** include the `pytorch` or `nvidia` channels on osx-arm64: they carry
stale ARM builds compiled against NumPy 1.x that conflict with the package's
NumPy 2.x dependency and break `import torch` at runtime. The `conda-forge`
channel provides PyTorch with Metal/MPS support natively.

```bash
conda create -n sieve -c lescailab -c bioconda -c conda-forge sieve
```

Metal/MPS GPU acceleration is available out of the box via PyTorch's MPS backend.

## Maintenance checklist

1. Keep `requirements: run` in `meta.yaml` aligned with `[project.dependencies]` in `pyproject.toml` where possible. If conda-forge/pytorch channel compatibility requires divergence (for example NumPy/Cython ABI constraints), document the reason in the recipe history.
2. Bump `version` in `meta.yaml` **and** `pyproject.toml` when the project version changes.
3. Keep command tests in `meta.yaml` aligned with `[project.scripts]` entry points.
4. Build and upload a package for each supported architecture (linux-64, linux-aarch64, osx-arm64) before announcing a release.
