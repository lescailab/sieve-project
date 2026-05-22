#!/usr/bin/env bash
# Build, MPS-verify, training-test, then upload the sieve conda package (osx-arm64).
# Must be run from the project root: bash conda/build_test_upload_macos.sh
#
# Requirements:
#   - sieve-build conda env exists with conda-build installed
#   - ANACONDA_API_TOKEN is set (exported from your shell profile, e.g.
#     ~/.bash_profile, ~/.zprofile, or ~/.zshrc depending on your shell)
#   - Apple Silicon Mac with Metal-capable GPU
set -euo pipefail

# ── platform guard ─────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This script targets osx-arm64 (Metal/MPS). For linux-64, use conda/build_test_upload.sh. For linux-aarch64, build manually with:"
    echo "  CONDA_SOLVER=libmamba conda build conda -c pytorch -c nvidia -c bioconda -c conda-forge --no-anaconda-upload --croot /tmp/sieve-conda-bld"
    exit 1
fi

# ── ensure all conda calls use sieve-build's (arm64-native) conda ─────────
# On multi-conda macOS setups (e.g., x86_64 anaconda + arm64 miniforge), the
# host PATH may point to a wrong-arch conda. Resolve sieve-build's prefix and
# put its bin first in PATH so every subsequent `conda` call resolves to one
# arm64-native conda binary that shares env directories with our build env.
SIEVE_BUILD_PREFIX="$(conda env list 2>/dev/null | awk '$1=="sieve-build"{print $NF}')"
if [[ -z "$SIEVE_BUILD_PREFIX" || ! -x "$SIEVE_BUILD_PREFIX/bin/conda" ]]; then
    echo "ERROR: sieve-build conda env not found. Create it per conda/README.md:"
    echo "  conda create -n sieve-build -c conda-forge python=3.11 conda>=26 conda-build>=26 anaconda-client"
    exit 1
fi
export PATH="$SIEVE_BUILD_PREFIX/bin:$PATH"

SIEVE_BUILD_ARCH="$(python -c 'import platform; print(platform.machine())')"
if [[ "$SIEVE_BUILD_ARCH" != "arm64" ]]; then
    echo "ERROR: sieve-build env is $SIEVE_BUILD_ARCH but this script targets osx-arm64."
    echo "Recreate sieve-build with an arm64-native conda installation."
    exit 1
fi

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$PROJECT_DIR/conda"
CROOT="/tmp/sieve-conda-bld"
TEST_OUTPUT="/tmp/sieve_train_mps_test"

# ── read version from meta.yaml ────────────────────────────────────────────
VERSION=$(grep '^{%' "$CONDA_DIR/meta.yaml" | grep 'set version' | sed 's/.*"\(.*\)".*/\1/')
TEST_ENV="sieve_test_${VERSION}"

# On osx-arm64 the pytorch/nvidia channels are not just no-ops: their stale
# osx-arm64 pytorch builds (e.g. 2.2.2 compiled against NumPy 1.x) outrank
# conda-forge's newer NumPy 2.x / MPS-enabled builds under strict channel
# priority and produce an env where `import torch` fails at runtime due to
# the NumPy ABI mismatch with the recipe's `numpy >=2.0,<3.0` pin. Drop
# pytorch/nvidia here; conda-forge carries Metal/MPS support natively.
CHANNELS="-c bioconda -c conda-forge"

echo "============================================================"
echo "  SIEVE conda package: build → test → upload (osx-arm64)"
echo "  Version : $VERSION"
echo "  Croot   : $CROOT"
echo "  Test env: $TEST_ENV"
echo "============================================================"

# ── 1. build ───────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Building package..."
conda run -n sieve-build \
    env CONDA_SOLVER=libmamba \
    conda build "$CONDA_DIR" \
    $CHANNELS \
    --no-anaconda-upload \
    --croot "$CROOT"

shopt -s nullglob
PKG_CANDIDATES=(
    "$CROOT/osx-arm64/sieve-${VERSION}-"*.conda
    "$CROOT/osx-arm64/sieve-${VERSION}-"*.tar.bz2
)
shopt -u nullglob

if [[ ${#PKG_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: built package not found under $CROOT/osx-arm64/"
    exit 1
fi
PACKAGE_PATH="${PKG_CANDIDATES[0]}"
echo "Built: $PACKAGE_PATH"

# ── 2. create isolated test environment ───────────────────────────────────
echo ""
echo "[2/5] Creating test environment: $TEST_ENV..."
conda env remove -n "$TEST_ENV" --yes 2>/dev/null || true
CONDA_SOLVER=libmamba conda create -n "$TEST_ENV" --yes \
    -c "file://$CROOT" \
    $CHANNELS \
    "sieve=$VERSION"

# Resolve the test env's prefix and invoke its binaries directly in steps 3
# and 4. `conda run -n <name>` is racy here: conda's env cache may not yet
# list the freshly-created env in the brief window between `conda create`
# completing and the next call, and a missed lookup silently falls through
# to the base env (where torch is absent), producing a confusing
# `ModuleNotFoundError: No module named 'torch'` even though the env on
# disk is correct.
TEST_ENV_PREFIX="$(dirname "$SIEVE_BUILD_PREFIX")/$TEST_ENV"
if [[ ! -x "$TEST_ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: test env python not found at $TEST_ENV_PREFIX/bin/python"
    exit 1
fi

# ── 3. verify Metal/MPS ───────────────────────────────────────────────────
echo ""
echo "[3/5] Verifying torch Metal/MPS..."
"$TEST_ENV_PREFIX/bin/python" -c "
import torch, sys
print(f'  torch version  : {torch.__version__}')
print(f'  MPS built      : {torch.backends.mps.is_built()}')
print(f'  MPS available  : {torch.backends.mps.is_available()}')
if not torch.backends.mps.is_built():
    print('FAIL: torch was built without MPS support.')
    print('Check channel order: conda-forge must be in the channel list on osx-arm64.')
    sys.exit(1)
if not torch.backends.mps.is_available():
    print('FAIL: MPS is built into torch but not available at runtime.')
    print('Verify this is an Apple Silicon Mac with a Metal-capable GPU and a recent macOS.')
    sys.exit(1)
print('MPS check PASSED')
"

# ── 4. run training on test_data/small with --device mps ──────────────────
echo ""
echo "[4/5] Running training test on test_data/small (device=mps)..."
rm -rf "$TEST_OUTPUT"
"$TEST_ENV_PREFIX/bin/sieve-train" \
    --preprocessed-data "$PROJECT_DIR/test_data/small/preprocessed_test.pt" \
    --level L3 \
    --cv 2 \
    --epochs 2 \
    --batch-size 4 \
    --output-dir "$TEST_OUTPUT" \
    --experiment-name mps_smoke_test \
    --device mps \
    --num-workers 0

echo "Training test PASSED"

# ── 5. upload ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Uploading to anaconda.org (lescailab, label=main)..."
if [[ -z "${ANACONDA_API_TOKEN:-}" ]]; then
    echo "ERROR: ANACONDA_API_TOKEN is not set. Export it from your shell profile (e.g. ~/.bash_profile, ~/.zprofile, ~/.zshrc) and reopen the shell."
    exit 1
fi
conda run -n sieve-build anaconda upload "$PACKAGE_PATH" \
    --user lescailab \
    --label main

# ── cleanup ────────────────────────────────────────────────────────────────
echo ""
echo "Cleaning up test environment and output..."
conda env remove -n "$TEST_ENV" --yes 2>/dev/null || true
rm -rf "$TEST_OUTPUT"

echo ""
echo "============================================================"
echo "  Done. sieve $VERSION (osx-arm64) uploaded to lescailab::sieve [main]"
echo "============================================================"
