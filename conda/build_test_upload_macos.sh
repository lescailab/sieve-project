#!/usr/bin/env bash
# Build, MPS-verify, training-test, then upload the sieve conda package (osx-arm64).
# Must be run from the project root: bash conda/build_test_upload_macos.sh
#
# Requirements:
#   - sieve-build conda env exists with conda-build installed
#   - ANACONDA_API_TOKEN is set (sourced from ~/.bash_profile)
#   - Apple Silicon Mac with Metal-capable GPU
set -euo pipefail

# ── platform guard ─────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This script targets osx-arm64 (Metal/MPS). For linux-64, use conda/build_test_upload.sh. For linux-aarch64, build manually with:"
    echo "  CONDA_SOLVER=libmamba conda build conda -c pytorch -c nvidia -c bioconda -c conda-forge --no-anaconda-upload --croot /tmp/sieve-conda-bld"
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

CHANNELS="-c pytorch -c nvidia -c bioconda -c conda-forge"

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

PACKAGE_PATH=$(ls "$CROOT/osx-arm64/sieve-${VERSION}-"*.conda 2>/dev/null \
    || ls "$CROOT/osx-arm64/sieve-${VERSION}-"*.tar.bz2 2>/dev/null \
    | head -1)

if [[ -z "$PACKAGE_PATH" ]]; then
    echo "ERROR: built package not found under $CROOT/osx-arm64/"
    exit 1
fi
echo "Built: $PACKAGE_PATH"

# ── 2. create isolated test environment ───────────────────────────────────
echo ""
echo "[2/5] Creating test environment: $TEST_ENV..."
conda env remove -n "$TEST_ENV" --yes 2>/dev/null || true
CONDA_SOLVER=libmamba conda create -n "$TEST_ENV" --yes \
    -c "file://$CROOT" \
    $CHANNELS \
    "sieve=$VERSION"

# ── 3. verify Metal/MPS ───────────────────────────────────────────────────
echo ""
echo "[3/5] Verifying torch Metal/MPS..."
conda run -n "$TEST_ENV" python -c "
import torch, sys
print(f'  torch version  : {torch.__version__}')
print(f'  MPS built      : {torch.backends.mps.is_built()}')
print(f'  MPS available  : {torch.backends.mps.is_available()}')
if not torch.backends.mps.is_built():
    print('FAIL: torch was built without MPS support.')
    print('Check channel order: pytorch and nvidia must come before conda-forge.')
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
conda run -n "$TEST_ENV" sieve-train \
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
    echo "ERROR: ANACONDA_API_TOKEN is not set. Source ~/.bash_profile or export it."
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
