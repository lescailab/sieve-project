#!/usr/bin/env bash
# Build, CUDA-verify, training-test, then upload the sieve conda package.
# Must be run from the project root: bash conda/build_test_upload.sh
#
# Requirements:
#   - sieve-build conda env exists with conda-build installed
#   - ANACONDA_API_TOKEN is set (sourced from ~/.bashrc)
#   - NVIDIA GPU present and CUDA driver installed
set -euo pipefail

# ── platform guard ─────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This script targets linux-64 (CUDA). For linux-aarch64 or macOS, build manually with:"
    echo "  CONDA_SOLVER=libmamba conda build conda -c pytorch -c nvidia -c bioconda -c conda-forge --no-anaconda-upload --croot /tmp/sieve-conda-bld"
    exit 1
fi

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$PROJECT_DIR/conda"
CROOT="/tmp/sieve-conda-bld"
TEST_OUTPUT="/tmp/sieve_train_cuda_test"

# ── read version from meta.yaml ────────────────────────────────────────────
VERSION=$(grep '^{%' "$CONDA_DIR/meta.yaml" | grep 'set version' | sed 's/.*"\(.*\)".*/\1/')
TEST_ENV="sieve_test_${VERSION}"

CHANNELS="-c pytorch -c nvidia -c bioconda -c conda-forge"

echo "============================================================"
echo "  SIEVE conda package: build → test → upload"
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

PACKAGE_PATH=$(ls "$CROOT/linux-64/sieve-${VERSION}-"*.conda 2>/dev/null \
    || ls "$CROOT/linux-64/sieve-${VERSION}-"*.tar.bz2 2>/dev/null \
    | head -1)

if [[ -z "$PACKAGE_PATH" ]]; then
    echo "ERROR: built package not found under $CROOT/linux-64/"
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

# ── 3. verify CUDA ────────────────────────────────────────────────────────
echo ""
echo "[3/5] Verifying torch CUDA..."
conda run -n "$TEST_ENV" python -c "
import torch, sys
print(f'  torch version : {torch.__version__}')
print(f'  CUDA version  : {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('FAIL: torch was built without CUDA support.')
    print('Check channel order: pytorch and nvidia must come before conda-forge.')
    sys.exit(1)
print('CUDA check PASSED')
"

# ── 4. run training on test_data/small with --device cuda ─────────────────
echo ""
echo "[4/5] Running training test on test_data/small (device=cuda)..."
rm -rf "$TEST_OUTPUT"
conda run -n "$TEST_ENV" sieve-train \
    --preprocessed-data "$PROJECT_DIR/test_data/small/preprocessed_test.pt" \
    --level L3 \
    --cv 2 \
    --epochs 2 \
    --batch-size 4 \
    --output-dir "$TEST_OUTPUT" \
    --experiment-name cuda_smoke_test \
    --device cuda \
    --num-workers 0

echo "Training test PASSED"

# ── 5. upload ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Uploading to anaconda.org (lescailab, label=main)..."
if [[ -z "${ANACONDA_API_TOKEN:-}" ]]; then
    echo "ERROR: ANACONDA_API_TOKEN is not set. Source ~/.bashrc or export it."
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
echo "  Done. sieve $VERSION uploaded to lescailab::sieve [main]"
echo "============================================================"
