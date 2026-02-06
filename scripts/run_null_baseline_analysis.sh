#!/bin/bash

# Null Baseline Attribution Analysis Pipeline
# ============================================
# This script runs the complete null baseline analysis:
# 1. Creates permuted dataset
# 2. Trains null model
# 3. Runs explainability on null model
# 4. Compares real vs null attributions

set -e  # Exit on error

# Find Python interpreter
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

# Default parameters (override with environment variables)
INPUT_DATA="${INPUT_DATA:-/home/shared/sieve-testing/preprocessed.pt}"
REAL_EXPERIMENT="${REAL_EXPERIMENT:-/home/shared/sieve-testing/experiments/CONFIG_G_FINAL}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/shared/sieve-testing}"
DEVICE="${DEVICE:-cuda}"

# Model parameters (should match the real model)
LR="${LR:-0.00001}"
LAMBDA_ATTR="${LAMBDA_ATTR:-0.1}"
LATENT_DIM="${LATENT_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-1}"

echo "=============================================="
echo "Null Baseline Attribution Analysis Pipeline"
echo "=============================================="
echo "Input data: $INPUT_DATA"
echo "Real experiment: $REAL_EXPERIMENT"
echo "Output base: $OUTPUT_BASE"
echo ""

# Validate that required paths exist
if [ ! -f "$INPUT_DATA" ]; then
    echo "ERROR: Input data not found at: $INPUT_DATA"
    echo "Please set INPUT_DATA environment variable or provide the correct path."
    exit 1
fi

if [ ! -d "$REAL_EXPERIMENT" ]; then
    echo "WARNING: Real experiment directory not found at: $REAL_EXPERIMENT"
    echo "This is only needed for comparison in Step 4."
    echo "Continuing with null model training..."
fi

# Step 1: Create permuted dataset
echo "[Step 1/4] Creating permuted dataset..."
NULL_DATA="${OUTPUT_BASE}/preprocessed_discovery_NULL.pt"

$PYTHON scripts/create_null_baseline.py \
    --input "$INPUT_DATA" \
    --output "$NULL_DATA" \
    --seed 42

echo ""

# Step 2: Train null model
echo "[Step 2/4] Training null model..."
NULL_EXPERIMENT="${OUTPUT_BASE}/experiments/NULL_BASELINE"

$PYTHON scripts/train.py \
    --preprocessed-data "$NULL_DATA" \
    --level L3 \
    --val-split 0.2 \
    --lr "$LR" \
    --lambda-attr "$LAMBDA_ATTR" \
    --early-stopping 15 \
    --epochs 100 \
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \
    --gradient-accumulation-steps 4 \
    --gradient-clip 1.0 \
    --latent-dim "$LATENT_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-attention-layers "$NUM_LAYERS" \
    --output-dir "${OUTPUT_BASE}/experiments" \
    --experiment-name NULL_BASELINE \
    --device "$DEVICE"

echo ""

# Step 3: Run explainability on null model
echo "[Step 3/4] Running explainability on null model..."
NULL_EXPLAIN_DIR="${OUTPUT_BASE}/results/null_attributions"

$PYTHON scripts/explain.py \
    --experiment-dir "$NULL_EXPERIMENT" \
    --preprocessed-data "$NULL_DATA" \
    --output-dir "$NULL_EXPLAIN_DIR" \
    --device "$DEVICE" \
    --is-null-baseline

echo ""

# Step 4: Compare real vs null
echo "[Step 4/4] Comparing real vs null attributions..."
REAL_RANKINGS="${REAL_EXPERIMENT}/explainability/sieve_variant_rankings.csv"
NULL_RANKINGS="${NULL_EXPLAIN_DIR}/sieve_variant_rankings.csv"
COMPARISON_DIR="${OUTPUT_BASE}/results/attribution_comparison"

# Check if real rankings exist
if [ ! -f "$REAL_RANKINGS" ]; then
    echo "Warning: Real rankings not found at $REAL_RANKINGS"
    echo "Looking for alternative locations..."

    # Try alternative paths
    ALT_PATHS=(
        "${OUTPUT_BASE}/results/explainability/sieve_variant_rankings.csv"
        "${REAL_EXPERIMENT}/sieve_variant_rankings.csv"
    )

    for alt in "${ALT_PATHS[@]}"; do
        if [ -f "$alt" ]; then
            REAL_RANKINGS="$alt"
            echo "Found real rankings at: $REAL_RANKINGS"
            break
        fi
    done
fi

$PYTHON scripts/compare_attributions.py \
    --real "$REAL_RANKINGS" \
    --null "$NULL_RANKINGS" \
    --output-dir "$COMPARISON_DIR" \
    --top-k 100

echo ""
echo "=============================================="
echo "Null Baseline Analysis Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Null model: $NULL_EXPERIMENT"
echo "  - Null attributions: $NULL_EXPLAIN_DIR"
echo "  - Comparison results: $COMPARISON_DIR"
echo ""
echo "Key files to review:"
echo "  - ${COMPARISON_DIR}/comparison_summary.yaml"
echo "  - ${COMPARISON_DIR}/real_vs_null_comparison.png"
echo "  - ${COMPARISON_DIR}/significant_variants_p01.csv"
