#!/bin/bash

# Null Baseline Attribution Analysis Pipeline
# ============================================
# This script runs the complete null baseline analysis:
# 1. Creates permuted dataset
# 2. Trains null model (identical conditions to real model, only labels permuted)
# 3. Runs explainability on null model
# 4. Compares real vs null attributions
#
# IMPORTANT: The null model must be trained under identical conditions to the
# real model. The only difference is the permuted labels. If the real model
# used --sex-map, the null model must too. Ploidy correction is baked into the
# preprocessed data, so using the same .pt file preserves it automatically.

set -e  # Exit on error

# Find Python interpreter
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

# Default parameters (override with environment variables)
INPUT_DATA="${INPUT_DATA:-/home/shared/sieve-run/data/preprocessed_ottawa_chr.pt}"
REAL_EXPERIMENT="${REAL_EXPERIMENT:-/home/shared/sieve-run/experiments/CONFIG_G_FINAL_CV/fold_4}"
REAL_RESULTS="${REAL_RESULTS:-/home/shared/sieve-run/results/fold4_explainability}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/shared/sieve-run}"
DEVICE="${DEVICE:-cuda}"

# Sex covariate and genome build (must match real model)
SEX_MAP="${SEX_MAP:-/home/shared/sieve-run/data/sample_sex.tsv}"
GENOME_BUILD="${GENOME_BUILD:-GRCh37}"

# Model parameters (MUST match real model exactly)
LR="${LR:-0.00001}"
LAMBDA_ATTR="${LAMBDA_ATTR:-0.1}"
LATENT_DIM="${LATENT_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-1}"

echo "=============================================="
echo "Null Baseline Attribution Analysis Pipeline"
echo "=============================================="
echo "Input data:       $INPUT_DATA"
echo "Real experiment:  $REAL_EXPERIMENT"
echo "Real results:     $REAL_RESULTS"
echo "Output base:      $OUTPUT_BASE"
echo "Sex map:          ${SEX_MAP:-<not set>}"
echo "Genome build:     $GENOME_BUILD"
echo ""

# Validate that required paths exist
if [ ! -f "$INPUT_DATA" ]; then
    echo "ERROR: Input data not found at: $INPUT_DATA"
    echo "Please set INPUT_DATA environment variable or provide the correct path."
    exit 1
fi

if [ -n "$SEX_MAP" ] && [ ! -f "$SEX_MAP" ]; then
    echo "ERROR: Sex map file not found at: $SEX_MAP"
    echo "Set SEX_MAP='' to disable sex covariate (not recommended if real model used it)."
    exit 1
fi

if [ ! -d "$REAL_EXPERIMENT" ]; then
    echo "WARNING: Real experiment directory not found at: $REAL_EXPERIMENT"
    echo "This is only needed for comparison in Step 4."
    echo "Continuing with null model training..."
fi

# Step 1: Create permuted dataset
echo "[Step 1/4] Creating permuted dataset..."
NULL_DATA="${OUTPUT_BASE}/data/preprocessed_ottawa_chr_NULL.pt"

$PYTHON scripts/create_null_baseline.py \
    --input "$INPUT_DATA" \
    --output "$NULL_DATA" \
    --seed 42

echo ""

# Step 2: Train null model
# NOTE: All hyperparameters MUST match the real model exactly.
# The only differences are:
#   - --preprocessed-data points to the null (permuted) .pt file
#   - --experiment-name is NULL_BASELINE
#   - --val-split 0.2 (single split instead of 5-fold CV — sufficient for null)
echo "[Step 2/4] Training null model..."
NULL_EXPERIMENT="${OUTPUT_BASE}/experiments/NULL_BASELINE"

# Build train.py command with conditional arguments
TRAIN_CMD=(
    $PYTHON scripts/train.py
    --preprocessed-data "$NULL_DATA"
    --level L3
    --val-split 0.2
    --lr "$LR"
    --lambda-attr "$LAMBDA_ATTR"
    --early-stopping 15
    --epochs 100
    --batch-size 16
    --chunk-size 3000
    --aggregation-method mean
    --gradient-accumulation-steps 4
    --gradient-clip 1.0
    --latent-dim "$LATENT_DIM"
    --hidden-dim "$HIDDEN_DIM"
    --num-attention-layers "$NUM_LAYERS"
    --output-dir "${OUTPUT_BASE}/experiments"
    --experiment-name NULL_BASELINE
    --genome-build "$GENOME_BUILD"
    --device "$DEVICE"
)

# Add --sex-map if set (critical for fair null comparison)
if [ -n "$SEX_MAP" ]; then
    TRAIN_CMD+=(--sex-map "$SEX_MAP")
fi

"${TRAIN_CMD[@]}"

echo ""

# Step 3: Run explainability on null model
echo "[Step 3/4] Running explainability on null model..."
NULL_EXPLAIN_DIR="${OUTPUT_BASE}/results/null_attributions"

$PYTHON scripts/explain.py \
    --experiment-dir "$NULL_EXPERIMENT" \
    --preprocessed-data "$NULL_DATA" \
    --output-dir "$NULL_EXPLAIN_DIR" \
    --genome-build "$GENOME_BUILD" \
    --device "$DEVICE" \
    --is-null-baseline

echo ""

# Step 4: Compare real vs null
echo "[Step 4/4] Comparing real vs null attributions..."
NULL_RANKINGS="${NULL_EXPLAIN_DIR}/sieve_variant_rankings.csv"
COMPARISON_DIR="${OUTPUT_BASE}/results/attribution_comparison"

# Locate real rankings — try several known locations
REAL_RANKINGS=""
CANDIDATE_PATHS=(
    "${REAL_RESULTS}/sieve_variant_rankings.csv"
    "${REAL_EXPERIMENT}/explainability/sieve_variant_rankings.csv"
    "${REAL_EXPERIMENT}/sieve_variant_rankings.csv"
    "${OUTPUT_BASE}/results/explainability/sieve_variant_rankings.csv"
    "${OUTPUT_BASE}/results/fold4_explainability/sieve_variant_rankings.csv"
)

for candidate in "${CANDIDATE_PATHS[@]}"; do
    if [ -f "$candidate" ]; then
        REAL_RANKINGS="$candidate"
        echo "Found real rankings at: $REAL_RANKINGS"
        break
    fi
done

if [ -z "$REAL_RANKINGS" ]; then
    echo "ERROR: Real rankings not found. Tried:"
    for candidate in "${CANDIDATE_PATHS[@]}"; do
        echo "  - $candidate"
    done
    echo "Set REAL_RESULTS to the directory containing sieve_variant_rankings.csv"
    exit 1
fi

$PYTHON scripts/compare_attributions.py \
    --real "$REAL_RANKINGS" \
    --null "$NULL_RANKINGS" \
    --output-dir "$COMPARISON_DIR" \
    --genome-build "$GENOME_BUILD" \
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
