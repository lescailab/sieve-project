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
# real model. The only difference is the permuted labels. Hyperparameters are
# read from the real experiment's config.yaml to guarantee an exact match.
# Ploidy correction is baked into the preprocessed data, so using the same
# .pt file preserves it automatically — only labels are permuted.
#
# Usage:
#   INPUT_DATA=/path/to/preprocessed.pt \
#   REAL_EXPERIMENT=/path/to/experiments/EXPERIMENT_NAME/fold_N \
#   REAL_RESULTS=/path/to/results/explainability \
#   OUTPUT_BASE=/path/to/output \
#   bash scripts/run_null_baseline_analysis.sh

set -e  # Exit on error

# Find Python interpreter
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

# Required parameters (set via environment variables)
INPUT_DATA="${INPUT_DATA:-}"
REAL_EXPERIMENT="${REAL_EXPERIMENT:-}"
REAL_RESULTS="${REAL_RESULTS:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
DEVICE="${DEVICE:-cuda}"

echo "=============================================="
echo "Null Baseline Attribution Analysis Pipeline"
echo "=============================================="
echo "Input data:       $INPUT_DATA"
echo "Real experiment:  $REAL_EXPERIMENT"
echo "Real results:     $REAL_RESULTS"
echo "Output base:      $OUTPUT_BASE"
echo ""

# -------------------------------------------------------------------
# Validate required parameters
# -------------------------------------------------------------------
if [ -z "$INPUT_DATA" ]; then
    echo "ERROR: INPUT_DATA is not set. Set it to the path of your preprocessed .pt file."
    exit 1
fi
if [ -z "$OUTPUT_BASE" ]; then
    echo "ERROR: OUTPUT_BASE is not set. Set it to the base output directory."
    exit 1
fi
if [ -z "$REAL_EXPERIMENT" ]; then
    echo "ERROR: REAL_EXPERIMENT is not set. Set it to the fold directory of the real experiment"
    echo "       (e.g. /path/to/experiments/CONFIG_G_FINAL_CV/fold_4)."
    exit 1
fi

if [ ! -f "$INPUT_DATA" ]; then
    echo "ERROR: Input data not found at: $INPUT_DATA"
    exit 1
fi
if [ ! -d "$REAL_EXPERIMENT" ]; then
    echo "ERROR: Real experiment directory not found at: $REAL_EXPERIMENT"
    exit 1
fi

# -------------------------------------------------------------------
# Read hyperparameters from the real experiment's config.yaml
# -------------------------------------------------------------------
# Look for config.yaml in the fold directory first, then parent
REAL_CONFIG=""
if [ -f "${REAL_EXPERIMENT}/config.yaml" ]; then
    REAL_CONFIG="${REAL_EXPERIMENT}/config.yaml"
elif [ -f "$(dirname "$REAL_EXPERIMENT")/config.yaml" ]; then
    REAL_CONFIG="$(dirname "$REAL_EXPERIMENT")/config.yaml"
fi

if [ -z "$REAL_CONFIG" ]; then
    echo "ERROR: No config.yaml found in $REAL_EXPERIMENT or its parent directory."
    echo "The null model needs the real experiment's config to match hyperparameters."
    exit 1
fi

echo "Reading hyperparameters from: $REAL_CONFIG"

# Extract all needed parameters from the YAML config using Python.
# This outputs KEY=VALUE lines that we eval to set shell variables.
CONFIG_VARS=$($PYTHON -c "
import yaml, shlex

with open('$REAL_CONFIG') as f:
    c = yaml.safe_load(f)

# Architecture parameters
print(f'CFG_LEVEL={shlex.quote(str(c.get(\"level\", \"L3\")))}')
print(f'CFG_LATENT_DIM={c.get(\"latent_dim\", 32)}')
print(f'CFG_HIDDEN_DIM={c.get(\"hidden_dim\", 64)}')
print(f'CFG_NUM_LAYERS={c.get(\"num_attention_layers\", 1)}')
print(f'CFG_NUM_HEADS={c.get(\"num_heads\", 4)}')
print(f'CFG_CHUNK_SIZE={c.get(\"chunk_size\", 3000)}')
print(f'CFG_CHUNK_OVERLAP={c.get(\"chunk_overlap\", 0)}')
print(f'CFG_AGGREGATION={shlex.quote(str(c.get(\"aggregation_method\", \"mean\")))}')

# Training parameters
print(f'CFG_LR={c.get(\"lr\", 0.00001)}')
print(f'CFG_LAMBDA_ATTR={c.get(\"lambda_attr\", 0.1)}')
print(f'CFG_BATCH_SIZE={c.get(\"batch_size\", 16)}')
print(f'CFG_GRAD_ACCUM={c.get(\"gradient_accumulation_steps\", 4)}')
grad_clip = c.get('gradient_clip')
print(f'CFG_GRAD_CLIP={grad_clip if grad_clip is not None else \"\"}')
print(f'CFG_EARLY_STOPPING={c.get(\"early_stopping\", 15)}')
print(f'CFG_EPOCHS={c.get(\"epochs\", 100)}')
print(f'CFG_SEED={c.get(\"seed\", 42)}')
print(f'CFG_GENOME_BUILD={shlex.quote(str(c.get(\"genome_build\", \"GRCh37\")))}')

# Sex covariate — read from config so null model matches exactly
sex_map = c.get('sex_map')
if sex_map and str(sex_map) not in ('None', 'null', ''):
    print(f'CFG_SEX_MAP={shlex.quote(str(sex_map))}')
else:
    print('CFG_SEX_MAP=')
")

eval "$CONFIG_VARS"

echo ""
echo "Config values read from real experiment:"
echo "  level:                $CFG_LEVEL"
echo "  latent_dim:           $CFG_LATENT_DIM"
echo "  hidden_dim:           $CFG_HIDDEN_DIM"
echo "  num_attention_layers: $CFG_NUM_LAYERS"
echo "  num_heads:            $CFG_NUM_HEADS"
echo "  chunk_size:           $CFG_CHUNK_SIZE"
echo "  aggregation_method:   $CFG_AGGREGATION"
echo "  lr:                   $CFG_LR"
echo "  lambda_attr:          $CFG_LAMBDA_ATTR"
echo "  batch_size:           $CFG_BATCH_SIZE"
echo "  gradient_clip:        ${CFG_GRAD_CLIP:-<none>}"
echo "  early_stopping:       $CFG_EARLY_STOPPING"
echo "  epochs:               $CFG_EPOCHS"
echo "  genome_build:         $CFG_GENOME_BUILD"
echo "  sex_map:              ${CFG_SEX_MAP:-<not set>}"
echo ""

# Validate sex map file if config says one was used
if [ -n "$CFG_SEX_MAP" ] && [ ! -f "$CFG_SEX_MAP" ]; then
    echo "ERROR: Sex map file from real experiment config not found at: $CFG_SEX_MAP"
    echo "The real model used sex as a covariate, so the null model must too."
    exit 1
fi

# -------------------------------------------------------------------
# Step 1: Create permuted dataset
# -------------------------------------------------------------------
echo "[Step 1/4] Creating permuted dataset..."
INPUT_BASENAME="$(basename "$INPUT_DATA" .pt)"
NULL_DATA="${OUTPUT_BASE}/data/${INPUT_BASENAME}_NULL.pt"

$PYTHON scripts/create_null_baseline.py \
    --input "$INPUT_DATA" \
    --output "$NULL_DATA" \
    --seed "$CFG_SEED"

echo ""

# -------------------------------------------------------------------
# Step 2: Train null model
# -------------------------------------------------------------------
# All hyperparameters are read from the real experiment's config.yaml.
# The only differences are:
#   - --preprocessed-data points to the null (permuted) .pt file
#   - --experiment-name is NULL_BASELINE
#   - --val-split 0.2 (single split instead of full CV — sufficient for null)
echo "[Step 2/4] Training null model..."
NULL_EXPERIMENT="${OUTPUT_BASE}/experiments/NULL_BASELINE"

TRAIN_CMD=(
    $PYTHON scripts/train.py
    --preprocessed-data "$NULL_DATA"
    --level "$CFG_LEVEL"
    --val-split 0.2
    --lr "$CFG_LR"
    --lambda-attr "$CFG_LAMBDA_ATTR"
    --early-stopping "$CFG_EARLY_STOPPING"
    --epochs "$CFG_EPOCHS"
    --batch-size "$CFG_BATCH_SIZE"
    --chunk-size "$CFG_CHUNK_SIZE"
    --chunk-overlap "$CFG_CHUNK_OVERLAP"
    --aggregation-method "$CFG_AGGREGATION"
    --gradient-accumulation-steps "$CFG_GRAD_ACCUM"
    --latent-dim "$CFG_LATENT_DIM"
    --hidden-dim "$CFG_HIDDEN_DIM"
    --num-heads "$CFG_NUM_HEADS"
    --num-attention-layers "$CFG_NUM_LAYERS"
    --seed "$CFG_SEED"
    --output-dir "${OUTPUT_BASE}/experiments"
    --experiment-name NULL_BASELINE
    --genome-build "$CFG_GENOME_BUILD"
    --device "$DEVICE"
)

# Add --gradient-clip if it was set in the real config
if [ -n "$CFG_GRAD_CLIP" ]; then
    TRAIN_CMD+=(--gradient-clip "$CFG_GRAD_CLIP")
fi

# Add --sex-map if the real model used it (critical for fair comparison)
if [ -n "$CFG_SEX_MAP" ]; then
    TRAIN_CMD+=(--sex-map "$CFG_SEX_MAP")
fi

"${TRAIN_CMD[@]}"

echo ""

# -------------------------------------------------------------------
# Step 3: Run explainability on null model
# -------------------------------------------------------------------
echo "[Step 3/4] Running explainability on null model..."
NULL_EXPLAIN_DIR="${OUTPUT_BASE}/results/null_attributions"

$PYTHON scripts/explain.py \
    --experiment-dir "$NULL_EXPERIMENT" \
    --preprocessed-data "$NULL_DATA" \
    --output-dir "$NULL_EXPLAIN_DIR" \
    --genome-build "$CFG_GENOME_BUILD" \
    --device "$DEVICE" \
    --is-null-baseline

echo ""

# -------------------------------------------------------------------
# Step 4: Compare real vs null
# -------------------------------------------------------------------
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
    --genome-build "$CFG_GENOME_BUILD" \
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
