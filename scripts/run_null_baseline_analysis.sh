#!/bin/bash

# Null Baseline Attribution Analysis Pipeline
# ============================================
# This script runs the complete null baseline analysis:
# 1. Creates permuted dataset
# 2. Trains null model (identical conditions to real model, only labels permuted)
# 3. Runs explainability on null model
# 4. Compares raw real vs raw null attributions,
#    producing empirical p-values and BH-FDR significance columns
#
# The null comparison operates on raw mean_attribution values from both models.
# Both models saw the same input data with the same chrX inflation — the only
# difference is the labels.  The raw attribution magnitude IS the signal, and
# the chrX inflation cancels in the empirical comparison because both sides
# are equally inflated.
#
# ChrX correction (correct_chrx_bias.py) should be applied SEPARATELY to the
# significance-annotated real rankings for ranking and visualisation purposes,
# AFTER this pipeline completes.
#
# This script produces null-contrasted rankings with empirical p-values and FDR
# correction.  The significance output directory (COMPARISON_DIR) is derived
# from OUTPUT_BASE:
#   - Preferred mode (PROJECT_DIR + LEVEL):
#       ${OUTPUT_BASE}/real_experiments/${LEVEL}/attributions/
#   - Legacy mode (OUTPUT_BASE set explicitly):
#       ${OUTPUT_BASE}/results/attribution_comparison/
# The following files are written to COMPARISON_DIR:
#   variant_rankings_with_significance.csv
#   gene_rankings_with_significance.csv
#   significance_summary.yaml
#
# IMPORTANT: The null model must be trained under identical conditions to the
# real model.  The only difference is the permuted labels.  Hyperparameters are
# read from the real experiment's config.yaml to guarantee an exact match.
# Ploidy correction is baked into the preprocessed data, so using the same
# .pt file preserves it automatically — only labels are permuted.
#
# Usage (preferred):
#   PROJECT_DIR=/path/to/CohortName \
#   LEVEL=L3 \
#   bash scripts/run_null_baseline_analysis.sh
#
# Usage (legacy, still supported):
#   INPUT_DATA=/path/to/preprocessed.pt \
#   REAL_EXPERIMENT=/path/to/training \
#   REAL_RESULTS=/path/to/attributions \
#   OUTPUT_BASE=/path/to/output \
#   bash scripts/run_null_baseline_analysis.sh
#
# Optional:
#   NULL_DATA=/path/to/permuted.pt \      # Skip permutation step if already generated
#   EXCLUDE_SEX_CHROMS=1 \                # Pass --exclude-sex-chroms to compare step

set -e  # Exit on error

# Resolve the repository root from this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find Python interpreter
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# Path derivation: PROJECT_DIR + LEVEL → all other paths
# ---------------------------------------------------------------------------
# When PROJECT_DIR and LEVEL are set, derive the standard layout paths.
# Explicit overrides (REAL_EXPERIMENT, REAL_RESULTS, OUTPUT_BASE, INPUT_DATA,
# NULL_DATA) take precedence if set before calling this script.
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-}"
LEVEL="${LEVEL:-}"

if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    # Real-side paths (read-only inputs)
    REAL_EXPERIMENT="${REAL_EXPERIMENT:-${PROJECT_DIR}/real_experiments/${LEVEL}/training}"
    REAL_RESULTS="${REAL_RESULTS:-${PROJECT_DIR}/real_experiments/${LEVEL}/attributions}"

    # Auto-detect input data from ${PROJECT_DIR}/data/ if not already set
    if [ -z "$INPUT_DATA" ]; then
        INPUT_DATA=$(find "${PROJECT_DIR}/data" -maxdepth 1 -name 'preprocessed*.pt' \
                     ! -name '*_NULL*' 2>/dev/null | head -1)
    fi

    # Auto-detect null data from ${PROJECT_DIR}/data/ if not already set
    if [ -z "$NULL_DATA" ]; then
        NULL_DATA_CANDIDATE=$(find "${PROJECT_DIR}/data" -maxdepth 1 -name '*_NULL.pt' \
                              2>/dev/null | head -1)
        [ -f "${NULL_DATA_CANDIDATE:-}" ] && NULL_DATA="$NULL_DATA_CANDIDATE"
    fi

    # Output base for null model and null attributions
    OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}}"
fi

# Required parameters (set via environment variables or derived above)
INPUT_DATA="${INPUT_DATA:-}"
REAL_EXPERIMENT="${REAL_EXPERIMENT:-}"
REAL_RESULTS="${REAL_RESULTS:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
NULL_DATA="${NULL_DATA:-}"
DEVICE="${DEVICE:-cuda}"
EXCLUDE_SEX_CHROMS="${EXCLUDE_SEX_CHROMS:-0}"

echo "=============================================="
echo "Null Baseline Attribution Analysis Pipeline"
echo "=============================================="
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    echo "Project dir:      $PROJECT_DIR"
    echo "Level:            $LEVEL"
fi
echo "Input data:       $INPUT_DATA"
echo "Real experiment:  $REAL_EXPERIMENT"
echo "Real results:     $REAL_RESULTS"
echo "Output base:      $OUTPUT_BASE"
echo "Null data:        ${NULL_DATA:-<will be generated>}"
echo ""

# -------------------------------------------------------------------
# Validate required parameters
# -------------------------------------------------------------------
if [ -z "$INPUT_DATA" ]; then
    echo "ERROR: INPUT_DATA is not set."
    echo "  Either set PROJECT_DIR and LEVEL (preferred), or set INPUT_DATA directly."
    exit 1
fi
if [ -z "$OUTPUT_BASE" ]; then
    echo "ERROR: OUTPUT_BASE is not set."
    echo "  Either set PROJECT_DIR and LEVEL (preferred), or set OUTPUT_BASE directly."
    exit 1
fi
if [ -z "$REAL_EXPERIMENT" ]; then
    echo "ERROR: REAL_EXPERIMENT is not set."
    echo "  Either set PROJECT_DIR and LEVEL (preferred), or set REAL_EXPERIMENT directly"
    echo "  to the training directory of the real experiment."
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
# The config path is passed via environment variable to avoid shell
# injection issues with special characters in paths.
CONFIG_VARS=$(REAL_CONFIG="$REAL_CONFIG" "$PYTHON" -c "
import os, sys, yaml, shlex

config_path = os.environ['REAL_CONFIG']
try:
    with open(config_path) as f:
        c = yaml.safe_load(f)
except Exception as e:
    print(f'ERROR: Failed to parse {config_path}: {e}', file=sys.stderr)
    sys.exit(1)

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
print(f'CFG_CLASS_WEIGHTING={shlex.quote(str(c.get(\"class_weighting\", \"auto\")))}')
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

pc_map = c.get('pc_map')
if pc_map and str(pc_map) not in ('None', 'null', ''):
    print(f'CFG_PC_MAP={shlex.quote(str(pc_map))}')
else:
    print('CFG_PC_MAP=')
print(f'CFG_NUM_PCS={int(c.get(\"num_pcs\", 0) or 0)}')

# Training protocol: CV folds vs single val-split
cv_folds = c.get('cv_folds', c.get('cv'))
val_split = c.get('val_split')
if cv_folds and str(cv_folds) not in ('None', 'null', '0', ''):
    print(f'CFG_CV_FOLDS={int(cv_folds)}')
    print('CFG_VAL_SPLIT=')
elif val_split and str(val_split) not in ('None', 'null', ''):
    print('CFG_CV_FOLDS=')
    print(f'CFG_VAL_SPLIT={val_split}')
else:
    print('CFG_CV_FOLDS=')
    print('CFG_VAL_SPLIT=0.2')
")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to read config from $REAL_CONFIG"
    exit 1
fi

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
echo "  class_weighting:      $CFG_CLASS_WEIGHTING"
echo "  gradient_clip:        ${CFG_GRAD_CLIP:-<none>}"
echo "  early_stopping:       $CFG_EARLY_STOPPING"
echo "  epochs:               $CFG_EPOCHS"
echo "  genome_build:         $CFG_GENOME_BUILD"
echo "  sex_map:              ${CFG_SEX_MAP:-<not set>}"
echo "  pc_map:               ${CFG_PC_MAP:-<not set>}"
echo "  num_pcs:              ${CFG_NUM_PCS}"
if [ -n "$CFG_CV_FOLDS" ]; then
    echo "  training_protocol:    CV (${CFG_CV_FOLDS} folds)"
elif [ -n "$CFG_VAL_SPLIT" ]; then
    echo "  training_protocol:    single split (val_split=${CFG_VAL_SPLIT})"
else
    echo "  training_protocol:    single split (val_split=0.2, fallback)"
fi
echo ""

# Validate that LEVEL (directory routing) matches CFG_LEVEL (model config).
# When PROJECT_DIR+LEVEL are used for path derivation, a mismatch would
# silently place outputs under the wrong level directory.
if [ -n "$LEVEL" ] && [ "$LEVEL" != "$CFG_LEVEL" ]; then
    echo "ERROR: LEVEL=${LEVEL} does not match the annotation level in the real experiment"
    echo "       config (level=${CFG_LEVEL} from ${REAL_CONFIG})."
    echo "  The directory layout uses LEVEL for path routing, but the model will be"
    echo "  trained with the config's level (${CFG_LEVEL}). These must match to avoid"
    echo "  placing outputs in the wrong directory."
    echo "  Either set LEVEL=${CFG_LEVEL} or correct the real experiment config."
    exit 1
fi

# Validate sex map file if config says one was used
if [ -n "$CFG_SEX_MAP" ] && [ ! -f "$CFG_SEX_MAP" ]; then
    echo "ERROR: Sex map file from real experiment config not found at: $CFG_SEX_MAP"
    echo "The real model used sex as a covariate, so the null model must too."
    exit 1
fi
if [ "$CFG_NUM_PCS" -gt 0 ] && [ -z "$CFG_PC_MAP" ]; then
    echo "ERROR: num_pcs=${CFG_NUM_PCS} but no pc_map path was found in the real config."
    exit 1
fi
if [ -n "$CFG_PC_MAP" ] && [ ! -f "$CFG_PC_MAP" ]; then
    echo "ERROR: PC map file from real experiment config not found at: $CFG_PC_MAP"
    exit 1
fi

# -------------------------------------------------------------------
# Step 1: Create permuted dataset (or use pre-existing one)
# -------------------------------------------------------------------
if [ -n "$NULL_DATA" ]; then
    if [ ! -f "$NULL_DATA" ]; then
        echo "ERROR: NULL_DATA was set but file not found at: $NULL_DATA"
        exit 1
    fi
    echo "[Step 1/4] Using pre-existing permuted dataset: $NULL_DATA"
else
    INPUT_BASENAME="$(basename "$INPUT_DATA" .pt)"
    NULL_DATA="${OUTPUT_BASE}/data/${INPUT_BASENAME}_NULL.pt"
    if [ -f "$NULL_DATA" ]; then
        echo "[Step 1/4] Reusing existing permuted dataset: $NULL_DATA"
    else
        echo "[Step 1/4] Creating permuted dataset..."
        mkdir -p "$(dirname "$NULL_DATA")"
        "$PYTHON" "$REPO_ROOT/scripts/create_null_baseline.py" \
            --input "$INPUT_DATA" \
            --output "$NULL_DATA" \
            --seed "$CFG_SEED"
    fi
fi

echo ""

# -------------------------------------------------------------------
# Step 2: Train null model
# -------------------------------------------------------------------
# All hyperparameters are read from the real experiment's config.yaml.
# The training protocol (CV folds vs single split) is matched to the
# real model: if the real run used CV with N folds, the null run uses
# CV with N folds.  If it used a single split, the null uses the same
# fraction.  The only differences from the real run are:
#   - --preprocessed-data points to the null (permuted) .pt file
#   - --experiment-name is "training" (nested under null_baselines/{LEVEL})
echo "[Step 2/4] Training null model..."
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    NULL_EXPERIMENT="${OUTPUT_BASE}/null_baselines/${LEVEL}/training"
else
    NULL_EXPERIMENT="${OUTPUT_BASE}/experiments/NULL_BASELINE"
fi

# Determine null model checkpoint path (depends on protocol)
if [ -n "$CFG_CV_FOLDS" ]; then
    # CV mode — checkpoint is selected by select_best_cv_fold.py after training
    NULL_MODEL_CHECKPOINT=""   # determined after training
else
    NULL_MODEL_CHECKPOINT="${NULL_EXPERIMENT}/best_model.pt"
fi

if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    TRAIN_OUTPUT_DIR="${OUTPUT_BASE}/null_baselines/${LEVEL}"
    TRAIN_EXPERIMENT_NAME="training"
else
    TRAIN_OUTPUT_DIR="${OUTPUT_BASE}/experiments"
    TRAIN_EXPERIMENT_NAME="NULL_BASELINE"
fi

TRAIN_CMD=(
    "$PYTHON" "$REPO_ROOT/scripts/train.py"
    --preprocessed-data "$NULL_DATA"
    --level "$CFG_LEVEL"
    --lr "$CFG_LR"
    --lambda-attr "$CFG_LAMBDA_ATTR"
    --early-stopping "$CFG_EARLY_STOPPING"
    --epochs "$CFG_EPOCHS"
    --batch-size "$CFG_BATCH_SIZE"
    --class-weighting "$CFG_CLASS_WEIGHTING"
    --chunk-size "$CFG_CHUNK_SIZE"
    --chunk-overlap "$CFG_CHUNK_OVERLAP"
    --aggregation-method "$CFG_AGGREGATION"
    --gradient-accumulation-steps "$CFG_GRAD_ACCUM"
    --latent-dim "$CFG_LATENT_DIM"
    --hidden-dim "$CFG_HIDDEN_DIM"
    --num-heads "$CFG_NUM_HEADS"
    --num-attention-layers "$CFG_NUM_LAYERS"
    --seed "$CFG_SEED"
    --output-dir "$TRAIN_OUTPUT_DIR"
    --experiment-name "$TRAIN_EXPERIMENT_NAME"
    --genome-build "$CFG_GENOME_BUILD"
    --device "$DEVICE"
)

# Match the real-model training protocol (CV vs single split)
if [ -n "$CFG_CV_FOLDS" ]; then
    TRAIN_CMD+=(--cv-folds "$CFG_CV_FOLDS")
elif [ -n "$CFG_VAL_SPLIT" ]; then
    TRAIN_CMD+=(--val-split "$CFG_VAL_SPLIT")
else
    TRAIN_CMD+=(--val-split 0.2)   # fallback: identical to previous default
fi

# Add --gradient-clip if it was set in the real config
if [ -n "$CFG_GRAD_CLIP" ]; then
    TRAIN_CMD+=(--gradient-clip "$CFG_GRAD_CLIP")
fi

# Add --sex-map if the real model used it (critical for fair comparison)
if [ -n "$CFG_SEX_MAP" ]; then
    TRAIN_CMD+=(--sex-map "$CFG_SEX_MAP")
fi
if [ "$CFG_NUM_PCS" -gt 0 ]; then
    TRAIN_CMD+=(--pc-map "$CFG_PC_MAP" --num-pcs "$CFG_NUM_PCS")
fi

# Check whether null training is already complete
if [ -n "$CFG_CV_FOLDS" ]; then
    # CV mode: training complete when cv_results.yaml exists in null experiment dir
    NULL_CV_RESULTS="${NULL_EXPERIMENT}/cv_results.yaml"
    if [ -f "$NULL_CV_RESULTS" ]; then
        echo "  Found existing null CV results at: $NULL_CV_RESULTS"
        echo "  Skipping null model training."
    else
        "${TRAIN_CMD[@]}"
    fi

    # Select best null fold by AUC from cv_results.yaml
    echo "  Selecting best null fold..."
    if [ ! -f "${NULL_EXPERIMENT}/cv_results.yaml" ]; then
        echo "ERROR: Expected null CV results file not found: ${NULL_EXPERIMENT}/cv_results.yaml"
        exit 1
    fi
    NULL_BEST_FOLD=$("$PYTHON" -c "
import yaml
with open('${NULL_EXPERIMENT}/cv_results.yaml') as f:
    r = yaml.safe_load(f)
best = max(range(len(r['fold_results'])), key=lambda i: r['fold_results'][i]['auc'])
print(best)
")

    NULL_MODEL_CHECKPOINT="${NULL_EXPERIMENT}/fold_${NULL_BEST_FOLD}/best_model.pt"
    NULL_EXPLAIN_INPUT_DIR="${NULL_EXPERIMENT}/fold_${NULL_BEST_FOLD}"
    echo "  Null best fold: ${NULL_BEST_FOLD} (checkpoint: ${NULL_MODEL_CHECKPOINT})"
else
    # Single-split mode
    if [ -f "$NULL_MODEL_CHECKPOINT" ]; then
        echo "  Found existing null model checkpoint at: $NULL_MODEL_CHECKPOINT"
        echo "  Skipping null model training."
    else
        "${TRAIN_CMD[@]}"
    fi
    NULL_EXPLAIN_INPUT_DIR="$NULL_EXPERIMENT"
fi

echo ""

# -------------------------------------------------------------------
# Step 3: Run explainability on null model
# -------------------------------------------------------------------
echo "[Step 3/4] Running explainability on null model..."
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    NULL_EXPLAIN_DIR="${OUTPUT_BASE}/null_baselines/${LEVEL}/attributions"
else
    NULL_EXPLAIN_DIR="${OUTPUT_BASE}/results/null_attributions"
fi
NULL_RAW_RANKINGS="${NULL_EXPLAIN_DIR}/sieve_variant_rankings.csv"

if [ -f "$NULL_RAW_RANKINGS" ]; then
    echo "  Found existing null explainability output at: $NULL_RAW_RANKINGS"
    echo "  Skipping null explainability."
else
    EXPLAIN_CMD=(
        "$PYTHON" "$REPO_ROOT/scripts/explain.py"
        --experiment-dir "$NULL_EXPLAIN_INPUT_DIR"
        --preprocessed-data "$NULL_DATA"
        --output-dir "$NULL_EXPLAIN_DIR"
        --genome-build "$CFG_GENOME_BUILD"
        --device "$DEVICE"
        --is-null-baseline
    )
    if [ "$CFG_NUM_PCS" -gt 0 ]; then
        EXPLAIN_CMD+=(--pc-map "$CFG_PC_MAP" --num-pcs "$CFG_NUM_PCS")
    fi
    "${EXPLAIN_CMD[@]}"
fi

echo ""

# -------------------------------------------------------------------
# Step 4: Compare raw real vs raw null attributions
# -------------------------------------------------------------------
# The comparison operates on raw mean_attribution from both models.
# Both models saw the same input data (same chrX inflation), so the
# chrX effect cancels in the empirical comparison.  ChrX correction
# should be applied separately to the significance-annotated real
# rankings for visualisation.
echo "[Step 4/4] Comparing raw real vs raw null attributions..."

REAL_RAW_RANKINGS="${REAL_RESULTS}/sieve_variant_rankings.csv"

if [ -z "$REAL_RESULTS" ]; then
    echo "ERROR: REAL_RESULTS is not set."
    echo "Set REAL_RESULTS to the directory containing sieve_variant_rankings.csv"
    echo "from the real model's explainability run."
    exit 1
fi

if [ ! -f "$REAL_RAW_RANKINGS" ]; then
    echo "ERROR: Real rankings not found at: $REAL_RAW_RANKINGS"
    echo "Ensure sieve_variant_rankings.csv exists in REAL_RESULTS."
    exit 1
fi

# Significance output goes into the real-side attributions directory
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    COMPARISON_DIR="${OUTPUT_BASE}/real_experiments/${LEVEL}/attributions"
else
    COMPARISON_DIR="${OUTPUT_BASE}/results/attribution_comparison"
fi

COMPARE_CMD=(
    "$PYTHON" "$REPO_ROOT/scripts/compare_attributions.py"
    --real "$REAL_RAW_RANKINGS"
    --null "$NULL_RAW_RANKINGS"
    --output-dir "$COMPARISON_DIR"
    --genome-build "$CFG_GENOME_BUILD"
)

# Optionally exclude sex chromosomes from the significance computation.
# Set EXCLUDE_SEX_CHROMS=1 in the calling environment to enable this.
if [ "${EXCLUDE_SEX_CHROMS:-0}" = "1" ]; then
    COMPARE_CMD+=(--exclude-sex-chroms)
fi

"${COMPARE_CMD[@]}"

echo ""
echo "=============================================="
echo "Null Baseline Analysis Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    echo "  - Null model:                ${OUTPUT_BASE}/null_baselines/${LEVEL}/training"
    echo "  - Null attributions:         ${OUTPUT_BASE}/null_baselines/${LEVEL}/attributions"
    echo "  - Significance results:      $COMPARISON_DIR"
else
    echo "  - Null model:                $NULL_EXPERIMENT"
    echo "  - Null attributions:         $NULL_EXPLAIN_DIR"
    echo "  - Significance results:      $COMPARISON_DIR"
fi
echo ""
echo "Key files to review:"
echo "  - ${COMPARISON_DIR}/variant_rankings_with_significance.csv"
echo "  - ${COMPARISON_DIR}/gene_rankings_with_significance.csv"
echo "  - ${COMPARISON_DIR}/significance_summary.yaml"
echo ""
echo "Next step: apply chrX correction to real rankings for visualisation:"
echo "  python scripts/correct_chrx_bias.py \\"
echo "      --rankings ${COMPARISON_DIR}/variant_rankings_with_significance.csv \\"
if [ -n "$PROJECT_DIR" ] && [ -n "$LEVEL" ]; then
    echo "      --project-dir ${PROJECT_DIR} \\"
else
    echo "      --output-dir ${REAL_RESULTS}/corrected \\"
fi
echo "      --include-sex-chroms \\"
echo "      --genome-build ${CFG_GENOME_BUILD}"
