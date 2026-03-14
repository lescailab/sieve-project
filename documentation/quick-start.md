# Quick Start

### For the Impatient

```bash
# 1. Install
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
pip install -e .

# 2. (Optional, recommended) Infer genetic sex for ploidy-aware encoding
python scripts/infer_sex.py \
    --vcf your_data.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37

# 3. Preprocess (once)
python scripts/preprocess.py \
    --vcf your_data.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37

# 4. Train
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --experiment-name my_model \
    --output-dir experiments \
    --device cuda

# 5. Explain
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability

# 6. Validate (null baseline)
# Set required environment variables first
export INPUT_DATA="preprocessed.pt"
export REAL_EXPERIMENT="experiments/my_model"            # or experiments/my_model/fold_0
export REAL_RESULTS="results/explainability"             # directory with sieve_variant_rankings.csv
export OUTPUT_BASE="results/null_baseline_run"           # where null outputs will be written
bash scripts/run_null_baseline_analysis.sh

# 7. (Optional) Correct chrX ploidy bias in rankings
python scripts/correct_chrx_bias.py \
    --rankings results/explainability/sieve_variant_rankings.csv \
    --null-rankings results/null_baseline_run/results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/explainability_corrected \
    --exclude-sex-chroms
```

### 5-Minute Test Run

```bash
# Use included test data
python scripts/train.py \
    --vcf test_data/small/test_data.vcf.gz \
    --phenotypes test_data/small/test_data_phenotypes.tsv \
    --level L3 \
    --epochs 20 \
    --batch-size 8 \
    --output-dir test_run
```

---

