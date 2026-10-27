# Quick Start

### For the Impatient

Installing SIEVE puts 24 `sieve-*` commands on your `PATH`. Those are the
primary form and are used throughout this guide. Every one of them is
equivalent to running its script directly, so contributors working from a
checkout can substitute `python scripts/<name>.py` anywhere a `sieve-*`
command appears. See the [Command Reference](command-reference.md) for the
full mapping.

```bash
# 1. Install
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
pip install -e .

# 2. Annotate your VCF with Ensembl VEP (if not already done)
#    Install VEP and download cache (once):
conda install -c bioconda ensembl-vep
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache

#    Run annotation:
vep \
    --input_file your_data.vcf.gz \
    --output_file your_data_vep.vcf.gz \
    --vcf \
    --compress_output bgzip \
    --symbol \
    --canonical \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache \
    --dir_cache /path/to/vep_cache \
    --fork 4
tabix -p vcf your_data_vep.vcf.gz

# 3. (Optional, recommended) Infer genetic sex for ploidy-aware encoding
sieve-infer-sex \
    --vcf your_data_vep.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37

# 4. Preprocess (once)
sieve-preprocess \
    --vcf your_data_vep.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37

# 5. Train
sieve-train \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --experiment-name my_model \
    --output-dir experiments \
    --device cuda

# 6. Explain
sieve-explain \
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

# 7. (Optional) Correct chrX ploidy bias for ranking/visualisation
#    Run AFTER step 6 on the significance-annotated file so that
#    empirical_p_variant and fdr_variant columns are preserved in the output.
sieve-correct-chrx-bias \
    --rankings results/null_baseline_run/results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/null_baseline_run/results/attribution_comparison/corrected \
    --include-sex-chroms
```

### 5-Minute Test Run

```bash
# Use included test data
sieve-train \
    --vcf test_data/small/test_data.vcf.gz \
    --phenotypes test_data/small/test_data_phenotypes.tsv \
    --level L3 \
    --epochs 20 \
    --batch-size 8 \
    --output-dir test_run
```

Working from a checkout instead of an installed environment? The same run is
`python scripts/train.py` with identical arguments.

---
