# Installation

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM for real datasets)
- **Storage**: ~100MB for software, ~15GB for VEP cache, ~10GB for large preprocessed datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large cohorts
- **Ensembl VEP**: Required to annotate your VCF before preprocessing (see below)

### Step 1: Clone Repository

```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
```

### Step 2: Create Environment

**Option A: Using conda** (recommended)
```bash
conda create -n sieve python=3.10
conda activate sieve
pip install -e .
```

**Option B: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run test suite
python test_vcf_parser.py
python test_encoding_pipeline.py
python test_model_architecture.py
python test_training_pipeline.py
```

All tests should complete without errors. You can also run `pytest` for a more detailed test report.

### Dependencies Installed

Core packages:
- **PyTorch** 2.0+ (deep learning)
- **NumPy**, **Pandas**, **SciPy** (data processing)
- **cyvcf2/pysam** (VCF parsing)
- **captum** (integrated gradients)
- **scikit-learn** (metrics, preprocessing)
- **matplotlib** (visualisation)
- **PyYAML** (configuration)

See `pyproject.toml` for complete list.

### Step 4: Install Ensembl VEP

SIEVE requires VCF files annotated with Ensembl VEP. If your VCF is not already
annotated, install VEP and download the cache:

```bash
# Install VEP from bioconda
conda install -c bioconda ensembl-vep

# Download the cache for your genome build (one-time, ~15 GB)
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache
```

See [Detailed Usage — How to Annotate Your VCF](detailed-usage.md#how-to-annotate-your-vcf-with-ensembl-vep)
for the full VEP command and required flags.

### Conda Package Workflow

If you install SIEVE as a conda package (instead of editable source install), use the
`sieve-*` commands exposed by the package. A complete command-based walkthrough is in:

- `conda/USAGE.md`

---

