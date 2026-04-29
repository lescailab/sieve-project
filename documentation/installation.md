# Installation

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM for real datasets)
- **Storage**: ~100MB for software, ~15GB for VEP cache, ~10GB for large preprocessed datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large cohorts
- **Ensembl VEP**: Required to annotate your VCF before preprocessing (see below)

---

### Route 1: conda package (recommended)

The easiest way to install SIEVE is via the pre-built conda package on the
[lescailab](https://anaconda.org/lescailab) Anaconda channel.

#### Step 1: Create a conda environment

```bash
conda create -n sieve python=3.10
conda activate sieve
```

#### Step 2: Install SIEVE

```bash
conda install -c lescailab -c pytorch -c nvidia -c bioconda -c conda-forge sieve
```

The channel order matters: `lescailab` must appear first so that the SIEVE package
takes precedence, followed by `pytorch`, `nvidia`, `bioconda`, and `conda-forge` for
all dependencies.

#### Step 3: Verify installation

```bash
sieve --help
```

All `sieve-*` commands exposed by the package should be available immediately.

---

### Route 2: source install (for developers)

Use this route if you want to modify the source code or work with an unreleased version.

#### Step 1: Clone the repository

```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
```

#### Step 2: Create an environment and install

**Option A: Using conda**
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

#### Step 3: Verify installation

```bash
# Run test suite
python test_vcf_parser.py
python test_encoding_pipeline.py
python test_model_architecture.py
python test_training_pipeline.py
```

All tests should complete without errors. You can also run `pytest` for a more detailed test report.

---

### Dependencies

Core packages installed by either route:
- **PyTorch** 2.0+ (deep learning)
- **NumPy**, **Pandas**, **SciPy** (data processing)
- **cyvcf2/pysam** (VCF parsing)
- **captum** (integrated gradients)
- **scikit-learn** (metrics, preprocessing)
- **matplotlib** (visualisation)
- **PyYAML** (configuration)

See `pyproject.toml` for the complete list.

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

If you installed SIEVE via conda (Route 1), use the `sieve-*` commands exposed by the
package. A complete command-based walkthrough is in:

- `conda/USAGE.md`

---

