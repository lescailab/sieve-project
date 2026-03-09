# Installation

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM for real datasets)
- **Storage**: ~100MB for software, ~10GB for large preprocessed datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large cohorts

### Step 1: Install SIEVE

**Option A: Conda package** (recommended)
```bash
conda create -n sieve -c lescailab -c conda-forge sieve
conda activate sieve
```

This installs a release build with all dependencies. Pipeline commands are available as `sieve-*` entry points (e.g. `sieve-train`, `sieve-explain`). See `conda/USAGE.md` for the full command-based walkthrough.

**Option B: Development install** (for contributors or latest features)
```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
conda create -n sieve python=3.10
conda activate sieve
pip install -e .
```

This installs from source in editable mode. Use `python scripts/...` to run pipeline steps.

**Option C: Using venv** (development install without conda)
```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -e .
```

### Step 2: Verify Installation

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

### Conda Package vs Development Install

| | Conda package (`lescailab::sieve`) | Development install (`pip install -e .`) |
|---|---|---|
| **Commands** | `sieve-train`, `sieve-explain`, etc. | `python scripts/train.py`, etc. |
| **Updates** | `conda update -c lescailab sieve` | `git pull` |
| **Use when** | Running analyses | Contributing or testing latest features |

For full conda command reference see `conda/USAGE.md`.

---
