# Claude Code Session Guide

**For AI Assistants Only** - End users should see USER_GUIDE.md

## Quick Reference

- **Repository**: https://github.com/lescailab/sieve-project
- **User Documentation**: USER_GUIDE.md (comprehensive end-user manual)
- **Branch naming**: `claude/<description>-<session-id>`

## Project Status (2026-02-04)

**Completed Phases**:
- ✅ Phase 1A-D: Data pipeline, model architecture, training, basic explainability
- ✅ Phase 2: Chunked processing for whole-genome coverage
- ✅ Phase 3: Null baseline attribution analysis pipeline

**Current Capabilities**:
- Multi-level annotation training (L0-L4)
- Chunked processing for memory efficiency
- Integrated gradients attribution
- Attention analysis for epistasis
- Null baseline statistical validation
- Biological validation (ClinVar, GWAS, GO)

## Critical Development Rules

### 1. ALWAYS Test Before Pushing

```
1. Write code
2. Run tests ← MANDATORY
3. Fix failures
4. Commit
5. Push
```

Never skip testing. Create tests if they don't exist.

### 2. Code Quality Standards

**Imports**:
- Remove unused imports (ruff will flag)
- Order: stdlib, third-party, local

**Random Number Generation**:
```python
# ✓ GOOD
rng = np.random.default_rng(seed)
result = rng.permutation(array)

# ✗ BAD
np.random.seed(seed)  # Mutates global state
result = np.random.permutation(array)
```

**matplotlib**:
```python
# ✓ GOOD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ✗ BAD
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Too late!
```

## File Structure

```
sieve-project/
├── USER_GUIDE.md          # Comprehensive end-user manual
├── README.md              # Project overview
├── CLAUDE.md              # Project context for AI
├── ARCHITECTURE.md        # Technical specifications
├── EXPERIMENTS.md         # Scientific protocol
├── src/                   # Source code
│   ├── data/              # VCF parsing, datasets
│   ├── encoding/          # Feature engineering
│   ├── models/            # Neural network components
│   ├── training/          # Training loop
│   └── explain/           # Attribution, ranking
├── scripts/               # Entry points
│   ├── preprocess.py
│   ├── train.py
│   ├── explain.py
│   ├── create_null_baseline.py
│   ├── compare_attributions.py
│   └── run_null_baseline_analysis.sh
└── tests/                 # Unit tests
```

## Common Tasks

### Adding a New Feature

1. Read relevant documentation (CLAUDE.md, ARCHITECTURE.md)
2. Implement with proper type hints and docstrings
3. **Write tests**
4. Run tests locally
5. Commit with semantic message
6. Push to feature branch

### Fixing a Bug

1. Reproduce the bug with a test
2. Fix the issue
3. Verify test passes
4. Add regression test if needed
5. Commit and push

### Updating Documentation

- **End-user docs**: Update USER_GUIDE.md
- **Developer docs**: Update CLAUDE.md or ARCHITECTURE.md
- **API changes**: Update docstrings

## Critical Known Issues

### Position Collision Bug (FIXED)

Variants on different chromosomes can have same position number.
Always use `(chrom, pos, gene)` as key, never just `(pos, gene)`.

### Memory Management

For large datasets use:
```bash
--batch-size 2 \
--gradient-accumulation-steps 16 \
--chunk-size 3000
```

### Chromosome Coverage

Chunked processing ensures ALL chromosomes are analyzed (not just chr1/chr2).
Verify with diagnostic output.

## Testing Patterns

### Unit Test
```python
def test_feature():
    """Test specific functionality."""
    result = function(input)
    assert result == expected
```

### Integration Test
```python
def test_pipeline():
    """Test end-to-end workflow."""
    # Use synthetic data
    # Run full pipeline
    # Verify outputs
```

## Commit Message Format

```
<type>: <description>

<detailed explanation>

<session URL>
```

Types: feat, fix, docs, test, refactor

Always include session URL.

## When Stuck

1. Check USER_GUIDE.md for end-user perspective
2. Check CLAUDE.md for project context
3. Check ARCHITECTURE.md for technical specs
4. Run existing tests
5. Create minimal reproduction

## Key Reminders

- Test before push (non-negotiable)
- No unused imports
- No global RNG state mutation
- matplotlib backend before pyplot import
- Use `(chrom, pos, gene)` keys to avoid collisions
- Validate paths in shell scripts

## Resources

- USER_GUIDE.md: Complete end-user documentation
- CLAUDE.md: Full project context
- ARCHITECTURE.md: Model specifications
- EXPERIMENTS.md: Scientific protocol
- GitHub: https://github.com/lescailab/sieve-project
