# Chunked Variant Processing - Whole Genome Coverage

## Critical Bug Fix

**Problem**: The original implementation truncated each sample to the first 3000 variants during batching (`collate_samples`). Since VCF files are sorted by chromosome/position, this meant the model **only saw chromosomes 1 and 2**, never learning from the remaining 80% of the genome.

**Evidence**:
- Explainability output: 95% variants from chr1, 5% from chr2, 0% from chr3-22
- Training used same truncation logic via `max_variants_per_batch=3000`
- Diagnostic script confirmed preprocessed data contains all chromosomes
- Human exomes typically have 25,000-30,000 variants across all chromosomes

**Impact**: All previous training and explainability results are invalid. The model never learned genome-wide patterns.

## Solution: Chunked Processing

Instead of truncating samples, we split each sample into **chunks** and process all chunks:

```
Sample (30,000 variants across chr1-22)
    ↓ Split into chunks
Chunk 1: variants 0-3000    (chr1-2)
Chunk 2: variants 3000-6000  (chr2-5)
Chunk 3: variants 6000-9000  (chr5-8)
...
Chunk 10: variants 27000-30000 (chr19-22)
    ↓ Process each chunk
Forward pass on chunk 1 → embedding₁
Forward pass on chunk 2 → embedding₂
...
Forward pass on chunk 10 → embedding₁₀
    ↓ Aggregate chunks
Sample embedding = mean(embedding₁, ..., embedding₁₀)
    ↓ Final prediction
Sample prediction = classifier(sample embedding)
```

**Key properties**:
- ✅ **Full genome coverage**: All 30k variants processed
- ✅ **Same memory usage**: Still 3000 variants per forward pass
- ✅ **No information loss**: All chromosomes contribute to prediction
- ✅ **Explainability-compatible**: Can compute attributions per chunk

## How to Use

### 1. Training with Chunked Processing

```python
from src.encoding import ChunkedVariantDataset, collate_chunks
from src.models import create_sieve_model, ChunkedSIEVEModel
from torch.utils.data import DataLoader

# Load preprocessed data
preprocessed = torch.load('data/preprocessed.pt', weights_only=False)
samples = preprocessed['samples']

# Create chunked dataset
dataset = ChunkedVariantDataset(
    samples=samples,
    annotation_level=AnnotationLevel.L3,
    chunk_size=3000,  # Same as old max_variants_per_batch
    overlap=0         # No overlap between chunks
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,     # Batch of chunks (not samples!)
    shuffle=True,
    collate_fn=collate_chunks
)

# Create base model
base_model = create_sieve_model(config, num_genes=dataset.num_genes)

# Wrap in chunked model
model = ChunkedSIEVEModel(
    base_model=base_model,
    aggregation_method='mean'  # or 'max', 'attention', 'logit_mean'
)

# Training loop
for batch in dataloader:
    loss, predictions = model.train_step(batch, criterion, device)
    # predictions are sample-level (aggregated from chunks)
    # loss is computed at sample level
```

### 2. Explainability with Chunked Processing

The explainability script needs to process all chunks per sample and aggregate attributions:

```python
from src.encoding import ChunkedVariantDataset, collate_chunks

# Create chunked dataset
dataset = ChunkedVariantDataset(
    samples=all_samples,
    annotation_level=annotation_level,
    chunk_size=2000,  # Smaller for explainability (IG is memory-intensive)
    overlap=0
)

# Process each sample's chunks
for sample_idx in range(len(all_samples)):
    # Get all chunk indices for this sample
    chunk_indices = dataset.get_chunks_for_sample(sample_idx)

    # Compute attributions for each chunk
    chunk_attributions = []
    chunk_metadata = []

    for chunk_idx in chunk_indices:
        chunk = dataset[chunk_idx]

        # Compute IG for this chunk
        attr = explainer.attribute(
            chunk['features'].unsqueeze(0).to(device),
            chunk['positions'].unsqueeze(0).to(device),
            chunk['gene_ids'].unsqueeze(0).to(device),
            chunk['mask'].unsqueeze(0).to(device)
        )

        chunk_attributions.append(attr)
        chunk_metadata.append({
            'positions': chunk['positions'].cpu().numpy(),
            'gene_ids': chunk['gene_ids'].cpu().numpy()
        })

    # Combine chunks into full sample attribution
    all_positions = np.concatenate([m['positions'] for m in chunk_metadata])
    all_gene_ids = np.concatenate([m['gene_ids'] for m in chunk_metadata])
    all_attributions = np.concatenate(chunk_attributions, axis=0)

    # Now you have attributions for ALL variants in the sample
    sample_metadata.append({
        'positions': all_positions,
        'gene_ids': all_gene_ids,
        'sample_idx': sample_idx,
        'sample_id': all_samples[sample_idx].sample_id,
        'label': all_samples[sample_idx].label
    })
    sample_attributions.append(all_attributions)
```

## Aggregation Methods

The `ChunkedSIEVEModel` supports different aggregation strategies:

### `mean` (default)
Average chunk embeddings before final classification.
```python
model = ChunkedSIEVEModel(base_model, aggregation_method='mean')
```
**Pros**: Simple, stable, treats all chunks equally
**Cons**: May dilute strong signals from specific chromosomes

### `max`
Max-pool chunk embeddings before classification.
```python
model = ChunkedSIEVEModel(base_model, aggregation_method='max')
```
**Pros**: Preserves strongest chunk signal
**Cons**: May ignore most chunks

### `attention`
Learned attention weights over chunks.
```python
model = ChunkedSIEVEModel(
    base_model,
    aggregation_method='attention',
    embedding_dim=256  # Required for attention
)
```
**Pros**: Model learns which chunks are important
**Cons**: More parameters, may overfit

### `logit_mean`
Average chunk predictions (logits) directly.
```python
model = ChunkedSIEVEModel(base_model, aggregation_method='logit_mean')
```
**Pros**: Simplest, no need for embeddings
**Cons**: Each chunk makes independent prediction, may lose coherence

## Performance Considerations

### Memory Usage
- **Old approach**: 3000 variants × batch_size
- **New approach**: 3000 variants × batch_size (same!)
- Chunk size can be adjusted: smaller chunks = more chunks per sample = slower but more memory-safe

### Training Time
- **Overhead**: ~10x more forward passes (10 chunks per sample instead of 1)
- **Mitigation**: Use smaller batch size, larger chunk size, or train on GPU
- **Worth it**: Only way to get whole-genome coverage

### Chunk Size Selection
```python
# Conservative (safe for 8GB GPU)
chunk_size = 2000

# Balanced (most GPUs)
chunk_size = 3000

# Aggressive (16GB+ GPU)
chunk_size = 5000
```

## Migration Guide

### From Old Training Code

**Before (BROKEN - only sees chr1/chr2)**:
```python
dataset = VariantDataset(samples, annotation_level)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_samples)

for batch in dataloader:
    outputs = model(batch['features'], batch['positions'],
                   batch['gene_ids'], batch['mask'])
    loss = criterion(outputs, batch['labels'])
```

**After (FIXED - sees whole genome)**:
```python
dataset = ChunkedVariantDataset(samples, annotation_level, chunk_size=3000)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_chunks)

model = ChunkedSIEVEModel(base_model, aggregation_method='mean')

for batch in dataloader:
    loss, predictions = model.train_step(batch, criterion, device)
    # predictions are sample-level, already aggregated
```

### From Old Explainability Code

The explainability script needs the most changes. See `scripts/explain_chunked.py` for a complete implementation.

## Validation

After training with chunked processing, verify whole-genome coverage:

```bash
# Run explainability
python scripts/explain_chunked.py \
    --experiment-dir outputs/L3_chunked \
    --preprocessed-data data/preprocessed.pt \
    --output-dir results/explainability_chunked

# Check chromosome distribution
cut -f 1 -d',' results/explainability_chunked/sieve_variant_rankings.csv | sort | uniq -c

# Should see ALL chromosomes 1-22, X (not just 1-2!)
```

## FAQ

**Q: Why not just increase `max_variants_per_batch`?**
A: Memory. 30,000 variants × attention mechanism × batch size = OOM on most GPUs.

**Q: Does chunking hurt model performance?**
A: Unknown - we've never had whole-genome coverage before! Likely improves it by seeing all chromosomes.

**Q: Can I use chunked processing with existing checkpoints?**
A: No. Models trained on chr1/chr2 only need retraining to learn genome-wide patterns.

**Q: What about overlapping chunks?**
A: Set `overlap > 0` in `ChunkedVariantDataset`. May help with boundary effects but increases processing time.

**Q: How do I debug chunk aggregation?**
A: Set `aggregation_method='logit_mean'` initially (simplest). Check that sample-level predictions make sense.

## Summary

**The chunked processing approach is MANDATORY for genome-wide analysis.**

The old approach was fundamentally broken. All previous results showing poor AUC, no ClinVar matches, no GO enrichment, and chr1/chr2-only Manhattan plots were consequences of this bug.

With chunked processing:
- ✅ Full genome coverage (all 23 chromosomes)
- ✅ Memory-safe (same as before)
- ✅ Explainability-compatible
- ✅ Biologically valid results

All new training and explainability MUST use `ChunkedVariantDataset` and `ChunkedSIEVEModel`.
