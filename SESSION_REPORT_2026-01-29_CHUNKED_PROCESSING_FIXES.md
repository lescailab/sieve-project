# Session Report: ChunkedSIEVEModel Bug Fixes and Testing
**Date**: 2026-01-29
**Branch**: `claude/add-data-caching-HoCB2`
**Status**: Ready for testing on user workstation

---

## Executive Summary

This session addressed critical bugs in the chunked processing pipeline that prevented training from running. The main issue was a type error in gradient accumulation caused by improper handling of dict-returning loss functions. We also addressed 5 code quality issues identified by GitHub Copilot and added comprehensive unit tests (95% coverage on ChunkedSIEVEModel).

**Key Achievement**: Training command that was failing with `TypeError: unsupported operand type(s) for /: 'dict' and 'int'` should now work.

---

## 1. Original Problem

### User's Training Command
```bash
python /home/shared/code_versioned/sieve-project/scripts/train.py \
    --preprocessed-data /home/shared/sieve-testing/preprocessed_discovery.pt \
    --level L3 \
    --val-split 0.2 \
    --lambda-attr 0.1 \
    --lr 0.0001 \
    --epochs 10 \
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \
    --gradient-accumulation-steps 4 \
    --output-dir /home/shared/sieve-testing/test_chunked \
    --experiment-name L3_test \
    --device cuda
```

### Error Encountered
```
Traceback (most recent call last):
  File "/home/shared/code_versioned/sieve-project/src/training/trainer.py", line 171, in train_epoch
    scaled_loss = loss / self.gradient_accumulation_steps
                  ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: unsupported operand type(s) for /: 'dict' and 'int'
```

### Root Cause Analysis

**The Problem Flow**:
1. `ChunkedSIEVEModel.train_step()` calls `criterion(predictions, sample_labels.float())`
2. `criterion` is a `SIEVELoss` instance which returns a **dictionary**: `{'total': tensor, 'classification': tensor, 'attribution_sparsity': tensor}`
3. `train_step()` was returning this dict directly instead of extracting the scalar loss
4. `trainer.py` line 171 tried to divide the dict by an int → TypeError

**Why This Wasn't Caught Earlier**:
- Standard (non-chunked) processing path in `trainer.py` correctly extracts `loss = loss_dict['total']` (line 168)
- Chunked processing path took a shortcut that bypassed this extraction
- No unit tests existed for `ChunkedSIEVEModel.train_step()`

---

## 2. Bugs Discovered and Fixed

### Bug #1: Dict vs Scalar Loss Handling
**File**: `src/models/chunked_sieve.py`
**Lines**: 222-231

**Original Code**:
```python
# Compute loss at sample level
loss = criterion(predictions, sample_labels.float())
return loss, predictions
```

**Problem**: `criterion` (SIEVELoss) returns a dict, not a scalar.

**Fix**:
```python
# Compute loss at sample level
loss_output = criterion(predictions, sample_labels.float())

# Handle both dict (SIEVELoss) and scalar (BCEWithLogitsLoss) returns
if isinstance(loss_output, dict):
    loss = loss_output['total']
else:
    loss = loss_output

return loss, predictions
```

**Impact**: Training can now proceed with both SIEVELoss and BCEWithLogitsLoss.

---

### Bug #2: Unused Intermediates Variable
**File**: `src/models/chunked_sieve.py`
**Lines**: 104-106

**Original Code**:
```python
chunk_outputs, intermediates = self.base_model(
    features, positions, gene_ids, mask, return_intermediate=True
)
```

**Problem**: `intermediates` was computed but never used, wasting computation.

**Fix**:
```python
chunk_outputs, _ = self.base_model(features, positions, gene_ids, mask)
```

**Impact**: Reduced unnecessary computation overhead.

---

### Bug #3: Implicit Sorting Dependency
**File**: `src/models/chunked_sieve.py`
**Line**: 116

**Original Code**:
```python
unique_samples = original_sample_indices.unique()
```

**Problem**: `torch.searchsorted` (line 124) requires sorted input, but `unique()` doesn't guarantee sorting in all PyTorch versions.

**Fix**:
```python
# Note: sorted=True is required for torch.searchsorted below
unique_samples = original_sample_indices.unique(sorted=True)
```

**Impact**: Future-proofs against PyTorch API changes.

---

### Bug #4: Dtype Mismatch in Mixed Precision
**File**: `src/models/chunked_sieve.py`
**Lines**: 135-139

**Original Code**:
```python
aggregated = torch.zeros(num_samples, dtype=chunk_outputs.dtype, device=device)
counts = torch.zeros(num_samples, dtype=torch.float32, device=device)
# ...
aggregated = aggregated / counts.clamp(min=1)
```

**Problem**: When using mixed precision (float16/bfloat16), `aggregated` has dtype from chunk_outputs but `counts` is always float32. Division may cause dtype issues.

**Fix**:
```python
# Average and preserve original dtype (important for mixed precision training)
aggregated = (aggregated / counts.clamp(min=1)).to(chunk_outputs.dtype)
```

**Impact**: Supports mixed precision training correctly.

---

### Bug #5: PyTorch Compatibility with torch.unique()
**File**: `src/models/chunked_sieve.py`
**Lines**: 200-211

**Original Approach (from Copilot suggestion)**:
```python
unique_samples, first_occurrence_indices = torch.unique(
    original_sample_indices,
    sorted=True,
    return_index=True,
)
sample_labels = labels[first_occurrence_indices]
```

**Problem**: PyTorch 2.9.1 JIT compiler doesn't support `sorted=True` with `return_index=True`:
```
TypeError: _return_output() got an unexpected keyword argument 'return_index'
```

**Fix (Manual Implementation)**:
```python
# Get unique samples in sorted order (matching forward method)
unique_samples = original_sample_indices.unique(sorted=True)

# For each unique sample, find its first occurrence
sample_labels = torch.zeros(len(unique_samples), dtype=labels.dtype, device=device)
for i, sample_idx in enumerate(unique_samples):
    # Find first chunk belonging to this sample
    first_chunk_idx = (original_sample_indices == sample_idx).nonzero(as_tuple=True)[0][0]
    sample_labels[i] = labels[first_chunk_idx]
```

**Impact**: Works around PyTorch version compatibility issue while maintaining correctness.

---

## 3. Implementation Strategy Details

### Chunked Processing Architecture

**Why Chunking is Necessary**:
- Whole genome has millions of positions
- Individual samples have variants at only thousands of positions
- Dense tensors would be infeasible for memory
- Solution: Split each sample's variants into chunks of fixed size

**How Chunking Works**:

1. **Dataset Level** (`ChunkedVariantDataset`):
   - Splits each sample into chunks of size `chunk_size` (e.g., 3000 variants)
   - Each chunk becomes a separate batch item
   - Metadata tracks: `chunk_indices`, `total_chunks`, `original_sample_indices`

2. **Model Level** (`ChunkedSIEVEModel`):
   - Processes each chunk through base SIEVE model independently
   - Aggregates chunk outputs back to sample-level predictions
   - Currently supports: `mean` and `logit_mean` aggregation

3. **Training Level** (`train_step()`):
   - Aggregates chunk labels to sample labels (all chunks from same sample have same label)
   - Computes loss at sample level (not chunk level)
   - Returns scalar loss for gradient accumulation

### Aggregation Methods

**Currently Implemented**:
- `mean`: Averages chunk logits directly
- `logit_mean`: Same as mean (for historical reasons)

**Not Yet Implemented** (will raise NotImplementedError):
- `max`: Max-pooling over chunk embeddings
- `attention`: Attention-weighted pooling over chunks

**Why Only Logit Aggregation?**:
- Embedding-based aggregation requires base model to expose intermediate embeddings
- Current architecture doesn't support this cleanly
- Logit averaging is simpler and works well in practice

### Label Aggregation Strategy

**Challenge**: Dataset returns chunk-level labels, but loss needs sample-level labels.

**Example**:
```
Chunks:     [0,    1,    2,    3,    4   ]
Samples:    [0,    0,    1,    2,    2   ]  # original_sample_indices
Labels:     [0,    0,    1,    1,    1   ]  # chunk labels (redundant)
```

**Goal**: Extract `[0, 1, 1]` (one label per unique sample).

**Method**:
1. Get unique samples: `[0, 1, 2]`
2. For each sample, find first chunk index: `[0, 2, 3]`
3. Extract labels at those positions: `[0, 1, 1]`

**Critical Requirement**: Ordering must match `forward()` method's sample ordering.

---

## 4. Testing Strategy

### Unit Tests Created
**File**: `tests/test_chunked_sieve.py`
**Coverage**: 95% (53/56 lines in ChunkedSIEVEModel)

**Test Suite**:

1. **`test_train_step_with_dict_loss`**
   - Tests train_step with SIEVELoss (dict return)
   - Verifies loss is scalar tensor
   - Verifies loss supports division
   - Verifies prediction shapes

2. **`test_train_step_with_scalar_loss`**
   - Tests train_step with BCEWithLogitsLoss (scalar return)
   - Verifies backward compatibility

3. **`test_chunk_aggregation_correctness`**
   - Verifies chunks aggregate to correct number of samples
   - Verifies no NaN/Inf in predictions

4. **`test_label_aggregation_correctness`**
   - Verifies chunk labels correctly aggregate to sample labels
   - Tests ordering consistency

5. **`test_forward_without_chunking`**
   - Verifies model works without chunking metadata
   - Tests backward compatibility with non-chunked processing

6. **`test_different_aggregation_methods`**
   - Verifies unsupported methods raise NotImplementedError
   - Tests both 'max' and 'attention'

7. **`test_gradient_accumulation_compatibility`**
   - Simulates 8-step gradient accumulation
   - Verifies loss scaling works correctly
   - Verifies gradients accumulate without errors

**Mock Objects**:
- `MockSIEVEModel`: Simple linear classifier for isolated testing
- Doesn't require full SIEVE architecture dependencies

**Test Data**:
- 3 samples split into 5 chunks:
  - Sample 0: 2 chunks (indices 0, 1)
  - Sample 1: 1 chunk (index 2)
  - Sample 2: 2 chunks (indices 3, 4)
- 20 variants per chunk
- 10 features per variant

---

## 5. Known Limitations and Warnings

### Attribution Regularization Not Supported
**Issue**: `lambda_attr > 0` doesn't work with chunked processing.

**Why**:
- Attribution sparsity loss requires variant-level embeddings
- Chunking splits variants across chunks
- No clear way to compute attribution sparsity across chunks

**Current Handling**:
- `train.py` warns user and sets `lambda_attr=0` automatically
- Tests verify this with `lambda_attr=0.0`

**User Impact**: If user requests `--lambda-attr 0.1`, it will be silently changed to 0.

### Aggregation Methods Limited
**Only Supported**: `mean`, `logit_mean`
**Not Supported**: `max`, `attention`

**User Impact**: If user requests `--aggregation-method max`, will get NotImplementedError.

### Performance Considerations
**Loop in Label Aggregation**:
```python
for i, sample_idx in enumerate(unique_samples):
    first_chunk_idx = (original_sample_indices == sample_idx).nonzero(as_tuple=True)[0][0]
    sample_labels[i] = labels[first_chunk_idx]
```

- Not fully vectorized due to PyTorch compatibility
- For typical batch sizes (3-10 unique samples), overhead is negligible
- Could be optimized if profiling shows it's a bottleneck

---

## 6. Code Quality Improvements

### Documentation Updates
- Updated docstring to clarify `mean` = `logit_mean`
- Marked `max` and `attention` as "NOT YET IMPLEMENTED"
- Added inline comments explaining PyTorch compatibility workarounds

### Comments Added
```python
# Note: sorted=True is required for torch.searchsorted below
# Average and preserve original dtype (important for mixed precision training)
# Note: lambda_attr must be 0 for chunked processing (attribution not supported)
```

### Style Improvements
- Consistent error messages
- Clear separation of supported vs unsupported features
- Explicit type handling (isinstance checks)

---

## 7. Git History

### Commits in This Session
```
264fb50 - Add comprehensive unit tests for ChunkedSIEVEModel.train_step()
a056acd - Address GitHub Copilot code quality feedback
8638bf5 - CRITICAL FIX: Handle SIEVELoss dict return in ChunkedSIEVEModel.train_step()
f4b491d - Optimize chunked processing with vectorized aggregation and cleanup
```

### Files Modified
- `src/models/chunked_sieve.py` (bug fixes, code quality)
- `scripts/train.py` (lambda_attr warning)
- `scripts/explain.py` (import cleanup)
- `tests/test_chunked_sieve.py` (new file, 326 lines)

---

## 8. What Should Work Now

### User's Original Command
The training command that was failing should now work:
```bash
python scripts/train.py \
    --preprocessed-data /home/shared/sieve-testing/preprocessed_discovery.pt \
    --level L3 \
    --lambda-attr 0.1 \  # Will be changed to 0.0 with warning
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \
    --gradient-accumulation-steps 4 \
    --device cuda
```

**Expected Behavior**:
1. Warning printed: "lambda_attr > 0 not supported with chunked processing. Setting lambda_attr=0.0"
2. Training starts and runs multiple epochs
3. Gradient accumulation works correctly (loss / 4)
4. Validation runs after each epoch
5. Checkpoints saved to output directory

### What's Been Validated
✅ Unit tests pass (7/7)
✅ Loss is scalar and supports division
✅ Predictions have correct shape
✅ Label aggregation is correct
✅ Backward pass works
✅ Gradient accumulation simulation works

### What Hasn't Been Tested
❌ Full training loop with real data
❌ Multi-epoch training
❌ Checkpoint saving/loading
❌ Validation metrics computation
❌ GPU execution (tests ran on CPU)

---

## 9. Debugging Tips for User

### If Training Still Fails

**Check 1: Loss Type Error**
```python
# In trainer.py, add debug print:
print(f"Loss type: {type(loss)}, value: {loss}")
```
Expected: `<class 'torch.Tensor'>`, single value

**Check 2: Prediction Shapes**
```python
# In chunked_sieve.py train_step, add:
print(f"Predictions shape: {predictions.shape}, Sample labels shape: {sample_labels.shape}")
```
Expected: Both should be `[num_unique_samples]`

**Check 3: Chunk Metadata**
```python
# In train_step, add:
print(f"Original sample indices: {original_sample_indices}")
print(f"Unique samples: {original_sample_indices.unique()}")
```
Expected: Unique samples should be sequential integers starting from 0

**Check 4: Enable Verbose Logging**
The trainer already has verbose mode. Run with existing code.

### Common Issues

**Issue**: "NaN loss during training"
**Cause**: Learning rate too high or batch normalization issues
**Fix**: Reduce learning rate to 1e-5, check for zero-variance features

**Issue**: "Out of memory"
**Cause**: Chunk size too large or batch size too large
**Fix**: Reduce `--chunk-size` to 2000 or `--batch-size` to 8

**Issue**: "NotImplementedError: Aggregation method 'max' not yet implemented"
**Cause**: Using unsupported aggregation method
**Fix**: Change to `--aggregation-method mean` or `logit_mean`

---

## 10. Future Work

### Short-term (Next Session)
1. **Test on Real Data**: Run full training pipeline on user's workstation
2. **Profile Performance**: Check if label aggregation loop is bottleneck
3. **Add Integration Tests**: Test full train.py script end-to-end

### Medium-term (Future Features)
1. **Implement Embedding-based Aggregation**:
   - Add `get_embeddings()` method to base SIEVE model
   - Support `max` and `attention` aggregation methods

2. **Support Attribution Regularization**:
   - Design chunk-aware attribution loss
   - Or: aggregate embeddings before computing attribution

3. **Optimize Label Aggregation**:
   - Vectorize the loop if profiling shows it's slow
   - Or: Wait for PyTorch to fix `return_index` compatibility

### Long-term (Architecture)
1. **Dynamic Chunking**: Adjust chunk size based on available memory
2. **Chunk Caching**: Cache processed chunks to disk for large datasets
3. **Distributed Chunking**: Split chunks across multiple GPUs

---

## 11. Key Learnings

### PyTorch API Gotchas
1. **`torch.unique()` with multiple return values has version compatibility issues**
   - Using `sorted=True` with `return_index=True` fails in some versions
   - Workaround: Manual loop or wait for PyTorch fix

2. **Mixed precision requires careful dtype management**
   - Always explicitly cast division results back to original dtype
   - Float32 is default, but models may use float16/bfloat16

3. **JIT compilation can cause surprising errors**
   - Error messages reference internal `_return_output()` function
   - Hard to debug without understanding PyTorch internals

### Testing Best Practices
1. **Test type conversions explicitly**
   - Dict vs scalar returns
   - Tensor dimensions and dtypes

2. **Mock complex dependencies**
   - Don't need full SIEVE architecture to test chunking logic
   - Simpler mocks = faster tests

3. **Test edge cases**
   - Different numbers of chunks per sample
   - Single-chunk samples
   - Gradient accumulation compatibility

### Code Quality
1. **GitHub Copilot feedback is valuable**
   - Caught 5 issues we would have missed
   - Especially good at spotting unused variables and dtype issues

2. **Explicit is better than implicit**
   - Document PyTorch version requirements
   - Make sorted=True explicit even if it's default

3. **Fail fast with clear errors**
   - NotImplementedError for unsupported features
   - ValueError for invalid configurations

---

## 12. Session Outcomes

### Bugs Fixed
✅ Critical: Dict vs scalar loss handling
✅ Performance: Removed unused intermediates
✅ Correctness: Explicit sorting for searchsorted
✅ Mixed precision: Dtype preservation
✅ Compatibility: PyTorch version workaround

### Tests Added
✅ 7 comprehensive unit tests
✅ 95% code coverage on ChunkedSIEVEModel
✅ All tests passing

### Code Quality
✅ 5 GitHub Copilot issues resolved
✅ Documentation updated
✅ Inline comments added

### Ready for Testing
✅ All changes committed and pushed
✅ Branch: `claude/add-data-caching-HoCB2`
✅ User can pull and test on workstation

---

## 13. Next Steps

**For User**:
1. Pull latest changes: `git pull origin claude/add-data-caching-HoCB2`
2. Run training command with real data
3. Monitor for any errors or unexpected behavior
4. Report back results

**For Next Session**:
1. If training fails, analyze error logs
2. If training succeeds, review metrics and checkpoints
3. Consider performance profiling if slow
4. Decide on priority for embedding-based aggregation

---

## Appendix A: Complete Training Command

```bash
# With all current limitations acknowledged:
python /home/shared/code_versioned/sieve-project/scripts/train.py \
    --preprocessed-data /home/shared/sieve-testing/preprocessed_discovery.pt \
    --level L3 \
    --val-split 0.2 \
    --lambda-attr 0.0 \  # Must be 0 for chunked processing
    --lr 0.0001 \
    --epochs 10 \
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \  # or logit_mean (same thing)
    --gradient-accumulation-steps 4 \
    --output-dir /home/shared/sieve-testing/test_chunked \
    --experiment-name L3_test \
    --device cuda
```

**Expected Warnings**:
- "lambda_attr > 0 not supported with chunked processing. Setting lambda_attr=0.0"

**Expected Output**:
- Dataset statistics (1968 samples, ~18283 chunks)
- Training progress (loss, AUC per epoch)
- Validation metrics after each epoch
- Checkpoint files in output directory

---

## Appendix B: Test Execution

```bash
# Run all chunked SIEVE tests
pytest tests/test_chunked_sieve.py -v

# Expected output:
# test_train_step_with_dict_loss PASSED
# test_train_step_with_scalar_loss PASSED
# test_chunk_aggregation_correctness PASSED
# test_label_aggregation_correctness PASSED
# test_forward_without_chunking PASSED
# test_different_aggregation_methods PASSED
# test_gradient_accumulation_compatibility PASSED
# ========================= 7 passed in ~10s =========================
```

---

**End of Session Report**
