# Artifact Filtering Integration - Summary

## Overview

Implemented fine-grained artifact filtering for STAGES cognitive prediction pipeline. Filters 30-second artifact segments **before** embedding generation, providing cleaner data for the SetTransformer model.

## Key Features

### 1. **Separate Module** (`artifact_filtering.py`)
- ✅ Standalone module - minimal changes to existing code
- ✅ Easy to enable/disable via configuration
- ✅ Handles resolution mismatch (30-sec masks vs 5-min chunks)
- ✅ Gracefully handles edge cases (all-artifact chunks)

### 2. **Resolution Handling**
- **Master masks**: 720 × 30-second segments per subject (6 hours)
- **HDF5 data**: Continuous signals at 128 Hz
- **Chunks**: 5-minute windows (38,400 samples)
- **Model patches**: 5-second patches (640 samples)
- **Mapping**: 1 chunk (5 min) = 10 mask segments (30 sec each)

### 3. **Filtering Strategy**
- Filters at native mask resolution (30-sec segments)
- Removes artifact segments, concatenates clean segments
- SetTransformer processes variable-length sequences naturally
- Skips chunks where all segments are artifacts

## Implementation Details

### Files Modified

1. **`artifact_filtering.py`** (NEW) - Core filtering module
   - `ArtifactFilter` class with mask loading and filtering
   - `create_artifact_filter()` factory function
   - Resolution mapping and edge case handling

2. **`dataset.py`** - Dataset integration
   - Added `artifact_filter` parameter to `SetTransformerDataset`
   - Modified `__getitem__()` to apply filtering after HDF5 load
   - Updated `collate_fn()` to handle:
     - None values (skipped all-artifact chunks)
     - Variable-length sequences (temporal padding)
     - 2D masks (channel + temporal padding)

3. **`generate_embeddings.py`** - Pipeline integration
   - Import artifact filtering module
   - Create artifact filter from config
   - Pass filter to dataset
   - Track filtered chunks in statistics

4. **`config_finetune_cognitive.yaml`** - Configuration
   - Added `artifact_filtering` section with all parameters
   - Currently **disabled by default** (`enabled: false`)

### Configuration

```yaml
artifact_filtering:
  enabled: false  # Set to true to enable
  master_masks_dir: '/home/boshra95/scratch/stages/stages/processed/master_masks'
  sampling_rate: 128
  segment_duration: 30
  all_artifact_policy: 'skip'
  min_clean_ratio: 0.0
  log_filtering_stats: true
```

## Usage

### Enable Filtering

In `config_finetune_cognitive.yaml`:
```yaml
artifact_filtering:
  enabled: true  # Change to true
```

### Run Embedding Generation

```bash
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

### Test the Module

```bash
cd sleepfm/stages_cognitive_prediction
python test_artifact_filtering.py
```

## How It Works

### Step-by-Step Flow

1. **Dataset loads chunk** from HDF5 (e.g., samples 0-38,400 for 5-min chunk)

2. **Extract subject ID** from file path (e.g., `subject123.hdf5` → `subject123`)

3. **Load master mask** from `{subject_id}_master_exclusion_mask.npy` (720 elements)
   - **File format**: 0 = clean signal (keep), 1 = artifact (exclude)
   - **Code inverts it**: True = clean (keep), False = artifact (exclude)

4. **Map chunk to mask indices**:
   - Chunk samples 0-38,400 → mask indices 0:10 (first 10 segments)
   - Chunk samples 38,400-76,800 → mask indices 10:20 (next 10 segments)

5. **Filter segments**:
   - Extract mask slice: `chunk_mask = master_mask[0:10]`
   - Example file: `[0,0,1,0,0,0,1,1,0,0]` → 7 clean (0s), 3 artifacts (1s)
   - After inversion: `[True,True,False,True,True,True,False,False,True,True]`
   - Keep only clean 30-sec segments (indices where mask==True)
   - Concatenate: shape changes from `(channels, 38400)` to `(channels, 26880)`

6. **Handle edge cases**:
   - If all segments are artifacts → return `None`
   - `collate_fn` filters out None values
   - Batch skipped if empty after filtering

7. **Variable-length batching**:
   - `collate_fn` pads to max length in batch
   - Creates 2D mask: `[max_channels, max_length]`
   - Marks padded channels AND padded time steps

8. **Model processing**:
   - SetTransformer receives padded data + mask
   - Attention mechanisms ignore masked positions
   - Generates embeddings from clean segments only

## Example

### Input
- Chunk: 5 minutes (38,400 samples at 128 Hz)
- Mask file: `[0,0,1,0,0,0,1,1,0,0]` for segments 0-9 (0=clean, 1=artifact)
- After inversion: `[True,True,False,True,True,True,False,False,True,True]`

### Filtering
- Segment 0 (clean): samples 0-3,840 → **KEEP**
- Segment 1 (clean): samples 3,840-7,680 → **KEEP**
- Segment 2 (artifact): samples 7,680-11,520 → **REMOVE**
- Segment 3 (clean): samples 11,520-15,360 → **KEEP**
- ... (continue for all 10 segments)

### Output
- Filtered chunk: 26,880 samples (7 clean segments concatenated)
- Shape: `(10 channels, 26880 samples)`
- Model processes only clean data

## Edge Cases Handled

1. **All artifacts in chunk**: Returns `None`, batch skipped
2. **No master mask file**: Warning logged, uses original data
3. **Invalid mask shape**: Warning logged, uses original data
4. **Filtering disabled**: Returns original data unchanged
5. **Variable sequence lengths**: Padded in collate_fn with proper masking

## Performance Considerations

### Memory
- **Mask caching**: Masks cached in memory after first load (720 bool = <1 KB each)
- **Variable lengths**: Padding adds some memory overhead, but minimal

### Computation
- **Filtering overhead**: Negligible (numpy slicing + concatenation)
- **Model efficiency**: Processes fewer samples (faster if many artifacts)
- **I/O**: One extra file read per subject (masks are small)

### Training Impact
- **Fewer batches**: If many chunks are all-artifacts
- **Better embeddings**: Model sees only clean patterns
- **Variable batch sizes**: After filtering empty batches

## Testing

### Unit Tests
Run `test_artifact_filtering.py` to verify:
- ✅ Chunk-to-mask index mapping
- ✅ Segment filtering correctness
- ✅ All-artifact chunk handling
- ✅ Disabled filter behavior

### Integration Test
1. Enable filtering in config
2. Run on a small subset (e.g., 5 subjects)
3. Check logs for filtering statistics
4. Verify embeddings are generated correctly

## Next Steps

### To Enable and Test

1. **Update config**:
   ```yaml
   artifact_filtering:
     enabled: true
   ```

2. **Test on small dataset**:
   ```yaml
   data:
     max_files: 5
   ```

3. **Run embedding generation**:
   ```bash
   python generate_embeddings.py --config config_finetune_cognitive.yaml
   ```

4. **Check logs** for filtering stats:
   ```
   Artifact filtering enabled
   Filtered chunk at 0: kept 7/10 segments (70.0% clean)
   Filtered all-artifact chunks: 3
   ```

5. **Run fine-tuning** to verify embeddings work correctly

### Potential Enhancements

- **Configurable min_clean_ratio**: Reject chunks with <50% clean segments
- **Per-modality masks**: Different masks for different channel types
- **Quality weighting**: Weight embeddings by clean ratio instead of hard filtering
- **Statistics logging**: Track and save detailed filtering statistics per subject

## Troubleshooting

### Issue: "Master mask not found"
- **Cause**: Mask file naming mismatch or missing files
- **Solution**: Check mask files exist at `{subject_id}_master_exclusion_mask.npy`

### Issue: "Invalid mask shape"
- **Cause**: Mask not 720 elements
- **Solution**: Verify mask preprocessing created 720-element arrays

### Issue: All batches filtered out
- **Cause**: Too many all-artifact chunks
- **Solution**: Lower `min_clean_ratio` or check mask quality

### Issue: Model errors with variable lengths
- **Cause**: Mask not properly applied
- **Solution**: Verify 2D masking in collate_fn is working

## Summary

✅ **Modular design** - Minimal changes to existing code  
✅ **Fine-grained filtering** - 30-sec resolution, not whole chunks  
✅ **Edge case handling** - All-artifact chunks, missing masks, etc.  
✅ **Configurable** - Easy to enable/disable  
✅ **Tested** - Unit tests verify correctness  
✅ **Efficient** - Minimal overhead, better quality embeddings  

The artifact filtering is **ready to use** - just set `enabled: true` in the config!
