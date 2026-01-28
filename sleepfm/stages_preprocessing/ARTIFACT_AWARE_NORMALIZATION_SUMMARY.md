# Artifact-Aware Normalization & Quality Tracking - Implementation Summary

## What Was Changed

This update implements **artifact-aware normalization** and **quality metadata tracking** to preserve the superior preprocessing quality of STAGES data throughout the SleepFM pipeline.

## Key Improvements

### 1. Artifact-Aware Normalization During Conversion

**What**: Load and apply normalization statistics that were computed on clean (non-artifact) segments only.

**Why**: Standard z-scoring on the whole signal (including artifacts) produces biased statistics. Using clean-segment-only statistics preserves the quality of your preprocessing.

**How**:
- Load `{CHANNEL}_normstats.json` for each channel
- Apply normalization: `normalized = (signal - mean) / std`
- Statistics were computed with: `computed_with_master_mask: true`

**Code Changes**:
- `convert_to_hdf5.py`:
  - Added `load_normalization_stats()` method
  - Added `apply_normalization()` method
  - Modified `convert_subject()` to load and apply normstats after resampling

**Example**:
```python
# Load normstats
normstats = {
    "mean": -0.0027,
    "std": 20.34,
    "n_clean_segments": 522,
    "n_total_segments": 720
}

# Apply to signal
normalized_signal = (raw_signal - normstats['mean']) / normstats['std']
```

---

### 2. Quality Metadata Creation

**What**: Create JSON files tracking which 30-sec windows are clean vs. artifact-contaminated.

**Why**: Enables filtering/weighting of embeddings during fine-tuning based on data quality.

**How**:
- Load `{SUBJECT}_master_exclusion_mask.npy` during conversion
- Map window-level mask (720 windows) to sample-level (2.76M samples at 128 Hz)
- Save quality metadata JSON alongside HDF5 file

**Code Changes**:
- `convert_to_hdf5.py`:
  - Added `load_master_mask()` method
  - Added `create_quality_metadata()` method
  - Modified `__init__()` to create `quality_metadata/` directory
  - Modified `convert_subject()` to save quality JSON

**Output**:
```
processed_hdf5/
├── BOGN00004.hdf5
└── quality_metadata/
    └── BOGN00004_quality.json
```

**Example Quality Metadata**:
```json
{
  "subject_id": "BOGN00004",
  "total_windows": 720,
  "clean_windows": [0, 1, 2, 5, 6, ...],
  "num_clean_windows": 522,
  "clean_ratio": 0.725
}
```

---

### 3. Validation on Clean Segments Only

**What**: Calculate validation statistics (mean, std) only on clean segments, not the whole signal.

**Why**: Artifact windows have different statistics that would contaminate validation checks. We want to verify that clean segments have mean ≈ 0, std ≈ 1.

**How**:
- Load quality metadata during validation
- Extract clean segments using `get_clean_segments()`
- Calculate statistics only on those segments

**Code Changes**:
- `validate_data.py`:
  - Added import for `json`
  - Added `quality_dir` path in `__init__()`
  - Modified `validate_signal_properties()` to load quality metadata
  - Added `get_clean_segments()` method
  - Updated normalization checks to only apply when `computed_on_clean=True`

**Example**:
```python
# Old: Stats on whole signal (with artifacts)
mean = np.mean(signal_data)  # Biased by artifacts!

# New: Stats on clean segments only
clean_data = get_clean_segments(signal_data, quality_metadata)
mean = np.mean(clean_data)  # Accurate representation of clean data
```

---

## File Changes Summary

### Modified Files

1. **`convert_to_hdf5.py`** (5 changes)
   - Added `quality_output` directory creation
   - Added `master_masks_dir` path
   - Added `load_normalization_stats()` method
   - Added `apply_normalization()` method
   - Added `load_master_mask()` method
   - Added `create_quality_metadata()` method
   - Modified `convert_subject()` to integrate all changes

2. **`validate_data.py`** (3 changes)
   - Added `json` import
   - Added `quality_dir` path
   - Modified `validate_signal_properties()` to use quality metadata
   - Added `get_clean_segments()` method

### New Files

3. **`QUALITY_METADATA.md`**
   - Comprehensive documentation on quality metadata format
   - Usage examples for fine-tuning
   - Integration guide with SleepFM pipeline

4. **`ARTIFACT_AWARE_NORMALIZATION_SUMMARY.md`** (this file)
   - Summary of all changes
   - Before/after comparison
   - Testing checklist

---

## Before vs. After Comparison

### Before: Standard Normalization

```python
# Conversion
continuous_signal = concatenate_segments(segmented_data)
resampled_signal = resample(continuous_signal, target_sr=128)
# No normalization!
save_to_hdf5(resampled_signal)

# Validation
mean = np.mean(signal_data)  # Includes artifacts
std = np.std(signal_data)    # Biased by artifacts
```

**Problems**:
- Not normalized (SleepFM pretrained model expects normalized input)
- No quality tracking
- Artifacts contaminate validation

### After: Artifact-Aware Pipeline

```python
# Conversion
continuous_signal = concatenate_segments(segmented_data)
resampled_signal = resample(continuous_signal, target_sr=128)

# Load artifact-aware normstats
normstats = load_normalization_stats(subject_id, channel_name)

# Apply normalization
normalized_signal = (resampled_signal - normstats['mean']) / normstats['std']
save_to_hdf5(normalized_signal)

# Save quality metadata
master_mask = load_master_mask(subject_id)
quality_metadata = create_quality_metadata(subject_id, master_mask)
save_quality_json(quality_metadata)

# Validation
quality_metadata = load_quality_metadata(subject_id)
clean_data = get_clean_segments(signal_data, quality_metadata)
mean = np.mean(clean_data)  # Clean segments only
std = np.std(clean_data)    # Accurate!
```

**Benefits**:
✅ Normalized using clean-segment statistics  
✅ Quality metadata created for fine-tuning  
✅ Validation checks clean segments only  
✅ Preserves preprocessing quality

---

## Testing Checklist

### Quick Test (Single Subject)

```bash
# Test conversion
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing
python test_single_subject.py --subject BOGN00004

# Check outputs
ls -lh ../processed_hdf5/BOGN00004.hdf5
ls -lh ../processed_hdf5/quality_metadata/BOGN00004_quality.json

# Inspect quality
python -c "
import json
with open('../processed_hdf5/quality_metadata/BOGN00004_quality.json', 'r') as f:
    q = json.load(f)
print(f'Clean ratio: {q[\"clean_ratio\"]:.1%}')
print(f'Clean windows: {q[\"num_clean_windows\"]}/{q[\"total_windows\"]}')
"

# Check normalization
python -c "
import h5py
import numpy as np
import json

# Load HDF5
with h5py.File('../processed_hdf5/BOGN00004.hdf5', 'r') as f:
    c3m2 = f['C3-M2'][:]

# Load quality
with open('../processed_hdf5/quality_metadata/BOGN00004_quality.json', 'r') as f:
    quality = json.load(f)

# Extract clean segments
samples_per_window = 30 * 128
clean_segments = []
for win_idx in quality['clean_windows']:
    start = win_idx * samples_per_window
    end = (win_idx + 1) * samples_per_window
    clean_segments.append(c3m2[start:end])

clean_data = np.concatenate(clean_segments)

print(f'C3-M2 stats on clean segments:')
print(f'  Mean: {clean_data.mean():.6f} (expected ~0)')
print(f'  Std:  {clean_data.std():.6f} (expected ~1)')
"
```

**Expected Output**:
```
Clean ratio: 72.5%
Clean windows: 522/720
C3-M2 stats on clean segments:
  Mean: -0.000032 (expected ~0)
  Std:  0.998741 (expected ~1)
```

### Full Pipeline Test

```bash
# Run on pilot subjects (10 subjects)
python convert_to_hdf5.py --config config_stages_conversion.yaml

# Validate
python validate_data.py --config config_stages_conversion.yaml

# Check logs
tail -n 50 ../processed_hdf5/logs/conversion_*.log
tail -n 50 ../processed_hdf5/logs/validation_*.log
```

---

## Data Flow Diagram

```
Input Data (100 Hz, segmented)
├── {CHANNEL}.npy (720, 3000)
├── {CHANNEL}_normstats.json (mean, std from clean segments)
└── {SUBJECT}_master_exclusion_mask.npy (720, bool)
    
    ↓ convert_to_hdf5.py
    
1. Load segmented data
2. Concatenate to continuous (2.16M samples)
3. Resample 100→128 Hz (2.76M samples)
4. Load normstats
5. Apply normalization: (signal - mean) / std
6. Load master mask
7. Create quality metadata
    
    ↓ Output
    
processed_hdf5/
├── {SUBJECT}.hdf5 (normalized, continuous @ 128 Hz)
└── quality_metadata/
    └── {SUBJECT}_quality.json (clean/artifact windows)
    
    ↓ validate_data.py
    
1. Load HDF5
2. Load quality metadata
3. Extract clean segments
4. Validate mean≈0, std≈1 on clean segments only
    
    ↓ generate_embeddings.py (unchanged)
    
1. Load HDF5 (normalized data)
2. Generate embeddings (SleepFM pretrained)
3. Save embeddings at 5-min resolution
    
    ↓ Fine-tuning (modified)
    
1. Load embeddings
2. Load quality metadata
3. Filter/weight embeddings by quality
4. Train cognitive prediction model
```

---

## Next Steps

1. **Test on pilot data** (10 subjects)
2. **Check quality distribution** across dataset
3. **Run full conversion** on all subjects
4. **Generate embeddings** using SleepFM
5. **Modify fine-tuning** code to use quality metadata
6. **Train cognitive models** with quality filtering/weighting

---

## Questions & Troubleshooting

### Q: What if normstats are missing for a channel?

**A**: The converter logs a warning and skips normalization for that channel. The signal is still resampled and saved, but without normalization.

```python
if normstats is None:
    logger.warning(f"{channel}: normstats not found, skipping normalization")
```

### Q: What if master mask is missing for a subject?

**A**: The converter logs a warning but continues. No quality metadata is created, and validation uses whole-signal statistics.

```python
if master_mask is None:
    logger.warning(f"Master mask not found for {subject_id}")
    # Continue without quality metadata
```

### Q: How do I know if normalization worked?

**A**: Check validation report. For normalized channels with quality metadata, stats should show:
- `mean`: -0.1 to +0.1
- `std`: 0.8 to 1.2
- `computed_on_clean`: true

### Q: Can I use the old validation without quality metadata?

**A**: Yes! If quality metadata doesn't exist, validation falls back to whole-signal statistics automatically.

---

## Summary of Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Normalization** | None | Artifact-aware z-score |
| **Statistics** | N/A | From clean segments only |
| **Quality Tracking** | None | Per-window clean/artifact labels |
| **Validation** | N/A | Clean segments only |
| **Fine-tuning** | All embeddings equal | Filter/weight by quality |
| **Model Robustness** | Lower | Higher (clean data focus) |

**🎯 Result**: Your superior STAGES preprocessing quality is now preserved throughout the SleepFM pipeline!
