# Quality Metadata Documentation

## Overview

The conversion pipeline creates **quality metadata** files that track artifact-free (clean) segments in the PSG recordings. These files are essential for:

1. **Validation**: Ensuring normalization statistics are computed on clean data only
2. **Fine-tuning**: Filtering or weighting embeddings based on data quality
3. **Analysis**: Understanding the quality distribution across subjects

## File Format

**Location**: `processed_hdf5/quality_metadata/{SUBJECT}_quality.json`

**Structure**:
```json
{
  "subject_id": "BOGN00004",
  "total_windows": 720,
  "clean_windows": [0, 1, 2, 5, 6, 7, ...],
  "artifact_windows": [3, 4, 8, 9, ...],
  "num_clean_windows": 522,
  "num_artifact_windows": 198,
  "clean_ratio": 0.725,
  "window_duration_sec": 30,
  "sampling_rate_hz": 128,
  "total_duration_hours": 6.0,
  "channels": ["C3-M2", "C4-M1", "O1-M2", ...],
  "sample_level_quality": {
    "total_samples": 2764800,
    "clean_samples": 2004480,
    "artifact_samples": 760320
  }
}
```

## Field Descriptions

### Window-Level Metadata
- **`total_windows`**: Total number of 30-sec epochs in the recording (typically 720 for 6 hours)
- **`clean_windows`**: List of window indices that are artifact-free (0-indexed)
- **`artifact_windows`**: List of window indices marked as artifacts
- **`num_clean_windows`**: Count of clean windows
- **`num_artifact_windows`**: Count of artifact windows
- **`clean_ratio`**: Fraction of clean windows (0.0 to 1.0)

### Recording Metadata
- **`subject_id`**: Subject identifier
- **`window_duration_sec`**: Duration of each window in seconds (30)
- **`sampling_rate_hz`**: Sampling rate of HDF5 data (128 Hz)
- **`total_duration_hours`**: Total recording duration
- **`channels`**: List of channels present in HDF5 file

### Sample-Level Metadata
- **`total_samples`**: Total number of samples at 128 Hz (720 windows × 3840 samples/window)
- **`clean_samples`**: Number of samples in clean windows
- **`artifact_samples`**: Number of samples in artifact windows

## How Normalization Uses Quality Metadata

### During Conversion (Artifact-Aware Normalization)

The normalization statistics (`*_normstats.json`) were computed on **clean segments only**:

```json
{
  "mean": -0.0027,
  "std": 20.34,
  "n_clean_segments": 522,
  "n_total_segments": 720,
  "computed_with_master_mask": true
}
```

During conversion, we apply:
```python
normalized_signal = (raw_signal - mean) / std
```

This ensures the signal is z-scored using statistics from artifact-free data, which is superior to z-scoring with artifacts included.

### During Validation

The validation script loads quality metadata and computes statistics **only on clean segments**:

```python
def get_clean_segments(signal_data, quality_metadata):
    samples_per_window = 30 * 128  # 3840 samples
    clean_windows = quality_metadata['clean_windows']
    
    clean_segments = []
    for window_idx in clean_windows:
        start_idx = window_idx * samples_per_window
        end_idx = (window_idx + 1) * samples_per_window
        clean_segments.append(signal_data[start_idx:end_idx])
    
    return np.concatenate(clean_segments)
```

This ensures validation checks (mean ≈ 0, std ≈ 1) are accurate.

## Using Quality Metadata for Fine-Tuning

### Option 1: Filter Embeddings (Quality Threshold)

Remove embeddings from low-quality windows:

```python
import json
import numpy as np

# Load quality metadata
with open(f'{subject_id}_quality.json', 'r') as f:
    quality = json.load(f)

# Embeddings are at 5-min resolution (10 windows per embedding)
# Each embedding covers windows [i*10 : (i+1)*10]
num_embeddings = quality['total_windows'] // 10

clean_embedding_mask = []
for emb_idx in range(num_embeddings):
    start_window = emb_idx * 10
    end_window = (emb_idx + 1) * 10
    
    # Count clean windows in this 5-min segment
    clean_count = sum(1 for w in range(start_window, end_window) 
                     if w in quality['clean_windows'])
    
    # Keep embedding if >= 50% windows are clean
    clean_embedding_mask.append(clean_count >= 5)

# Filter embeddings
embeddings_filtered = embeddings[clean_embedding_mask]
labels_filtered = labels[clean_embedding_mask]
```

### Option 2: Weight Embeddings (Quality Weighting)

Weight loss by segment quality:

```python
# Calculate quality weight for each embedding
weights = []
for emb_idx in range(num_embeddings):
    start_window = emb_idx * 10
    end_window = (emb_idx + 1) * 10
    
    clean_count = sum(1 for w in range(start_window, end_window) 
                     if w in quality['clean_windows'])
    
    # Weight = clean_ratio for this segment
    weight = clean_count / 10.0
    weights.append(weight)

weights = torch.tensor(weights)

# Use in loss function
loss = criterion(predictions, labels)
weighted_loss = (loss * weights).mean()
```

### Option 3: Subject-Level Filtering

Filter out subjects with very low quality:

```python
# Load all quality metadata
subject_quality = {}
for subject_id in subject_list:
    with open(f'{subject_id}_quality.json', 'r') as f:
        quality = json.load(f)
    subject_quality[subject_id] = quality['clean_ratio']

# Keep subjects with >= 50% clean data
good_subjects = [s for s, ratio in subject_quality.items() if ratio >= 0.5]
```

## Quality Statistics Example

For the STAGES dataset, typical quality distributions:

| Subject | Total Windows | Clean Windows | Clean Ratio | Quality Category |
|---------|--------------|---------------|-------------|------------------|
| BOGN00004 | 720 | 522 | 72.5% | Good |
| BOGN00010 | 720 | 680 | 94.4% | Excellent |
| BOGN00023 | 720 | 310 | 43.1% | Fair |
| BOGN00045 | 720 | 120 | 16.7% | Poor |

**Recommended thresholds**:
- **Excellent**: ≥ 80% clean
- **Good**: 60-80% clean
- **Fair**: 40-60% clean
- **Poor**: < 40% clean (consider excluding)

## Integration with SleepFM Pipeline

### 1. Conversion
```bash
python convert_to_hdf5.py --config config_stages_conversion.yaml
# Creates: processed_hdf5/{SUBJECT}.hdf5
#          processed_hdf5/quality_metadata/{SUBJECT}_quality.json
```

### 2. Validation
```bash
python validate_data.py --config config_stages_conversion.yaml
# Uses quality metadata to validate on clean segments only
```

### 3. Embedding Generation
```bash
python generate_embeddings.py --config config_stages_conversion.yaml
# Standard SleepFM embedding (no modification needed)
# Creates embeddings at 5-min resolution
```

### 4. Fine-Tuning (Modified)
```python
# In your fine-tuning code:
from pathlib import Path
import json

def load_subject_with_quality(subject_id, embeddings_dir, quality_dir):
    # Load embeddings
    emb_path = embeddings_dir / f"{subject_id}.hdf5"
    with h5py.File(emb_path, 'r') as f:
        embeddings = f['embeddings'][:]
    
    # Load quality
    quality_path = quality_dir / f"{subject_id}_quality.json"
    with open(quality_path, 'r') as f:
        quality = json.load(f)
    
    # Calculate per-embedding quality
    num_embeddings = len(embeddings)
    emb_quality = []
    for i in range(num_embeddings):
        start_win = i * 10
        end_win = (i + 1) * 10
        clean_count = sum(1 for w in range(start_win, end_win) 
                         if w in quality['clean_windows'])
        emb_quality.append(clean_count / 10.0)
    
    return embeddings, emb_quality

# Use in dataset
embeddings, quality_weights = load_subject_with_quality(...)
# Filter or weight as needed
```

## Summary

✅ **Quality metadata tracks clean vs. artifact windows**  
✅ **Normalization uses artifact-aware statistics**  
✅ **Validation checks stats on clean segments only**  
✅ **Fine-tuning can filter/weight embeddings by quality**  
✅ **Quality-aware training improves model robustness**

This ensures the preprocessing quality from STAGES is preserved throughout the SleepFM pipeline!
