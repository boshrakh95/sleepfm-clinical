# STAGES Data Preparation for SleepFM

This directory contains scripts to convert CogPSGFormerPP preprocessed STAGES data into the HDF5 format required by SleepFM.

## Overview

Your preprocessed STAGES data (30-sec segmented NumPy arrays at 100 Hz) needs to be converted to continuous HDF5 files at 128 Hz for SleepFM.

## Directory Structure

```
stages_preprocessing/
├── config_stages_conversion.yaml  # Configuration file (EDIT THIS!)
├── convert_to_hdf5.py             # Convert NumPy → HDF5
├── prepare_labels.py              # Prepare cognitive targets & demographics
├── create_splits.py               # Create train/val/test splits
├── validate_data.py               # Validate converted data
├── run_pipeline.sh                # Run complete pipeline
└── README.md                      # This file
```

## Quick Start

### 1. Edit Configuration

Open `config_stages_conversion.yaml` and verify/update paths:

```yaml
input:
  base_dir: /home/boshra95/scratch/stages/stages/processed
  
output:
  base_dir: /home/boshra95/scratch/stages/sleepfm_format
```

### 2. Pilot Run (Recommended)

Test on 10 subjects first:

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing

# Enable pilot mode in config
# Set: pilot_mode: true, pilot_count: 10

python convert_to_hdf5.py --config config_stages_conversion.yaml
python prepare_labels.py --config config_stages_conversion.yaml
python create_splits.py --config config_stages_conversion.yaml
python validate_data.py --config config_stages_conversion.yaml
```

Or use the pipeline script:

```bash
bash run_pipeline.sh --pilot
```

### 3. Full Run

After validating pilot results:

```bash
# Disable pilot mode in config
# Set: pilot_mode: false

bash run_pipeline.sh
```

## Configuration Options

Key parameters in `config_stages_conversion.yaml`:

### Subject Selection

```yaml
subjects:
  pilot_mode: true          # Process only first N subjects
  pilot_count: 10           # Number of subjects in pilot
  
  include: []               # Specific subjects to include (empty = all)
  exclude: []               # Subjects to exclude
```

### Processing Parameters

```yaml
processing:
  current_sample_rate: 100     # Your data sampling rate
  target_sample_rate: 128      # SleepFM requirement
  resampling_method: linear    # 'linear', 'cubic', or 'fft'
  dtype: float16               # Data type for storage
  compression: gzip            # HDF5 compression
  compression_opts: 4          # Compression level (0-9)
```

### Channel Mapping

```yaml
channel_mapping:
  # Your name → SleepFM name
  LOC: EOG(L)
  ROC: EOG(R)
  FLOW: Flow
  THOR: Thor
  ABDM: ABD
  # ... (other channels keep same names)
```

### Train/Val/Test Split

```yaml
split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
  stratify_by: overall_cognitive  # Ensure balanced distribution
```

## Scripts Description

### 1. convert_to_hdf5.py

Converts 30-sec segmented NumPy arrays to continuous HDF5 files.

**Operations:**
- Loads segmented data (720 windows × 3000 samples at 100 Hz)
- Concatenates into continuous signals (216,000 samples)
- Resamples to 128 Hz (276,480 samples)
- Renames channels to SleepFM conventions
- Saves as compressed HDF5 files

**Key features:**
- Validates all required modalities present
- Checks data quality (NaN, Inf)
- Verifies normalization (mean ≈ 0, std ≈ 1)
- Resume capability (skips existing files)
- Detailed logging

**Output:**
```
output/hdf5_data/
├── SUBJECT001.hdf5
├── SUBJECT002.hdf5
└── ...
```

### 2. prepare_labels.py

Prepares cognitive targets and demographics in SleepFM format.

**Operations:**
- Loads cognitive_targets.csv and demographics_final.csv
- Filters to subjects with HDF5 files
- Normalizes demographics (age z-score, sex encoding)
- Creates unified CSV with "Study ID" column
- Generates distribution plots

**Output:**
```
output/labels/labels_with_demographics.csv
```

Format:
```
Study ID,sustained_attention,working_memory,...,nsrr_age,nsrr_sex
SUBJECT001,0.45,0.32,...,0.12,1
```

### 3. create_splits.py

Creates stratified train/val/test splits.

**Operations:**
- Loads labels with cognitive scores
- Stratifies by cognitive score (ensures balanced distribution)
- Creates 70/15/15 train/val/test split
- Generates SleepFM-compatible JSON
- Validates split distributions

**Output:**
```
output/splits/
├── dataset_split.json          # For SleepFM
├── train_subjects.txt          # For reference
├── val_subjects.txt
└── test_subjects.txt
```

### 4. validate_data.py

Validates converted HDF5 files.

**Validation checks:**
- HDF5 structure (all channels present)
- Modality coverage (all 4 modalities)
- Signal properties (length, duration)
- Data normalization (mean, std)
- Data quality (NaN, Inf)
- Comparison with original NumPy (sample)

**Output:**
```
output/validation/
├── validation_report.txt
├── validation_results.json
└── validation_statistics.png
```

## Expected Output

After successful conversion:

```
sleepfm_format/
├── hdf5_data/              # HDF5 files (one per subject)
│   ├── SUBJECT001.hdf5
│   └── ...
├── labels/                 # Label files
│   └── labels_with_demographics.csv
├── splits/                 # Train/val/test splits
│   ├── dataset_split.json
│   ├── train_subjects.txt
│   ├── val_subjects.txt
│   └── test_subjects.txt
├── validation/             # Validation reports
│   ├── validation_report.txt
│   ├── validation_results.json
│   └── validation_statistics.png
└── logs/                   # Detailed logs
    ├── conversion_*.log
    ├── label_preparation_*.log
    ├── split_creation_*.log
    └── validation_*.log
```

## Data Format Details

### Input Format (Your Data)

```python
# EEG: eeg_segmented/SUBJECT/C3-M2.npy
shape: [720, 3000]  # 720 windows × 3000 samples
sampling_rate: 100 Hz
duration: 30 sec per window → 6 hours total
```

### Output Format (SleepFM)

```python
# HDF5: hdf5_data/SUBJECT.hdf5
with h5py.File('SUBJECT.hdf5', 'r') as f:
    c3_m2 = f['C3-M2'][:]  # shape: [276480]
    
# sampling_rate: 128 Hz
# duration: 6 hours = 21,600 sec = 276,480 samples
```

## Troubleshooting

### Issue: "No subjects found"

**Solution:** Check paths in config:
```yaml
input:
  base_dir: /home/boshra95/scratch/stages/stages/processed
  eeg_dir: eeg_segmented  # Should contain subject folders
```

### Issue: "Missing required modality"

**Solution:** Some subjects may be missing entire modalities. They will be excluded. Check logs for details.

### Issue: "Normalization warning"

If you see warnings about mean/std:
- This is expected if your preprocessing differs from SleepFM
- Your artifact-aware normalization is actually better
- Warnings can be ignored if mean is close to 0 and std close to 1

### Issue: Memory errors

**Solution:** Reduce batch size or enable cache clearing:
```yaml
options:
  num_workers: 4  # Reduce from 8
  
advanced:
  clear_cache_per_subject: true
```

### Issue: File size too large

**Solution:** Increase compression or use float16:
```yaml
processing:
  dtype: float16           # Saves ~50% space vs float32
  compression: gzip
  compression_opts: 9      # Max compression (slower but smaller)
```

## Next Steps

After successful data preparation:

### 1. Generate Embeddings

Use SleepFM's pretrained transformer to generate embeddings:

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/pipeline

python generate_embeddings.py \
  --data_dir /home/boshra95/scratch/stages/sleepfm_format/hdf5_data \
  --output_dir /home/boshra95/scratch/stages/sleepfm_format/embeddings \
  --config ../checkpoints/model_base/config.json \
  --checkpoint ../checkpoints/model_base/best.pt \
  --split_json /home/boshra95/scratch/stages/sleepfm_format/splits/dataset_split.json
```

### 2. Fine-tune for Cognitive Prediction

Create a fine-tuning script adapted from `finetune_disease_prediction.py`:
- Replace Cox loss with MSE (regression) or CrossEntropy (classification)
- Use cognitive scores as targets
- Train LSTM model on embeddings

## Performance Notes

### Pilot Run (10 subjects)
- Conversion: ~2-5 minutes
- Label prep: <1 minute
- Splits: <1 minute
- Validation: ~1-2 minutes
- **Total: ~5-10 minutes**

### Full Run (1500 subjects)
- Conversion: ~3-6 hours (depends on I/O)
- Label prep: <5 minutes
- Splits: ~1 minute
- Validation: ~10-20 minutes
- **Total: ~3-7 hours**

### Storage
- Original NumPy: ~168 GB (float32)
- Converted HDF5: ~30-40 GB (float16 + gzip)
- **Savings: ~75%**

## Support

If you encounter issues:

1. Check logs in `output/logs/`
2. Review validation report in `output/validation/validation_report.txt`
3. Verify a single HDF5 file manually:

```python
import h5py

with h5py.File('output/hdf5_data/SUBJECT001.hdf5', 'r') as f:
    print("Channels:", list(f.keys()))
    print("C3-M2 shape:", f['C3-M2'].shape)
    print("C3-M2 mean:", f['C3-M2'][:].mean())
    print("C3-M2 std:", f['C3-M2'][:].std())
```

## References

- SleepFM paper: Understanding sleep through multimodal signals
- Your preprocessing: CogPSGFormerPP
- Data analysis documents: 
  - DATA_PREPARATION_ANALYSIS.md
  - CHANNEL_COMPATIBILITY_ANALYSIS.md
