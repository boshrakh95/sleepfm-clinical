# STAGES to SleepFM - Pre-Flight Checklist

## Before You Start

### ✅ Verify Paths in Config

Open `config_stages_conversion.yaml` and verify:

- [ ] **Input base directory** exists:
  ```yaml
  input:
    base_dir: /home/boshra95/scratch/stages/stages/processed
  ```
  Check: `ls /home/boshra95/scratch/stages/stages/processed/eeg_segmented/`

- [ ] **Output directory** is correct:
  ```yaml
  output:
    base_dir: /home/boshra95/scratch/stages/sleepfm_format
  ```
  Note: Will be created automatically

- [ ] **CSV files** exist:
  - `cognitive_targets.csv`
  - `demographics_final.csv`
  - `simple_channels.csv`

### ✅ Install Dependencies

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing
bash setup_environment.sh
```

Or manually:
```bash
pip install loguru pyyaml h5py scipy tqdm matplotlib seaborn scikit-learn
```

### ✅ Choose Processing Mode

Edit `config_stages_conversion.yaml`:

**For testing (recommended first):**
```yaml
subjects:
  pilot_mode: true
  pilot_count: 10
```

**For full processing:**
```yaml
subjects:
  pilot_mode: false
```

## Running the Pipeline

### Option 1: Quick Start (Recommended)

```bash
bash getting_started.sh
```

This will:
1. Check dependencies
2. Verify configuration
3. Test single subject conversion
4. Guide you through next steps

### Option 2: Manual Step-by-Step

#### Step 1: Test Single Subject
```bash
python test_single_subject.py --auto
```

**Expected output:**
```
Processing subject: BOGN00004
✓ Conversion successful
  C3-M2       : shape=(276480,), mean=0.001, std=0.998
  C4-M1       : shape=(276480,), mean=-0.003, std=1.002
  ...
✓ All tests passed
```

**Time:** ~10-30 seconds

#### Step 2: Run Pilot (10 subjects)
```bash
# Ensure pilot_mode: true in config
bash run_pipeline.sh
```

**This runs:**
1. `convert_to_hdf5.py` - Convert 10 subjects
2. `prepare_labels.py` - Prepare cognitive/demographics
3. `create_splits.py` - Create train/val/test splits
4. `validate_data.py` - Validate everything

**Time:** ~5-10 minutes

#### Step 3: Review Results
```bash
# Check validation report
cat /home/boshra95/scratch/stages/sleepfm_format/validation/validation_report.txt

# View logs
ls -lh /home/boshra95/scratch/stages/sleepfm_format/logs/

# Count converted files
ls /home/boshra95/scratch/stages/sleepfm_format/hdf5_data/*.hdf5 | wc -l
```

#### Step 4: Full Run (if pilot successful)
```bash
# Edit config: pilot_mode: false
nano config_stages_conversion.yaml

# Run full pipeline
bash run_pipeline.sh
```

**Time:** ~3-7 hours for 1500 subjects

### Option 3: Individual Scripts

Run each step separately:

```bash
# 1. Convert to HDF5
python convert_to_hdf5.py --config config_stages_conversion.yaml

# 2. Prepare labels
python prepare_labels.py --config config_stages_conversion.yaml

# 3. Create splits
python create_splits.py --config config_stages_conversion.yaml

# 4. Validate
python validate_data.py --config config_stages_conversion.yaml --detailed 5
```

## What to Check

### After Single Subject Test

- [ ] HDF5 file created in `output/hdf5_data/`
- [ ] File size reasonable (~20-30 MB per subject)
- [ ] All 13 channels present
- [ ] Mean ≈ 0, Std ≈ 1 for each channel
- [ ] Duration = 6 hours
- [ ] No errors in console output

### After Pilot Run

- [ ] 10 HDF5 files created
- [ ] `labels_with_demographics.csv` created
- [ ] `dataset_split.json` created
- [ ] Validation report shows 100% valid files
- [ ] Plots generated in `validation/`
- [ ] All modalities present (BAS, RESP, EKG, EMG)

### After Full Run

- [ ] ~1500 HDF5 files created
- [ ] Total size ~30-40 GB
- [ ] Labels file has all subjects
- [ ] Splits are balanced (check `split_distributions.png`)
- [ ] <5% failed conversions (check validation report)

## Troubleshooting

### "ModuleNotFoundError: No module named 'loguru'"
```bash
pip install loguru
# Or run: bash setup_environment.sh
```

### "FileNotFoundError: cognitive_targets.csv"
Check that input paths in config match your data location:
```bash
ls /home/boshra95/scratch/stages/stages/processed/
```

### "Subject missing required modality"
Some subjects may lack entire modalities. They're automatically excluded. This is normal.

### "Memory error"
Reduce workers in config:
```yaml
options:
  num_workers: 4  # Reduce from 8
```

### Conversion very slow
Check I/O speed. Consider:
- Running on faster storage
- Reducing compression: `compression_opts: 2`
- Using fewer workers to reduce I/O contention

## Expected Output Structure

```
/home/boshra95/scratch/stages/sleepfm_format/
├── hdf5_data/              # ← SleepFM will use these
│   ├── BOGN00004.hdf5
│   └── ...
├── labels/
│   └── labels_with_demographics.csv  # ← For fine-tuning
├── splits/
│   └── dataset_split.json            # ← For embedding generation
├── validation/
│   ├── validation_report.txt         # ← Read this!
│   ├── validation_statistics.png
│   └── ...
└── logs/
    └── *.log
```

## After Successful Conversion

### Generate Embeddings

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/pipeline

python generate_embeddings.py \
  --data_dir /home/boshra95/scratch/stages/sleepfm_format/hdf5_data \
  --output_dir /home/boshra95/scratch/stages/sleepfm_format/embeddings \
  --config ../checkpoints/model_base/config.json \
  --checkpoint ../checkpoints/model_base/best.pt \
  --split_json /home/boshra95/scratch/stages/sleepfm_format/splits/dataset_split.json
```

### Fine-tune for Cognitive Prediction

Create adaptation of `finetune_disease_prediction.py`:
- Load embeddings from `embeddings/` directory
- Load labels from `labels_with_demographics.csv`
- Replace Cox loss with MSE or CrossEntropy
- Train LSTM model

## Performance Notes

### Single Subject
- Time: ~10-30 seconds
- Output size: ~20-30 MB
- Memory: <2 GB

### Pilot (10 subjects)
- Time: ~5-10 minutes
- Output size: ~200-300 MB
- Memory: <4 GB

### Full Run (1500 subjects)
- Time: ~3-7 hours (depends on I/O)
- Output size: ~30-40 GB
- Memory: <8 GB (with cache clearing)

## Configuration Tips

### For Testing
```yaml
subjects:
  pilot_mode: true
  pilot_count: 10

options:
  skip_existing: false  # Re-process for testing
  verbose: true
  save_validation_plots: true
```

### For Production
```yaml
subjects:
  pilot_mode: false

options:
  skip_existing: true   # Resume capability
  num_workers: 8
  save_validation_plots: true

advanced:
  clear_cache_per_subject: true  # Prevent memory buildup
```

### For Debugging
```yaml
subjects:
  include: ["BOGN00004", "BOGN00008"]  # Specific subjects

options:
  verbose: true
  
advanced:
  check_for_nans: true
  check_for_infs: true
  verify_normalization: true
```

## File Checklist

All files created in `stages_preprocessing/`:

- [ ] `__init__.py` - Package marker
- [ ] `config_stages_conversion.yaml` - **Main configuration**
- [ ] `convert_to_hdf5.py` - Conversion script
- [ ] `prepare_labels.py` - Label preparation
- [ ] `create_splits.py` - Split creation
- [ ] `validate_data.py` - Validation
- [ ] `test_single_subject.py` - Quick test
- [ ] `run_pipeline.sh` - Full pipeline
- [ ] `setup_environment.sh` - Dependency installer
- [ ] `getting_started.sh` - Quick setup
- [ ] `requirements_stages_prep.txt` - Dependencies
- [ ] `README.md` - Documentation
- [ ] `IMPLEMENTATION_SUMMARY.md` - Overview
- [ ] `CHECKLIST.md` - This file

## Support

### Check Logs
All operations are logged in detail:
```bash
tail -f /home/boshra95/scratch/stages/sleepfm_format/logs/conversion_*.log
```

### Manual Inspection
```python
import h5py

# Check HDF5 file
with h5py.File('output/hdf5_data/BOGN00004.hdf5', 'r') as f:
    print("Channels:", list(f.keys()))
    print("C3-M2 shape:", f['C3-M2'].shape)
    print("C3-M2 stats:", f['C3-M2'][:].mean(), f['C3-M2'][:].std())
```

### Get Help
1. Check validation report first
2. Review logs for specific errors
3. Test single subject to isolate issues
4. Verify config paths are correct

## Ready to Start!

Recommended workflow:

1. ✅ **Run getting started:**
   ```bash
   bash getting_started.sh
   ```

2. ✅ **Review test output**

3. ✅ **Run pilot (10 subjects):**
   ```bash
   # Edit config: pilot_mode: true
   bash run_pipeline.sh
   ```

4. ✅ **Check validation report**

5. ✅ **Run full pipeline when ready:**
   ```bash
   # Edit config: pilot_mode: false
   bash run_pipeline.sh
   ```

Good luck! 🚀
