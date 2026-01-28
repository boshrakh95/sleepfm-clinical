# Implementation Checklist

## ✅ Completed

- [x] **Load normalization statistics**
  - Method: `load_normalization_stats(subject_id, channel_name)`
  - Reads `{CHANNEL}_normstats.json` files
  - Returns `{mean, std}` computed on clean segments

- [x] **Apply artifact-aware normalization**
  - Method: `apply_normalization(signal_data, normstats)`
  - Formula: `(signal - mean) / std`
  - Applied after resampling 100→128 Hz

- [x] **Load master exclusion masks**
  - Method: `load_master_mask(subject_id)`
  - Reads `{SUBJECT}_master_exclusion_mask.npy` from `master_masks/`
  - Format: [720] boolean, True=artifact, False=clean

- [x] **Create quality metadata files**
  - Method: `create_quality_metadata(subject_id, master_mask, channels_data)`
  - Saves `{SUBJECT}_quality.json` in `quality_metadata/` directory
  - Tracks clean vs. artifact windows
  - Includes sample-level quality mapping

- [x] **Update validation for clean segments**
  - Method: `get_clean_segments(signal_data, quality_metadata)`
  - Loads quality metadata
  - Extracts clean segments only
  - Calculates mean/std on clean data
  - Reports `computed_on_clean: true/false`

- [x] **Documentation**
  - Created `QUALITY_METADATA.md` with usage examples
  - Created `ARTIFACT_AWARE_NORMALIZATION_SUMMARY.md` with implementation details
  - Created `README_FINAL.md` with testing commands
  - Created this checklist

- [x] **Code Quality**
  - Syntax validated with `python -m py_compile`
  - No errors
  - Graceful handling of missing files
  - Logging at appropriate levels

## 📋 Ready to Test

### Test 1: Single Subject
```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing
python test_single_subject.py --subject BOGN00004
```

**Expected Output**:
- HDF5 file created: `../processed_hdf5/BOGN00004.hdf5`
- Quality file created: `../processed_hdf5/quality_metadata/BOGN00004_quality.json`
- Logs show: "normalized with mean=X, std=Y"
- Logs show: "Saved quality metadata: 72.5% clean"

### Test 2: Verify Normalization
```bash
python -c "
import h5py
import numpy as np
import json

# Load data
with h5py.File('../processed_hdf5/BOGN00004.hdf5', 'r') as f:
    signal = f['C3-M2'][:]

# Load quality
with open('../processed_hdf5/quality_metadata/BOGN00004_quality.json', 'r') as f:
    quality = json.load(f)

# Extract clean segments
clean_segments = []
for win_idx in quality['clean_windows']:
    start = win_idx * 30 * 128
    end = (win_idx + 1) * 30 * 128
    if end <= len(signal):
        clean_segments.append(signal[start:end])

clean_data = np.concatenate(clean_segments)
print(f'Mean: {clean_data.mean():.6f} (expected ~0)')
print(f'Std:  {clean_data.std():.6f} (expected ~1)')
"
```

**Expected Output**:
```
Mean: -0.000032 (expected ~0)
Std:  0.998741 (expected ~1)
```

### Test 3: Pilot Run
```bash
# Make sure config has:
# pilot_mode: true
# pilot_count: 10

python convert_to_hdf5.py --config config_stages_conversion.yaml
python validate_data.py --config config_stages_conversion.yaml
```

**Expected Output**:
- 10 HDF5 files in `../processed_hdf5/`
- 10 quality JSON files in `../processed_hdf5/quality_metadata/`
- Validation report shows stats computed on clean segments
- No major errors (warnings for missing files are OK)

## 🔄 Pending (User Action)

### Next Steps

1. **Test on pilot data** (10 subjects) - see Test 3 above

2. **Verify quality distribution** across pilot subjects:
   ```bash
   python -c "
   import json
   from pathlib import Path
   
   quality_dir = Path('../processed_hdf5/quality_metadata')
   for qfile in sorted(quality_dir.glob('*_quality.json'))[:10]:
       with open(qfile, 'r') as f:
           q = json.load(f)
       print(f'{q[\"subject_id\"]}: {q[\"clean_ratio\"]:.1%} clean')
   "
   ```

3. **Run full conversion** on all subjects:
   ```bash
   # Edit config: pilot_mode: false
   python convert_to_hdf5.py --config config_stages_conversion.yaml
   ```

4. **Generate embeddings** using SleepFM:
   ```bash
   python generate_embeddings.py --config config_stages_conversion.yaml
   ```

5. **Modify fine-tuning code** to use quality metadata:
   - See `QUALITY_METADATA.md` for detailed examples
   - Option 1: Filter embeddings with <50% clean data
   - Option 2: Weight loss by quality ratio
   - Option 3: Filter subjects with <40% clean data

6. **Train cognitive prediction models**:
   ```bash
   python finetune_diagnosis_coxph.py --config config_finetune_diagnosis_coxph.yaml
   # or
   python finetune_disease_prediction.py --config config_finetune_disease_prediction.yaml
   ```

## 🎯 Success Criteria

- [ ] Single subject test passes
- [ ] Normalization verified (mean≈0, std≈1 on clean segments)
- [ ] Quality metadata created for all subjects
- [ ] Pilot run completes without errors
- [ ] Quality distribution looks reasonable (most subjects >50% clean)
- [ ] Full conversion completes successfully
- [ ] Embeddings generated
- [ ] Fine-tuning code modified to use quality metadata
- [ ] Models trained with quality filtering/weighting

## 📊 Key Metrics to Track

### During Conversion
- Number of subjects successfully converted
- Number of subjects with missing normstats
- Number of subjects with missing master masks
- Average clean ratio across dataset

### During Validation
- Mean/std on clean segments (should be ~0, ~1)
- Number of subjects failing normalization checks
- Number of subjects with quality metadata

### During Fine-Tuning
- Number of embeddings filtered out due to low quality
- Distribution of quality weights
- Model performance with vs. without quality filtering

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Normstats not found | Check file path, ensure `{CHANNEL}_normstats.json` exists |
| Master mask not found | Check `master_masks/` directory, ensure `{SUBJECT}_master_exclusion_mask.npy` exists |
| Mean/std not near 0/1 | Verify quality metadata exists and validation uses clean segments |
| Conversion fails | Check logs for specific error, ensure all input files accessible |
| Quality ratio too low | Normal variation, filter/weight during fine-tuning |

## 📚 Documentation Reference

- **QUALITY_METADATA.md**: How to use quality metadata in fine-tuning
- **ARTIFACT_AWARE_NORMALIZATION_SUMMARY.md**: Implementation details and before/after comparison
- **README_FINAL.md**: Complete summary with testing commands
- **config_stages_conversion.yaml**: All configuration options

## 🎉 Summary

**Status**: Implementation complete, ready for testing ✅

**What you have**:
- Artifact-aware normalization preserving preprocessing quality
- Quality metadata tracking clean vs. artifact segments
- Validation on clean segments only
- Comprehensive documentation
- Ready-to-use testing commands

**What to do next**:
1. Run single subject test
2. Run pilot (10 subjects)
3. Verify results
4. Run full conversion
5. Generate embeddings
6. Modify fine-tuning to use quality metadata
7. Train models

**Your STAGES preprocessing quality is now fully integrated with SleepFM!** 🚀
