# Data Preparation Analysis: CogPSGFormerPP → SleepFM

## Executive Summary

Your processed PSG data from **CogPSGFormerPP** is fundamentally **incompatible** with SleepFM's expected format. However, you have **all the necessary raw components** to prepare data for SleepFM fine-tuning. This document provides a comprehensive analysis of the differences and a step-by-step conversion strategy.

---

## 1. CURRENT DATA FORMAT (CogPSGFormerPP)

### **1.1 Data Structure**
```
/scratch/stages/stages/processed/
├── eeg_segmented/
│   └── BOGN00004/
│       ├── C3-M2.npy          # Shape: [720, 3000]
│       ├── C4-M1.npy          # Shape: [720, 3000]
│       ├── O1-M2.npy          # Shape: [720, 3000]
│       ├── O2-M1.npy          # Shape: [720, 3000]
│       ├── *_normstats.json   # Per-channel normalization stats
│       └── reference_metadata.json
├── ecg_segmented/
│   └── BOGN00004/
│       ├── EKG.npy            # Shape: [720, 3000]
│       └── EKG_normstats.json
├── respiratory_segmented/
│   └── BOGN00004/
│       ├── FLOW.npy           # Shape: [720, 3000]
│       ├── THOR.npy           # Shape: [720, 3000]
│       ├── ABDM.npy           # Shape: [720, 3000]
│       └── *_normstats.json
├── eog_segmented/
│   └── BOGN00004/
│       ├── LOC.npy            # Shape: [720, 3000]
│       ├── ROC.npy            # Shape: [720, 3000]
│       └── *_normstats.json
├── emg_segmented/
│   └── BOGN00004/
│       ├── CHIN.npy           # Shape: [720, 3000]
│       ├── RLEG.npy           # Shape: [720, 3000]
│       ├── LLEG.npy           # Shape: [720, 3000]
│       └── *_normstats.json
├── cognitive_targets.csv      # Cognitive test scores
├── demographics_final.csv     # Age, sex, BMI
└── master_masks/              # Artifact masks
```

### **1.2 Key Characteristics**

| **Aspect** | **Your Data** |
|------------|---------------|
| **Format** | NumPy arrays (.npy) |
| **Organization** | Separate files per channel, per modality |
| **Temporal structure** | Pre-segmented into 30-second windows |
| **Shape** | `[num_windows, samples_per_window]` = `[720, 3000]` |
| **Sampling rate** | 100 Hz (downsampled from original) |
| **Total duration** | 720 windows × 30 sec = 21,600 sec = 6 hours |
| **Samples per window** | 30 sec × 100 Hz = 3,000 samples |
| **Normalization** | Z-score normalized per channel (stats saved separately) |
| **Channel naming** | Standardized (C3-M2, C4-M1, etc.) |
| **Channels per modality** | Variable (4 EEG, 1 ECG, 3 RESP, 2 EOG, 3 EMG) |

---

## 2. SLEEPFM EXPECTED FORMAT

### **2.1 Data Structure**
```
data/
├── subject_001.hdf5
│   ├── C3 (dataset): [total_samples] at 128 Hz
│   ├── C4 (dataset): [total_samples] at 128 Hz
│   ├── O1 (dataset): [total_samples] at 128 Hz
│   ├── O2 (dataset): [total_samples] at 128 Hz
│   ├── EKG (dataset): [total_samples] at 128 Hz
│   ├── Airflow (dataset): [total_samples] at 128 Hz
│   ├── ... (one continuous dataset per channel)
└── subject_002.hdf5
    └── ...
```

### **2.2 Key Requirements**

| **Aspect** | **SleepFM Expects** |
|------------|---------------------|
| **Format** | HDF5 (.hdf5) |
| **Organization** | One HDF5 file per subject, all channels as datasets |
| **Temporal structure** | **Continuous time series** (NOT pre-segmented) |
| **Shape per channel** | `[total_samples]` (1D array) |
| **Sampling rate** | **128 Hz** (fixed) |
| **Example duration** | 6 hours = 21,600 sec → 2,764,800 samples @ 128 Hz |
| **Normalization** | **Z-score per channel** (applied inline, NOT separate) |
| **Channel naming** | Raw channel names (C3, C4, not C3-M2) |
| **Chunking** | Done by SleepFM dynamically (5-min chunks for pretraining) |

---

## 3. CRITICAL DIFFERENCES

### **3.1 Format: NumPy → HDF5**
- **Your data**: Separate `.npy` files per channel
- **SleepFM**: Single `.hdf5` file per subject with all channels as internal datasets

### **3.2 Temporal Structure: Segmented → Continuous**
- **Your data**: Pre-segmented into `[720, 3000]` (windows × samples)
- **SleepFM**: Continuous `[2,764,800]` (single 1D time series)
- **Impact**: You must **concatenate all 720 windows** back into a continuous signal

### **3.3 Sampling Rate: 100 Hz → 128 Hz**
- **Your data**: Downsampled to 100 Hz
- **SleepFM**: Requires **128 Hz**
- **Impact**: You must **resample from 100 Hz to 128 Hz**

### **3.4 Channel Naming: Referenced → Raw**
- **Your data**: C3-M2, C4-M1, O1-M2, O2-M1 (already referenced)
- **SleepFM channel groups**: Expects raw names like C3, C4, O1, O2
- **Impact**: **Channel name mismatch** - SleepFM's `channel_groups.json` uses different names

### **3.5 Normalization: Separate Stats → Inline**
- **Your data**: Normalization stats stored in JSON files
- **SleepFM**: Expects normalized data embedded in HDF5
- **Impact**: Data is already normalized (good!), but stats are external

### **3.6 Channel Organization: Modality Folders → Single File**
- **Your data**: Channels grouped by modality in separate folders
- **SleepFM**: All channels for one subject in one HDF5 file
- **Impact**: Must merge data from multiple modality folders

---

## 4. MODALITY MAPPING

### **4.1 Your Channels → SleepFM Modality Groups**

Based on SleepFM's `channel_groups.json`, here's how your channels map:

| **Your Channels** | **SleepFM Modality** | **Expected Names in SleepFM** | **Action Required** |
|-------------------|----------------------|-------------------------------|---------------------|
| C3-M2, C4-M1, O1-M2, O2-M1 | **BAS** (Brain Activity Signals) | C3, C4, O1, O2, F3, F4, EOG(L), EOG(R), etc. | ✅ Compatible (rename to remove reference) |
| FLOW, THOR, ABDM | **RESP** (Respiratory) | Airflow, THOR, ABD, RespRate, etc. | ✅ Compatible (FLOW → Airflow, THOR/ABDM OK) |
| EKG | **EKG** (Cardiac) | ECG, EKG, HR, etc. | ✅ Compatible |
| CHIN, RLEG, LLEG | **EMG** (Muscle) | CHIN, CHIN1, CHIN2, LEG, LAT, RAT, etc. | ✅ Compatible |
| LOC, ROC | **BAS** or separate EOG | EOG(L), EOG(R), E1, E2, etc. | ⚠️ EOG treated as BAS in SleepFM |

### **4.2 Channel Naming Strategy**

**Option 1: Minimal Changes (Recommended)**
```python
# Map your channels to SleepFM-compatible names
channel_mapping = {
    # EEG
    "C3-M2": "C3",
    "C4-M1": "C4",
    "O1-M2": "O1",
    "O2-M1": "O2",
    
    # ECG
    "EKG": "EKG",
    
    # Respiratory
    "FLOW": "Airflow",
    "THOR": "THOR",
    "ABDM": "ABD",
    
    # EOG (treated as BAS in SleepFM)
    "LOC": "EOG(L)",
    "ROC": "EOG(R)",
    
    # EMG
    "CHIN": "CHIN",
    "RLEG": "LEG1",  # Or "RLEG" if you update channel_groups.json
    "LLEG": "LEG2",  # Or "LLEG"
}
```

**Option 2: Update SleepFM's `channel_groups.json`**
Add your exact channel names to the appropriate modality groups.

---

## 5. PREPROCESSING REQUIREMENTS FOR SLEEPFM

### **5.1 What SleepFM's Preprocessing Does**

From `sleepfm/preprocessing/preprocessing.py`:
1. **Load EDF files**
2. **Resample to 128 Hz** (using linear interpolation)
3. **Z-score normalize** per channel: `(signal - mean) / std`
4. **Save to HDF5** with compression

### **5.2 What You've Already Done (Can Skip)**

✅ **Channel selection and filtering** (your data is clean)  
✅ **Z-score normalization** (already applied)  
✅ **Artifact detection** (you have master masks)  
✅ **Sleep stage cropping** (you've extracted only sleep periods)

### **5.3 What You Still Need to Do**

❌ **Concatenate 30-sec windows** into continuous signals  
❌ **Resample from 100 Hz → 128 Hz**  
❌ **Convert NumPy → HDF5** format  
❌ **Organize all channels** into single HDF5 per subject  
❌ **Ensure normalization** is embedded (already done, just verify)

---

## 6. STEP-BY-STEP DATA PREPARATION PLAN

### **STAGE 1: Reconstruct Continuous Signals**

For each subject and each channel:

1. **Load all 30-sec segments**
   ```python
   segments = np.load("eeg_segmented/BOGN00004/C3-M2.npy")  # [720, 3000]
   ```

2. **Concatenate into continuous signal**
   ```python
   continuous = segments.reshape(-1)  # [2,160,000] at 100 Hz
   # 720 windows × 3000 samples = 2,160,000 samples
   # Duration: 2,160,000 / 100 Hz = 21,600 sec = 6 hours
   ```

3. **Handle potential gaps** (if any windows were excluded)
   - Check artifact masks in `master_masks/`
   - Option A: Zero-pad excluded windows
   - Option B: Skip subjects with too many artifacts

### **STAGE 2: Resample to 128 Hz**

```python
from scipy.signal import resample

# Current: 2,160,000 samples @ 100 Hz (6 hours)
# Target: 2,764,800 samples @ 128 Hz (6 hours)
target_samples = int(len(continuous) * 128 / 100)
resampled = resample(continuous, target_samples)
```

**Alternative (better for long signals):**
```python
from scipy.interpolate import interp1d

old_time = np.arange(len(continuous)) / 100  # Time points at 100 Hz
new_time = np.arange(target_samples) / 128   # Time points at 128 Hz
interpolator = interp1d(old_time, continuous, kind='linear')
resampled = interpolator(new_time)
```

### **STAGE 3: Verify Normalization**

Your data is already z-score normalized. Verify:
```python
# Should be close to 0
print("Mean:", resampled.mean())  
# Should be close to 1
print("Std:", resampled.std())
```

If not, re-normalize:
```python
resampled = (resampled - resampled.mean()) / resampled.std()
```

### **STAGE 4: Create HDF5 Files**

For each subject, create one HDF5 file with all channels:

```python
import h5py

subject_id = "BOGN00004"Ok write me a script that can prepare my preprocessed data for sleepfm. 
output_path = f"data_for_sleepfm/{subject_id}.hdf5"

with h5py.File(output_path, 'w') as hf:
    # EEG channels (BAS modality)
    hf.create_dataset("C3", data=eeg_c3_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("C4", data=eeg_c4_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("O1", data=eeg_o1_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("O2", data=eeg_o2_resampled, 
                      dtype='float16', compression="gzip")
    
    # EOG channels (BAS modality in SleepFM)
    hf.create_dataset("EOG(L)", data=eog_loc_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("EOG(R)", data=eog_roc_resampled, 
                      dtype='float16', compression="gzip")
    
    # ECG channel (EKG modality)
    hf.create_dataset("EKG", data=ecg_resampled, 
                      dtype='float16', compression="gzip")
    
    # Respiratory channels (RESP modality)
    hf.create_dataset("Airflow", data=resp_flow_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("THOR", data=resp_thor_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("ABD", data=resp_abdm_resampled, 
                      dtype='float16', compression="gzip")
    
    # EMG channels (EMG modality)
    hf.create_dataset("CHIN", data=emg_chin_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("LEG1", data=emg_rleg_resampled, 
                      dtype='float16', compression="gzip")
    hf.create_dataset("LEG2", data=emg_lleg_resampled, 
                      dtype='float16', compression="gzip")
```

### **STAGE 5: Prepare Label Files**

#### **5.1 Cognitive Targets**

Your `cognitive_targets.csv` is already well-formatted! Minor adjustments:

```python
import pandas as pd

# Load cognitive targets
cog_df = pd.read_csv("cognitive_targets.csv")

# Rename subject ID column to match SleepFM convention
cog_df = cog_df.rename(columns={"subject_id": "Study ID"})

# Select columns for your task (e.g., composite scores)
cog_labels = cog_df[[
    "Study ID",
    "sustained_attention",
    "working_memory",
    "episodic_memory",
    "executive_functioning",
    "overall_cognitive"
]]

# Save for SleepFM
cog_labels.to_csv("cognitive_labels_for_sleepfm.csv", index=False)
```

#### **5.2 Demographics**

```python
demo_df = pd.read_csv("demographics_final.csv")

# Convert sex to numeric (required for SleepFM)
demo_df["nsrr_sex"] = demo_df["nsrr_sex"].map({"male": 1, "female": 0})

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
demo_df[["nsrr_age", "nsrr_bmi"]] = scaler.fit_transform(
    demo_df[["nsrr_age", "nsrr_bmi"]]
)

# Keep only age and gender (SleepFM default expects 2 features)
demo_final = demo_df[["subject_id", "nsrr_age", "nsrr_sex"]].rename(
    columns={"subject_id": "Study ID"}
)

demo_final.to_csv("demographics_for_sleepfm.csv", index=False)
```

#### **5.3 Train/Val/Test Split**

Create a JSON file mapping splits to HDF5 paths:

```python
import json
from sklearn.model_selection import train_test_split

# Get all subject IDs
subject_ids = cog_df["subject_id"].tolist()

# Split: 70% train, 15% val, 15% test
train_ids, temp_ids = train_test_split(subject_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Create paths (relative to data directory)
split_dict = {
    "train": [f"data_for_sleepfm/{sid}.hdf5" for sid in train_ids],
    "validation": [f"data_for_sleepfm/{sid}.hdf5" for sid in val_ids],
    "test": [f"data_for_sleepfm/{sid}.hdf5" for sid in test_ids]
}

with open("cognitive_split.json", "w") as f:
    json.dump(split_dict, f, indent=2)
```

### **STAGE 6: Generate Embeddings**

**CRITICAL**: You must generate embeddings using SleepFM's pretrained transformer before fine-tuning!

```bash
python sleepfm/pipeline/generate_embeddings.py \
    --model_path sleepfm/checkpoints/model_base \
    --dataset_name stages \
    --split_path cognitive_split.json \
    --splits train,validation,test \
    --num_workers 8 \
    --batch_size 64
```

**Output:**
```
embeddings/
├── stages/
│   ├── BOGN00004.hdf5  # Contains BAS, RESP, EKG, EMG embeddings
│   └── ...
└── stages_5min_agg/
    ├── BOGN00004.hdf5  # 5-min aggregated embeddings (for cognitive task)
    └── ...
```

---

## 7. POTENTIAL ISSUES & SOLUTIONS

### **Issue 1: Missing Channels**

**Problem**: Some subjects might not have all channels.

**Solution**:
- Filter subjects to only those with required modalities
- OR: Use SleepFM's masking mechanism (channels marked as padding)

### **Issue 2: Sampling Rate Conversion Artifacts**

**Problem**: Resampling from 100 Hz → 128 Hz may introduce minor artifacts.

**Solution**:
- Use high-quality interpolation (scipy's `resample` or `interp1d`)
- Apply anti-aliasing filter before upsampling
- Verify power spectra before/after resampling

### **Issue 3: Normalization Consistency**

**Problem**: Your normalization was computed with artifact masks; SleepFM doesn't use masks during preprocessing.

**Solution**:
- Keep your current normalization (it's cleaner!)
- OR: Re-normalize without masks for consistency

### **Issue 4: Memory Constraints**

**Problem**: Loading 1500+ subjects × 13 channels into memory for conversion.

**Solution**:
- Process subjects one at a time
- Use parallel processing with multiprocessing
- Stream data directly from NumPy → HDF5 without full load

### **Issue 5: Channel Name Mismatches**

**Problem**: SleepFM's pretrained model expects specific channel names.

**Solution A** (Easier): Update your channel names to match SleepFM  
**Solution B**: Modify `sleepfm/configs/channel_groups.json` to include your names

---

## 8. DATA VALIDATION CHECKLIST

Before using your converted data with SleepFM, verify:

- [ ] **HDF5 structure**: One file per subject, channels as datasets
- [ ] **Sampling rate**: All channels at 128 Hz
- [ ] **Data type**: float16 or float32 (not float64)
- [ ] **Compression**: gzip compression applied
- [ ] **Shape**: Each channel is 1D array `[num_samples]`
- [ ] **Duration**: Consistent across channels (same length)
- [ ] **Normalization**: Mean ≈ 0, Std ≈ 1 for each channel
- [ ] **Channel names**: Match SleepFM's `channel_groups.json`
- [ ] **No NaNs/Infs**: All values are valid numbers
- [ ] **Label files**: CSV with "Study ID" column
- [ ] **Split file**: JSON with valid HDF5 paths
- [ ] **Embeddings**: Generated for all subjects in split

---

## 9. RECOMMENDED WORKFLOW

### **Phase 1: Pilot Conversion (1-2 days)**
1. Select 10 subjects
2. Convert to HDF5 format
3. Generate embeddings
4. Test with SleepFM fine-tuning code
5. Validate results

### **Phase 2: Full Conversion (3-5 days)**
1. Convert all 1500+ subjects
2. Quality control checks
3. Generate embeddings for all subjects
4. Create final train/val/test splits

### **Phase 3: Model Development (1-2 weeks)**
1. Adapt SleepFM's disease prediction code for cognitive prediction
2. Implement custom model architecture (if needed)
3. Train and validate models
4. Hyperparameter tuning

---

## 10. ALTERNATIVE APPROACH: Skip Preprocessing

**Option**: Use your existing NumPy format and create a custom SleepFM dataset class.

**Pros**:
- No need to convert to HDF5
- Preserve 30-sec windowing
- Faster iteration

**Cons**:
- Cannot use SleepFM's pretrained transformer directly
- Must train transformer from scratch
- More code modifications needed

**Not recommended** unless you need very specific custom processing.

---

## 11. SUMMARY OF ACTION ITEMS

### **Must Do (Critical Path)**
1. ✅ **Concatenate** 30-sec windows → continuous signals
2. ✅ **Resample** 100 Hz → 128 Hz
3. ✅ **Convert** NumPy → HDF5 (one file per subject)
4. ✅ **Rename** channels to match SleepFM convention
5. ✅ **Prepare** cognitive labels CSV
6. ✅ **Prepare** demographics CSV (age, sex normalized)
7. ✅ **Create** train/val/test split JSON
8. ✅ **Generate** embeddings using SleepFM's pretrained model
9. ✅ **Adapt** fine-tuning code for cognitive prediction

### **Nice to Have (Optional)**
- [ ] Update `channel_groups.json` with your exact channel names
- [ ] Apply artifact masks during conversion
- [ ] Add sleep stage annotations (if available)
- [ ] Create visualization scripts for quality control

---

## 12. ESTIMATED EFFORT

| **Task** | **Effort** | **Complexity** |
|----------|------------|----------------|
| Write conversion script | 1 day | Medium |
| Test on pilot subjects | 0.5 day | Low |
| Convert all subjects | 2 days | Low (mostly compute) |
| Generate embeddings | 1-2 days | Low (mostly compute) |
| Adapt fine-tuning code | 2-3 days | Medium |
| **Total** | **6-8 days** | **Medium** |

---

## Next Steps

When you're ready, I can write the conversion script that:
1. Loads your NumPy segmented data
2. Concatenates into continuous signals
3. Resamples to 128 Hz
4. Saves as HDF5 in SleepFM format
5. Prepares all label files
6. Creates split JSON

**Would you like me to proceed with writing the code?**
