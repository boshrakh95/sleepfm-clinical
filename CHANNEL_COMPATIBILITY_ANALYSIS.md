# Channel Compatibility Analysis: Your Data vs SleepFM

## Executive Summary

**Good News**: Your channel names are **COMPATIBLE** with SleepFM, but you need to understand how SleepFM handles channels and missing modalities.

---

## 1. HOW SLEEPFM HANDLES CHANNELS

### **1.1 Channel-Agnostic Architecture (Key Insight!)**

From the paper and code, SleepFM is **channel-agnostic**, meaning:
- ✅ It **does NOT require specific channel names**
- ✅ It groups channels by **modality type** (BAS, RESP, EKG, EMG)
- ✅ It uses **masking** to handle variable numbers of channels
- ✅ It can work with **any channels** as long as they map to the 4 modalities

### **1.2 The Modality System**

SleepFM has **4 modality types**:

```json
{
  "BAS": [...],   // Brain Activity Signals (EEG + EOG)
  "RESP": [...],  // Respiratory signals
  "EKG": [...],   // Cardiac signals
  "EMG": [...]    // Muscle activity
}
```

**CRITICAL**: The channel names in `channel_groups.json` are just **lookup lists**. SleepFM checks:
1. Does this channel name appear in any modality list?
2. If YES → assign to that modality
3. If NO → ignore the channel

---

## 2. YOUR CHANNELS → SLEEPFM MODALITIES

### **2.1 Current Mapping**

| **Your Channels** | **Found in channel_groups.json?** | **SleepFM Modality** | **Status** |
|-------------------|-----------------------------------|----------------------|------------|
| C3-M2 | ✅ YES (line 206: "C3-M2") | BAS | ✅ Compatible |
| C4-M1 | ✅ YES (line 199: "C4-M1") | BAS | ✅ Compatible |
| O1-M2 | ✅ YES (line 207: "O1-M2") | BAS | ✅ Compatible |
| O2-M1 | ✅ YES (line 200: "O2-M1") | BAS | ✅ Compatible |
| LOC | ❌ NO (but variants exist) | BAS (EOG) | ⚠️ Need to add |
| ROC | ❌ NO (but variants exist) | BAS (EOG) | ⚠️ Need to add |
| EKG | ✅ YES (multiple entries) | EKG | ✅ Compatible |
| FLOW | ✅ YES (line 445: "Flow") | RESP | ⚠️ Use "Flow" |
| THOR | ✅ YES (line 446: "Thor") | RESP | ⚠️ Use "Thor" |
| ABDM | ❌ NO (but "ABD" exists) | RESP | ⚠️ Use "ABD" |
| CHIN | ✅ YES (many variants) | EMG | ✅ Compatible |
| RLEG | ✅ YES (line 729: "RLEG") | EMG | ✅ Compatible |
| LLEG | ✅ YES (line 728: "LLEG") | EMG | ✅ Compatible |

### **2.2 Recommended Channel Naming for HDF5**

**Option A: Add your names to `channel_groups.json`** (Easier)
```json
{
  "BAS": [
    "LOC",
    "ROC",
    ...existing channels...
  ],
  "RESP": [
    "FLOW",
    "ABDM",
    ...existing channels...
  ]
}
```

**Option B: Rename during conversion** (What I recommend)
```python
channel_mapping = {
    # Your name → SleepFM-compatible name
    "C3-M2": "C3-M2",    # Already in list
    "C4-M1": "C4-M1",    # Already in list
    "O1-M2": "O1-M2",    # Already in list
    "O2-M1": "O2-M1",    # Already in list
    "LOC": "EOG(L)",     # Use SleepFM's EOG format
    "ROC": "EOG(R)",     # Use SleepFM's EOG format
    "EKG": "EKG",        # Already in list
    "FLOW": "Flow",      # Capitalization matters!
    "THOR": "Thor",      # Capitalization matters!
    "ABDM": "ABD",       # Use SleepFM's name
    "CHIN": "CHIN",      # Already in list
    "RLEG": "RLEG",      # Already in list
    "LLEG": "LLEG",      # Already in list
}
```

---

## 3. HOW SLEEPFM HANDLES MISSING CHANNELS

### **3.1 The Masking Mechanism**

From `dataset.py` lines 144-168:

```python
# For each modality, SleepFM:
max_channels = max(data[modality_index].shape[0] for data in batch)

for data in batch:
    channels, length = modality_data.shape
    pad_channels = max_channels - channels
    
    # Create mask: 0 for real values, 1 for padded values
    mask = torch.cat((torch.zeros(channels), torch.ones(pad_channels)), dim=0)
    
    # Pad the channel dimension
    pad_channel_tensor = torch.zeros((pad_channels, length))
    modality_data = torch.cat((modality_data, pad_channel_tensor), dim=0)
```

**What this means:**
- ✅ Subjects can have **different numbers of channels** per modality
- ✅ Missing channels are **zero-padded**
- ✅ Masks tell the model which channels are real (0) vs padding (1)
- ✅ The model learns to **ignore padding** via attention masking

### **3.2 Requirement: At Least One Channel Per Modality**

From `index_file_helper` (lines 27-31):

```python
flag = True
for modality, channels in modality_to_channels.items():
    if len(channels) == 0:  # If ANY modality has 0 channels
        flag = False        # Skip this file!
        break
if flag:  # Only process if ALL modalities have at least 1 channel
    ...
```

**CRITICAL RULE**: 
- ❌ If a subject is **missing an entire modality**, the file is **excluded**
- ✅ If a subject has **some channels** in each modality, it's **included**

**Example:**
```
Subject A:
✅ BAS: C3-M2, C4-M1 (2 channels)
✅ RESP: Flow (1 channel - OK!)
✅ EKG: EKG (1 channel)
✅ EMG: CHIN (1 channel - OK!)
→ ACCEPTED

Subject B:
✅ BAS: C3-M2, C4-M1, O1-M2, O2-M1
❌ RESP: (0 channels - MISSING MODALITY!)
✅ EKG: EKG
✅ EMG: CHIN, RLEG, LLEG
→ REJECTED
```

---

## 4. YOUR DATA: WHAT WILL HAPPEN

### **4.1 Your Channel Availability**

From your preprocessed data, you have:
```
BAS (Brain): C3-M2, C4-M1, O1-M2, O2-M1, LOC, ROC (6 channels)
RESP: FLOW, THOR, ABDM (3 channels)
EKG: EKG (1 channel)
EMG: CHIN, RLEG, LLEG (3 channels)
```

**Result**: ✅ **ALL 4 modalities present** → Your subjects will be **ACCEPTED**

### **4.2 Channel Count Configuration**

In your config YAML, you need to set:

```yaml
# Max channels per modality (for padding)
BAS_CHANNELS: 10   # You have 6 (C3-M2, C4-M1, O1-M2, O2-M1, LOC, ROC)
RESP_CHANNELS: 7   # You have 3 (Flow, Thor, ABD)
EKG_CHANNELS: 2    # You have 1 (EKG)
EMG_CHANNELS: 4    # You have 3 (CHIN, RLEG, LLEG)
```

**What happens:**
- Your 6 BAS channels → padded to 10 (4 channels of zeros)
- Your 3 RESP channels → padded to 7 (4 channels of zeros)
- Your 1 EKG channel → padded to 2 (1 channel of zeros)
- Your 3 EMG channels → padded to 4 (1 channel of zeros)

The model **ignores the padding** via masks.

---

## 5. FINAL HDF5 STRUCTURE (What You Should Create)

### **5.1 Recommended Structure**

```
BOGN00004.hdf5
├── C3-M2 (dataset): [2,764,800] at 128 Hz, float16
├── C4-M1 (dataset): [2,764,800] at 128 Hz, float16
├── O1-M2 (dataset): [2,764,800] at 128 Hz, float16
├── O2-M1 (dataset): [2,764,800] at 128 Hz, float16
├── EOG(L) (dataset): [2,764,800] at 128 Hz, float16  # renamed from LOC
├── EOG(R) (dataset): [2,764,800] at 128 Hz, float16  # renamed from ROC
├── EKG (dataset): [2,764,800] at 128 Hz, float16
├── Flow (dataset): [2,764,800] at 128 Hz, float16    # renamed from FLOW
├── Thor (dataset): [2,764,800] at 128 Hz, float16    # renamed from THOR
├── ABD (dataset): [2,764,800] at 128 Hz, float16     # renamed from ABDM
├── CHIN (dataset): [2,764,800] at 128 Hz, float16
├── RLEG (dataset): [2,764,800] at 128 Hz, float16
└── LLEG (dataset): [2,764,800] at 128 Hz, float16
```

### **5.2 What SleepFM Will Do**

1. **Open HDF5 file**: Read all dataset names
2. **Group by modality**: 
   - BAS: [C3-M2, C4-M1, O1-M2, O2-M1, EOG(L), EOG(R)]
   - RESP: [Flow, Thor, ABD]
   - EKG: [EKG]
   - EMG: [CHIN, RLEG, LLEG]
3. **Load 5-minute chunks**: Extract data for each channel
4. **Pad to max**: Pad each modality to its max channel count
5. **Create masks**: Mark real channels (0) vs padding (1)
6. **Process through model**: Generate embeddings

---

## 6. HANDLING SUBJECTS WITH MISSING CHANNELS

### **6.1 Subjects Missing Some Channels (Within Modality)**

**Scenario**: Subject has C3-M2, C4-M1 but missing O1-M2, O2-M1

**What happens:**
```
BAS modality has 2 channels instead of 6
→ Model pads to 10 channels (8 zeros)
→ Model processes normally
→ ✅ Subject INCLUDED
```

### **6.2 Subjects Missing Entire Modality**

**Scenario**: Subject has no respiratory channels (FLOW, THOR, ABDM all missing)

**What happens:**
```
RESP modality has 0 channels
→ File is SKIPPED during indexing
→ ❌ Subject EXCLUDED
```

**From your `simple_channels.csv`**: 
Check which subjects have all modalities. Filter before conversion.

---

## 7. RECOMMENDED APPROACH

### **Step 1: Update channel_groups.json** (One-time)

Add your exact channel names:

```json
{
  "BAS": [
    "LOC",
    "ROC",
    ...existing...
  ],
  "RESP": [
    "FLOW",
    "ABDM",
    ...existing...
  ]
}
```

**OR** use the mapping during conversion (Option B above).

### **Step 2: Filter Subjects**

Before conversion, identify subjects with all 4 modalities:

```python
import pandas as pd

# Load channel availability
channels_df = pd.read_csv("simple_channels.csv")

def has_all_modalities(channels_str):
    channels = set(channels_str.split(','))
    has_bas = any(c in ['C3M2', 'C4M1', 'O1M2', 'O2M1', 'E1M2', 'E2M2'] 
                  for c in channels)
    has_resp = any(c in ['FLOW', 'THOR', 'ABDM', 'PTAF'] 
                   for c in channels)
    has_ekg = 'EKG' in channels
    has_emg = any(c in ['CHIN', 'RLEG', 'LLEG'] 
                  for c in channels)
    
    return has_bas and has_resp and has_ekg and has_emg

valid_subjects = channels_df[
    channels_df['channels'].apply(has_all_modalities)
]['subject_id'].tolist()

print(f"Valid subjects: {len(valid_subjects)}")
```

### **Step 3: Convert Only Valid Subjects**

```python
for subject_id in valid_subjects:
    convert_subject_to_hdf5(subject_id)
```

---

## 8. ANSWER TO YOUR QUESTIONS

### **Q1: Are my channel names okay?**

**A**: Almost! You need to:
1. ✅ Keep: C3-M2, C4-M1, O1-M2, O2-M1, EKG, CHIN, RLEG, LLEG
2. ⚠️ Rename: LOC→EOG(L), ROC→EOG(R), FLOW→Flow, THOR→Thor, ABDM→ABD
3. **OR**: Add your names to `channel_groups.json`

### **Q2: What about missing channels?**

**A**: 
- ✅ **Missing some channels within a modality**: OK (padded)
- ❌ **Missing entire modality**: Subject excluded
- ✅ **Your data**: All subjects have all 4 modalities → All OK

### **Q3: How should the final data look?**

**A**: HDF5 file per subject with:
- ✅ One 1D dataset per channel
- ✅ Each dataset: `[num_samples]` at 128 Hz
- ✅ float16 dtype
- ✅ gzip compression
- ✅ Channel names matching `channel_groups.json` (or added to it)

---

## 9. VALIDATION CHECKLIST

Before generating embeddings:

- [ ] All channel names in `channel_groups.json` OR mapped to existing names
- [ ] All subjects have at least 1 channel in each of BAS, RESP, EKG, EMG
- [ ] HDF5 files created with correct structure
- [ ] All channels at 128 Hz sampling rate
- [ ] All channels are 1D continuous arrays
- [ ] Data is z-score normalized (mean≈0, std≈1)
- [ ] No NaN or Inf values
- [ ] Test with 1-2 subjects before full conversion

---

## 10. SUMMARY

**Your situation:**
- ✅ You have all 4 required modalities
- ✅ Most channel names are compatible
- ⚠️ Need minor renaming for 5 channels
- ✅ Your data will work with SleepFM

**Action items:**
1. Update `channel_groups.json` OR rename during conversion
2. Filter subjects to those with all 4 modalities
3. Convert to HDF5 format
4. Validate with pilot subjects
5. Generate embeddings

**You're in good shape!** The conversion script will handle everything.
