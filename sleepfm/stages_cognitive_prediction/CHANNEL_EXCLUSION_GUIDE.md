# Channel Exclusion Feature Guide

## Overview

The pipeline now supports excluding specific channels from embedding generation and fine-tuning. This is useful when you want to:

- Exclude channels with known quality issues (e.g., FLOW channel)
- Test the impact of specific channels on model performance
- Match channels used in other studies or publications
- Ablation studies on channel importance

## How It Works

### 1. Configuration

Add channel names to the `exclude_channels` list in your config file:

```yaml
data:
  exclude_channels: ["Flow"]  # Case-insensitive matching
```

**Important Notes:**
- Matching is **case-insensitive**: `"Flow"`, `"FLOW"`, and `"flow"` all match the same channel
- You only need to specify one variant of the channel name
- Empty list `[]` means no channels are excluded (default behavior)

### 2. Common Examples

**Exclude FLOW channel:**
```yaml
data:
  exclude_channels: ["Flow"]
```

**Exclude multiple channels:**
```yaml
data:
  exclude_channels: ["Flow", "CHIN", "EKG"]
```

**Exclude all EMG channels:**
```yaml
data:
  exclude_channels: ["CHIN", "RLEG", "LLEG"]
```

**No exclusions (default):**
```yaml
data:
  exclude_channels: []
```

## Implementation Details

### Where Filtering Happens

**Embedding Generation (`generate_embeddings.py`):**
- Channels are filtered when loading HDF5 files
- Excluded channels are completely skipped during data loading
- Embeddings are generated without excluded channels

**Dataset Classes (`dataset.py`):**
- Store the exclusion configuration
- Pass through to embedding generation (if using on-the-fly mode)
- No additional filtering needed for pre-computed embeddings

### Logging

When channels are excluded, the pipeline logs:

```
INFO: Excluding channels: ['Flow']
DEBUG: Excluding 1 channel(s) from SUBJECT001: ['Flow']
```

This helps you verify that exclusions are working correctly.

## Workflow

### Step 1: Update Configuration

Edit `config_finetune_cognitive.yaml`:

```yaml
data:
  exclude_channels: ["Flow"]  # Add channels to exclude
```

### Step 2: Generate Embeddings

**Important:** You must regenerate embeddings after changing excluded channels!

```bash
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

The exclusion happens during embedding generation. Each embedding is computed from the remaining (non-excluded) channels.

### Step 3: Fine-tune

Train your model using the new embeddings:

```bash
python finetune_cognitive.py --config config_finetune_cognitive.yaml
```

The model will use embeddings that were generated without the excluded channels.

## Important Considerations

### 1. Regenerating Embeddings

**You must regenerate embeddings if you:**
- Add or remove channels from `exclude_channels`
- Change which channels to exclude
- Switch between excluding and not excluding channels

**You do NOT need to regenerate if you:**
- Change other config parameters (learning rate, batch size, etc.)
- Change the cognitive target
- Change model architecture (LSTM layers, hidden dim, etc.)

### 2. Embedding Cache Location

Embeddings are saved to:
```
/home/boshra95/scratch/stages/sleepfm_format/embeddings/
```

To force regeneration:
```bash
# Option 1: Delete existing embeddings
rm -rf /home/boshra95/scratch/stages/sleepfm_format/embeddings/*.npy

# Option 2: Generate to new directory
# Update config: preprocessing.embeddings_dir: '/path/to/new/embeddings/'
```

### 3. Model Compatibility

If you exclude channels:
- The embedding dimension stays the same (128)
- The model architecture doesn't need to change
- Previous checkpoints trained WITH excluded channels won't be directly comparable
- For fair comparison, use consistent channel sets across experiments

## Use Cases

### Example 1: Exclude FLOW Channel

Many sleep studies have poor FLOW channel quality. To exclude it:

```yaml
data:
  exclude_channels: ["Flow"]
```

Then regenerate embeddings and retrain.

### Example 2: Ablation Study

Test impact of different channel groups:

**Experiment 1: All channels**
```yaml
data:
  exclude_channels: []
```

**Experiment 2: No FLOW**
```yaml
data:
  exclude_channels: ["Flow"]
```

**Experiment 3: No EMG**
```yaml
data:
  exclude_channels: ["CHIN", "RLEG", "LLEG"]
```

**Experiment 4: BAS and RESP only**
```yaml
data:
  exclude_channels: ["EKG", "CHIN", "RLEG", "LLEG"]
```

For each experiment:
1. Update config
2. Generate embeddings to separate directory
3. Train model
4. Compare results

### Example 3: Match Published Study

If a published study used only specific channels:

```yaml
data:
  # Exclude channels not used in reference study
  exclude_channels: ["Flow", "ABD", "RLEG", "LLEG"]
```

## Troubleshooting

### Issue: Embeddings not changing

**Problem:** You updated `exclude_channels` but results are the same.

**Solution:** Delete old embeddings and regenerate:
```bash
rm -rf /home/boshra95/scratch/stages/sleepfm_format/embeddings/*.npy
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

### Issue: Channel not being excluded

**Problem:** Channel still appears in data.

**Solutions:**
1. Check spelling in `exclude_channels` list
2. Verify channel name matches exactly (case-insensitive)
3. Check logs for "Excluding channels: [...]" message
4. Make sure you regenerated embeddings

### Issue: All channels excluded

**Problem:** No channels remain after exclusion.

**Solution:** Review your exclusion list - you may have accidentally excluded too many channels or used incorrect names.

## Technical Details

### Code Changes

**Files Modified:**
1. `config_finetune_cognitive.yaml` - Added `exclude_channels` parameter
2. `generate_embeddings.py` - Added filtering logic in `generate_embeddings_for_file()`
3. `dataset.py` - Store exclusion list for compatibility
4. `README.md` - Documentation update

### Filtering Logic

```python
# Create case-insensitive set
excluded_set = set([ch.lower() for ch in exclude_channels])

# Filter channels
for channel in available_channels:
    if channel.lower() in excluded_set:
        continue  # Skip excluded channel
    # ... process channel
```

### Performance Impact

- **Embedding generation:** Minimal overhead (just a set lookup per channel)
- **Training:** No impact (embeddings are pre-computed)
- **Storage:** No change (embedding dimension stays the same)

## Summary

The channel exclusion feature provides:
- ✅ Flexible channel selection
- ✅ Easy configuration via YAML
- ✅ Case-insensitive matching
- ✅ Logging for verification
- ✅ Compatible with existing pipeline
- ✅ No performance overhead during training

**Key Reminder:** Always regenerate embeddings after changing `exclude_channels`!
