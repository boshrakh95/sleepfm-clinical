# Channel Exclusion Guide

## How to Exclude Channels from Analysis

You can now exclude specific channels (e.g., FLOW channels) from the embedding generation by adding them to the `exclude_channels` list in your config file.

## Example: Exclude FLOW Channels

To exclude FLOW-related channels from your analysis, edit your config file:

```yaml
preprocessing:
  embeddings_dir: '/path/to/embeddings'
  save_granular_embeddings: false
  exclude_channels: 
    - 'AIRFLOW'
    - 'AIRFLOW-0'
    - 'AIRFLOW-1'
    - 'Airflow'
    - 'NEW AIR'
    - 'NEWAIR'
    - 'New A'
    - 'New AIR'
    - 'New Air'
    - 'new air'
```

## Common Channel Exclusions

### Exclude all FLOW variants:
```yaml
exclude_channels: ['AIRFLOW', 'AIRFLOW-0', 'AIRFLOW-1', 'Airflow', 'NEW AIR', 'NEWAIR', 'New A', 'New AIR', 'New Air', 'new air']
```

### Exclude specific respiratory channels:
```yaml
exclude_channels: ['Nasal', 'NASAL', 'Therm', 'Oral Therm']
```

### Exclude SpO2:
```yaml
exclude_channels: ['SpO2', 'SaO2', 'tcSpO2', 'SPO2', 'SAO2']
```

## Channel Groups Reference

Channels are grouped into 4 modalities:
- **BAS**: EOG, EEG channels
- **RESP**: Chest, Abdomen, SpO2, Nasal, Airflow, Snore, etc.
- **EKG**: ECG channels
- **EMG**: Leg, Chin EMG channels

See `/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json` for the complete list.

## How It Works

1. The code reads `exclude_channels` from your config
2. Before creating the dataset, it filters out excluded channels from all modality groups
3. Only the remaining channels are used for embedding generation
4. The pretrained model processes only the non-excluded channels

## Note on Data Loading Efficiency

The current implementation already efficiently loads data:
- HDF5 files use chunk caching (300 MB cache per file)
- Data is read in 5-minute chunks on-demand
- No need to load entire subject data into memory
- This approach balances memory usage and I/O efficiency

For your use case (excluding FLOW channels), no data loading optimization is needed beyond the current implementation.
