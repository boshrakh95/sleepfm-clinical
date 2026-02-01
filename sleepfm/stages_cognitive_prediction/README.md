# STAGES Cognitive Prediction Fine-tuning Pipeline

This directory contains the complete pipeline for fine-tuning SleepFM on STAGES cognitive prediction tasks.

## Overview

The pipeline consists of:

1. **Configuration** (`config_finetune_cognitive.yaml`) - Comprehensive YAML config controlling all aspects
2. **Dataset** (`dataset.py`) - Custom dataset classes for cognitive prediction
3. **Models** (`models.py`) - LSTM-based models for regression/classification
4. **Embedding Generation** (`generate_embeddings.py`) - Pre-compute embeddings for faster training
5. **Fine-tuning** (`finetune_cognitive.py`) - Main training script

## Quick Start

### 1. Configure Your Experiment

Edit `config_finetune_cognitive.yaml`:

```yaml
# Choose your target
task:
  target: 'sustained_attention'  # or working_memory, episodic_memory, etc.
  task_type: 'regression'  # or 'classification'

# Choose pre-trained model
model:
  pretrained_model: 'base'  # or 'diagnosis'
  pretrained_checkpoint: '/path/to/checkpoint.pt'

# Model architecture
model:
  name: 'CognitiveLSTMWithDemo'  # Includes demographics
```

### 2. Generate Embeddings (Recommended)

Pre-compute embeddings for faster training:

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction

python generate_embeddings.py --config config_finetune_cognitive.yaml
```

This will:
- Load your pre-trained SetTransformer model
- Process all HDF5 files (train/val/test)
- Generate embeddings for each 5-min chunk
- Save to `/home/boshra95/scratch/stages/sleepfm_format/embeddings/`

**Note:** Embeddings are specific to the pretrained model. If you change the model, regenerate embeddings.

### 3. Fine-tune the Model

Run fine-tuning:

```bash
python finetune_cognitive.py --config config_finetune_cognitive.yaml
```

This will:
- Load cached embeddings
- Train LSTM model with demographics
- Validate on validation set
- Save checkpoints and best model
- Evaluate on test set
- Save predictions

## Configuration Options

### Task Configuration

```yaml
task:
  task_type: 'regression'  # or 'classification'
  target: 'sustained_attention'  # cognitive target to predict
  use_demographics: true  # include age/gender
  use_quality_filtering: true  # filter by artifact quality
  min_clean_ratio: 0.5  # minimum clean segments ratio
```

### Model Options

**Model Architectures:**
- `CognitiveRegressionLSTM` - LSTM for regression (sleep embeddings only)
- `CognitiveClassificationLSTM` - LSTM for classification (sleep embeddings only)
- `CognitiveLSTMWithDemo` - LSTM with demographics (recommended)
- `CognitiveEmbeddingLSTM` - Works with pre-computed embeddings (fastest)

**Pre-trained Models:**
- `base` - Base SetTransformer pretrained on multi-dataset PSG
- `diagnosis` - Diagnosis model fine-tuned for disease prediction

```yaml
model:
  name: 'CognitiveLSTMWithDemo'
  pretrained_model: 'base'
  pretrained_checkpoint: '/path/to/checkpoint.pt'
  
  params:
    embed_dim: 128  # Must match pretrained model
    lstm_hidden_dim: 128
    lstm_num_layers: 2
    lstm_bidirectional: true
    dropout: 0.3
```

### Training Configuration

```yaml
training:
  batch_size: 16
  epochs: 50
  lr: 0.0001
  optimizer: 'AdamW'
  weight_decay: 0.01
  
  # Learning rate scheduler
  scheduler: 'CosineAnnealingLR'
  
  # Mixed precision training
  use_amp: true
  
  # Gradient settings
  accumulation_steps: 4
  max_grad_norm: 1.0
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
  
  # Loss function
  loss_function: 'MSE'  # or 'MAE', 'Huber' for regression
                        # or 'BCE', 'CrossEntropy' for classification
```

### Evaluation Metrics

```yaml
evaluation:
  # Regression metrics
  metrics:
    - 'MSE'
    - 'MAE'
    - 'R2'
    - 'PearsonR'
    - 'SpearmanR'
  
  # Classification metrics (if task_type='classification')
  # metrics:
  #   - 'Accuracy'
  #   - 'F1'
  #   - 'AUC'
  #   - 'Precision'
  #   - 'Recall'
  
  primary_metric: 'R2'  # For model selection
  higher_is_better: true
```

## Pipeline Workflow

### Complete Workflow

```bash
# 1. Data preprocessing (already done)
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing
python convert_to_hdf5.py --config config_stages_conversion.yaml
python prepare_labels.py --config config_stages_conversion.yaml
python create_splits.py --config config_stages_conversion.yaml

# 2. Generate embeddings (once per pretrained model)
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction
python generate_embeddings.py --config config_finetune_cognitive.yaml

# 3. Fine-tune for each cognitive target
# Edit config to set target='sustained_attention'
python finetune_cognitive.py --config config_finetune_cognitive.yaml

# Edit config to set target='working_memory'
python finetune_cognitive.py --config config_finetune_cognitive.yaml

# ... repeat for all targets
```

### Training Multiple Targets

To train models for all 8 cognitive targets, you can use a shell script:

```bash
#!/bin/bash

TARGETS=("sustained_attention" "working_memory" "episodic_memory" "executive_functioning" "CPF_A.CPF_CR" "CPF_A.CPF_FP" "CPF_A.CPF_TPRT")

for TARGET in "${TARGETS[@]}"; do
    echo "Training model for $TARGET"
    
    # Update config
    sed -i "s/target: .*/target: '$TARGET'/" config_finetune_cognitive.yaml
    sed -i "s/split_path: .*/split_path: '\/home\/boshra95\/scratch\/stages\/sleepfm_format\/splits\/dataset_split_${TARGET}.json'/" config_finetune_cognitive.yaml
    
    # Run training
    python finetune_cognitive.py --config config_finetune_cognitive.yaml
done
```

## Output Structure

```
/home/boshra95/scratch/stages/sleepfm_format/
├── embeddings/                    # Pre-computed embeddings
│   ├── SUBJECT001.npy
│   ├── SUBJECT002.npy
│   └── ...
│
└── cognitive_models/              # Model outputs
    ├── stages_cognitive_20260201_143022/  # Experiment directory
    │   ├── config.yaml            # Saved configuration
    │   ├── training.log           # Training logs
    │   ├── best_model.pth         # Best model checkpoint
    │   ├── checkpoint_epoch_10.pth
    │   ├── checkpoint_epoch_20.pth
    │   ├── test_predictions.csv   # Predictions on test set
    │   └── test_metrics.json      # Test performance metrics
    │
    └── stages_cognitive_20260201_154531/  # Another experiment
        └── ...
```

## Model Checkpoints

Checkpoints contain:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - LR scheduler state
- `epoch` - Current epoch
- `metrics` - Validation metrics
- `config` - Full configuration

Load a checkpoint:

```python
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Advanced Features

### Quality-Aware Training

Use quality metadata to filter or weight embeddings:

```yaml
task:
  use_quality_filtering: true
  min_clean_ratio: 0.5  # Exclude subjects with <50% clean data
  quality_weighting: false  # Weight by quality (experimental)
```

### Freeze Encoder

Freeze the SetTransformer encoder and only train LSTM:

```yaml
training:
  freeze_encoder: true
  freeze_epochs: 0  # Or freeze for first N epochs
```

### Hyperparameter Tuning

Key hyperparameters to tune:
- `training.lr` (learning rate)
- `training.batch_size`
- `model.params.lstm_hidden_dim`
- `model.params.lstm_num_layers`
- `model.params.dropout`
- `training.weight_decay`

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
1. Reduce `batch_size`
2. Increase `accumulation_steps` to maintain effective batch size
3. Set `freeze_encoder: true` to save memory
4. Reduce `lstm_hidden_dim` or `lstm_num_layers`

### Poor Performance

If model performs poorly:
1. Check data quality (use quality filtering)
2. Try different loss functions (Huber instead of MSE)
3. Adjust learning rate
4. Try different model architecture
5. Include demographics (`use_demographics: true`)
6. Increase model capacity (more LSTM layers)

### Embeddings Take Too Long

Embedding generation takes ~1-2 hours for 1500 subjects:
- Use GPU (`device: 'cuda'`)
- Run on compute cluster with `salloc`
- Embeddings only need to be generated once per pretrained model

## Citation

If you use this pipeline, please cite:

```bibtex
@article{sleepfm2024,
  title={SleepFM: Multi-Modal Representation Learning for Sleep Staging},
  author={...},
  journal={...},
  year={2024}
}
```

## Support

For questions or issues:
1. Check the configuration YAML for typos
2. Review the training logs
3. Check that embeddings were generated correctly
4. Verify split files exist for your target
5. Ensure labels file contains your target column

## Notes

- **Per-target splits**: Each cognitive target has its own split file because different subjects have valid data for different targets
- **Quality metadata**: Tracks which 30-sec windows are clean vs artifact, used for filtering
- **Demographics**: Age and gender are normalized (age scaled, gender 0/1)
- **Embeddings**: One embedding per 5-min chunk (30 sec × 10 windows)
- **Aggregation**: Can aggregate multiple chunks for temporal modeling
