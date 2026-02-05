# Random Hyperparameter Search for STAGES Cognitive Prediction

This directory contains scripts for running random hyperparameter search without writing HPO code.

## Files

- **`run_hpo.sh`**: Main script to submit multiple SLURM jobs with random hyperparameters
- **`analyze_hpo_results.py`**: Analyze results and find best configuration
- **`finetune_cognitive.py`**: Training script (modified to accept command-line overrides)
- **`config_finetune_cognitive.yaml`**: Base configuration (will be overridden by HPO)

## Quick Start

### 1. Configure Search Space

Edit `run_hpo.sh` and modify the search space arrays:

```bash
# Example: Customize which hyperparameters to search
LR_OPTIONS=(0.0001 0.00005 0.0005)
BATCH_SIZE_OPTIONS=(4 8 16)
EPOCHS_OPTIONS=(50 70 100)
WEIGHT_DECAY_OPTIONS=(0.001 0.01 0.0001)
```

Uncomment additional parameters you want to include:

```bash
# Uncomment to search over dropout
DROPOUT_OPTIONS=(0.1 0.2 0.3)

# Uncomment to search over gradient clipping
MAX_GRAD_NORM_OPTIONS=(1.0 2.0 5.0)

# Uncomment to search over warmup epochs
WARMUP_EPOCHS_OPTIONS=(0 2 5)

# Uncomment to search across tasks
TARGET_OPTIONS=("sustained_attention" "working_memory" "episodic_memory")
```

### 2. Set Number of Trials

In `run_hpo.sh`:

```bash
NUM_TRIALS=20  # Number of random configurations to try
```

### 3. Configure SLURM Resources

In `run_hpo.sh`:

```bash
PARTITION="gpu"
GPUS=1
CPUS=8
MEM="32G"
TIME="8:00:00"
```

### 4. Run HPO

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction
bash run_hpo.sh
```

The script will:
1. Randomly sample hyperparameters from your search space
2. Generate a SLURM job script for each trial
3. Submit all jobs to the cluster
4. Save logs to `/home/boshra95/scratch/stages/sleepfm_format/hpo_logs/`

### 5. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check a specific job's output (while running)
tail -f /home/boshra95/scratch/stages/sleepfm_format/hpo_logs/hpo_*_JOBID.out

# Cancel all jobs if needed
scancel -u $USER
```

### 6. Analyze Results

After jobs complete:

```bash
python analyze_hpo_results.py \
    --results_dir /home/boshra95/scratch/stages/sleepfm_format/cognitive_models \
    --metric val_Accuracy \
    --output hpo_results.csv
```

This will:
- Extract results from all completed trials
- Find the best configuration
- Save a summary CSV with all results
- Print top 5 configurations

Example output:
```
================================================================================
BEST CONFIGURATION
================================================================================
Experiment: hpo_20260205_143021_1234
Best val_Accuracy: 0.8542

Hyperparameters:
  learning rate: 0.0001
  batch_size: 8
  epochs: 70
  weight_decay: 0.001

Performance:
  train_Accuracy: 0.8621
  train_F1: 0.8543
  val_Accuracy: 0.8542
  val_F1: 0.8421
  test_Accuracy: 0.8312
  test_F1: 0.8201
```

## Available Hyperparameters

### Already Implemented (can override from command line):

- `--lr`: Learning rate
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--weight_decay`: L2 regularization strength
- `--warmup_epochs`: Number of warmup epochs
- `--dropout`: Dropout rate
- `--max_grad_norm`: Gradient clipping threshold
- `--target`: Cognitive target task
- `--task_type`: Classification or regression

### To Add More Parameters:

1. **Edit `run_hpo.sh`**: Add the parameter to the job script command
2. **Edit `finetune_cognitive.py`**: Add the argument parser if not already there

## Example: Custom Search Space

Search over learning rate, batch size, and dropout:

```bash
# In run_hpo.sh

# Define search space
LR_OPTIONS=(0.0001 0.00005 0.0005 0.00001)
BATCH_SIZE_OPTIONS=(4 8 16)
DROPOUT_OPTIONS=(0.1 0.2 0.3 0.4)

# Sample hyperparameters
LR=$(random_choice "${LR_OPTIONS[@]}")
BATCH_SIZE=$(random_choice "${BATCH_SIZE_OPTIONS[@]}")
DROPOUT=$(random_choice "${DROPOUT_OPTIONS[@]}")

# In the job script, add:
python finetune_cognitive.py \
    --config ${CONFIG_FILE} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --dropout ${DROPOUT} \
    --epochs 50
```

## Tips

1. **Start Small**: Try 5-10 trials first to make sure everything works
2. **Fixed vs Random**: Keep some parameters fixed (like epochs) while searching others
3. **Resource Management**: Adjust `TIME` based on your dataset size and epochs
4. **Log Organization**: All logs go to `hpo_logs/` - review them if jobs fail
5. **Best Practices**: 
   - Use validation metric for selection (e.g., `val_Accuracy`)
   - Report test performance only for final best model
   - Save all predictions with `save_predictions: true` in config

## Troubleshooting

### Jobs not starting?
```bash
# Check SLURM queue
squeue -u $USER

# Check if partition exists
sinfo

# Check job reason
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### Jobs failing?
```bash
# Check error logs
cat /home/boshra95/scratch/stages/sleepfm_format/hpo_logs/hpo_*_JOBID.err

# Check output logs
less /home/boshra95/scratch/stages/sleepfm_format/hpo_logs/hpo_*_JOBID.out
```

### No results found?
- Make sure `save_predictions: true` in config
- Check that jobs completed successfully
- Verify output directory matches `--results_dir` in analysis script

## Advanced: Grid Search

To do grid search instead of random search, modify `run_hpo.sh`:

```bash
# Instead of random sampling, iterate through all combinations
for LR in "${LR_OPTIONS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_OPTIONS[@]}"; do
        for WEIGHT_DECAY in "${WEIGHT_DECAY_OPTIONS[@]}"; do
            # Submit job with this configuration
            ...
        done
    done
done
```

Note: Grid search can create many jobs! With 4 LR × 3 batch_size × 3 weight_decay = 36 trials.
