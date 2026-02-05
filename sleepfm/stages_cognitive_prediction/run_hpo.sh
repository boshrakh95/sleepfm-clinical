#!/bin/bash
# ==============================================================================
# Random Hyperparameter Search for STAGES Cognitive Prediction
# ==============================================================================
# This script submits multiple SLURM jobs with randomly sampled hyperparameters
# 
# Usage:
#   bash run_hpo.sh
#
# Customize the search space below by modifying the arrays of possible values
# ==============================================================================

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Number of random configurations to try
NUM_TRIALS=10

# Base configuration file (will be overridden by command-line args)
CONFIG_FILE="config_finetune_cognitive.yaml"

# SLURM settings
PARTITION="gpu"
GPUS=1
CPUS=8
MEM="32G"
TIME="8:00:00"

# Output directory for logs
LOG_DIR="/home/boshra95/scratch/stages/sleepfm_format/hpo_logs"
mkdir -p "$LOG_DIR"

# ==============================================================================
# HYPERPARAMETER SEARCH SPACE
# ==============================================================================
# Define arrays of possible values for each hyperparameter
# Add or remove parameters as needed

# Learning rate options
LR_OPTIONS=(0.0001 0.00005 0.0005 0.00001)

# Batch size options
BATCH_SIZE_OPTIONS=(4 8 16)

# Number of epochs
EPOCHS_OPTIONS=(50 70 100)

# Weight decay (L2 regularization)
WEIGHT_DECAY_OPTIONS=(0.001 0.01 0.0001)

# Dropout rate (requires modifying config file manually for now)
# DROPOUT_OPTIONS=(0.1 0.2 0.3)

# Gradient clipping
# MAX_GRAD_NORM_OPTIONS=(1.0 2.0 5.0)

# Learning rate scheduler warmup epochs
# WARMUP_EPOCHS_OPTIONS=(0 2 5)

# Target task (if you want to search across tasks)
# TARGET_OPTIONS=("sustained_attention" "working_memory" "episodic_memory")

# Task type
# TASK_TYPE_OPTIONS=("classification" "regression")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Function to randomly select an element from an array
random_choice() {
    local array=("$@")
    local size=${#array[@]}
    local index=$((RANDOM % size))
    echo "${array[$index]}"
}

# Function to generate a unique experiment name
generate_exp_name() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local rand_id=$((RANDOM % 10000))
    echo "hpo_${timestamp}_${rand_id}"
}

# ==============================================================================
# MAIN LOOP: Submit jobs with random hyperparameters
# ==============================================================================

echo "=========================================="
echo "Starting Random Hyperparameter Search"
echo "=========================================="
echo "Number of trials: $NUM_TRIALS"
echo "Configuration file: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
echo ""

for trial in $(seq 1 $NUM_TRIALS); do
    echo "----------------------------------------"
    echo "Trial $trial/$NUM_TRIALS"
    echo "----------------------------------------"
    
    # Sample random hyperparameters
    LR=$(random_choice "${LR_OPTIONS[@]}")
    BATCH_SIZE=$(random_choice "${BATCH_SIZE_OPTIONS[@]}")
    EPOCHS=$(random_choice "${EPOCHS_OPTIONS[@]}")
    WEIGHT_DECAY=$(random_choice "${WEIGHT_DECAY_OPTIONS[@]}")
    
    # Uncomment to include these in search:
    # TARGET=$(random_choice "${TARGET_OPTIONS[@]}")
    # TASK_TYPE=$(random_choice "${TASK_TYPE_OPTIONS[@]}")
    # DROPOUT=$(random_choice "${DROPOUT_OPTIONS[@]}")
    # MAX_GRAD_NORM=$(random_choice "${MAX_GRAD_NORM_OPTIONS[@]}")
    # WARMUP_EPOCHS=$(random_choice "${WARMUP_EPOCHS_OPTIONS[@]}")
    
    # Generate unique experiment name
    EXP_NAME=$(generate_exp_name)
    
    # Print sampled configuration
    echo "Configuration:"
    echo "  LR: $LR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Weight decay: $WEIGHT_DECAY"
    echo "  Experiment name: $EXP_NAME"
    
    # Create SLURM job script
    JOB_SCRIPT="${LOG_DIR}/${EXP_NAME}_job.sh"
    
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${EXP_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${EXP_NAME}_%j.err

# Print job info
echo "========================================"
echo "SLURM Job: \${SLURM_JOB_ID}"
echo "Node: \${SLURM_NODELIST}"
echo "========================================"
echo "Hyperparameters:"
echo "  LR: ${LR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "========================================"
echo ""

# Activate environment
source /home/boshra95/sleepfm_env/bin/activate

# Change to working directory
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction

# Run training with sampled hyperparameters
python finetune_cognitive.py \\
    --config ${CONFIG_FILE} \\
    --lr ${LR} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS}

# Add more parameters as needed:
# --target \${TARGET} \\
# --task_type \${TASK_TYPE} \\

# Note: Some parameters like weight_decay, dropout, warmup_epochs
# need to be added as command-line arguments in finetune_cognitive.py
# if you want to override them

echo ""
echo "========================================"
echo "Job completed"
echo "========================================"
EOF
    
    # Make job script executable
    chmod +x "$JOB_SCRIPT"
    
    # Submit job
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "  Submitted job: $JOB_ID"
    echo "  Job script: $JOB_SCRIPT"
    echo ""
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Total trials: $NUM_TRIALS"
echo "Check logs in: $LOG_DIR"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs: scancel -u \$USER"
echo "=========================================="
