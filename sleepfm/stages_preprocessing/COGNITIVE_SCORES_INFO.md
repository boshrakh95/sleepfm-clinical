# STAGES Cognitive Scores - Data Structure

## Summary

Cognitive scores are available in **two formats**:
- **Regression** (continuous z-scores): in `cognitive_targets.csv`
- **Classification** (binary 0/1): in `individual_scores/processed_*_classification.csv`

## File Locations

### Regression Labels (Continuous Scores)
- **`cognitive_targets.csv`**
  - Location: `/home/boshra95/scratch/stages/stages/processed/`
  - Subjects: 1,531
  - Format: Z-scores (normalized, mean≈0, std≈1)
  - Contains: Raw test scores + 8 composite scores

### Classification Labels (Binary)
- **`individual_scores/processed_*_classification.csv`**
  - Location: `/home/boshra95/scratch/stages/stages/processed/individual_scores/`
  - Format: Binary (0.0 = low, 1.0 = high)
  - Balanced: ~50/50 split for each score
  - Example: `processed_sustained_attention_classification.csv`

## Configuration

Set `task_type` in [config_stages_conversion.yaml](config_stages_conversion.yaml):

```yaml
labels:
  task_type: classification  # Options: 'regression', 'classification', 'both'
```

### Options:
- **`regression`**: Use continuous z-scores from `cognitive_targets.csv`
- **`classification`**: Use binary labels from `individual_scores/` (recommended for classification tasks)
- **`both`**: Include both types (regression + classification with `_class` suffix)

## Available Composite Scores

The following 8 composite scores are in `cognitive_targets.csv`:

1. **`sustained_attention`** (1,443 subjects with data)
   - Measures ability to maintain focus
   
2. **`working_memory`** (1,317 subjects)
   - Short-term memory and manipulation
   
3. **`episodic_memory`** (1,461 subjects)
   - Long-term memory for events
   
4. **`executive_functioning`** (1,420 subjects)
   - High-level cognitive control
   
5. **`nback_working_memory_capacity`** (1,317 subjects)
   - N-back task: memory capacity
   
6. **`nback_speed`** (1,301 subjects)
   - N-back task: response speed
   
7. **`nback_accuracy`** (1,369 subjects)
   - N-back task: correctness
   
8. **`nback_impulsivity`** (1,369 subjects)
   - N-back task: impulsive responses

## Classification Label Distribution

All classification scores are balanced (~50/50):

**Example (sustained_attention):**
- Class 0 (low): 721 subjects (50.0%)
- Class 1 (high): 722 subjects (50.0%)

All 8 scores follow similar balanced distribution.

## Data Completeness

- **Total subjects in dataset:** 1,531
- **Most complete score:** episodic_memory (1,461 = 95%)
- **Least complete score:** nback_speed (1,301 = 85%)

### Regression Scores
- Already **normalized** (z-scores with mean ≈ 0, std ≈ 1)
- Source: `cognitive_targets.csv`

### Classification Scores
- Binary labels: **0** (below median) or **1** (above median)
- Balanced: ~50/50 split
- Source: `individual_scores/processed_*_classification.csv`

## Updated Configuration

The config now supports both task types:

```yaml
labels:
  # Set to 'classification' for binary classification tasks
  task_type: classification
  
  cognitive_scores:
    - sustained_attention
    - working_memory
    - episodic_memory
    - executive_functioning
    # ... etc
```

## Notes

- Scores were processed by `cognitive_targets.py` in CogPSGFormerPP
- Individual score files in `individual_scores/` are reference only
- The pipeline will use `cognitive_targets.csv` directly
- Some subjects have missing scores for certain tests (handled by pipeline)
