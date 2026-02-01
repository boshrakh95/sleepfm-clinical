# Quick Start Guide

## TL;DR

1. **Always use the same model:** `CognitiveEmbeddingLSTM`
2. **Toggle task type** via config: `task_type: 'regression'` or `'classification'`
3. **Toggle demographics** via config: `use_demographics: true` or `false`
4. **Never change model name** - it adapts automatically!

## Complete Workflow

### Step 1: Generate Embeddings (ONCE)

```bash
cd /home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

⏱️ Takes ~1-2 hours. Only run once (or when changing excluded channels).

### Step 2: Configure Your Task

Edit `config_finetune_cognitive.yaml`:

```yaml
task:
  task_type: 'classification'  # or 'regression'
  target: 'sustained_attention'  # your cognitive target
  use_demographics: true  # or false

model:
  name: 'CognitiveEmbeddingLSTM'  # ALWAYS use this

  params:
    num_classes: 2  # 2 for classification, 1 for regression

preprocessing:
  generate_embeddings: false  # MUST be false
```

### Step 3: Train

```bash
python finetune_cognitive.py --config config_finetune_cognitive.yaml
```

⏱️ Takes ~10-30 minutes depending on data size.

## All Supported Combinations

The **same model** (`CognitiveEmbeddingLSTM`) works for all these:

| Task Type | Demographics | Config | num_classes |
|-----------|--------------|--------|-------------|
| Regression | Yes | `task_type: 'regression'`<br>`use_demographics: true` | 1 |
| Regression | No | `task_type: 'regression'`<br>`use_demographics: false` | 1 |
| Classification | Yes | `task_type: 'classification'`<br>`use_demographics: true` | 2 |
| Classification | No | `task_type: 'classification'`<br>`use_demographics: false` | 2 |

**Model name is ALWAYS:** `CognitiveEmbeddingLSTM`

## Common Workflows

### Run Multiple Targets

```bash
# Edit config for each target and run
for target in sustained_attention working_memory episodic_memory; do
    # Update config
    sed -i "s/target: .*/target: '$target'/" config_finetune_cognitive.yaml
    sed -i "s|split_path: .*|split_path: '/home/boshra95/scratch/stages/sleepfm_format/splits/dataset_split_${target}.json'|" config_finetune_cognitive.yaml
    
    # Train
    python finetune_cognitive.py --config config_finetune_cognitive.yaml
done
```

### Switch from Classification to Regression

Just edit config:

```yaml
# Change this
task:
  task_type: 'regression'  # was 'classification'

# And this
model:
  params:
    num_classes: 1  # was 2
```

No need to regenerate embeddings! Just retrain.

### Test With/Without Demographics

```yaml
# Try 1: With demographics
task:
  use_demographics: true
  
# Try 2: Without demographics  
task:
  use_demographics: false
```

Same model name, different performance. Compare results!

## Key Files & Locations

```
Embeddings:     /home/boshra95/scratch/stages/sleepfm_format/embeddings/
Labels:         /home/boshra95/scratch/stages/sleepfm_format/labels/
Splits:         /home/boshra95/scratch/stages/sleepfm_format/splits/
Models output:  /home/boshra95/scratch/stages/sleepfm_format/cognitive_models/
```

## Troubleshooting

**Q: Do I need different models for regression vs classification?**  
A: No! Always use `CognitiveEmbeddingLSTM`. It adapts automatically.

**Q: What about demographics on/off?**  
A: Same answer. One model, controlled by config.

**Q: When do I regenerate embeddings?**  
A: Only when you:
- Change pretrained model (base ↔ diagnosis)
- Change excluded channels
- Change modalities

**Q: My config says `num_classes: 2` but I want regression**  
A: Change to `num_classes: 1` and `task_type: 'regression'`

**Q: Can I use the shell script?**  
A: Yes! `./run_cognitive_finetuning.sh` trains all targets automatically.

## That's It!

One model (`CognitiveEmbeddingLSTM`), many use cases. Configure via YAML, never change model name.
