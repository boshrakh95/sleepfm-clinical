# Model Architecture Clarification

## The Confusion

You asked great questions about the difference between models and the `generate_embeddings` setting. Here's the clear explanation:

## Two-Stage Pipeline

The pipeline actually works in **TWO separate stages**:

### Stage 1: Embedding Generation (ONCE)
**Script:** `generate_embeddings.py`
**Purpose:** Convert raw PSG → embeddings using pretrained SetTransformer
**Input:** Raw HDF5 PSG files
**Output:** Cached .npy embedding files

```bash
# Run this ONCE to create embeddings
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

This uses:
- `pretrained_model` (base or diagnosis)
- `pretrained_checkpoint` path
- SetTransformer encoder
- Saves embeddings to `embeddings_dir`

### Stage 2: Fine-tuning (MANY TIMES)
**Script:** `finetune_cognitive.py`
**Purpose:** Train LSTM model for cognitive prediction
**Input:** Pre-computed embeddings from Stage 1
**Output:** Trained cognitive prediction model

```bash
# Run this to train (can run many times with different hyperparameters)
python finetune_cognitive.py --config config_finetune_cognitive.yaml
```

This uses:
- Pre-computed embeddings (loads .npy files)
- LSTM model (`CognitiveEmbeddingLSTM`)
- Does NOT use SetTransformer
- Does NOT load pretrained checkpoint

## The Models Explained

### CognitiveEmbeddingLSTM ✅ (THE ONLY MODEL YOU NEED)

**Status:** ✅ Fully implemented and working for ALL use cases

**What it does:**
1. Loads pre-computed embeddings from .npy files
2. LSTM processes the sequence of embeddings
3. Attention pooling over time
4. Optional demographics (age, gender) - controlled by `task.use_demographics`
5. Output layer - adapts based on `task.task_type`

**Automatically adapts to:**
- **Regression** (`task_type: 'regression'`): Single output value
- **Classification** (`task_type: 'classification'`): Binary logits (2 classes)
- **With demographics** (`use_demographics: true`): Includes demographics embedding
- **Without demographics** (`use_demographics: false`): Sleep embeddings only

**Forward pass:**
```python
embeddings [batch, seq_len, 128]
    ↓
LSTM
    ↓
Attention Pooling → sleep_features
    ↓
if use_demographics:
    demographics → demo_embedding
    concat(sleep_features, demo_embedding)
    ↓
Output layer (adapts to regression/classification)
    ↓
predictions (1 output for regression, 2 for classification)
```

**Config examples:**

```yaml
# Regression with demographics
task:
  task_type: 'regression'
  use_demographics: true
model:
  name: 'CognitiveEmbeddingLSTM'  # Same name!
  params:
    num_classes: 1

# Classification without demographics
task:
  task_type: 'classification'
  use_demographics: false
model:
  name: 'CognitiveEmbeddingLSTM'  # Same name!
  params:
    num_classes: 2
```

**Key Point: ALWAYS use `CognitiveEmbeddingLSTM` - it adapts automatically!**

## The generate_embeddings Setting Explained

```yaml
preprocessing:
  generate_embeddings: false  # This is REQUIRED
```

This setting is confusing because:

**`generate_embeddings: false`** (CORRECT) means:
- Use pre-computed cached embeddings
- Load .npy files from `embeddings_dir`
- Fast training (no SetTransformer computation)
- ✅ **THIS IS WHAT YOU WANT**

**`generate_embeddings: true`** (INCORRECT) would mean:
- Generate embeddings on-the-fly during training
- Run SetTransformer every batch
- ❌ **NOT IMPLEMENTED - WILL CRASH**

## Correct Workflow

### Step 1: Generate Embeddings (ONCE)

```yaml
# In config_finetune_cognitive.yaml
model:
  pretrained_model: 'base'
  pretrained_checkpoint: '/path/to/checkpoint.pt'
  
preprocessing:
  embeddings_dir: '/path/to/output/embeddings/'
```

```bash
python generate_embeddings.py --config config_finetune_cognitive.yaml
```

This creates:
```
/path/to/output/embeddings/
├── SUBJECT001.npy  # [num_chunks, 128]
├── SUBJECT002.npy
└── ...
```

### Step 2: Fine-tune Model (AS MANY TIMES AS NEEDED)

```yaml
# In config_finetune_cognitive.yaml
model:
  name: 'CognitiveEmbeddingLSTM'  # The only working model
  
task:
  task_type: 'classification'  # or 'regression'
  use_demographics: true  # optional
  
preprocessing:
  generate_embeddings: false  # MUST BE FALSE
  embeddings_dir: '/path/to/output/embeddings/'  # From step 1
```

```bash
python finetune_cognitive.py --config config_finetune_cognitive.yaml
```

You can run Step 2 many times with different:
- Hyperparameters (learning rate, batch size, etc.)
- LSTM architecture (num layers, hidden dim)
- Targets (different cognitive scores)
- Loss functions

WITHOUT regenerating embeddings!

## Why This Design?

### Advantages of Pre-computed Embeddings:

1. **Speed:** SetTransformer only runs once, not every epoch
2. **Flexibility:** Try different LSTM architectures quickly
3. **GPU memory:** Smaller models during fine-tuning
4. **Reproducibility:** Same embeddings across experiments

### When to Regenerate Embeddings:

You MUST regenerate embeddings if you change:
- ✅ Pretrained model (base → diagnosis)
- ✅ Pretrained checkpoint
- ✅ Excluded channels (`exclude_channels`)
- ✅ Modalities (`modality_types`)
- ✅ Chunk size or sampling parameters

You do NOT need to regenerate if you change:
- ❌ LSTM parameters (layers, hidden dim, dropout)
- ❌ Learning rate, batch size, optimizer
- ❌ Cognitive target
- ❌ Demographics usage
- ❌ Loss function

## Summary

**What you should do:**

1. Run `generate_embeddings.py` **ONCE** to create embeddings
2. Use `CognitiveEmbeddingLSTM` model
3. Set `generate_embeddings: false` in config
4. Run `finetune_cognitive.py` as many times as needed

**Key takeaway:** The pretrained SetTransformer is ONLY used in Stage 1 (embedding generation), not during fine-tuning!
