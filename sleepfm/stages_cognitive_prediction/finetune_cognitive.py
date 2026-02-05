#!/usr/bin/env python3
"""
STAGES Cognitive Prediction Fine-tuning
========================================

Fine-tune SleepFM for cognitive prediction tasks on STAGES dataset.

This script:
1. Loads pre-trained SleepFM (base or diagnosis model)
2. Creates cognitive prediction model with LSTM head
3. Loads STAGES data with cognitive labels and demographics
4. Trains/validates with quality-aware filtering
5. Evaluates on test set
6. Saves predictions and model checkpoints

Usage:
    python finetune_cognitive.py --config config_finetune_cognitive.yaml

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger
import random
from typing import Dict, List, Tuple, Optional

# Add sleepfm directory to path for absolute imports
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.utils import load_config, load_data, save_data, count_parameters
from sleepfm.models.models import SetTransformer
from sleepfm.stages_cognitive_prediction.models import (
    CognitivePredictionModel,
    create_cognitive_model
)
from sleepfm.stages_cognitive_prediction.dataset import (
    CognitivePredictionDataset,
    cognitive_collate_fn
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained_model(config: Dict, device: torch.device) -> SetTransformer:
    """Load pre-trained SetTransformer or diagnosis model.
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Pretrained model
    """
    pretrained_type = config['model']['pretrained_model']
    checkpoint_path = config['model']['pretrained_checkpoint']
    
    logger.info(f"Loading pretrained {pretrained_type} model from {checkpoint_path}")
    
    if pretrained_type == 'base':
        # Load base SetTransformer model
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Try to load config from checkpoint directory
        checkpoint_dir = Path(checkpoint_path).parent
        config_file = checkpoint_dir / "config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                checkpoint_config = json.load(f)
            
            # Use checkpoint config for model architecture
            num_layers = checkpoint_config.get('num_layers', 6)
            embed_dim = checkpoint_config.get('embed_dim', 128)
            num_heads = checkpoint_config.get('num_heads', 8)
            pooling_head = checkpoint_config.get('pooling_head', 8)
            dropout = checkpoint_config.get('dropout', 0.3)
            
            logger.info(f"Using checkpoint config: num_layers={num_layers}, embed_dim={embed_dim}")
        else:
            # Fallback to config file
            logger.warning(f"Config file not found at {config_file}, using config from YAML")
            model_config = config['model']['params']
            num_layers = model_config.get('num_layers', 6)
            embed_dim = model_config['embed_dim']
            num_heads = model_config['num_heads']
            pooling_head = model_config['pooling_head']
            dropout = model_config['dropout']
        
        model = SetTransformer(
            in_channels=4,  # Number of modalities
            patch_size=config['data']['chunk_size'],
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pooling_head=pooling_head,
            dropout=dropout,
            max_seq_length=config['model']['params']['max_seq_length']
        )
        
        # Extract state dict and handle DataParallel 'module.' prefix
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel training)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from checkpoint keys")
        
        model.load_state_dict(state_dict)
        logger.info(f"Loaded base SetTransformer model with {num_layers} layers")
        
    elif pretrained_type == 'diagnosis':
        # Load diagnosis fine-tuned model
        # Extract the SetTransformer encoder from diagnosis model
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create SetTransformer
        model_config = config['model']['params']
        model = SetTransformer(
            in_channels=4,
            patch_size=config['data']['chunk_size'],
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            pooling_head=model_config['pooling_head'],
            dropout=model_config['dropout'],
            max_seq_length=model_config['max_seq_length']
        )
        
        # Extract SetTransformer weights from diagnosis model
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from checkpoint keys")
        
        # Filter state dict to only SetTransformer components
        transformer_state = {}
        for key, value in state_dict.items():
            if key.startswith('set_transformer.') or key.startswith('encoder.'):
                new_key = key.replace('set_transformer.', '').replace('encoder.', '')
                transformer_state[new_key] = value
        
        if transformer_state:
            model.load_state_dict(transformer_state, strict=False)
            logger.info("Loaded diagnosis model encoder")
        else:
            logger.warning("Could not extract SetTransformer from diagnosis model, using random init")
    
    else:
        raise ValueError(f"Unknown pretrained_model: {pretrained_type}")
    
    model = model.to(device)
    model.eval()  # Set to eval mode for embedding generation
    
    return model


def create_model(
    config: Dict,
    device: torch.device
) -> nn.Module:
    """Create cognitive prediction model using new architecture.
    
    Args:
        config: Configuration dictionary
        device: Device to create model on
    
    Returns:
        Cognitive prediction model
    """
    logger.info(f"Creating model: {config['model']['name']}")
    
    # Use the factory function from models.py
    model = create_cognitive_model(config)
    model = model.to(device)
    
    # Apply DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    
    # Count parameters
    total_layers, total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters:")
    logger.info(f"  Total layers: {total_layers}")
    logger.info(f"  Total params: {total_params / 1e6:.2f} million")
    logger.info(f"  Trainable: {trainable_params / 1e6:.2f} million")
    
    return model


def get_loss_function(config: Dict) -> nn.Module:
    """Get loss function based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Loss function
    """
    task_type = config['task']['task_type']
    loss_name = config['training']['loss_function']
    
    if task_type == 'regression':
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE' or loss_name == 'L1':
            return nn.L1Loss()
        elif loss_name == 'Huber' or loss_name == 'SmoothL1':
            delta = config['training'].get('huber_delta', 1.0)
            return nn.SmoothL1Loss(beta=delta)
        else:
            raise ValueError(f"Unknown regression loss: {loss_name}")
    
    else:  # classification
        if loss_name == 'BCE':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'CrossEntropy' or loss_name == 'CE':
            return nn.CrossEntropyLoss()
        elif loss_name == 'FocalLoss':
            # Implement focal loss
            alpha = config['training'].get('focal_alpha', 0.25)
            gamma = config['training'].get('focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown classification loss: {loss_name}")


class FocalLoss(nn.Module):
    """Focal Loss for classification."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task_type: str,
    metrics_list: List[str]
) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        task_type: 'regression' or 'classification'
        metrics_list: List of metrics to compute
    
    Returns:
        Dictionary of metric values
    """
    from scipy import stats
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    )
    
    results = {}
    
    if task_type == 'regression':
        if 'MSE' in metrics_list:
            results['MSE'] = mean_squared_error(targets, predictions)
        if 'RMSE' in metrics_list:
            results['RMSE'] = np.sqrt(mean_squared_error(targets, predictions))
        if 'MAE' in metrics_list:
            results['MAE'] = mean_absolute_error(targets, predictions)
        if 'R2' in metrics_list:
            results['R2'] = r2_score(targets, predictions)
        if 'PearsonR' in metrics_list:
            r, p = stats.pearsonr(predictions, targets)
            results['PearsonR'] = r
            results['PearsonR_pval'] = p
        if 'SpearmanR' in metrics_list:
            r, p = stats.spearmanr(predictions, targets)
            results['SpearmanR'] = r
            results['SpearmanR_pval'] = p
    
    else:  # classification
        # Convert logits to predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = predictions.argmax(axis=1)
            pred_probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()[:, 1]
        else:
            pred_probs = torch.sigmoid(torch.from_numpy(predictions.flatten())).numpy()
            pred_classes = (pred_probs > 0.5).astype(int)
        
        if 'Accuracy' in metrics_list:
            results['Accuracy'] = accuracy_score(targets, pred_classes)
        if 'F1' in metrics_list:
            results['F1'] = f1_score(targets, pred_classes, average='binary')
        if 'AUC' in metrics_list:
            results['AUC'] = roc_auc_score(targets, pred_probs)
        if 'Precision' in metrics_list:
            results['Precision'] = precision_score(targets, pred_classes, average='binary')
        if 'Recall' in metrics_list:
            results['Recall'] = recall_score(targets, pred_classes, average='binary')
    
    return results


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        config: Configuration
        scaler: Gradient scaler for mixed precision
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    accumulation_steps = config['training']['accumulation_steps']
    use_amp = config['training']['use_amp']
    log_interval = config['logging']['log_interval']
    use_demographics = config['task']['use_demographics']
    task_type = config['task']['task_type']
    loss_name = config['training']['loss_function']
    metrics_list = config['evaluation']['metrics']
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch: (x_data, labels, demographics, masks, subject_ids)
        x_data, labels, demographics, masks, subject_ids = batch
        
        # Move to device
        x_data = x_data.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        if demographics is not None:
            demographics = demographics.to(device)
        
        # Forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(x_data, masks, demographics)
                
                # Handle output/label shape based on task and loss
                if task_type == 'regression':
                    outputs = outputs.squeeze(-1)  # [B, 1] -> [B]
                    labels_processed = labels
                elif task_type == 'classification':
                    # For BCE loss, keep outputs as [B, 1] and reshape labels to [B, 1]
                    if loss_name in ['BCE', 'BCEWithLogitsLoss', 'FocalLoss']:
                        labels_processed = labels.unsqueeze(-1).float()  # [B] -> [B, 1]
                    else:
                        # For CrossEntropy, squeeze outputs to [B] for multi-class
                        outputs = outputs.squeeze(-1)
                        labels_processed = labels
                else:
                    labels_processed = labels
                
                loss = criterion(outputs, labels_processed)
                loss = loss / accumulation_steps
        else:
            outputs = model(x_data, masks, demographics)
            
            # Handle output/label shape based on task and loss
            if task_type == 'regression':
                outputs = outputs.squeeze(-1)  # [B, 1] -> [B]
                labels_processed = labels
            elif task_type == 'classification':
                # For BCE loss, keep outputs as [B, 1] and reshape labels to [B, 1]
                if loss_name in ['BCE', 'BCEWithLogitsLoss', 'FocalLoss']:
                    labels_processed = labels.unsqueeze(-1).float()  # [B] -> [B, 1]
                else:
                    # For CrossEntropy, squeeze outputs to [B] for multi-class
                    outputs = outputs.squeeze(-1)
                    labels_processed = labels
            else:
                labels_processed = labels
            
            loss = criterion(outputs, labels_processed)
            loss = loss / accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Collect predictions and targets for metrics (detach to avoid memory issues)
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    avg_loss = total_loss / num_batches
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, task_type, metrics_list)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    split: str = 'val'
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        config: Configuration
        split: Split name for logging
    
    Returns:
        metrics: Dictionary of metrics
        predictions: Array of predictions
        targets: Array of ground truth labels
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    use_demographics = config['task']['use_demographics']
    task_type = config['task']['task_type']
    loss_name = config['training']['loss_function']
    metrics_list = config['evaluation']['metrics']
    
    pbar = tqdm(dataloader, desc=f"[{split.upper()}]")
    
    with torch.no_grad():
        for batch in pbar:
            # Unpack batch: (x_data, labels, demographics, masks, subject_ids)
            x_data, labels, demographics, masks, subject_ids = batch
            
            # Move to device
            x_data = x_data.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            if demographics is not None:
                demographics = demographics.to(device)
            
            # Forward pass
            outputs = model(x_data, masks, demographics)
            
            # Handle output/label shape based on task and loss
            if task_type == 'regression':
                outputs = outputs.squeeze(-1)  # [B, 1] -> [B]
                labels_processed = labels
            elif task_type == 'classification':
                # For BCE loss, keep outputs as [B, 1] and reshape labels to [B, 1]
                if loss_name in ['BCE', 'BCEWithLogitsLoss', 'FocalLoss']:
                    labels_processed = labels.unsqueeze(-1).float()  # [B] -> [B, 1]
                else:
                    # For CrossEntropy, squeeze outputs to [B] for multi-class
                    outputs = outputs.squeeze(-1)
                    labels_processed = labels
            else:
                labels_processed = labels
            
            loss = criterion(outputs, labels_processed)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, task_type, metrics_list)
    metrics['loss'] = avg_loss
    
    return metrics, predictions, targets


def save_predictions_to_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    subject_ids: List[str],
    task_type: str,
    output_path: Path,
    split: str = 'test'
):
    """Save predictions to CSV file.
    
    Args:
        predictions: Model predictions (logits for classification, values for regression)
        targets: Ground truth labels/values
        subject_ids: List of subject IDs
        task_type: 'classification' or 'regression'
        output_path: Path to save CSV file
        split: Split name (train/val/test) for filename
    """
    # Flatten predictions if needed
    if len(predictions.shape) > 1:
        pred_values = predictions[:, 0]
    else:
        pred_values = predictions
    
    # Create dataframe
    df_dict = {
        'subject_id': subject_ids,
        'true_label': targets,
    }
    
    if task_type == 'classification':
        # For classification: convert logits to probabilities using sigmoid
        pred_probs = torch.sigmoid(torch.from_numpy(pred_values)).numpy()
        df_dict['predicted_probability'] = pred_probs
        # Convert probability to class (threshold 0.5 for binary)
        df_dict['predicted_class'] = (pred_probs > 0.5).astype(int)
    else:
        # For regression: just save predicted value
        df_dict['predicted_value'] = pred_values
    
    predictions_df = pd.DataFrame(df_dict)
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Saved {split} predictions to: {output_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    is_best: bool = False,
    checkpoint_dir: Path = None
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        metrics: Validation metrics
        config: Configuration
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoint
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path(config['logging']['output_dir'])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint
    if not config['logging']['save_best_only']:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model: {best_path}")


def main(config):
    """Main training function.
    
    Args:
        config: Configuration dictionary (either loaded from YAML or passed directly)
    """
    # If config is a string path, load it
    if isinstance(config, str):
        config = load_config(config)
    
    # Set up logging
    output_dir = Path(config['logging']['output_dir'])
    experiment_name = config['logging']['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = exp_dir / "training.log"
    logger.add(log_file, rotation="100 MB", retention="10 days", level="DEBUG")
    
    logger.info("="*80)
    logger.info("STAGES Cognitive Prediction Fine-tuning")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"Target: {config['task']['target']}")
    logger.info(f"Task type: {config['task']['task_type']}")
    
    # Save configuration
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to: {config_save_path}")
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Set device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create cognitive prediction model (works with pre-computed embeddings)
    model = create_model(config, device)
    
    # Create datasets
    logger.info("Loading datasets...")
    
    # Load datasets from pre-computed embeddings
    train_dataset = CognitivePredictionDataset(config, split='train')
    val_dataset = CognitivePredictionDataset(config, split='val')
    test_dataset = CognitivePredictionDataset(config, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=cognitive_collate_fn,
        prefetch_factor=config['system'].get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=cognitive_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=cognitive_collate_fn
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Create optimizer
    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Create learning rate scheduler
    scheduler_name = config['training'].get('scheduler', None)
    scheduler = None
    warmup_scheduler = None
    
    if scheduler_name == 'CosineAnnealingLR':
        params = config['training']['scheduler_params']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'],
            eta_min=params.get('eta_min', 0)
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        params = config['training']['scheduler_params']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 5),
            verbose=True
        )
    
    # Add warmup scheduler if configured
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    if warmup_epochs > 0 and scheduler is not None:
        warmup_start_lr = config['training'].get('warmup_start_lr', config['training']['lr'] / 10)
        # Linear warmup from warmup_start_lr to main lr
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_lr / config['training']['lr'],
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # Chain warmup with main scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )
        logger.info(f"Using warmup scheduler: {warmup_epochs} epochs from LR {warmup_start_lr} to {config['training']['lr']}")
    
    logger.info(f"Learning rate scheduler: {scheduler_name if scheduler_name else 'None'}")
    
    # Create loss function
    criterion = get_loss_function(config)
    logger.info(f"Loss function: {criterion.__class__.__name__}")
    
    # Gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Training loop
    best_metric = -float('inf') if config['evaluation']['higher_is_better'] else float('inf')
    patience_counter = 0
    primary_metric = config['evaluation']['primary_metric']
    
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80 + "\n")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")
        logger.info("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler, epoch
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}")
        for metric_name, metric_value in train_metrics.items():
            if metric_name != 'loss':
                logger.info(f"Train - {metric_name}: {metric_value:.4f}")
        
        # Validate
        if epoch % config['evaluation']['eval_interval'] == 0:
            val_metrics, val_preds, val_targets = validate(
                model, val_loader, criterion, device, config, split='val'
            )
            
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}")
            for metric_name, metric_value in val_metrics.items():
                if metric_name != 'loss':
                    logger.info(f"Val - {metric_name}: {metric_value:.4f}")
            
            # Check if best model
            current_metric = val_metrics[primary_metric]
            
            is_best = False
            if config['evaluation']['higher_is_better']:
                if current_metric > best_metric:
                    best_metric = current_metric
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if is_best:
                logger.info(f"★ New best {primary_metric}: {best_metric:.4f}")
            
            # Save checkpoint
            if epoch % config['logging']['save_checkpoint_interval'] == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, config,
                    is_best=is_best, checkpoint_dir=exp_dir
                )
            
            # Early stopping
            if config['training']['early_stopping']['enabled']:
                if patience_counter >= config['training']['early_stopping']['patience']:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.2e}")
    
    # Final evaluation on all splits with best model
    logger.info("\n" + "="*80)
    logger.info("Final Evaluation with Best Model")
    logger.info("="*80)
    
    # Load best model
    best_checkpoint = torch.load(exp_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on all splits
    all_metrics = {}
    
    # Train set
    logger.info("\nTrain Set Evaluation:")
    train_metrics, train_preds, train_targets = validate(
        model, train_loader, criterion, device, config, split='train'
    )
    for metric_name, metric_value in train_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    all_metrics['train'] = train_metrics
    
    # Validation set
    logger.info("\nValidation Set Evaluation:")
    val_metrics, val_preds, val_targets = validate(
        model, val_loader, criterion, device, config, split='val'
    )
    for metric_name, metric_value in val_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    all_metrics['val'] = val_metrics
    
    # Test set
    logger.info("\nTest Set Evaluation:")
    test_metrics, test_preds, test_targets = validate(
        model, test_loader, criterion, device, config, split='test'
    )
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    all_metrics['test'] = test_metrics
    
    # Save predictions for all splits
    if config['evaluation']['save_predictions']:
        logger.info("\nSaving predictions for all splits...")
        
        # Train predictions
        save_predictions_to_csv(
            train_preds, train_targets, train_dataset.subject_ids,
            config['task']['task_type'], exp_dir / "train_predictions.csv", split='train'
        )
        
        # Validation predictions
        save_predictions_to_csv(
            val_preds, val_targets, val_dataset.subject_ids,
            config['task']['task_type'], exp_dir / "val_predictions.csv", split='val'
        )
        
        # Test predictions
        save_predictions_to_csv(
            test_preds, test_targets, test_dataset.subject_ids,
            config['task']['task_type'], exp_dir / "test_predictions.csv", split='test'
        )
    
    # Save all metrics
    metrics_file = exp_dir / "final_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSaved all metrics to: {metrics_file}")
    
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SleepFM for STAGES cognitive prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Override cognitive target (e.g., sustained_attention, working_memory, episodic_memory)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=['classification', 'regression'],
        help="Override task type (classification or regression)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.target is not None:
        config['task']['target'] = args.target
        # Update split path to match target
        config['data']['split_path'] = config['data']['split_path'].replace(
            config['task']['target'], args.target
        )
        logger.info(f"Overriding target: {args.target}")
    
    if args.task_type is not None:
        config['task']['task_type'] = args.task_type
        logger.info(f"Overriding task_type: {args.task_type}")
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")
    
    if args.lr is not None:
        config['training']['lr'] = args.lr
        logger.info(f"Overriding learning rate: {args.lr}")
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    main(config)
