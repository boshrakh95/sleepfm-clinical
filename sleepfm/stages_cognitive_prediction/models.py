"""
STAGES Cognitive Prediction Models
===================================

Model architectures following DiagnosisFinetuneFullLSTMCOXPHWithDemo structure.
Adapted for cognitive prediction (regression/classification with demographics).

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from einops import rearrange
from typing import Dict, Optional
import sys
import os

# Add sleepfm directory to path for absolute imports
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.models.models import AttentionPooling, PositionalEncoding


class CognitivePredictionModel(nn.Module):
    """
    Model for cognitive prediction using pre-computed embeddings.
    Based on DiagnosisFinetuneFullLSTMCOXPHWithDemo architecture.
    
    Architecture:
    1. Spatial pooling across modalities (BAS, RESP, EKG, EMG)
    2. LSTM for temporal modeling
    3. Temporal pooling (mean over valid sequence)
    4. Optional: demographics embedding
    5. Task head (regression or classification)
    
    Input: [B, num_modalities, seq_len, embed_dim]
    Output: [B, num_classes] for classification or [B] for regression
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        task_type: str = 'regression',
        num_classes: int = 1,
        use_demographics: bool = False,
        pooling_head: int = 4,
        dropout: float = 0.1,
        max_seq_length: Optional[int] = None
    ):
        """
        Initialize model.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads for spatial pooling
            num_layers: Number of LSTM layers
            task_type: 'regression' or 'classification'
            num_classes: Number of classes (for classification) or 1 (for regression)
            use_demographics: Whether to use demographics features
            pooling_head: Number of heads for attention pooling
            dropout: Dropout rate
            max_seq_length: Maximum sequence length (for positional encoding)
        """
        super(CognitivePredictionModel, self).__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.use_demographics = use_demographics
        self.embed_dim = embed_dim
        
        # Spatial pooling across modalities
        self.spatial_pooling = AttentionPooling(
            embed_dim, 
            num_heads=pooling_head, 
            dropout=dropout
        )
        
        # Positional encoding for temporal sequences
        if max_seq_length is None:
            max_seq_length = 20000
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Demographics embedding (if used)
        if use_demographics:
            self.demo_embedding = nn.Sequential(
                nn.Linear(2, embed_dim // 4),  # 2 = age + gender
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            task_input_dim = embed_dim + embed_dim // 4
        else:
            self.demo_embedding = None
            task_input_dim = embed_dim
        
        # Task head
        if task_type == 'classification':
            self.task_head = nn.Linear(task_input_dim, num_classes)
        else:  # regression
            self.task_head = nn.Linear(task_input_dim, 1)
    
    def forward(self, x, mask, demo_features=None):
        """
        Forward pass.
        
        Args:
            x: [B, num_modalities, seq_len, embed_dim]
            mask: [B, num_modalities, seq_len] - 0 for valid, 1 for padding
            demo_features: [B, 2] - age and gender (if use_demographics=True)
        
        Returns:
            output: [B, num_classes] for classification or [B] for regression
        """
        B, C, S, E = x.shape
        
        # Spatial pooling across modalities
        # Rearrange: [B, C, S, E] -> [(B*S), C, E]
        x = rearrange(x, 'b c s e -> (b s) c e')
        
        # Create mask for spatial pooling
        mask_spatial = mask[:, :, 0]  # [B, C]
        mask_spatial = mask_spatial.unsqueeze(1).expand(-1, S, -1)  # [B, S, C]
        mask_spatial = rearrange(mask_spatial, 'b t c -> (b t) c')  # [(B*S), C]
        mask_spatial = mask_spatial.to(dtype=torch.bool)
        
        # Apply spatial pooling: [(B*S), C, E] -> [(B*S), E]
        x = self.spatial_pooling(x, mask_spatial)
        x = x.view(B, S, E)  # [B, S, E]
        
        # Temporal processing with LSTM
        # Create mask for temporal dimension
        mask_temporal = mask[:, 0, :]  # [B, S]
        
        # Compute lengths for packing
        lengths = (mask_temporal == 0).sum(dim=1).cpu()  # Number of valid timesteps per sample
        
        # Pack sequence for LSTM
        packed_x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_out, _ = self.lstm(packed_x)
        
        # Unpack sequence
        x, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)  # [B, S, E]
        
        # Temporal pooling (mean over valid lengths)
        x = torch.stack([x[i, :lengths[i]].mean(dim=0) for i in range(B)])  # [B, E]
        
        # Add demographics if used
        if self.use_demographics and demo_features is not None:
            demo_embed = self.demo_embedding(demo_features)  # [B, E//4]
            x = torch.cat([x, demo_embed], dim=1)  # [B, E + E//4]
        
        # Task head
        output = self.task_head(x)  # [B, num_classes] or [B, 1]
        
        # For regression, squeeze the last dimension
        if self.task_type == 'regression':
            output = output.squeeze(-1)  # [B]
        
        return output


def create_cognitive_model(config: Dict) -> CognitivePredictionModel:
    """
    Factory function to create cognitive prediction model from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: CognitivePredictionModel instance
    """
    model_params = config['model']['params']
    task_config = config['task']
    
    # Determine number of classes based on task type and loss function
    if task_config['task_type'] == 'classification':
        # Check loss function to determine output size
        loss_fn = config.get('training', {}).get('loss_function', 'BCE')
        if loss_fn in ['BCE', 'BCEWithLogitsLoss', 'FocalLoss']:
            # Binary classification with BCE-style losses: single output
            num_classes = 1
        else:
            # Multi-class or CrossEntropy: 2 outputs for binary
            num_classes = 2
    else:
        # For regression
        num_classes = 1
    
    model = CognitivePredictionModel(
        embed_dim=model_params['embed_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        task_type=task_config['task_type'],
        num_classes=num_classes,
        use_demographics=task_config['use_demographics'],
        pooling_head=model_params['pooling_head'],
        dropout=model_params['dropout'],
        max_seq_length=model_params.get('max_seq_length')
    )
    
    return model
