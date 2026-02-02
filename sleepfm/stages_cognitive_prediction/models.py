"""
STAGES Cognitive Prediction Models
===================================

Model architectures for fine-tuning SleepFM on cognitive prediction tasks.

Models:
1. CognitiveRegressionLSTM: LSTM-based regression model
2. CognitiveClassificationLSTM: LSTM-based classification model
3. CognitiveLSTMWithDemo: LSTM with demographics features

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os

# Add sleepfm directory to path for absolute imports
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.models.models import SetTransformer, AttentionPooling


class CognitiveRegressionLSTM(nn.Module):
    """LSTM-based model for cognitive regression on sleep embeddings.
    
    Architecture:
    1. Load pre-trained SetTransformer (frozen or fine-tuned)
    2. Generate embeddings from PSG data
    3. LSTM to model temporal dependencies
    4. Attention pooling over time
    5. Regression head
    """
    
    def __init__(
        self,
        pretrained_model: SetTransformer,
        embed_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.embed_dim = embed_dim
        self.freeze_encoder = freeze_encoder
        
        # Freeze pre-trained model if requested
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=lstm_bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_out_dim = lstm_hidden_dim * 2 if lstm_bidirectional else lstm_hidden_dim
        
        # Attention pooling over time
        self.attention_pool = AttentionPooling(lstm_out_dim, num_heads=4, dropout=dropout)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, lstm_out_dim // 4),
            nn.LayerNorm(lstm_out_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 4, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input PSG data [batch, channels, time]
            mask: Channel mask [batch, channels]
            padding_mask: Temporal padding mask [batch, seq_len] (optional)
        
        Returns:
            predictions: Regression predictions [batch, 1]
        """
        # Generate embeddings using pre-trained model
        # x shape: [batch, channels, sequence, samples]
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeddings, _ = self.pretrained_model(x, mask)  # [batch, embed_dim]
        
        # embeddings is [batch, embed_dim] for single chunk
        # For sequence of chunks, we need to reshape
        # Assume x is already chunked: [batch, num_chunks, channels, chunk_size]
        # We need to process each chunk separately
        
        # If embeddings is [batch, embed_dim], expand to [batch, 1, embed_dim]
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(1)
        
        # LSTM processing: [batch, seq_len, embed_dim] -> [batch, seq_len, lstm_out_dim]
        lstm_out, _ = self.lstm(embeddings)
        
        # Attention pooling: [batch, seq_len, lstm_out_dim] -> [batch, lstm_out_dim]
        if padding_mask is not None:
            # Invert mask for attention pooling (True=padding, False=valid)
            attention_mask = ~padding_mask
            pooled = self.attention_pool(lstm_out, attention_mask)
        else:
            pooled = self.attention_pool(lstm_out)
        
        # Regression head: [batch, lstm_out_dim] -> [batch, 1]
        predictions = self.regressor(pooled)
        
        return predictions.squeeze(-1)  # [batch]


class CognitiveClassificationLSTM(nn.Module):
    """LSTM-based model for cognitive classification on sleep embeddings.
    
    Similar to regression model but with classification head.
    """
    
    def __init__(
        self,
        pretrained_model: SetTransformer,
        embed_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Freeze pre-trained model if requested
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=lstm_bidirectional
        )
        
        lstm_out_dim = lstm_hidden_dim * 2 if lstm_bidirectional else lstm_hidden_dim
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_out_dim, num_heads=4, dropout=dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, lstm_out_dim // 4),
            nn.LayerNorm(lstm_out_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 4, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        # Generate embeddings
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeddings, _ = self.pretrained_model(x, mask)
        
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(embeddings)
        
        # Attention pooling
        if padding_mask is not None:
            attention_mask = ~padding_mask
            pooled = self.attention_pool(lstm_out, attention_mask)
        else:
            pooled = self.attention_pool(lstm_out)
        
        # Classification head
        logits = self.classifier(pooled)
        
        return logits


class CognitiveLSTMWithDemo(nn.Module):
    """LSTM model with demographics features for cognitive prediction.
    
    Combines sleep embeddings with demographics (age, gender) for improved prediction.
    """
    
    def __init__(
        self,
        pretrained_model: SetTransformer,
        embed_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = True,
        demo_embed_dim: int = 16,
        num_classes: int = 1,  # 1 for regression, 2+ for classification
        task_type: str = 'regression',
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.embed_dim = embed_dim
        self.task_type = task_type
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Freeze pre-trained model if requested
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Demographics embedding
        self.demo_embedding = nn.Sequential(
            nn.Linear(2, demo_embed_dim),  # age, gender
            nn.LayerNorm(demo_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling of sleep embeddings
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=lstm_bidirectional
        )
        
        lstm_out_dim = lstm_hidden_dim * 2 if lstm_bidirectional else lstm_hidden_dim
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_out_dim, num_heads=4, dropout=dropout)
        
        # Fusion layer (combine sleep features + demographics)
        fusion_dim = lstm_out_dim + demo_embed_dim
        
        # Output head
        if task_type == 'regression':
            self.output_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, fusion_dim // 4),
                nn.LayerNorm(fusion_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 4, 1)
            )
        else:  # classification
            self.output_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, fusion_dim // 4),
                nn.LayerNorm(fusion_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 4, num_classes)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        demographics: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input PSG data
            mask: Channel mask
            demographics: Demographics features [batch, 2] (age, gender)
            padding_mask: Temporal padding mask (optional)
        
        Returns:
            predictions: Regression predictions [batch] or classification logits [batch, num_classes]
        """
        # Generate sleep embeddings
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeddings, _ = self.pretrained_model(x, mask)
        
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(embeddings)
        
        # Attention pooling
        if padding_mask is not None:
            attention_mask = ~padding_mask
            sleep_features = self.attention_pool(lstm_out, attention_mask)
        else:
            sleep_features = self.attention_pool(lstm_out)
        
        # Demographics embedding
        demo_features = self.demo_embedding(demographics)
        
        # Fusion
        combined_features = torch.cat([sleep_features, demo_features], dim=-1)
        
        # Output
        output = self.output_head(combined_features)
        
        if self.task_type == 'regression':
            return output.squeeze(-1)  # [batch]
        else:
            return output  # [batch, num_classes]


class CognitiveEmbeddingLSTM(nn.Module):
    """LSTM model that works directly with pre-computed embeddings.
    
    This is faster for training as it doesn't require running the
    SetTransformer encoder during training.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        lstm_bidirectional: bool = True,
        demo_embed_dim: int = 16,
        num_classes: int = 1,
        task_type: str = 'regression',
        use_demographics: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.task_type = task_type
        self.num_classes = num_classes
        self.use_demographics = use_demographics
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=lstm_bidirectional
        )
        
        lstm_out_dim = lstm_hidden_dim * 2 if lstm_bidirectional else lstm_hidden_dim
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_out_dim, num_heads=4, dropout=dropout)
        
        # Demographics embedding (optional)
        if use_demographics:
            self.demo_embedding = nn.Sequential(
                nn.Linear(2, demo_embed_dim),
                nn.LayerNorm(demo_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fusion_dim = lstm_out_dim + demo_embed_dim
        else:
            fusion_dim = lstm_out_dim
        
        # Output head
        if task_type == 'regression':
            self.output_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, fusion_dim // 4),
                nn.LayerNorm(fusion_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 4, 1)
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, fusion_dim // 4),
                nn.LayerNorm(fusion_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 4, num_classes)
            )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        quality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with pre-computed embeddings.
        
        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, embed_dim]
            demographics: Demographics features [batch, 2] (optional)
            padding_mask: Temporal padding mask [batch, seq_len] (optional)
            quality_mask: Quality mask [batch, seq_len] (optional)
        
        Returns:
            predictions: Regression [batch] or classification logits [batch, num_classes]
        """
        # LSTM processing
        lstm_out, _ = self.lstm(embeddings)
        
        # Combine padding and quality masks
        if padding_mask is not None and quality_mask is not None:
            combined_mask = padding_mask & quality_mask
        elif padding_mask is not None:
            combined_mask = padding_mask
        elif quality_mask is not None:
            combined_mask = quality_mask
        else:
            combined_mask = None
        
        # Attention pooling
        if combined_mask is not None:
            attention_mask = ~combined_mask  # Invert for attention
            sleep_features = self.attention_pool(lstm_out, attention_mask)
        else:
            sleep_features = self.attention_pool(lstm_out)
        
        # Add demographics if provided
        if self.use_demographics and demographics is not None:
            demo_features = self.demo_embedding(demographics)
            combined_features = torch.cat([sleep_features, demo_features], dim=-1)
        else:
            combined_features = sleep_features
        
        # Output
        output = self.output_head(combined_features)
        
        if self.task_type == 'regression':
            return output.squeeze(-1)
        else:
            return output
