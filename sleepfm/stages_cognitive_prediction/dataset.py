"""
STAGES Cognitive Prediction Dataset
====================================

Dataset class for fine-tuning SleepFM on cognitive prediction tasks.

This dataset:
1. Loads HDF5 PSG files
2. Generates embeddings using pre-trained SleepFM
3. Loads cognitive labels and demographics
4. Filters by quality metadata (optional)
5. Returns aggregated embeddings and labels for training

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
import os

# Add sleepfm directory to path for absolute imports
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.models.dataset import SetTransformerDataset, collate_fn
from sleepfm.utils import load_data


class CognitivePredictionDataset(Dataset):
    """Dataset for cognitive prediction fine-tuning.
    
    This dataset handles:
    - Loading PSG data from HDF5 files
    - Generating embeddings using pre-trained SetTransformer
    - Loading cognitive labels and demographics
    - Quality-based filtering and weighting
    - Aggregating embeddings over time windows
    """
    
    def __init__(
        self,
        config: Dict,
        channel_groups: Dict,
        split: str = 'train',
        pretrained_model=None
    ):
        """Initialize dataset.
        
        Args:
            config: Configuration dictionary
            channel_groups: Channel group mappings
            split: Data split ('train', 'val', 'test')
            pretrained_model: Pre-trained SetTransformer model for embedding generation
        """
        self.config = config
        self.channel_groups = channel_groups
        self.split = split
        self.pretrained_model = pretrained_model
        
        # Paths
        self.data_path = Path(config['data']['data_path'])
        self.labels_path = Path(config['data']['labels_path'])
        self.quality_path = Path(config['data'].get('quality_path', ''))
        
        # Task configuration
        self.target = config['task']['target']
        self.task_type = config['task']['task_type']
        self.use_demographics = config['task']['use_demographics']
        self.use_quality_filtering = config['task']['use_quality_filtering']
        self.min_clean_ratio = config['task']['min_clean_ratio']
        
        # Data configuration
        self.chunk_duration = config['data']['chunk_duration']
        self.chunk_size = config['data']['chunk_size']
        self.aggregation_window = config['data']['aggregation_window']
        self.exclude_channels = config['data'].get('exclude_channels', [])
        
        # Create set of excluded channel names (case-insensitive)
        self.excluded_set = set([ch.lower() for ch in self.exclude_channels])
        
        # Load split
        split_path = config['data']['split_path']
        split_data = load_data(split_path)
        self.hdf5_paths = split_data[split]
        
        # Load labels
        self.labels_df = self._load_labels()
        
        # Filter subjects based on labels availability
        self.subjects = self._filter_subjects()
        
        logger.info(f"{split} split: {len(self.subjects)} subjects with valid labels")
        
        # Load quality metadata if using quality filtering
        if self.use_quality_filtering:
            self.quality_metadata = self._load_quality_metadata()
        else:
            self.quality_metadata = {}
    
    def _load_labels(self) -> pd.DataFrame:
        """Load labels and demographics."""
        labels_file = self.labels_path / "labels_with_demographics.csv"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        df = pd.read_csv(labels_file)
        
        # Check if target exists
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found in labels. "
                           f"Available targets: {df.columns.tolist()}")
        
        return df
    
    def _filter_subjects(self) -> List[str]:
        """Filter subjects that have both HDF5 files and valid labels."""
        # Extract subject IDs from HDF5 paths
        hdf5_subjects = set([Path(p).stem for p in self.hdf5_paths])
        
        # Get subjects with valid labels (non-NaN)
        valid_label_subjects = set(
            self.labels_df[self.labels_df[self.target].notna()]['Study ID'].values
        )
        
        # Intersection
        subjects = sorted(list(hdf5_subjects & valid_label_subjects))
        
        return subjects
    
    def _load_quality_metadata(self) -> Dict:
        """Load quality metadata for all subjects."""
        quality_meta = {}
        
        for subject_id in self.subjects:
            quality_file = self.quality_path / f"{subject_id}_quality.json"
            
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality_meta[subject_id] = json.load(f)
            else:
                logger.warning(f"Quality metadata not found for {subject_id}")
        
        return quality_meta
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get item by index.
        
        Returns:
            embeddings: Aggregated embeddings [seq_len, embed_dim]
            label: Target value (float for regression, int for classification)
            demographics: Demographics features [demo_dim] (if use_demographics=True)
            quality_mask: Quality mask [seq_len] (if use_quality_filtering=True)
            subject_id: Subject identifier
        """
        subject_id = self.subjects[idx]
        
        # Load HDF5 file
        hdf5_path = self.data_path / f"{subject_id}.hdf5"
        
        # Get label
        subject_row = self.labels_df[self.labels_df['Study ID'] == subject_id].iloc[0]
        
        if self.task_type == 'regression':
            label = float(subject_row[self.target])
        else:  # classification
            label = int(subject_row[self.target])
        
        # Get demographics if requested
        demographics = None
        if self.use_demographics:
            age = float(subject_row['nsrr_age'])
            gender = float(subject_row['nsrr_sex'])  # Already encoded as 0/1
            demographics = torch.tensor([age, gender], dtype=torch.float32)
        
        # Load PSG data and generate embeddings
        # This will be done by the model during forward pass
        # For now, we just return the file path and let the model handle it
        
        # Get quality mask if requested
        quality_mask = None
        if self.use_quality_filtering and subject_id in self.quality_metadata:
            quality_info = self.quality_metadata[subject_id]
            
            # Filter by clean ratio
            clean_ratio = quality_info['clean_ratio']
            
            if clean_ratio < self.min_clean_ratio:
                logger.warning(f"Subject {subject_id} has low clean ratio: {clean_ratio:.2%}")
            
            # Create mask for clean windows
            total_windows = quality_info['total_windows']
            clean_windows = set(quality_info['clean_windows'])
            
            # Create binary mask (True=clean, False=artifact)
            quality_mask = torch.tensor(
                [i in clean_windows for i in range(total_windows)],
                dtype=torch.bool
            )
        
        return {
            'hdf5_path': str(hdf5_path),
            'subject_id': subject_id,
            'label': torch.tensor(label, dtype=torch.float32 if self.task_type == 'regression' else torch.long),
            'demographics': demographics,
            'quality_mask': quality_mask
        }


class CognitivePredictionDatasetWithEmbeddings(Dataset):
    """Dataset that uses pre-computed embeddings instead of raw PSG.
    
    This is faster for training as embeddings are pre-computed and cached.
    """
    
    def __init__(
        self,
        config: Dict,
        split: str = 'train',
        embeddings_dir: Optional[str] = None
    ):
        """Initialize dataset with embeddings.
        
        Args:
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test')
            embeddings_dir: Directory containing pre-computed embeddings
        """
        self.config = config
        self.split = split
        
        # Paths
        self.labels_path = Path(config['data']['labels_path'])
        self.quality_path = Path(config['data'].get('quality_path', ''))
        
        if embeddings_dir is None:
            embeddings_dir = config['preprocessing']['embeddings_dir']
        self.embeddings_dir = Path(embeddings_dir)
        
        # Task configuration
        self.target = config['task']['target']
        self.task_type = config['task']['task_type']
        self.use_demographics = config['task']['use_demographics']
        self.use_quality_filtering = config['task']['use_quality_filtering']
        self.min_clean_ratio = config['task']['min_clean_ratio']
        
        # Load split
        split_path = config['data']['split_path']
        split_data = load_data(split_path)
        self.hdf5_paths = split_data[split]
        
        # Load labels
        self.labels_df = self._load_labels()
        
        # Filter subjects
        self.subjects = self._filter_subjects()
        
        logger.info(f"{split} split: {len(self.subjects)} subjects with embeddings")
        
        # Load quality metadata if using
        if self.use_quality_filtering:
            self.quality_metadata = self._load_quality_metadata()
        else:
            self.quality_metadata = {}
    
    def _load_labels(self) -> pd.DataFrame:
        """Load labels and demographics."""
        labels_file = self.labels_path / "labels_with_demographics.csv"
        df = pd.read_csv(labels_file)
        return df
    
    def _filter_subjects(self) -> List[str]:
        """Filter subjects with embeddings and valid labels."""
        # Get subjects with embeddings
        embedding_files = list(self.embeddings_dir.glob("*.npy"))
        embedding_subjects = set([f.stem for f in embedding_files])
        
        # Get subjects with valid labels
        valid_label_subjects = set(
            self.labels_df[self.labels_df[self.target].notna()]['Study ID'].values
        )
        
        # Intersection
        subjects = sorted(list(embedding_subjects & valid_label_subjects))
        
        return subjects
    
    def _load_quality_metadata(self) -> Dict:
        """Load quality metadata."""
        quality_meta = {}
        
        for subject_id in self.subjects:
            quality_file = self.quality_path / f"{subject_id}_quality.json"
            
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality_meta[subject_id] = json.load(f)
        
        return quality_meta
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item with pre-computed embeddings."""
        subject_id = self.subjects[idx]
        
        # Load embeddings
        embedding_path = self.embeddings_dir / f"{subject_id}.npy"
        embeddings = np.load(embedding_path)
        embeddings = torch.from_numpy(embeddings).float()
        
        # Get label
        subject_row = self.labels_df[self.labels_df['Study ID'] == subject_id].iloc[0]
        
        if self.task_type == 'regression':
            label = float(subject_row[self.target])
        else:
            label = int(subject_row[self.target])
        
        # Get demographics
        demographics = None
        if self.use_demographics:
            age = float(subject_row['nsrr_age'])
            gender = float(subject_row['nsrr_sex'])
            demographics = torch.tensor([age, gender], dtype=torch.float32)
        
        # Get quality mask
        quality_mask = None
        if self.use_quality_filtering and subject_id in self.quality_metadata:
            quality_info = self.quality_metadata[subject_id]
            total_windows = quality_info['total_windows']
            clean_windows = set(quality_info['clean_windows'])
            
            # Adjust for aggregation window
            num_agg_windows = total_windows // self.config['data']['aggregation_window']
            quality_mask = []
            
            for i in range(num_agg_windows):
                start = i * self.config['data']['aggregation_window']
                end = (i + 1) * self.config['data']['aggregation_window']
                window_indices = set(range(start, end))
                
                # Check if any window in this aggregation is clean
                is_clean = bool(window_indices & clean_windows)
                quality_mask.append(is_clean)
            
            quality_mask = torch.tensor(quality_mask, dtype=torch.bool)
        
        return {
            'embeddings': embeddings,
            'subject_id': subject_id,
            'label': torch.tensor(label, dtype=torch.float32 if self.task_type == 'regression' else torch.long),
            'demographics': demographics,
            'quality_mask': quality_mask
        }


def cognitive_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for cognitive prediction dataset.
    
    Handles variable-length sequences and optional demographics/quality masks.
    """
    # Separate components
    embeddings = [item['embeddings'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    subject_ids = [item['subject_id'] for item in batch]
    
    # Pad embeddings to same length
    max_len = max([e.shape[0] for e in embeddings])
    embed_dim = embeddings[0].shape[1]
    
    padded_embeddings = []
    padding_masks = []
    
    for emb in embeddings:
        seq_len = emb.shape[0]
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            padding = torch.zeros(pad_len, embed_dim)
            padded_emb = torch.cat([emb, padding], dim=0)
        else:
            padded_emb = emb
        
        # Mask: True for valid positions, False for padding
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        
        padded_embeddings.append(padded_emb)
        padding_masks.append(mask)
    
    padded_embeddings = torch.stack(padded_embeddings)
    padding_masks = torch.stack(padding_masks)
    
    # Handle demographics
    demographics = None
    if batch[0]['demographics'] is not None:
        demographics = torch.stack([item['demographics'] for item in batch])
    
    # Handle quality masks
    quality_masks = None
    if batch[0]['quality_mask'] is not None:
        # Pad quality masks
        max_qual_len = max([item['quality_mask'].shape[0] for item in batch if item['quality_mask'] is not None])
        padded_quality_masks = []
        
        for item in batch:
            if item['quality_mask'] is not None:
                qual_mask = item['quality_mask']
                qual_len = qual_mask.shape[0]
                pad_len = max_qual_len - qual_len
                
                if pad_len > 0:
                    padding = torch.zeros(pad_len, dtype=torch.bool)
                    padded_qual = torch.cat([qual_mask, padding])
                else:
                    padded_qual = qual_mask
                
                padded_quality_masks.append(padded_qual)
        
        if padded_quality_masks:
            quality_masks = torch.stack(padded_quality_masks)
    
    return {
        'embeddings': padded_embeddings,
        'labels': labels,
        'demographics': demographics,
        'padding_mask': padding_masks,
        'quality_mask': quality_masks,
        'subject_ids': subject_ids
    }
