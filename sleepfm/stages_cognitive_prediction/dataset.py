"""
Dataset classes for STAGES Cognitive Prediction
================================================

Loads pre-computed embeddings from HDF5 files (following demo.py format).
Each HDF5 file contains embeddings with modalities as separate datasets.

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import os
import sys
import json
import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from loguru import logger

# Add sleepfm directory to path
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.models.dataset import SetTransformerDataset
from sleepfm.utils import load_data


class CognitivePredictionDataset(Dataset):
    """
    Dataset for cognitive prediction using pre-computed embeddings.
    Follows the same structure as DiagnosisFinetuneFullCOXPHWithDemoDataset.
    
    Embeddings are stored in HDF5 files where each modality is a separate dataset.
    Format: [seq_len, embed_dim] per modality.
    """
    
    def __init__(
        self,
        config: Dict,
        split: str = "train",
        embeddings_dir: Optional[str] = None,
        labels_path: Optional[str] = None,
        split_path: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
            embeddings_dir: Directory containing HDF5 embedding files
            labels_path: Path to labels CSV file
            split_path: Path to split JSON file
        """
        self.config = config
        self.split = split
        
        # Get paths from config if not provided
        if embeddings_dir is None:
            embeddings_dir = config['preprocessing']['embeddings_dir']
        if labels_path is None:
            labels_path = config['data']['labels_path']
        if split_path is None:
            split_path = config['data']['split_path']
        
        self.embeddings_dir = Path(embeddings_dir)
        
        # Load labels
        logger.info(f"Loading labels from {labels_path}")
        self.labels_df = pd.read_csv(labels_path)
        
        # Handle different column name formats (Study ID vs subject_id)
        if 'Study ID' in self.labels_df.columns:
            self.labels_df = self.labels_df.set_index('Study ID')
        elif 'subject_id' in self.labels_df.columns:
            self.labels_df = self.labels_df.set_index('subject_id')
        else:
            raise ValueError(f"Labels CSV must have either 'Study ID' or 'subject_id' column. Found: {self.labels_df.columns.tolist()}")
        
        # Load split or use available embeddings
        use_available = config['data'].get('use_available_subjects', False)
        
        if use_available:
            # Use all available embeddings, ignoring split file
            logger.info(f"Using all available embeddings (ignoring split file)")
            all_embedding_files = list(self.embeddings_dir.glob("*.hdf5"))
            all_subjects = [f.stem for f in all_embedding_files if f.stem in self.labels_df.index]
            
            # Create splits automatically (60% train, 20% val, 20% test)
            import random
            random.seed(config['system'].get('seed', 42))
            random.shuffle(all_subjects)
            
            n_train = int(0.6 * len(all_subjects))
            n_val = int(0.2 * len(all_subjects))
            
            if split == 'train':
                subject_ids = all_subjects[:n_train]
            elif split == 'val':
                subject_ids = all_subjects[n_train:n_train+n_val]
            else:  # test
                subject_ids = all_subjects[n_train+n_val:]
            
            valid_subjects = subject_ids
            logger.info(f"{split} split (auto-generated): {len(valid_subjects)} subjects from {len(all_subjects)} total")
        else:
            # Use predefined split file
            logger.info(f"Loading split from {split_path}")
            split_data = load_data(split_path)
            subject_paths = split_data[split]
            
            # Extract subject IDs from file paths (handles both full paths and bare subject IDs)
            subject_ids = []
            for item in subject_paths:
                if isinstance(item, str):
                    # If it's a file path, extract the stem (filename without extension)
                    if '/' in item or '\\' in item or item.endswith('.hdf5'):
                        subject_id = Path(item).stem
                    else:
                        # Already a bare subject ID
                        subject_id = item
                    subject_ids.append(subject_id)
            
            # Filter by subjects that have both labels and embeddings
            valid_subjects = []
            for subject_id in subject_ids:
                emb_file = self.embeddings_dir / f"{subject_id}.hdf5"
                if emb_file.exists() and subject_id in self.labels_df.index:
                    valid_subjects.append(subject_id)
            
            logger.info(f"{split} split: {len(subject_ids)} subjects in split, {len(valid_subjects)} have embeddings and labels")
        
        # Apply quality filtering if enabled
        if config['task'].get('use_quality_filter', False):
            quality_threshold = config['task'].get('quality_threshold', 0.8)
            valid_subjects = [
                s for s in valid_subjects
                if self.labels_df.loc[s, 'quality'] >= quality_threshold
            ]
            logger.info(f"After quality filtering (>={quality_threshold}): {len(valid_subjects)} subjects")
        
        # Limit number of subjects if specified (for testing)
        max_files = config['data'].get('max_files', None)
        if max_files is not None and max_files > 0:
            valid_subjects = valid_subjects[:max_files]
            logger.info(f"Limited to {max_files} subjects for testing")
        
        self.subject_ids = valid_subjects
        
        # Get task parameters
        self.target = config['task']['target']
        self.task_type = config['task']['task_type']
        self.use_demographics = config['task']['use_demographics']
        
        # Find demographic column names if needed
        if self.use_demographics:
            # Find age column (try: age, Age, AGE, nsrr_age, NSRR_age)
            self.age_col = None
            for possible in ['age', 'Age', 'AGE', 'nsrr_age', 'NSRR_age', 'NSRR_AGE']:
                if possible in self.labels_df.columns:
                    self.age_col = possible
                    break
            if self.age_col is None:
                raise ValueError(f"Could not find age column in labels. Available columns: {list(self.labels_df.columns)}")
            
            # Find gender/sex column (try: gender, Gender, GENDER, sex, Sex, SEX, nsrr_sex, nsrr_gender)
            self.gender_col = None
            for possible in ['gender', 'Gender', 'GENDER', 'sex', 'Sex', 'SEX', 'nsrr_sex', 'NSRR_sex', 'nsrr_gender', 'NSRR_gender']:
                if possible in self.labels_df.columns:
                    self.gender_col = possible
                    break
            if self.gender_col is None:
                raise ValueError(f"Could not find gender/sex column in labels. Available columns: {list(self.labels_df.columns)}")
            
            logger.info(f"Using demographic columns: {self.age_col}, {self.gender_col}")
        
        # Get modality types from config
        self.modality_types = config['data']['modality_types']
        self.max_seq_length = config['model']['params'].get('max_seq_length', None)
        self.max_channels = len(self.modality_types)  # One "channel" per modality
        
        logger.info(f"Dataset initialized: {len(self.subject_ids)} subjects")
        logger.info(f"Target: {self.target}, Task: {self.task_type}, Use demographics: {self.use_demographics}")
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Load embeddings and labels for one subject.
        
        Returns:
            x_data: [num_modalities, seq_len, embed_dim]
            label: scalar (regression) or int (classification)
            demo_features: [2] if use_demographics else None
            subject_id: str
        """
        subject_id = self.subject_ids[idx]
        emb_file = self.embeddings_dir / f"{subject_id}.hdf5"
        
        # Load embeddings from HDF5 (each modality is a separate dataset)
        x_data = []
        with h5py.File(emb_file, 'r') as hf:
            for modality_type in self.modality_types:
                if modality_type in hf:
                    emb = hf[modality_type][:]  # [seq_len, embed_dim]
                    x_data.append(emb)
                else:
                    # If modality is missing, create zeros
                    # Get embed_dim from first available modality
                    if x_data:
                        embed_dim = x_data[0].shape[1]
                    else:
                        embed_dim = 128  # Default
                    emb = np.zeros((1, embed_dim), dtype=np.float32)
                    x_data.append(emb)
        
        # Stack modalities: [num_modalities, seq_len, embed_dim]
        x_data = np.array(x_data, dtype=np.float32)
        
        # Get label
        label_value = self.labels_df.loc[subject_id, self.target]
        
        if self.task_type == 'classification':
            label = int(label_value)
        else:  # regression
            label = float(label_value)
        
        # Get demographics if needed
        if self.use_demographics:
            age = self.labels_df.loc[subject_id, self.age_col]
            gender = self.labels_df.loc[subject_id, self.gender_col]
            demo_features = np.array([age, gender], dtype=np.float32)
        else:
            demo_features = None
        
        # Convert to tensors
        x_data = torch.from_numpy(x_data)
        label = torch.tensor(label)
        if demo_features is not None:
            demo_features = torch.from_numpy(demo_features)
        
        return x_data, label, demo_features, subject_id


def cognitive_collate_fn(batch):
    """
    Collate function for cognitive prediction dataset.
    Handles variable-length sequences by padding.
    
    Args:
        batch: List of (x_data, label, demo_features, subject_id)
    
    Returns:
        x_data_padded: [B, num_modalities, max_seq_len, embed_dim]
        labels: [B]
        demo_features: [B, 2] or None
        masks: [B, num_modalities, max_seq_len] - 0 for valid, 1 for padding
        subject_ids: List[str]
    """
    x_data_list, labels, demo_features_list, subject_ids = zip(*batch)
    
    # Get dimensions
    batch_size = len(batch)
    num_modalities = x_data_list[0].shape[0]
    embed_dim = x_data_list[0].shape[2]
    
    # Find max sequence length in this batch
    max_seq_len = max([x.shape[1] for x in x_data_list])
    
    # Initialize padded tensors
    x_data_padded = torch.zeros(batch_size, num_modalities, max_seq_len, embed_dim)
    masks = torch.ones(batch_size, num_modalities, max_seq_len)  # 1 = padding
    
    # Fill in data
    for i, x in enumerate(x_data_list):
        c, s, e = x.shape
        s = min(s, max_seq_len)  # Truncate if too long
        
        x_data_padded[i, :c, :s, :e] = x[:c, :s, :e]
        masks[i, :c, :s] = 0  # 0 = valid data
    
    # Stack labels
    labels = torch.stack(labels)
    
    # Stack demographics if present
    if demo_features_list[0] is not None:
        demo_features = torch.stack(demo_features_list)
    else:
        demo_features = None
    
    return x_data_padded, labels, demo_features, masks, subject_ids
