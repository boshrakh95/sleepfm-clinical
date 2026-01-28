#!/usr/bin/env python3
"""
Create Train/Val/Test Splits for SleepFM
=========================================

This script creates stratified train/validation/test splits for SleepFM training.

Operations:
1. Load labels with cognitive scores
2. Stratify by cognitive score to ensure balanced distribution
3. Create train/val/test splits
4. Generate JSON file with HDF5 file paths for each split
5. Validate split distributions

Author: Generated for STAGES data preparation
Date: January 2026
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class STAGESSplitCreator:
    """Create train/val/test splits for SleepFM."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Directories
        self.output_base = Path(self.config['output']['base_dir'])
        self.hdf5_dir = self.output_base / self.config['output']['hdf5_dir']
        self.labels_dir = self.output_base / self.config['output']['labels_dir']
        self.splits_dir = self.output_base / self.config['output']['splits_dir']
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Split parameters
        self.train_ratio = self.config['split']['train_ratio']
        self.val_ratio = self.config['split']['val_ratio']
        self.test_ratio = self.config['split']['test_ratio']
        self.random_seed = self.config['split']['random_seed']
        
        np.random.seed(self.random_seed)
    
    def load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        # Resolve relative to script directory if not absolute
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging."""
        log_dir = Path(self.config['output']['base_dir']) / self.config['output']['logs_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"split_creation_{timestamp}.log"
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="DEBUG")
        
        logger.info("="*80)
        logger.info("STAGES Train/Val/Test Split Creation")
        logger.info("="*80)
    
    def load_labels(self) -> pd.DataFrame:
        """Load prepared labels."""
        labels_file = self.labels_dir / "labels_with_demographics.csv"
        
        if not labels_file.exists():
            logger.error(f"Labels file not found: {labels_file}")
            logger.error("Run prepare_labels.py first!")
            sys.exit(1)
        
        logger.info(f"Loading labels from: {labels_file}")
        
        df = pd.read_csv(labels_file)
        logger.info(f"Loaded {len(df)} subjects")
        
        return df
    
    def verify_hdf5_files(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Verify that HDF5 files exist for all subjects."""
        logger.info("Verifying HDF5 files...")
        
        valid_subjects = []
        missing_subjects = []
        
        for subject_id in labels_df['Study ID']:
            hdf5_file = self.hdf5_dir / f"{subject_id}.hdf5"
            
            if hdf5_file.exists():
                valid_subjects.append(subject_id)
            else:
                missing_subjects.append(subject_id)
        
        if missing_subjects:
            logger.warning(f"Missing HDF5 files for {len(missing_subjects)} subjects:")
            for subject_id in missing_subjects[:10]:
                logger.warning(f"  {subject_id}")
            if len(missing_subjects) > 10:
                logger.warning(f"  ... and {len(missing_subjects)-10} more")
        
        # Filter to subjects with HDF5 files
        labels_df = labels_df[labels_df['Study ID'].isin(valid_subjects)]
        
        logger.info(f"Valid subjects with HDF5 files: {len(labels_df)}")
        
        return labels_df
    
    def create_stratification_bins(self, labels_df: pd.DataFrame) -> np.ndarray:
        """Create stratification bins based on cognitive score."""
        stratify_col = self.config['split']['stratify_by']
        
        if stratify_col not in labels_df.columns:
            logger.warning(f"Stratification column '{stratify_col}' not found. "
                          "Using random split.")
            return None
        
        # Remove NaN values
        valid_mask = labels_df[stratify_col].notna()
        
        if valid_mask.sum() < len(labels_df):
            logger.warning(f"{(~valid_mask).sum()} subjects have missing {stratify_col}. "
                          "These will be randomly assigned.")
        
        # Create bins
        n_bins = self.config['split']['stratify_bins']
        
        bins = np.zeros(len(labels_df), dtype=int)
        
        # Bin subjects with valid scores
        valid_scores = labels_df.loc[valid_mask, stratify_col].values
        quantiles = np.percentile(valid_scores, np.linspace(0, 100, n_bins+1))
        
        bins[valid_mask] = np.digitize(valid_scores, quantiles[1:-1])
        
        # Random bin for subjects with missing scores
        bins[~valid_mask] = np.random.randint(0, n_bins, size=(~valid_mask).sum())
        
        logger.info(f"Created {n_bins} stratification bins based on {stratify_col}")
        
        # Log bin distribution
        for bin_idx in range(n_bins):
            count = (bins == bin_idx).sum()
            logger.info(f"  Bin {bin_idx}: {count} subjects")
        
        return bins
    
    def create_splits(self, labels_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create stratified train/val/test splits."""
        logger.info("Creating splits...")
        logger.info(f"  Train: {self.train_ratio*100:.0f}%")
        logger.info(f"  Val: {self.val_ratio*100:.0f}%")
        logger.info(f"  Test: {self.test_ratio*100:.0f}%")
        
        # Get subject IDs
        subject_ids = labels_df['Study ID'].values
        
        # Create stratification bins
        stratify_bins = self.create_stratification_bins(labels_df)
        
        # First split: separate test set
        train_val_ids, test_ids = train_test_split(
            subject_ids,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=stratify_bins if stratify_bins is not None else None
        )
        
        # Second split: separate train and val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        if stratify_bins is not None:
            # Get bins for train_val subset
            train_val_indices = labels_df['Study ID'].isin(train_val_ids)
            train_val_bins = stratify_bins[train_val_indices]
        else:
            train_val_bins = None
        
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=train_val_bins
        )
        
        splits = {
            'train': train_ids.tolist(),
            'val': val_ids.tolist(),
            'test': test_ids.tolist()
        }
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_ids)} subjects")
        logger.info(f"  Val: {len(val_ids)} subjects")
        logger.info(f"  Test: {len(test_ids)} subjects")
        
        return splits
    
    def create_sleepfm_split_json(self, splits: Dict[str, List[str]]):
        """Create JSON file in SleepFM format."""
        logger.info("Creating SleepFM split JSON...")
        
        sleepfm_splits = {}
        
        for split_name, subject_ids in splits.items():
            # Convert to HDF5 file paths
            hdf5_paths = [str(self.hdf5_dir / f"{subject_id}.hdf5") 
                         for subject_id in subject_ids]
            
            sleepfm_splits[split_name] = hdf5_paths
        
        # Save JSON
        output_file = self.splits_dir / "dataset_split.json"
        
        with open(output_file, 'w') as f:
            json.dump(sleepfm_splits, f, indent=2)
        
        logger.info(f"Saved SleepFM split JSON to: {output_file}")
        
        return output_file
    
    def create_subject_id_splits(self, splits: Dict[str, List[str]]):
        """Create simple subject ID split files (for reference)."""
        logger.info("Creating subject ID split files...")
        
        for split_name, subject_ids in splits.items():
            output_file = self.splits_dir / f"{split_name}_subjects.txt"
            
            with open(output_file, 'w') as f:
                for subject_id in subject_ids:
                    f.write(f"{subject_id}\n")
            
            logger.info(f"  {split_name}: {output_file}")
    
    def validate_splits(self, labels_df: pd.DataFrame, splits: Dict[str, List[str]]):
        """Validate that splits have similar distributions."""
        logger.info("Validating split distributions...")
        
        stratify_col = self.config['split']['stratify_by']
        
        if stratify_col not in labels_df.columns:
            logger.warning(f"Cannot validate: {stratify_col} not in labels")
            return
        
        # Get scores for each split
        split_scores = {}
        
        for split_name, subject_ids in splits.items():
            split_df = labels_df[labels_df['Study ID'].isin(subject_ids)]
            scores = split_df[stratify_col].dropna()
            
            split_scores[split_name] = scores
            
            logger.info(f"{split_name}:")
            logger.info(f"  Mean: {scores.mean():.3f}")
            logger.info(f"  Std: {scores.std():.3f}")
            logger.info(f"  Min: {scores.min():.3f}")
            logger.info(f"  Max: {scores.max():.3f}")
        
        # Create distribution plot
        if self.config['options']['save_validation_plots']:
            self.plot_split_distributions(split_scores)
    
    def plot_split_distributions(self, split_scores: Dict[str, pd.Series]):
        """Plot distributions for each split."""
        validation_dir = self.output_base / self.config['output']['validation_dir']
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (split_name, scores) in enumerate(split_scores.items()):
            ax = axes[idx]
            
            sns.histplot(scores, kde=True, ax=ax, bins=30)
            ax.set_title(f"{split_name.capitalize()} Set\n"
                        f"(n={len(scores)}, mean={scores.mean():.2f})")
            ax.set_xlabel(self.config['split']['stratify_by'])
            ax.set_ylabel("Count")
        
        plt.tight_layout()
        
        plot_file = validation_dir / "split_distributions.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved split distribution plots to: {plot_file}")
    
    def run(self):
        """Run split creation pipeline."""
        # Load labels
        labels_df = self.load_labels()
        
        # Verify HDF5 files
        labels_df = self.verify_hdf5_files(labels_df)
        
        if len(labels_df) == 0:
            logger.error("No valid subjects found!")
            return
        
        # Create splits
        splits = self.create_splits(labels_df)
        
        # Save in SleepFM format
        self.create_sleepfm_split_json(splits)
        
        # Save subject ID lists (for reference)
        self.create_subject_id_splits(splits)
        
        # Validate distributions
        self.validate_splits(labels_df, splits)
        
        logger.info("="*80)
        logger.info("Split creation complete!")
        logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for SleepFM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_stages_conversion.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Run split creation
    creator = STAGESSplitCreator(args.config)
    creator.run()


if __name__ == "__main__":
    main()
