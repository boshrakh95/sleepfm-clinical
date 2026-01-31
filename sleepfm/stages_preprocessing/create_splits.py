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
        
        # Exclude subjects from config
        excluded_subjects = self.config['subjects'].get('exclude', [])
        if excluded_subjects:
            n_before = len(df)
            df = df[~df['Study ID'].isin(excluded_subjects)]
            n_after = len(df)
            logger.info(f"Excluded {n_before - n_after} subjects from config: {n_before} → {n_after}")
        
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
    
    def create_stratification_bins(self, labels_df: pd.DataFrame, stratify_col: str) -> np.ndarray:
        """Create stratification bins based on cognitive score.
        
        Args:
            labels_df: DataFrame with labels
            stratify_col: Column name to stratify by
        
        Returns:
            Array of bin indices for each subject
        """
        if stratify_col not in labels_df.columns:
            logger.warning(f"Stratification column '{stratify_col}' not found. "
                          "Using random split.")
            return None
        
        # Remove NaN values
        valid_mask = labels_df[stratify_col].notna()
        
        if valid_mask.sum() == 0:
            logger.warning(f"No valid values for {stratify_col}. Using random split.")
            return None
        
        if valid_mask.sum() < len(labels_df):
            logger.warning(f"{(~valid_mask).sum()} subjects have missing {stratify_col}. "
                          "These will be excluded from this split.")
        
        # Create bins only for valid subjects
        n_bins = self.config['split']['stratify_bins']
        
        valid_scores = labels_df.loc[valid_mask, stratify_col].values
        
        # Handle case where all values are the same
        if len(np.unique(valid_scores)) < n_bins:
            logger.warning(f"Only {len(np.unique(valid_scores))} unique values for {stratify_col}. "
                          f"Reducing bins to {len(np.unique(valid_scores))}")
            n_bins = len(np.unique(valid_scores))
        
        quantiles = np.percentile(valid_scores, np.linspace(0, 100, n_bins+1))
        bins = np.digitize(valid_scores, quantiles[1:-1])
        
        logger.info(f"Created {n_bins} stratification bins for {stratify_col}")
        
        # Log bin distribution
        for bin_idx in range(n_bins):
            count = (bins == bin_idx).sum()
            logger.info(f"  Bin {bin_idx}: {count} subjects")
        
        return bins
    
    def create_splits_for_target(self, labels_df: pd.DataFrame, target: str) -> Dict[str, List[str]]:
        """Create stratified train/val/test splits for a specific cognitive target.
        
        Args:
            labels_df: DataFrame with all labels
            target: Cognitive target column name
        
        Returns:
            Dictionary with train/val/test subject IDs
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Creating splits for target: {target}")
        logger.info(f"{'='*80}")
        
        # Filter to subjects with valid data for this target
        valid_mask = labels_df[target].notna()
        target_df = labels_df[valid_mask].copy()
        
        if len(target_df) == 0:
            logger.error(f"No subjects with valid {target} data!")
            return None
        
        logger.info(f"Subjects with valid {target} data: {len(target_df)}/{len(labels_df)}")
        
        # Get subject IDs
        subject_ids = target_df['Study ID'].values
        
        # Create stratification bins
        stratify_bins = self.create_stratification_bins(target_df, target)
        
        logger.info(f"Creating splits:")
        logger.info(f"  Train: {self.train_ratio*100:.0f}%")
        logger.info(f"  Val: {self.val_ratio*100:.0f}%")
        logger.info(f"  Test: {self.test_ratio*100:.0f}%")
        
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
            train_val_mask = target_df['Study ID'].isin(train_val_ids)
            train_val_bins = stratify_bins[train_val_mask.values]
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
        
        logger.info(f"Split sizes for {target}:")
        logger.info(f"  Train: {len(train_ids)} subjects ({len(train_ids)/len(subject_ids)*100:.1f}%)")
        logger.info(f"  Val: {len(val_ids)} subjects ({len(val_ids)/len(subject_ids)*100:.1f}%)")
        logger.info(f"  Test: {len(test_ids)} subjects ({len(test_ids)/len(subject_ids)*100:.1f}%)")
        
        # Validate distributions
        self.validate_target_splits(target_df, splits, target)
        
        return splits
    
    def create_sleepfm_split_json(self, splits: Dict[str, List[str]], target: str = None):
        """Create JSON file in SleepFM format.
        
        Args:
            splits: Dictionary with train/val/test subject IDs
            target: Cognitive target name (for filename), None for generic split
        """
        logger.info("Creating SleepFM split JSON...")
        
        sleepfm_splits = {}
        
        for split_name, subject_ids in splits.items():
            # Convert to HDF5 file paths
            hdf5_paths = [str(self.hdf5_dir / f"{subject_id}.hdf5") 
                         for subject_id in subject_ids]
            
            sleepfm_splits[split_name] = hdf5_paths
        
        # Save JSON
        if target:
            output_file = self.splits_dir / f"dataset_split_{target}.json"
        else:
            output_file = self.splits_dir / "dataset_split.json"
        
        with open(output_file, 'w') as f:
            json.dump(sleepfm_splits, f, indent=2)
        
        logger.info(f"Saved SleepFM split JSON to: {output_file}")
        
        return output_file
    
    def create_subject_id_splits(self, splits: Dict[str, List[str]], target: str = None):
        """Create simple subject ID split files (for reference).
        
        Args:
            splits: Dictionary with train/val/test subject IDs
            target: Cognitive target name (for filename), None for generic split
        """
        logger.info("Creating subject ID split files...")
        
        for split_name, subject_ids in splits.items():
            if target:
                output_file = self.splits_dir / f"{split_name}_subjects_{target}.txt"
            else:
                otarget_splits(self, labels_df: pd.DataFrame, splits: Dict[str, List[str]], target: str):
        """Validate that splits have similar distributions for a target.
        
        Args:
            labels_df: DataFrame with labels (already filtered to valid subjects)
            splits: Dictionary with train/val/test subject IDs
            target: Cognitive target column name
        """
        logger.info(f"Validating split distributions for {target}...")
        
        # Get scores for each split
        split_scores = {}
        
        for split_name, subject_ids in splits.items():
            split_df = labels_df[labels_df['Study ID'].isin(subject_ids)]
            scores = split_df[target].dropna()
            
            split_scores[split_name] = scores
            
            logger.info(f"  {split_name}:")
            logger.info(f"    Mean: {scores.mean():.3f} ± {scores.std():.3f}")
            logger.info(f"    Range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Create distribution plot
        if self.config['options']['save_validation_plots']:
            self.plot_split_distributions(split_scores, target
            logger.info(f"  Mean: {scores.mean():.3f}")
            logger.info(f"  Std: {scores.std():.3f}")
            logger.info(f"  Min: {scores.min():.3f}")
            logger.info(f"  Max: {scores.max():.3f}")
        
        # Create distribution plot
        if self.config['options']['save_validation_plots']:
            self.plot_split_distributions(split_scores)
    
    def plot_split_distributions(self, split_scores: Dict[str, pd.Series]):
        """Plot distributions for each split."""
        validation_dir = self.output_base / self.config['output']['valida, target: str):
        """Plot distributions for each split.
        
        Args:
            split_scores: Dictionary with split names and score series
            target: Cognitive target name
        """
        validation_dir = self.output_base / self.config['output']['validation_dir']
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (split_name, scores) in enumerate(split_scores.items()):
            ax = axes[idx]
            
            sns.histplot(scores, kde=True, ax=ax, bins=30)
            ax.set_title(f"{split_name.capitalize()} Set\n"
                        f"(n={len(scores)}, mean={scores.mean():.2f}±{scores.std():.2f})")
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
        
        plt.suptitle(f"Split Distributions: {target}", fontsize=14, y=1.02)
        plt.tight_layout()
        
        plot_file = validation_dir / f"split_distributions_{target}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"  Saved plot
        """Run split creation pipeline."""
        # Load labels
        labels_df = self.load_labels()
        
        # Verify HDF5 files
        laGet cognitive targets from config
        cognitive_targets = self.config['labels']['cognitive_scores']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Creating separate splits for {len(cognitive_targets)} cognitive targets")
        logger.info(f"{'='*80}\n")
        
        # Create summary of subjects per target
        summary_data = []
        
        for target in cognitive_targets:
            if target not in labels_df.columns:
                logger.warning(f"Target '{target}' not found in labels, skipping")
                continue
            
            # Create splits for this target
            splits = self.create_splits_for_target(labels_df, target)
            
            if splits is None:
                logger.error(f"Failed to create splits for {target}")
                continue
            
            # Save in SleepFM format
            self.create_sleepfm_split_json(splits, target=target)
            
            # Save subject ID lists (for reference)
            self.create_subject_id_splits(splits, target=target)
            
            # Track summary
            n_valid = labels_df[target].notna().sum()
            summary_data.append({
                'target': target,
                'total_subjects': n_valid,
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            })
        
        # Print summary table
        self.print_summary_table(summary_data)
        
        logger.info("\n" + "="*80)
        logger.info("Split creation complete!")
        logger.info("="*80)
    
    def print_summary_table(self, summary_data: List[Dict]):
        """Print summary table of splits across all targets."""
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY: Splits Across All Targets")
        logger.info(f"{'='*80}\n")
        
        # Header
        logger.info(f"{'Target':<35} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
        logger.info("-" * 80)
        
        # Rows
        for row in summary_data:
            logger.info(f"{row['target']:<35} "
                       f"{row['total_subjects']:>8} "
                       f"{row['train']:>8} "
                       f"{row['val']:>8} "
                       f"{row['test']:>8}")
        
        logger.info(""
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
