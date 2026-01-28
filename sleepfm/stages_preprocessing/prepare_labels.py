#!/usr/bin/env python3
"""
Prepare Labels for SleepFM Training
====================================

This script prepares cognitive targets and demographics files in the format
expected by SleepFM.

Operations:
1. Load cognitive_targets.csv and demographics_final.csv
2. Filter to subjects that have HDF5 files
3. Normalize demographics (age, encode sex)
4. Create unified labels CSV with "Study ID" column
5. Validate label distributions

Author: Generated for STAGES data preparation
Date: January 2026
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class STAGESLabelPreparator:
    """Prepare labels for SleepFM training."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Directories
        self.input_base = Path(self.config['input']['base_dir'])
        self.output_base = Path(self.config['output']['base_dir'])
        self.hdf5_dir = self.output_base / self.config['output']['hdf5_dir']
        self.labels_dir = self.output_base / self.config['output']['labels_dir']
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging."""
        log_dir = Path(self.config['output']['base_dir']) / self.config['output']['logs_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"label_preparation_{timestamp}.log"
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="DEBUG")
        
        logger.info("="*80)
        logger.info("STAGES Label Preparation")
        logger.info("="*80)
    
    def get_available_subjects(self) -> List[str]:
        """Get list of subjects with HDF5 files."""
        hdf5_files = list(self.hdf5_dir.glob("*.hdf5"))
        subjects = [f.stem for f in hdf5_files]
        
        logger.info(f"Found {len(subjects)} subjects with HDF5 files")
        
        return subjects
    
    def load_cognitive_targets(self) -> pd.DataFrame:
        """Load cognitive targets CSV."""
        cognitive_file = self.input_base / self.config['input']['cognitive_targets']
        
        logger.info(f"Loading cognitive targets from: {cognitive_file}")
        
        df = pd.read_csv(cognitive_file)
        logger.info(f"Loaded {len(df)} subjects with cognitive scores")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def load_demographics(self) -> pd.DataFrame:
        """Load demographics CSV."""
        demographics_file = self.input_base / self.config['input']['demographics']
        
        logger.info(f"Loading demographics from: {demographics_file}")
        
        df = pd.read_csv(demographics_file)
        logger.info(f"Loaded {len(df)} subjects with demographics")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def prepare_cognitive_labels(self, available_subjects: List[str]) -> pd.DataFrame:
        """Prepare cognitive targets in SleepFM format."""
        # Load raw data
        cognitive_df = self.load_cognitive_targets()
        
        # Get subject ID column name (flexible)
        id_columns = [col for col in cognitive_df.columns if 'id' in col.lower() or 'subject' in col.lower()]
        if not id_columns:
            # Assume first column is ID
            id_col = cognitive_df.columns[0]
        else:
            id_col = id_columns[0]
        
        logger.info(f"Using '{id_col}' as subject identifier")
        
        # Rename to "Study ID" for SleepFM compatibility
        cognitive_df = cognitive_df.rename(columns={id_col: 'Study ID'})
        
        # Convert Study ID to string
        cognitive_df['Study ID'] = cognitive_df['Study ID'].astype(str)
        
        # Filter to available subjects
        cognitive_df = cognitive_df[cognitive_df['Study ID'].isin(available_subjects)]
        
        logger.info(f"Retained {len(cognitive_df)} subjects with HDF5 files")
        
        # Select cognitive score columns
        score_columns = self.config['labels']['cognitive_scores']
        
        # Check which columns exist
        available_scores = [col for col in score_columns if col in cognitive_df.columns]
        missing_scores = [col for col in score_columns if col not in cognitive_df.columns]
        
        if missing_scores:
            logger.warning(f"Missing cognitive scores: {missing_scores}")
        
        logger.info(f"Available cognitive scores: {available_scores}")
        
        # Select columns
        output_columns = ['Study ID'] + available_scores
        cognitive_df = cognitive_df[output_columns]
        
        # Log statistics
        for col in available_scores:
            mean_val = cognitive_df[col].mean()
            std_val = cognitive_df[col].std()
            missing_pct = cognitive_df[col].isna().sum() / len(cognitive_df) * 100
            
            logger.info(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, "
                       f"missing={missing_pct:.1f}%")
        
        return cognitive_df
    
    def prepare_demographics(self, available_subjects: List[str]) -> pd.DataFrame:
        """Prepare demographics in SleepFM format."""
        # Load raw data
        demographics_df = self.load_demographics()
        
        # Get subject ID column
        id_columns = [col for col in demographics_df.columns if 'id' in col.lower() or 'subject' in col.lower()]
        if not id_columns:
            id_col = demographics_df.columns[0]
        else:
            id_col = id_columns[0]
        
        logger.info(f"Using '{id_col}' as subject identifier")
        
        # Rename to "Study ID"
        demographics_df = demographics_df.rename(columns={id_col: 'Study ID'})
        demographics_df['Study ID'] = demographics_df['Study ID'].astype(str)
        
        # Filter to available subjects
        demographics_df = demographics_df[demographics_df['Study ID'].isin(available_subjects)]
        
        logger.info(f"Retained {len(demographics_df)} subjects with HDF5 files")
        
        # Select demographic features
        demo_features = self.config['labels']['demographics_features']
        
        # Check which columns exist
        available_demo = [col for col in demo_features if col in demographics_df.columns]
        missing_demo = [col for col in demo_features if col not in demographics_df.columns]
        
        if missing_demo:
            logger.warning(f"Missing demographics: {missing_demo}")
        
        # Process demographics
        output_df = demographics_df[['Study ID']].copy()
        
        for col in available_demo:
            if 'age' in col.lower():
                # Normalize age if configured
                if self.config['labels']['normalize_demographics']:
                    age_mean = demographics_df[col].mean()
                    age_std = demographics_df[col].std()
                    output_df[col] = (demographics_df[col] - age_mean) / age_std
                    logger.info(f"  {col}: normalized (mean={age_mean:.1f}, std={age_std:.1f})")
                else:
                    output_df[col] = demographics_df[col]
                    logger.info(f"  {col}: raw values")
            
            elif 'sex' in col.lower() or 'gender' in col.lower():
                # Encode sex as numeric
                gender_encoding = self.config['labels']['gender_encoding']
                
                # Handle different possible values
                sex_map = {}
                for value in demographics_df[col].unique():
                    if pd.isna(value):
                        continue
                    
                    value_lower = str(value).lower()
                    if 'male' in value_lower and 'female' not in value_lower:
                        sex_map[value] = gender_encoding['male']
                    elif 'female' in value_lower:
                        sex_map[value] = gender_encoding['female']
                    elif value in ['1', 1, '2', 2]:
                        # Already numeric - keep as is
                        sex_map[value] = int(value) if value in ['1', '2'] else value
                
                output_df[col] = demographics_df[col].map(sex_map)
                logger.info(f"  {col}: encoded as {sex_map}")
            
            else:
                # Keep other demographics as is
                output_df[col] = demographics_df[col]
                logger.info(f"  {col}: raw values")
        
        return output_df
    
    def merge_labels_and_demographics(self, 
                                      cognitive_df: pd.DataFrame, 
                                      demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Merge cognitive and demographic data."""
        logger.info("Merging cognitive scores and demographics...")
        
        merged_df = pd.merge(cognitive_df, demographics_df, on='Study ID', how='inner')
        
        logger.info(f"Merged dataset: {len(merged_df)} subjects")
        
        # Check for missing values
        for col in merged_df.columns:
            if col == 'Study ID':
                continue
            
            missing_count = merged_df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"  {col}: {missing_count} missing values "
                             f"({missing_count/len(merged_df)*100:.1f}%)")
        
        return merged_df
    
    def create_visualization(self, labels_df: pd.DataFrame):
        """Create visualization of label distributions."""
        if not self.config['options']['save_validation_plots']:
            return
        
        logger.info("Creating label distribution plots...")
        
        validation_dir = self.output_base / self.config['output']['validation_dir']
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Get numeric columns (exclude Study ID)
        numeric_cols = [col for col in labels_df.columns 
                       if col != 'Study ID' and pd.api.types.is_numeric_dtype(labels_df[col])]
        
        # Create subplots
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Remove NaN values for plotting
            data = labels_df[col].dropna()
            
            # Histogram with KDE
            sns.histplot(data, kde=True, ax=ax, bins=30)
            ax.set_title(f"{col}\n(n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        plot_file = validation_dir / "label_distributions.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distribution plots to: {plot_file}")
    
    def save_labels(self, labels_df: pd.DataFrame):
        """Save processed labels."""
        output_file = self.labels_dir / "labels_with_demographics.csv"
        
        labels_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved labels to: {output_file}")
        logger.info(f"  Subjects: {len(labels_df)}")
        logger.info(f"  Features: {len(labels_df.columns)-1}")  # Excluding Study ID
    
    def run(self):
        """Run label preparation pipeline."""
        # Get subjects with HDF5 files
        available_subjects = self.get_available_subjects()
        
        if not available_subjects:
            logger.error("No HDF5 files found! Run conversion first.")
            return
        
        # Prepare cognitive labels
        cognitive_df = self.prepare_cognitive_labels(available_subjects)
        
        # Prepare demographics
        demographics_df = self.prepare_demographics(available_subjects)
        
        # Merge
        labels_df = self.merge_labels_and_demographics(cognitive_df, demographics_df)
        
        # Create visualizations
        self.create_visualization(labels_df)
        
        # Save
        self.save_labels(labels_df)
        
        logger.info("="*80)
        logger.info("Label preparation complete!")
        logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare labels for SleepFM training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_stages_conversion.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Run preparation
    preparator = STAGESLabelPreparator(args.config)
    preparator.run()


if __name__ == "__main__":
    main()
