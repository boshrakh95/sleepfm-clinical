#!/usr/bin/env python3
"""
Convert STAGES Preprocessed Data to SleepFM HDF5 Format
========================================================

This script converts CogPSGFormerPP preprocessed data (30-sec segmented NumPy arrays)
into continuous HDF5 files compatible with SleepFM.

Main operations:
1. Load 30-sec segmented data from NumPy arrays
2. Concatenate segments into continuous signals
3. Resample from 100 Hz to 128 Hz
4. Rename channels to match SleepFM conventions
5. Save as HDF5 with proper compression

Author: Generated for STAGES data preparation
Date: January 2026
"""

import os
import sys
import argparse
import yaml
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import interp1d
import multiprocessing as mp
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


class STAGEStoSleepFMConverter:
    """Convert STAGES preprocessed data to SleepFM HDF5 format."""
    
    def __init__(self, config_path: str):
        """Initialize converter with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        self.channel_mapping = self.config['channel_mapping']
        self.modality_definitions = self.config['modalities']
        
        # Processing parameters
        self.current_sr = self.config['processing']['current_sample_rate']
        self.target_sr = self.config['processing']['target_sample_rate']
        self.resampling_method = self.config['processing']['resampling_method']
        self.dtype = np.float16 if self.config['processing']['dtype'] == 'float16' else np.float32
        
        # Directories
        self.input_base = Path(self.config['input']['base_dir'])
        self.output_base = Path(self.config['output']['base_dir'])
        self.hdf5_output = self.output_base / self.config['output']['hdf5_dir']
        
        # Statistics
        self.stats = {
            'total_subjects': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['output']['base_dir']) / self.config['output']['logs_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"conversion_{timestamp}.log"
        
        logger.remove()
        logger.add(sys.stderr, level="INFO" if self.config['options']['verbose'] else "WARNING")
        logger.add(log_file, rotation="100 MB", retention="10 days", level="DEBUG")
        
        logger.info("="*80)
        logger.info("STAGES to SleepFM Data Conversion")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
    
    def setup_directories(self):
        """Create output directories."""
        base_dir = Path(self.config['output']['base_dir'])
        
        for subdir in ['hdf5_dir', 'logs_dir', 'validation_dir']:
            dir_path = base_dir / self.config['output'][subdir]
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def get_subject_list(self) -> List[str]:
        """Get list of subjects to process."""
        # Get all subjects from EEG directory
        eeg_dir = self.input_base / self.config['input']['eeg_dir']
        all_subjects = [d.name for d in eeg_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(all_subjects)} subjects in EEG directory")
        
        # Apply include/exclude filters
        if self.config['subjects']['include']:
            all_subjects = [s for s in all_subjects if s in self.config['subjects']['include']]
            logger.info(f"Filtered to {len(all_subjects)} included subjects")
        
        if self.config['subjects']['exclude']:
            all_subjects = [s for s in all_subjects if s not in self.config['subjects']['exclude']]
            logger.info(f"After exclusions: {len(all_subjects)} subjects")
        
        # Pilot mode
        if self.config['subjects']['pilot_mode']:
            pilot_count = self.config['subjects']['pilot_count']
            all_subjects = all_subjects[:pilot_count]
            logger.warning(f"PILOT MODE: Processing only {len(all_subjects)} subjects")
        
        # Filter by channel availability
        all_subjects = self.filter_subjects_by_channels(all_subjects)
        
        return sorted(all_subjects)
    
    def filter_subjects_by_channels(self, subjects: List[str]) -> List[str]:
        """Filter subjects that have all required modalities."""
        logger.info("Filtering subjects by channel availability...")
        
        valid_subjects = []
        
        for subject in tqdm(subjects, desc="Checking channels"):
            if self.subject_has_all_modalities(subject):
                valid_subjects.append(subject)
        
        logger.info(f"Valid subjects with all modalities: {len(valid_subjects)}/{len(subjects)}")
        
        return valid_subjects
    
    def subject_has_all_modalities(self, subject_id: str) -> bool:
        """Check if subject has at least one channel in each required modality."""
        modality_dirs = {
            'BAS': ['eeg_dir', 'eog_dir'],  # Check both EEG and EOG
            'RESP': ['respiratory_dir'],
            'EKG': ['ecg_dir'],
            'EMG': ['emg_dir']
        }
        
        for modality, dir_keys in modality_dirs.items():
            has_channel = False
            
            for dir_key in dir_keys:
                dir_path = self.input_base / self.config['input'][dir_key] / subject_id
                
                if dir_path.exists():
                    # Check for .npy files (excluding normstats and metadata)
                    npy_files = [f for f in dir_path.glob("*.npy")]
                    if npy_files:
                        has_channel = True
                        break
            
            if not has_channel:
                logger.debug(f"Subject {subject_id} missing {modality} modality")
                return False
        
        return True
    
    def load_channel_data(self, subject_id: str, channel_name: str, modality_type: str) -> Optional[np.ndarray]:
        """Load channel data from NumPy file."""
        # Determine which directory to look in
        dir_mapping = {
            'eeg': 'eeg_dir',
            'eog': 'eog_dir',
            'ecg': 'ecg_dir',
            'resp': 'respiratory_dir',
            'emg': 'emg_dir'
        }
        
        # Try to find the file
        for dir_key in dir_mapping.values():
            file_path = self.input_base / self.config['input'][dir_key] / subject_id / f"{channel_name}.npy"
            
            if file_path.exists():
                try:
                    data = np.load(file_path)
                    return data
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    return None
        
        return None
    
    def concatenate_segments(self, segmented_data: np.ndarray) -> np.ndarray:
        """Concatenate 30-sec segments into continuous signal.
        
        Args:
            segmented_data: Array of shape [num_segments, samples_per_segment]
        
        Returns:
            Continuous signal of shape [total_samples]
        """
        if segmented_data.ndim == 2:
            # Flatten along time axis
            continuous = segmented_data.reshape(-1)
        else:
            continuous = segmented_data
        
        return continuous
    
    def resample_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Resample signal from 100 Hz to 128 Hz.
        
        Args:
            signal_data: Signal at 100 Hz
        
        Returns:
            Signal at 128 Hz
        """
        current_length = len(signal_data)
        current_duration = current_length / self.current_sr
        target_length = int(current_duration * self.target_sr)
        
        if self.resampling_method == 'linear':
            # Linear interpolation (fast, good quality)
            old_time = np.arange(current_length) / self.current_sr
            new_time = np.arange(target_length) / self.target_sr
            
            interpolator = interp1d(old_time, signal_data, kind='linear', 
                                   fill_value='extrapolate')
            resampled = interpolator(new_time)
            
        elif self.resampling_method == 'cubic':
            # Cubic spline interpolation (higher quality, slower)
            old_time = np.arange(current_length) / self.current_sr
            new_time = np.arange(target_length) / self.target_sr
            
            interpolator = interp1d(old_time, signal_data, kind='cubic', 
                                   fill_value='extrapolate')
            resampled = interpolator(new_time)
            
        elif self.resampling_method == 'fft':
            # FFT-based resampling (most accurate for band-limited signals)
            resampled = signal.resample(signal_data, target_length)
        
        else:
            raise ValueError(f"Unknown resampling method: {self.resampling_method}")
        
        return resampled.astype(self.dtype)
    
    def validate_signal(self, signal_data: np.ndarray, channel_name: str) -> Tuple[bool, str]:
        """Validate signal quality.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for NaNs
        if self.config['advanced']['check_for_nans']:
            if np.isnan(signal_data).any():
                return False, f"Signal contains NaN values"
        
        # Check for Infs
        if self.config['advanced']['check_for_infs']:
            if np.isinf(signal_data).any():
                return False, f"Signal contains Inf values"
        
        # Check normalization
        if self.config['advanced']['verify_normalization']:
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            
            mean_thresh = self.config['processing']['normalization_mean_threshold']
            std_range = self.config['processing']['normalization_std_threshold']
            
            if abs(mean) > mean_thresh:
                logger.warning(f"{channel_name}: Mean {mean:.3f} outside threshold ±{mean_thresh}")
            
            if not (std_range[0] <= std <= std_range[1]):
                logger.warning(f"{channel_name}: Std {std:.3f} outside range {std_range}")
        
        return True, ""
    
    def convert_subject(self, subject_id: str) -> bool:
        """Convert one subject's data to HDF5 format.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            True if successful, False otherwise
        """
        output_file = self.hdf5_output / f"{subject_id}.hdf5"
        
        # Skip if exists and skip_existing is True
        if output_file.exists() and self.config['options']['skip_existing']:
            logger.info(f"Skipping {subject_id} (already exists)")
            self.stats['skipped'] += 1
            return True
        
        logger.info(f"Processing subject: {subject_id}")
        
        try:
            # Collect all channels for this subject
            channels_data = {}
            
            # Process each channel
            for original_name, sleepfm_name in self.channel_mapping.items():
                # Load segmented data
                segmented = self.load_channel_data(subject_id, original_name, 'unknown')
                
                if segmented is None:
                    logger.debug(f"  {original_name} -> not found")
                    continue
                
                logger.debug(f"  {original_name} -> {sleepfm_name}: shape {segmented.shape}")
                
                # Concatenate segments
                continuous = self.concatenate_segments(segmented)
                
                # Resample to 128 Hz
                resampled = self.resample_signal(continuous)
                
                # Validate
                is_valid, error_msg = self.validate_signal(resampled, sleepfm_name)
                if not is_valid:
                    logger.warning(f"  {sleepfm_name}: {error_msg}")
                    continue
                
                channels_data[sleepfm_name] = resampled
            
            # Check if we have all required modalities
            if not self.validate_modalities(channels_data):
                logger.error(f"Subject {subject_id} missing required modalities")
                return False
            
            # Save to HDF5
            with h5py.File(output_file, 'w') as hf:
                for channel_name, channel_data in channels_data.items():
                    hf.create_dataset(
                        channel_name,
                        data=channel_data,
                        dtype=self.dtype,
                        compression=self.config['processing']['compression'],
                        compression_opts=self.config['processing']['compression_opts'],
                        chunks=(self.config['processing']['chunk_size'],)
                    )
            
            logger.info(f"✓ Saved {subject_id}: {len(channels_data)} channels, "
                       f"{len(list(channels_data.values())[0])} samples")
            
            self.stats['successful'] += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Error processing {subject_id}: {e}")
            self.stats['failed'] += 1
            self.stats['errors'].append((subject_id, str(e)))
            return False
    
    def validate_modalities(self, channels_data: Dict[str, np.ndarray]) -> bool:
        """Check if all required modalities are present."""
        modality_groups = {
            'BAS': ['C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'EOG(L)', 'EOG(R)'],
            'RESP': ['Flow', 'Thor', 'ABD'],
            'EKG': ['EKG'],
            'EMG': ['CHIN', 'RLEG', 'LLEG']
        }
        
        for modality, channel_list in modality_groups.items():
            modality_config = self.modality_definitions[modality]
            
            if not modality_config['required']:
                continue
            
            # Count how many channels from this modality are present
            present_channels = [ch for ch in channel_list if ch in channels_data]
            
            if len(present_channels) < modality_config['min_channels']:
                logger.debug(f"Modality {modality}: only {len(present_channels)} channels "
                           f"(need {modality_config['min_channels']})")
                return False
        
        return True
    
    def process_all_subjects(self):
        """Process all subjects."""
        subjects = self.get_subject_list()
        self.stats['total_subjects'] = len(subjects)
        
        logger.info(f"Processing {len(subjects)} subjects...")
        logger.info(f"Output directory: {self.hdf5_output}")
        
        # Process subjects
        for subject_id in tqdm(subjects, desc="Converting subjects"):
            self.convert_subject(subject_id)
            
            # Clear cache if configured
            if self.config['advanced']['clear_cache_per_subject']:
                import gc
                gc.collect()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print conversion summary."""
        logger.info("="*80)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total subjects: {self.stats['total_subjects']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        
        if self.stats['errors']:
            logger.warning(f"\nErrors encountered:")
            for subject_id, error in self.stats['errors'][:10]:  # Show first 10
                logger.warning(f"  {subject_id}: {error}")
            if len(self.stats['errors']) > 10:
                logger.warning(f"  ... and {len(self.stats['errors'])-10} more")
        
        # Save summary to file
        summary_file = (Path(self.config['output']['base_dir']) / 
                       self.config['output']['logs_dir'] / 
                       f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(summary_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert STAGES preprocessed data to SleepFM HDF5 format"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_stages_conversion.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Run conversion
    converter = STAGEStoSleepFMConverter(args.config)
    converter.process_all_subjects()


if __name__ == "__main__":
    main()
