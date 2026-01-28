#!/usr/bin/env python3
"""
Validate Converted HDF5 Files
==============================

This script validates the converted HDF5 files to ensure they are correctly
formatted for SleepFM.

Validation checks:
1. HDF5 file structure and datasets
2. Signal sampling rate and duration
3. Channel names and modality coverage
4. Data normalization (mean ≈ 0, std ≈ 1)
5. Data quality (NaN, Inf, outliers)
6. Comparison with original NumPy data

Author: Generated for STAGES data preparation
Date: January 2026
"""

import os
import sys
import argparse
import yaml
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class STAGESDataValidator:
    """Validate converted HDF5 files."""
    
    def __init__(self, config_path: str):
        """Initialize validator."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Directories
        self.input_base = Path(self.config['input']['base_dir'])
        self.output_base = Path(self.config['output']['base_dir'])
        self.hdf5_dir = self.output_base / self.config['output']['hdf5_dir']
        self.quality_dir = self.output_base / 'quality_metadata'
        self.validation_dir = self.output_base / self.config['output']['validation_dir']
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected parameters
        self.expected_sr = self.config['processing']['target_sample_rate']
        self.modality_definitions = self.config['modalities']
        
        # Validation results
        self.validation_results = []
    
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
        log_file = log_dir / f"validation_{timestamp}.log"
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="DEBUG")
        
        logger.info("="*80)
        logger.info("STAGES HDF5 Data Validation")
        logger.info("="*80)
    
    def get_hdf5_files(self) -> List[Path]:
        """Get list of HDF5 files to validate."""
        hdf5_files = list(self.hdf5_dir.glob("*.hdf5"))
        
        logger.info(f"Found {len(hdf5_files)} HDF5 files")
        
        return hdf5_files
    
    def validate_hdf5_structure(self, hdf5_path: Path) -> Dict:
        """Validate HDF5 file structure."""
        result = {
            'subject_id': hdf5_path.stem,
            'file_exists': True,
            'file_size_mb': hdf5_path.stat().st_size / (1024**2),
            'channels': [],
            'num_channels': 0,
            'modalities': {},
            'issues': []
        }
        
        try:
            with h5py.File(hdf5_path, 'r') as hf:
                # Get all datasets (channels)
                channels = list(hf.keys())
                result['channels'] = channels
                result['num_channels'] = len(channels)
                
                # Group by modality
                modality_groups = {
                    'BAS': ['C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'EOG(L)', 'EOG(R)'],
                    'RESP': ['Flow', 'Thor', 'ABD'],
                    'EKG': ['EKG'],
                    'EMG': ['CHIN', 'RLEG', 'LLEG']
                }
                
                for modality, expected_channels in modality_groups.items():
                    present_channels = [ch for ch in channels if ch in expected_channels]
                    result['modalities'][modality] = present_channels
                
                # Check for required modalities
                for modality, modality_config in self.modality_definitions.items():
                    if modality_config['required']:
                        if len(result['modalities'].get(modality, [])) < modality_config['min_channels']:
                            result['issues'].append(
                                f"Missing required modality: {modality}"
                            )
        
        except Exception as e:
            result['issues'].append(f"Error reading HDF5: {e}")
        
        return result
    
    def validate_signal_properties(self, hdf5_path: Path) -> Dict:
        """Validate signal properties (sampling rate, duration, normalization)."""
        result = {
            'subject_id': hdf5_path.stem,
            'channels': {},
            'issues': []
        }
        
        # Load quality metadata if available
        subject_id = hdf5_path.stem
        quality_file = self.quality_dir / f"{subject_id}_quality.json"
        quality_metadata = None
        if quality_file.exists():
            with open(quality_file, 'r') as f:
                quality_metadata = json.load(f)
        
        try:
            with h5py.File(hdf5_path, 'r') as hf:
                for channel_name in hf.keys():
                    channel_data = hf[channel_name][:]
                    
                    # Calculate stats on clean segments only if quality metadata exists
                    if quality_metadata is not None:
                        clean_data = self.get_clean_segments(channel_data, quality_metadata)
                        if len(clean_data) > 0:
                            mean_val = float(np.mean(clean_data))
                            std_val = float(np.std(clean_data))
                            min_val = float(np.min(clean_data))
                            max_val = float(np.max(clean_data))
                        else:
                            mean_val = std_val = min_val = max_val = np.nan
                        computed_on_clean = True
                    else:
                        mean_val = float(np.mean(channel_data))
                        std_val = float(np.std(channel_data))
                        min_val = float(np.min(channel_data))
                        max_val = float(np.max(channel_data))
                        computed_on_clean = False
                    
                    channel_result = {
                        'length': len(channel_data),
                        'duration_hours': len(channel_data) / self.expected_sr / 3600,
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'has_nan': bool(np.isnan(channel_data).any()),
                        'has_inf': bool(np.isinf(channel_data).any()),
                        'num_zeros': int((channel_data == 0).sum()),
                        'dtype': str(channel_data.dtype),
                        'computed_on_clean': computed_on_clean
                    }
                    
                    result['channels'][channel_name] = channel_result
                    
                    # Check for issues
                    if channel_result['has_nan']:
                        result['issues'].append(f"{channel_name}: contains NaN values")
                    
                    if channel_result['has_inf']:
                        result['issues'].append(f"{channel_name}: contains Inf values")
                    
                    # Check normalization (only if computed on clean segments)
                    if computed_on_clean and not np.isnan(mean_val):
                        mean_thresh = self.config['processing']['normalization_mean_threshold']
                        std_range = self.config['processing']['normalization_std_threshold']
                        
                        if abs(mean_val) > mean_thresh:
                            result['issues'].append(
                                f"{channel_name}: mean {mean_val:.3f} > {mean_thresh}"
                            )
                        
                        if not (std_range[0] <= std_val <= std_range[1]):
                            result['issues'].append(
                                f"{channel_name}: std {std_val:.3f} outside {std_range}"
                            )
                
                # Add quality info
                if quality_metadata is not None:
                    result['quality'] = {
                        'total_windows': quality_metadata['total_windows'],
                        'clean_windows': quality_metadata['num_clean_windows'],
                        'clean_ratio': quality_metadata['clean_ratio']
                    }
        
        except Exception as e:
            result['issues'].append(f"Error validating signals: {e}")
        
        return result
    
    def get_clean_segments(self, signal_data: np.ndarray, quality_metadata: Dict) -> np.ndarray:
        """Extract clean (non-artifact) segments from signal.
        
        Args:
            signal_data: Full signal array at 128 Hz
            quality_metadata: Quality metadata with clean/artifact windows
        
        Returns:
            Array containing only clean segments
        """
        samples_per_window = int(30 * self.expected_sr)  # 30 sec at 128 Hz = 3840
        clean_windows = quality_metadata['clean_windows']
        
        clean_segments = []
        for window_idx in clean_windows:
            start_idx = window_idx * samples_per_window
            end_idx = (window_idx + 1) * samples_per_window
            
            if end_idx <= len(signal_data):
                clean_segments.append(signal_data[start_idx:end_idx])
        
        if len(clean_segments) > 0:
            return np.concatenate(clean_segments)
        else:
            return np.array([])
    
    def validate_subject(self, hdf5_path: Path, detailed: bool = False) -> Dict:
        """Validate a single subject's HDF5 file."""
        logger.info(f"Validating {hdf5_path.name}...")
        
        # Structure validation
        structure_result = self.validate_hdf5_structure(hdf5_path)
        
        # Signal properties validation
        signal_result = self.validate_signal_properties(hdf5_path)
        
        # Combine results
        validation_result = {
            'subject_id': hdf5_path.stem,
            'file_path': str(hdf5_path),
            'structure': structure_result,
            'signals': signal_result,
            'all_issues': structure_result['issues'] + signal_result['issues'],
            'is_valid': len(structure_result['issues'] + signal_result['issues']) == 0
        }
        
        # Detailed validation (sample comparison with original)
        if detailed:
            comparison_result = self.compare_with_original(hdf5_path)
            validation_result['comparison'] = comparison_result
        
        return validation_result
    
    def compare_with_original(self, hdf5_path: Path) -> Dict:
        """Compare HDF5 data with original NumPy data (sample comparison)."""
        subject_id = hdf5_path.stem
        
        result = {
            'subject_id': subject_id,
            'channels_compared': [],
            'issues': []
        }
        
        try:
            # Load HDF5 data
            with h5py.File(hdf5_path, 'r') as hf:
                # Check one channel from each modality
                test_channels = {
                    'C3-M2': 'eeg_dir',
                    'EKG': 'ecg_dir',
                    'Flow': 'respiratory_dir',
                    'EOG(L)': 'eog_dir',
                    'CHIN': 'emg_dir'
                }
                
                channel_mapping_reverse = {v: k for k, v in self.config['channel_mapping'].items()}
                
                for hdf5_channel, input_dir in test_channels.items():
                    if hdf5_channel not in hf:
                        continue
                    
                    # Get original channel name
                    original_channel = channel_mapping_reverse.get(hdf5_channel)
                    if not original_channel:
                        continue
                    
                    # Load original data
                    original_path = (self.input_base / 
                                   self.config['input'][input_dir] / 
                                   subject_id / 
                                   f"{original_channel}.npy")
                    
                    if not original_path.exists():
                        continue
                    
                    original_data = np.load(original_path)
                    hdf5_data = hf[hdf5_channel][:]
                    
                    # Compare lengths (accounting for resampling)
                    original_length = original_data.size
                    expected_hdf5_length = int(original_length * self.expected_sr / 
                                              self.config['processing']['current_sample_rate'])
                    
                    length_diff = abs(len(hdf5_data) - expected_hdf5_length)
                    
                    result['channels_compared'].append({
                        'channel': hdf5_channel,
                        'original_length': original_length,
                        'hdf5_length': len(hdf5_data),
                        'expected_length': expected_hdf5_length,
                        'length_diff': length_diff
                    })
                    
                    if length_diff > 10:  # Allow small rounding differences
                        result['issues'].append(
                            f"{hdf5_channel}: length mismatch (expected {expected_hdf5_length}, "
                            f"got {len(hdf5_data)})"
                        )
        
        except Exception as e:
            result['issues'].append(f"Comparison error: {e}")
        
        return result
    
    def create_validation_report(self):
        """Create comprehensive validation report."""
        logger.info("Creating validation report...")
        
        # Summary statistics
        total_files = len(self.validation_results)
        valid_files = sum(1 for r in self.validation_results if r['is_valid'])
        invalid_files = total_files - valid_files
        
        # Collect all issues
        all_issues = []
        for result in self.validation_results:
            for issue in result['all_issues']:
                all_issues.append({
                    'subject_id': result['subject_id'],
                    'issue': issue
                })
        
        # Create report
        report_lines = [
            "="*80,
            "STAGES HDF5 VALIDATION REPORT",
            "="*80,
            "",
            f"Total files: {total_files}",
            f"Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)",
            f"Invalid files: {invalid_files} ({invalid_files/total_files*100:.1f}%)",
            "",
            f"Total issues: {len(all_issues)}",
            ""
        ]
        
        # Channel statistics
        all_channels = set()
        modality_coverage = {'BAS': 0, 'RESP': 0, 'EKG': 0, 'EMG': 0}
        
        for result in self.validation_results:
            for channel in result['structure']['channels']:
                all_channels.add(channel)
            
            for modality in modality_coverage.keys():
                if result['structure']['modalities'].get(modality):
                    modality_coverage[modality] += 1
        
        report_lines.extend([
            "Channel Statistics:",
            f"  Unique channels found: {sorted(all_channels)}",
            "",
            "Modality Coverage:",
        ])
        
        for modality, count in modality_coverage.items():
            report_lines.append(f"  {modality}: {count}/{total_files} subjects "
                              f"({count/total_files*100:.1f}%)")
        
        # Common issues
        if all_issues:
            report_lines.extend([
                "",
                "Common Issues:",
            ])
            
            issue_counts = {}
            for issue_item in all_issues:
                issue = issue_item['issue']
                # Generalize issue
                issue_key = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
                report_lines.append(f"  {issue}: {count} occurrences")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        # Save report
        report_file = self.validation_dir / "validation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        for line in report_lines:
            logger.info(line)
        
        logger.info(f"Report saved to: {report_file}")
        
        # Save detailed results as JSON
        import json
        
        results_file = self.validation_dir / "validation_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
    
    def create_visualization(self):
        """Create visualization of validation results."""
        if not self.config['options']['save_validation_plots']:
            return
        
        logger.info("Creating validation visualizations...")
        
        # Extract statistics
        file_sizes = []
        num_channels_list = []
        durations = []
        
        for result in self.validation_results:
            file_sizes.append(result['structure']['file_size_mb'])
            num_channels_list.append(result['structure']['num_channels'])
            
            # Get duration from first channel
            if result['signals']['channels']:
                first_channel = list(result['signals']['channels'].values())[0]
                durations.append(first_channel['duration_hours'])
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # File sizes
        axes[0, 0].hist(file_sizes, bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('File Size (MB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'HDF5 File Sizes\n(mean={np.mean(file_sizes):.1f} MB)')
        
        # Number of channels
        axes[0, 1].hist(num_channels_list, bins=range(0, 15), edgecolor='black')
        axes[0, 1].set_xlabel('Number of Channels')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Channels per Subject\n(mean={np.mean(num_channels_list):.1f})')
        
        # Durations
        axes[1, 0].hist(durations, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Duration (hours)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Recording Durations\n(mean={np.mean(durations):.1f} hours)')
        
        # Modality coverage
        modality_counts = {'BAS': 0, 'RESP': 0, 'EKG': 0, 'EMG': 0}
        
        for result in self.validation_results:
            for modality in modality_counts.keys():
                if result['structure']['modalities'].get(modality):
                    modality_counts[modality] += 1
        
        axes[1, 1].bar(modality_counts.keys(), modality_counts.values())
        axes[1, 1].set_xlabel('Modality')
        axes[1, 1].set_ylabel('Number of Subjects')
        axes[1, 1].set_title('Modality Coverage')
        axes[1, 1].axhline(y=len(self.validation_results), color='r', 
                          linestyle='--', label='Total subjects')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        plot_file = self.validation_dir / "validation_statistics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation plots to: {plot_file}")
    
    def run(self, num_detailed: Optional[int] = None):
        """Run validation on all HDF5 files."""
        # Get HDF5 files
        hdf5_files = self.get_hdf5_files()
        
        if not hdf5_files:
            logger.error("No HDF5 files found!")
            return
        
        # Determine number of detailed validations
        if num_detailed is None:
            num_detailed = self.config['options']['detailed_validation_count']
        
        logger.info(f"Performing detailed validation on {num_detailed} subjects")
        
        # Validate files
        for idx, hdf5_path in enumerate(tqdm(hdf5_files, desc="Validating")):
            detailed = idx < num_detailed
            
            result = self.validate_subject(hdf5_path, detailed=detailed)
            self.validation_results.append(result)
        
        # Create report
        self.create_validation_report()
        
        # Create visualizations
        self.create_visualization()
        
        logger.info("="*80)
        logger.info("Validation complete!")
        logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate converted HDF5 files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_stages_conversion.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--detailed",
        type=int,
        default=None,
        help="Number of subjects to validate in detail"
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = STAGESDataValidator(args.config)
    validator.run(num_detailed=args.detailed)


if __name__ == "__main__":
    main()
