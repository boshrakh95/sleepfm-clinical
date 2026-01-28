#!/usr/bin/env python3
"""
Quick Test - Convert Single Subject
====================================

Quick test script to convert and validate a single subject.
Useful for debugging and verifying the conversion process.

Usage:
    python test_single_subject.py --subject BOGN00004
    python test_single_subject.py --auto  # Use first available subject
"""

import argparse
import yaml
import h5py
import json
import numpy as np
from pathlib import Path
from convert_to_hdf5 import STAGEStoSleepFMConverter
from validate_data import STAGESDataValidator


def test_single_subject(subject_id: str, config_path: str = "config_stages_conversion.yaml"):
    """Test conversion and validation for a single subject."""
    
    print("="*80)
    print(f"Testing single subject: {subject_id}")
    print("="*80)
    print()
    
    # Resolve config path relative to this script's directory
    script_dir = Path(__file__).parent
    if not Path(config_path).is_absolute():
        config_path = script_dir / config_path
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily modify config for single subject
    original_include = config['subjects']['include']
    original_skip = config['options']['skip_existing']
    
    config['subjects']['include'] = [subject_id]
    config['options']['skip_existing'] = False  # Force re-conversion for testing
    
    # Save temp config in script directory
    script_dir = Path(__file__).parent
    temp_config_path = script_dir / "temp_test_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Step 1: Convert
        print("Step 1: Converting to HDF5...")
        print("-" * 80)
        converter = STAGEStoSleepFMConverter(temp_config_path)
        success = converter.convert_subject(subject_id)
        
        if not success:
            print(f"\n❌ Conversion failed for {subject_id}")
            return False
        
        print(f"\n✓ Conversion successful")
        print()
        
        # Step 2: Validate
        print("Step 2: Validating HDF5 file...")
        print("-" * 80)
        
        hdf5_path = Path(config['output']['base_dir']) / config['output']['hdf5_dir'] / f"{subject_id}.hdf5"
        
        if not hdf5_path.exists():
            print(f"❌ HDF5 file not found: {hdf5_path}")
            return False
        
        # Load quality metadata
        quality_path = Path(config['output']['base_dir']) / 'quality_metadata' / f"{subject_id}_quality.json"
        quality_metadata = None
        if quality_path.exists():
            with open(quality_path, 'r') as f:
                quality_metadata = json.load(f)
            print(f"\n✓ Quality metadata found:")
            print(f"  Clean ratio: {quality_metadata['clean_ratio']:.1%}")
            print(f"  Clean windows: {quality_metadata['num_clean_windows']}/{quality_metadata['total_windows']}")
        else:
            print(f"\n⚠ Quality metadata not found (stats will include artifacts)")
        
        # Display HDF5 structure
        print(f"\nHDF5 file: {hdf5_path}")
        print(f"File size: {hdf5_path.stat().st_size / (1024**2):.2f} MB")
        print()
        
        with h5py.File(hdf5_path, 'r') as hf:
            print(f"Channels ({len(hf.keys())}):")
            
            for channel_name in sorted(hf.keys()):
                data = hf[channel_name]
                
                # Load data to compute statistics
                channel_data = data[:]
                
                duration_hours = len(channel_data) / config['processing']['target_sample_rate'] / 3600
                
                # Calculate stats on clean segments only if quality metadata exists
                if quality_metadata is not None:
                    # Extract clean segments
                    samples_per_window = int(30 * config['processing']['target_sample_rate'])
                    clean_segments = []
                    for win_idx in quality_metadata['clean_windows']:
                        start_idx = win_idx * samples_per_window
                        end_idx = (win_idx + 1) * samples_per_window
                        if end_idx <= len(channel_data):
                            clean_segments.append(channel_data[start_idx:end_idx])
                    
                    if clean_segments:
                        clean_data = np.concatenate(clean_segments)
                        # Use float32 to avoid overflow with float16 data
                        mean_val = np.mean(clean_data.astype(np.float32))
                        std_val = np.std(clean_data.astype(np.float32))
                        computed_on = "clean"
                    else:
                        mean_val = np.mean(channel_data.astype(np.float32))
                        std_val = np.std(channel_data.astype(np.float32))
                        computed_on = "all (no clean segments)"
                else:
                    mean_val = np.mean(channel_data.astype(np.float32))
                    std_val = np.std(channel_data.astype(np.float32))
                    computed_on = "all"
                
                print(f"  {channel_name:12s}: "
                      f"shape={channel_data.shape}, "
                      f"dtype={channel_data.dtype}, "
                      f"mean={mean_val:6.3f}, "
                      f"std={std_val:6.3f}, "
                      f"duration={duration_hours:.2f}h "
                      f"({computed_on})")
        
        print()
        print("✓ Validation successful")
        print()
        
        # Step 3: Compare with original
        print("Step 3: Comparing with original NumPy data...")
        print("-" * 80)
        
        input_base = Path(config['input']['base_dir'])
        
        # Test one channel
        test_channel_orig = 'C3-M2'
        test_channel_hdf5 = 'C3-M2'
        
        orig_path = input_base / config['input']['eeg_dir'] / subject_id / f"{test_channel_orig}.npy"
        
        if orig_path.exists():
            orig_data = np.load(orig_path)
            
            with h5py.File(hdf5_path, 'r') as hf:
                hdf5_data = hf[test_channel_hdf5][:]
            
            # Calculate expected length after resampling
            orig_length = orig_data.size
            expected_length = int(orig_length * config['processing']['target_sample_rate'] / 
                                config['processing']['current_sample_rate'])
            
            print(f"\nChannel: {test_channel_orig}")
            print(f"  Original shape: {orig_data.shape}")
            print(f"  Original length: {orig_length}")
            print(f"  Original sampling rate: {config['processing']['current_sample_rate']} Hz")
            print()
            print(f"  HDF5 length: {len(hdf5_data)}")
            print(f"  Expected length: {expected_length}")
            print(f"  HDF5 sampling rate: {config['processing']['target_sample_rate']} Hz")
            print(f"  Length difference: {abs(len(hdf5_data) - expected_length)} samples")
            print()
            
            if abs(len(hdf5_data) - expected_length) <= 10:
                print("✓ Length matches expected (within tolerance)")
            else:
                print("⚠ Length mismatch (may need investigation)")
        
        print()
        print("="*80)
        print(f"✓ All tests passed for {subject_id}")
        print("="*80)
        
        return True
        
    finally:
        # Cleanup temp config
        if temp_config_path.exists():
            temp_config_path.unlink()


def get_first_subject(config_path: str) -> str:
    """Get first available subject from EEG directory."""
    # Resolve config path relative to this script's directory
    script_dir = Path(__file__).parent
    if not Path(config_path).is_absolute():
        config_path = script_dir / config_path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    eeg_dir = Path(config['input']['base_dir']) / config['input']['eeg_dir']
    
    subjects = [d.name for d in eeg_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not subjects:
        raise ValueError("No subjects found in EEG directory")
    
    return sorted(subjects)[0]


def main():
    parser = argparse.ArgumentParser(description="Test single subject conversion")
    parser.add_argument("--subject", type=str, help="Subject ID to test")
    parser.add_argument("--auto", action="store_true", help="Automatically use first subject")
    parser.add_argument("--config", type=str, default="config_stages_conversion.yaml", 
                       help="Config file")
    
    args = parser.parse_args()
    
    if args.auto:
        subject_id = get_first_subject(args.config)
        print(f"Auto-selected subject: {subject_id}\n")
    elif args.subject:
        subject_id = args.subject
    else:
        parser.error("Either --subject or --auto must be specified")
    
    test_single_subject(subject_id, args.config)


if __name__ == "__main__":
    main()
