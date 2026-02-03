#!/usr/bin/env python3
"""
Test Artifact Filtering Module
===============================

Quick test to verify artifact filtering works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stages_cognitive_prediction.artifact_filtering import ArtifactFilter


def test_artifact_filter():
    """Test basic artifact filtering functionality."""
    
    print("="*80)
    print("Testing Artifact Filtering Module")
    print("="*80)
    
    # Create a dummy artifact filter (pointing to real directory but we'll mock data)
    artifact_filter = ArtifactFilter(
        master_masks_dir="/home/boshra95/scratch/stages/stages/processed/master_masks",
        sampling_rate=128,
        segment_duration=30,
        enabled=True
    )
    
    print("\n1. Test chunk mask index calculation:")
    print("-" * 40)
    
    # Test: 5-min chunk starting at sample 0
    chunk_samples = 5 * 60 * 128  # 38,400 samples
    start_idx, end_idx = artifact_filter.get_chunk_mask_indices(0, chunk_samples)
    print(f"Chunk 0-5min (samples 0-38400): mask indices {start_idx}:{end_idx}")
    assert start_idx == 0 and end_idx == 10, "First 5-min chunk should map to mask[0:10]"
    
    # Test: Next 5-min chunk
    start_idx, end_idx = artifact_filter.get_chunk_mask_indices(38400, chunk_samples)
    print(f"Chunk 5-10min (samples 38400-76800): mask indices {start_idx}:{end_idx}")
    assert start_idx == 10 and end_idx == 20, "Second 5-min chunk should map to mask[10:20]"
    
    print("✓ Chunk index calculation correct!")
    
    print("\n2. Test filtering with mock data:")
    print("-" * 40)
    
    # Create mock chunk data: 10 channels × 38,400 samples (5 min at 128 Hz)
    chunk_data = np.random.randn(10, 38400).astype(np.float32)
    
    # Create mock master mask: 720 segments (6 hours)
    # First 5-min chunk (segments 0-9): [1,1,0,1,1,1,0,0,1,1] - 7 clean, 3 artifacts
    master_mask = np.ones(720, dtype=bool)
    master_mask[2] = False  # segment 2 is artifact
    master_mask[6:8] = False  # segments 6-7 are artifacts
    
    print(f"Mock mask for first 10 segments: {master_mask[:10].astype(int)}")
    
    # Filter the chunk
    filtered_data, segment_mask = artifact_filter.filter_chunk(chunk_data, 0, master_mask)
    
    print(f"Original chunk shape: {chunk_data.shape}")
    print(f"Filtered chunk shape: {filtered_data.shape}")
    print(f"Segment mask: {segment_mask.astype(int)}")
    
    # Expected: 7 clean segments × 3,840 samples = 26,880 samples
    expected_samples = 7 * 30 * 128
    assert filtered_data.shape[1] == expected_samples, f"Expected {expected_samples} samples, got {filtered_data.shape[1]}"
    print(f"✓ Filtered correctly: kept 7/10 segments ({filtered_data.shape[1]} samples)")
    
    print("\n3. Test all-artifact chunk:")
    print("-" * 40)
    
    # Create a mask where all segments in the chunk are artifacts
    master_mask_all_artifacts = np.ones(720, dtype=bool)
    master_mask_all_artifacts[0:10] = False  # First 10 segments all artifacts
    
    filtered_data, segment_mask = artifact_filter.filter_chunk(
        chunk_data, 0, master_mask_all_artifacts
    )
    
    print(f"Filtered data: {filtered_data}")
    print(f"Segment mask: {segment_mask.astype(int)}")
    
    assert filtered_data is None, "All-artifact chunk should return None"
    print("✓ All-artifact chunk correctly returns None")
    
    print("\n4. Test disabled filtering:")
    print("-" * 40)
    
    artifact_filter_disabled = ArtifactFilter(
        master_masks_dir="/home/boshra95/scratch/stages/stages/processed/master_masks",
        sampling_rate=128,
        segment_duration=30,
        enabled=False
    )
    
    filtered_data, segment_mask = artifact_filter_disabled.filter_chunk(
        chunk_data, 0, master_mask
    )
    
    print(f"Disabled filter - original shape: {chunk_data.shape}")
    print(f"Disabled filter - filtered shape: {filtered_data.shape}")
    
    assert np.array_equal(filtered_data, chunk_data), "Disabled filter should return original data"
    print("✓ Disabled filter returns original data")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)


if __name__ == "__main__":
    test_artifact_filter()
