#!/usr/bin/env python3
"""
Artifact Filtering Module for STAGES Cognitive Prediction
==========================================================

This module provides functionality to filter artifact segments from PSG data
using master exclusion masks before embedding generation.

Key Features:
- Load master exclusion masks (30-sec resolution, 720 segments per subject)
- Filter artifact segments from 5-min chunks before model processing
- Handle variable-length sequences after filtering
- Gracefully handle edge cases (all-artifact chunks)
- Toggleable via configuration

Resolution Mapping:
- Master mask: 30-second segments (720 total = 6 hours)
- HDF5 chunks: 5-minute chunks (5 * 60 * 128 = 38,400 samples at 128 Hz)
- Model patches: 5-second patches (640 samples at 128 Hz)
- 1 chunk (5 min) = 10 master_mask segments (30-sec each)
- 1 mask segment (30 sec) = 3,840 samples = 6 patches (5-sec each)

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import torch


class ArtifactFilter:
    """Filter artifacts from PSG chunks using master exclusion masks."""
    
    def __init__(
        self,
        master_masks_dir: str,
        sampling_rate: int = 128,
        segment_duration: int = 30,
        enabled: bool = True
    ):
        """
        Initialize artifact filter.
        
        Args:
            master_masks_dir: Directory containing master exclusion mask files
                             Files should be named: {subject_id}_master_exclusion_mask.npy
            sampling_rate: Sampling rate of HDF5 data (default: 128 Hz)
            segment_duration: Duration of each mask segment in seconds (default: 30)
            enabled: Whether filtering is enabled (default: True)
        """
        self.master_masks_dir = Path(master_masks_dir)
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.segment_samples = segment_duration * sampling_rate  # 30 * 128 = 3,840
        self.enabled = enabled
        
        # Cache for loaded masks to avoid repeated disk I/O
        self._mask_cache = {}
        
        # Statistics tracking per subject
        self._subject_stats = {}  # {subject_id: {'total_segments': int, 'clean_segments': int, 'chunks_processed': int}}
        
        logger.info(f"ArtifactFilter initialized:")
        logger.info(f"  Master masks dir: {self.master_masks_dir}")
        logger.info(f"  Sampling rate: {self.sampling_rate} Hz")
        logger.info(f"  Segment duration: {self.segment_duration} sec")
        logger.info(f"  Segment samples: {self.segment_samples}")
        logger.info(f"  Filtering enabled: {self.enabled}")
    
    def load_master_mask(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Load master exclusion mask for a subject.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            Boolean array of shape (720,) where True=clean, False=artifact
            (inverted from file: file has 0=signal/clean, 1=artifact/exclude)
            Returns None if mask file not found
        """
        # Check cache first
        if subject_id in self._mask_cache:
            return self._mask_cache[subject_id]
        
        # Construct mask file path
        mask_file = self.master_masks_dir / f"{subject_id}_master_exclusion_mask.npy"
        
        if not mask_file.exists():
            logger.warning(f"Master mask not found for {subject_id}: {mask_file}")
            return None
        
        try:
            # Load mask
            mask = np.load(mask_file)
            
            # Validate shape
            if len(mask) != 720:
                logger.warning(f"Invalid mask shape for {subject_id}: {mask.shape}, expected (720,)")
                return None
            
            # IMPORTANT: Invert the mask!
            # File format: 0=signal (clean), 1=artifact (exclude)
            # We want: True=clean (keep), False=artifact (exclude)
            mask = (mask == 0)
            
            # Cache the mask
            self._mask_cache[subject_id] = mask
            
            return mask
            
        except Exception as e:
            logger.error(f"Error loading mask for {subject_id}: {e}")
            return None
    
    def get_chunk_mask_indices(
        self,
        chunk_start: int,
        chunk_samples: int
    ) -> Tuple[int, int]:
        """
        Calculate which master mask indices correspond to a chunk.
        
        Args:
            chunk_start: Starting sample index of the chunk in HDF5
            chunk_samples: Number of samples in the chunk
        
        Returns:
            (start_idx, end_idx): Indices into master_mask array
        
        Example:
            chunk_start=0, chunk_samples=38400 (5 min)
            -> segments 0-9 (indices 0:10)
            
            chunk_start=38400, chunk_samples=38400 (next 5 min)
            -> segments 10-19 (indices 10:20)
        """
        # Convert sample indices to segment indices
        start_segment = chunk_start // self.segment_samples
        end_segment = (chunk_start + chunk_samples - 1) // self.segment_samples + 1
        
        return start_segment, end_segment
    
    def filter_chunk(
        self,
        chunk_data: np.ndarray,
        chunk_start: int,
        master_mask: np.ndarray,
        subject_id: str = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Filter artifact segments from a chunk.
        
        Args:
            chunk_data: Signal data of shape (channels, samples)
            chunk_start: Starting sample index of chunk in original recording
            master_mask: Master exclusion mask (720,) where True=clean, False=artifact
                        (already inverted from file format in load_master_mask)
            subject_id: Optional subject ID for statistics tracking
        
        Returns:
            filtered_data: Clean segments concatenated, shape (channels, clean_samples)
                          Returns None if all segments are artifacts
            segment_mask: Boolean mask of shape (num_segments,) indicating which 
                         segments were kept (True=kept, False=removed)
        
        Example:
            Input: chunk_data shape (10, 38400) for 5-min chunk
                   master_mask[0:10] = [True,True,False,True,True,True,False,False,True,True]
            Output: filtered_data shape (10, 23040) - 6 clean segments × 3840 samples
                    segment_mask = [True, True, False, True, True, True, False, False, True, True]
        """
        if not self.enabled:
            # If filtering disabled, return original data
            num_segments = chunk_data.shape[1] // self.segment_samples
            segment_mask = np.ones(num_segments, dtype=bool)
            return chunk_data, segment_mask
        
        # Get which mask segments this chunk corresponds to
        start_idx, end_idx = self.get_chunk_mask_indices(chunk_start, chunk_data.shape[1])
        
        # Extract relevant mask segments
        chunk_mask = master_mask[start_idx:end_idx]
        
        # Calculate number of complete segments in this chunk
        num_segments = len(chunk_mask)
        
        # Find clean segments
        clean_segment_indices = np.where(chunk_mask)[0]
        
        # Track statistics if subject_id provided
        if subject_id is not None:
            if subject_id not in self._subject_stats:
                self._subject_stats[subject_id] = {
                    'total_segments': 0,
                    'clean_segments': 0,
                    'artifact_segments': 0,
                    'chunks_processed': 0,
                    'all_artifact_chunks': 0
                }
            
            # Only count each chunk once (avoid counting per modality)
            # Use chunk_start as unique identifier
            if not hasattr(self, '_processed_chunks'):
                self._processed_chunks = {}
            if subject_id not in self._processed_chunks:
                self._processed_chunks[subject_id] = set()
            
            if chunk_start not in self._processed_chunks[subject_id]:
                self._subject_stats[subject_id]['total_segments'] += num_segments
                self._subject_stats[subject_id]['clean_segments'] += len(clean_segment_indices)
                self._subject_stats[subject_id]['artifact_segments'] += (num_segments - len(clean_segment_indices))
                self._subject_stats[subject_id]['chunks_processed'] += 1
                if len(clean_segment_indices) == 0:
                    self._subject_stats[subject_id]['all_artifact_chunks'] += 1
                self._processed_chunks[subject_id].add(chunk_start)
        
        if len(clean_segment_indices) == 0:
            # All segments are artifacts
            return None, np.zeros(num_segments, dtype=bool)
        
        # Extract clean segments
        clean_segments = []
        for seg_idx in clean_segment_indices:
            seg_start = seg_idx * self.segment_samples
            seg_end = seg_start + self.segment_samples
            
            # Handle edge case: last segment might be incomplete
            if seg_end > chunk_data.shape[1]:
                seg_end = chunk_data.shape[1]
            
            clean_segments.append(chunk_data[:, seg_start:seg_end])
        
        # Concatenate clean segments
        filtered_data = np.concatenate(clean_segments, axis=1)
        
        # Create segment mask (True=kept, False=removed)
        segment_mask = chunk_mask.astype(bool)
        
        return filtered_data, segment_mask
    
    def filter_batch(
        self,
        batch_data: List[np.ndarray],
        chunk_starts: List[int],
        subject_ids: List[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Filter artifacts from a batch of chunks.
        
        Args:
            batch_data: List of arrays, each shape (channels, samples)
            chunk_starts: List of chunk start indices
            subject_ids: List of subject identifiers
        
        Returns:
            filtered_batch: List of filtered arrays or None for all-artifact chunks
            segment_masks: List of segment masks or None
        """
        filtered_batch = []
        segment_masks = []
        
        for data, chunk_start, subject_id in zip(batch_data, chunk_starts, subject_ids):
            # Load master mask for this subject
            master_mask = self.load_master_mask(subject_id)
            
            if master_mask is None:
                # No mask available, keep original data
                logger.warning(f"No mask for {subject_id}, keeping all segments")
                num_segments = data.shape[1] // self.segment_samples
                segment_mask = np.ones(num_segments, dtype=bool)
                filtered_batch.append(data)
                segment_masks.append(segment_mask)
                continue
            
            # Filter chunk
            filtered_data, segment_mask = self.filter_chunk(data, chunk_start, master_mask, subject_id)
            
            filtered_batch.append(filtered_data)
            segment_masks.append(segment_mask)
        
        return filtered_batch, segment_masks
    
    def get_subject_stats(self, subject_id: str) -> Optional[Dict]:
        """Get filtering statistics for a specific subject."""
        return self._subject_stats.get(subject_id, None)
    
    def get_all_stats(self) -> Dict:
        """Get all filtering statistics."""
        return {
            'enabled': self.enabled,
            'cached_masks': len(self._mask_cache),
            'subjects_processed': len(self._subject_stats),
            'subject_stats': self._subject_stats,
            'master_masks_dir': str(self.master_masks_dir),
            'segment_duration': self.segment_duration,
            'sampling_rate': self.sampling_rate
        }
    
    def print_subject_summary(self, subject_id: str):
        """Print filtering summary for a subject."""
        stats = self.get_subject_stats(subject_id)
        if stats is None:
            return
        
        total = stats['total_segments']
        clean = stats['clean_segments']
        artifact = stats['artifact_segments']
        clean_pct = (clean / total * 100) if total > 0 else 0
        
        logger.info(
            f"Subject {subject_id} filtering: "
            f"{clean}/{total} segments clean ({clean_pct:.1f}%), "
            f"{stats['chunks_processed']} chunks processed, "
            f"{stats['all_artifact_chunks']} all-artifact chunks"
        )
    
    def save_stats_report(self, output_file: str):
        """Save filtering statistics to a CSV file."""
        import pandas as pd
        
        if not self._subject_stats:
            logger.warning("No statistics to save")
            return
        
        # Convert to DataFrame
        records = []
        for subject_id, stats in self._subject_stats.items():
            total = stats['total_segments']
            clean = stats['clean_segments']
            clean_pct = (clean / total * 100) if total > 0 else 0
            
            records.append({
                'subject_id': subject_id,
                'total_segments': total,
                'clean_segments': clean,
                'artifact_segments': stats['artifact_segments'],
                'clean_percentage': clean_pct,
                'chunks_processed': stats['chunks_processed'],
                'all_artifact_chunks': stats['all_artifact_chunks']
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved filtering statistics to {output_file}")
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics (legacy method)."""
        return self.get_all_stats()
    
    def clear_cache(self):
        """Clear the mask cache to free memory."""
        self._mask_cache.clear()
        if hasattr(self, '_processed_chunks'):
            self._processed_chunks.clear()
        logger.info("Cleared artifact filter mask cache")


def extract_subject_id_from_path(file_path: str) -> str:
    """
    Extract subject ID from HDF5 file path.
    
    Args:
        file_path: Path to HDF5 file (e.g., "/path/to/subject123.hdf5")
    
    Returns:
        Subject ID (e.g., "subject123")
    """
    # Get filename without extension
    filename = Path(file_path).stem
    return filename


def create_artifact_filter(config: Dict) -> Optional[ArtifactFilter]:
    """
    Create artifact filter from configuration.
    
    Args:
        config: Configuration dictionary with 'artifact_filtering' section
    
    Returns:
        ArtifactFilter instance or None if disabled
    """
    # Check if artifact filtering is configured
    if 'artifact_filtering' not in config:
        logger.info("No artifact_filtering config found, filtering disabled")
        return None
    
    filter_config = config['artifact_filtering']
    
    # Check if enabled
    if not filter_config.get('enabled', False):
        logger.info("Artifact filtering disabled in config")
        return None
    
    # Get master masks directory
    master_masks_dir = filter_config.get('master_masks_dir')
    if master_masks_dir is None:
        logger.warning("master_masks_dir not specified in config, filtering disabled")
        return None
    
    # Get parameters
    sampling_rate = filter_config.get('sampling_rate', 128)
    segment_duration = filter_config.get('segment_duration', 30)
    
    # Create filter
    artifact_filter = ArtifactFilter(
        master_masks_dir=master_masks_dir,
        sampling_rate=sampling_rate,
        segment_duration=segment_duration,
        enabled=True
    )
    
    return artifact_filter
