#!/usr/bin/env python3
"""
Generate Embeddings for STAGES Cognitive Prediction
====================================================

Generate and cache embeddings using pre-trained SleepFM model.

This script:
1. Loads pre-trained SetTransformer model
2. Processes all HDF5 files in the dataset
3. Generates embeddings for each PSG file
4. Saves embeddings to disk for faster training

Usage:
    python generate_embeddings.py --config config_finetune_cognitive.yaml

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import os
import sys
import argparse
import yaml
import torch
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, load_data
from models.models import SetTransformer
from models.dataset import SetTransformerDataset, collate_fn
from torch.utils.data import DataLoader


def load_pretrained_model(config: Dict, device: torch.device) -> SetTransformer:
    """Load pre-trained SetTransformer model."""
    pretrained_type = config['model']['pretrained_model']
    checkpoint_path = config['model']['pretrained_checkpoint']
    
    logger.info(f"Loading pretrained {pretrained_type} model from {checkpoint_path}")
    
    model_params = config['model']['params']
    
    model = SetTransformer(
        in_channels=len(config['data']['modality_types']),
        patch_size=config['data']['chunk_size'],
        embed_dim=model_params['embed_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        pooling_head=model_params['pooling_head'],
        dropout=model_params['dropout'],
        max_seq_length=model_params['max_seq_length']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info("Loaded pretrained model")
    
    return model


def generate_embeddings_for_file(
    file_path: str,
    model: SetTransformer,
    channel_groups: Dict,
    config: Dict,
    device: torch.device
) -> np.ndarray:
    """Generate embeddings for a single HDF5 file.
    
    Args:
        file_path: Path to HDF5 file
        model: Pre-trained SetTransformer
        channel_groups: Channel group mappings
        config: Configuration dictionary
        device: Device to run model on
    
    Returns:
        embeddings: Array of embeddings [num_chunks, embed_dim]
    """
    chunk_size = config['data']['chunk_size']
    modality_types = config['data']['modality_types']
    max_channels_config = config['data']['max_channels']
    exclude_channels = config['data'].get('exclude_channels', [])
    
    # Create set of excluded channel names (case-insensitive)
    excluded_set = set([ch.lower() for ch in exclude_channels])
    
    # Log excluded channels if any
    if excluded_set:
        logger.debug(f"Excluding channels: {exclude_channels}")
    
    embeddings_list = []
    
    with h5py.File(file_path, 'r') as hf:
        # Get available channels
        available_channels = list(hf.keys())
        
        # Track excluded channels found in this file
        excluded_found = [ch for ch in available_channels if ch.lower() in excluded_set]
        if excluded_found:
            logger.debug(f"Excluding {len(excluded_found)} channel(s) from {Path(file_path).stem}: {excluded_found}")
        
        # Group by modality
        modality_to_channels = {mod: [] for mod in modality_types}
        
        for channel in available_channels:
            # Skip excluded channels
            if channel.lower() in excluded_set:
                continue
                
            for modality in modality_types:
                if channel in channel_groups[modality]:
                    modality_to_channels[modality].append(channel)
                    break
        
        # Get signal length
        first_channel = available_channels[0]
        signal_length = hf[first_channel].shape[0]
        
        # Calculate number of chunks
        num_chunks = signal_length // chunk_size
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size
            
            # Load data for this chunk
            modality_data_list = []
            modality_masks = []
            
            for modality in modality_types:
                channels = modality_to_channels[modality]
                max_channels = max_channels_config[modality]
                
                # Initialize with zeros
                data = np.zeros((max_channels, chunk_size), dtype=np.float32)
                mask = np.ones(max_channels, dtype=bool)  # True = padding
                
                # Fill with actual data
                for idx, channel in enumerate(channels[:max_channels]):
                    signal = hf[channel][chunk_start:chunk_end]
                    data[idx, :len(signal)] = signal
                    mask[idx] = False  # False = valid data
                
                modality_data_list.append(torch.from_numpy(data).unsqueeze(0))  # [1, C, T]
                modality_masks.append(torch.from_numpy(mask).unsqueeze(0))  # [1, C]
            
            # Move to device
            modality_data_list = [d.to(device) for d in modality_data_list]
            modality_masks = [m.to(device) for m in modality_masks]
            
            # Generate embedding
            with torch.no_grad():
                # Stack modalities: [1, num_modalities, C, T]
                x = torch.stack(modality_data_list, dim=1).squeeze(0)  # [num_modalities, C, T]
                x = x.unsqueeze(0)  # [1, num_modalities, C, T]
                
                # Stack masks: [1, num_modalities, C]
                mask = torch.stack(modality_masks, dim=1).squeeze(0)  # [num_modalities, C]
                mask = mask.unsqueeze(0)  # [1, num_modalities, C]
                
                # Get embedding
                embedding, _ = model(x, mask)  # [1, embed_dim]
                
                embeddings_list.append(embedding.cpu().numpy())
        
    # Stack all embeddings
    if embeddings_list:
        embeddings = np.concatenate(embeddings_list, axis=0)  # [num_chunks, embed_dim]
    else:
        embeddings = np.array([])
    
    return embeddings


def main(config_path: str):
    """Main embedding generation function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    output_dir = Path(config['preprocessing']['embeddings_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"embedding_generation_{timestamp}.log"
    logger.add(log_file, rotation="100 MB", retention="10 days", level="DEBUG")
    
    logger.info("="*80)
    logger.info("STAGES Embedding Generation")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    
    # Log channel exclusions if any
    exclude_channels = config['data'].get('exclude_channels', [])
    if exclude_channels:
        logger.info(f"Excluding channels: {exclude_channels}")
    else:
        logger.info("No channels excluded (using all available channels)")
    
    # Set device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load channel groups
    channel_groups = load_data(config['data']['channel_groups_path'])
    
    # Load pretrained model
    model = load_pretrained_model(config, device)
    
    # Get all HDF5 files from all splits
    split_path = config['data']['split_path']
    split_data = load_data(split_path)
    
    all_files = []
    for split in ['train', 'val', 'test']:
        all_files.extend(split_data[split])
    
    # Make absolute paths
    data_path = Path(config['data']['data_path'])
    all_files = [str(data_path / f) if not Path(f).is_absolute() else f for f in all_files]
    
    logger.info(f"Total files to process: {len(all_files)}")
    
    # Generate embeddings
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in tqdm(all_files, desc="Generating embeddings"):
        file_path = Path(file_path)
        subject_id = file_path.stem
        
        # Check if already exists
        output_file = output_dir / f"{subject_id}.npy"
        
        if output_file.exists():
            logger.debug(f"Skipping {subject_id} (already exists)")
            skipped_count += 1
            continue
        
        try:
            # Generate embeddings
            embeddings = generate_embeddings_for_file(
                str(file_path),
                model,
                channel_groups,
                config,
                device
            )
            
            if len(embeddings) == 0:
                logger.warning(f"No embeddings generated for {subject_id}")
                error_count += 1
                continue
            
            # Save embeddings
            np.save(output_file, embeddings)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} files...")
        
        except Exception as e:
            logger.error(f"Error processing {subject_id}: {e}")
            error_count += 1
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Embedding Generation Summary")
    logger.info("="*80)
    logger.info(f"Total files: {len(all_files)}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings for STAGES cognitive prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    main(args.config)
