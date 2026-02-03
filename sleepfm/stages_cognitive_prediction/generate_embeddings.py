#!/usr/bin/env python3
"""
Generate Embeddings for STAGES Cognitive Prediction
====================================================

Generate and cache embeddings using pre-trained SleepFM model.

This script follows the demo.py approach:
1. Loads pre-trained SetTransformer model
2. Processes each modality separately (BAS, RESP, EKG, EMG)
3. Generates both 5-min aggregated and granular 5-sec embeddings
4. Saves embeddings to HDF5 files (one per subject, with modalities as datasets)

Usage:
    python generate_embeddings.py --config config_finetune_cognitive.yaml

Author: Generated for STAGES cognitive prediction
Date: February 2026
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from typing import Dict, List

# Add sleepfm directory to path for absolute imports
sleepfm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sleepfm_root not in sys.path:
    sys.path.insert(0, sleepfm_root)

from sleepfm.utils import load_config, load_data, count_parameters
from sleepfm.models.models import SetTransformer
from sleepfm.models.dataset import SetTransformerDataset, collate_fn
from sleepfm.stages_cognitive_prediction.artifact_filtering import (
    create_artifact_filter,
    extract_subject_id_from_path
)
from torch.utils.data import DataLoader


def load_pretrained_model(config: Dict, device: torch.device) -> SetTransformer:
    """Load pre-trained SetTransformer model (following demo.py approach)."""
    checkpoint_path = config['model']['pretrained_checkpoint']
    
    logger.info(f"Loading pretrained model from {checkpoint_path}")
    
    # Load checkpoint config
    checkpoint_dir = Path(checkpoint_path).parent
    config_file = checkpoint_dir / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")
    
    with open(config_file, 'r') as f:
        checkpoint_config = json.load(f)
    
    # Extract model parameters from checkpoint config
    in_channels = checkpoint_config['in_channels']
    patch_size = checkpoint_config['patch_size']
    embed_dim = checkpoint_config['embed_dim']
    num_heads = checkpoint_config['num_heads']
    num_layers = checkpoint_config['num_layers']
    pooling_head = checkpoint_config.get('pooling_head', 8)
    max_seq_length = checkpoint_config.get('max_seq_length', 128)
    dropout = 0.0  # Set to 0 for inference
    
    logger.info(f"Model architecture: num_layers={num_layers}, embed_dim={embed_dim}, patch_size={patch_size}, max_seq_length={max_seq_length}")
    
    # Create model (matching demo.py exactly)
    model = SetTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling_head=pooling_head,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Always strip 'module.' prefix if present (checkpoint was saved from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        logger.info("Removed 'module.' prefix from checkpoint keys")
    
    # Move to device first
    model = model.to(device)
    
    # Load state dict into non-wrapped model
    model.load_state_dict(state_dict)
    
    # Apply DataParallel AFTER loading state dict
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model.eval()
    
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')
    
    return model, checkpoint_config


def main(config_path: str):
    """Main embedding generation function (following demo.py approach)."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    output_dir = Path(config['preprocessing']['embeddings_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Option to save granular (5-sec) embeddings in addition to aggregated (5-min)
    save_granular = config['preprocessing'].get('save_granular_embeddings', False)
    
    # Create separate directories for aggregated and granular (like demo)
    if save_granular:
        output_dir_granular = output_dir / "granular"
        output_dir_granular.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"embedding_generation_{timestamp}.log"
    logger.add(log_file, rotation="100 MB", retention="10 days", level="DEBUG")
    
    logger.info("="*80)
    logger.info("STAGES Embedding Generation (Demo-Compatible)")
    logger.info("="*80)
    logger.info(f"Output directory (aggregated): {output_dir}")
    if save_granular:
        logger.info(f"Output directory (granular): {output_dir_granular}")
    
    # Set device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load channel groups
    channel_groups = load_data(config['data']['channel_groups_path'])
    
    # Get channels to exclude (if specified)
    exclude_channels = config['preprocessing'].get('exclude_channels', [])
    if exclude_channels:
        logger.info(f"Excluding channels: {exclude_channels}")
        # Remove excluded channels from all modality groups
        for modality_type in channel_groups:
            channel_groups[modality_type] = [
                ch for ch in channel_groups[modality_type] 
                if ch not in exclude_channels
            ]
        logger.info(f"Channel groups after exclusion: {', '.join([f'{k}({len(v)})' for k, v in channel_groups.items()])}")
    
    # Load pretrained model
    model, checkpoint_config = load_pretrained_model(config, device)
    
    # Get modality types from checkpoint config
    modality_types = checkpoint_config['modality_types']
    embed_dim = checkpoint_config['embed_dim']
    logger.info(f"Modality types: {modality_types}")
    
    # Initialize artifact filter (if enabled in config)
    artifact_filter = create_artifact_filter(config)
    if artifact_filter is not None:
        logger.info("Artifact filtering enabled:")
        logger.info(f"  Master masks dir: {artifact_filter.master_masks_dir}")
        logger.info(f"  Segment duration: {artifact_filter.segment_duration} sec")
        logger.info(f"  Sampling rate: {artifact_filter.sampling_rate} Hz")
    else:
        logger.info("Artifact filtering disabled")
    
    # Get all HDF5 files from splits
    split_path = config['data']['split_path']
    split_data = load_data(split_path)
    
    all_files = []
    for split in ['train', 'val', 'test']:
        if split in split_data:
            all_files.extend(split_data[split])
    
    # Make absolute paths
    data_path = Path(config['data']['data_path'])
    all_files = [str(data_path / f) if not Path(f).is_absolute() else f for f in all_files]
    
    # Limit number of files if specified (for testing)
    max_files = config['data'].get('max_files', None)
    if max_files is not None and max_files > 0:
        all_files = all_files[:max_files]
        logger.info(f"Limited to {max_files} files for testing")
    
    logger.info(f"Total files to process: {len(all_files)}")
    
    # Create dataset (for proper data loading like demo)
    dataset = SetTransformerDataset(
        checkpoint_config,
        channel_groups,
        hdf5_paths=all_files,
        split="test",  # Doesn't matter for embedding generation
        artifact_filter=artifact_filter  # Pass artifact filter
    )
    
    # Create dataloader
    batch_size = config.get('system', {}).get('batch_size', 16)
    num_workers = config.get('system', {}).get('num_workers', 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}")
    
    # Generate embeddings
    processed_subjects = set()
    skipped_count = 0
    filtered_chunks_count = 0
    error_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            try:
                # Unpack batch (following demo.py)
                batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
                
                # Check if batch was filtered out (all artifacts)
                if batch_data is None:
                    filtered_chunks_count += 1
                    continue
                
                # Separate modalities
                modality_data = {}
                modality_masks = {}
                
                for idx, modality_type in enumerate(modality_types):
                    modality_data[modality_type] = batch_data[idx].to(device, dtype=torch.float)
                    modality_masks[modality_type] = mask_list[idx].to(device, dtype=torch.bool)
                
                # Generate embeddings for each modality separately (exactly like demo)
                embeddings = [
                    model(modality_data[modality_types[0]], modality_masks[modality_types[0]]),
                    model(modality_data[modality_types[1]], modality_masks[modality_types[1]]),
                    model(modality_data[modality_types[2]], modality_masks[modality_types[2]]),
                    model(modality_data[modality_types[3]], modality_masks[modality_types[3]]),
                ]
                
                # Model returns tuple: (5-min aggregated, granular 5-sec)
                # Extract aggregated embeddings: e[0] is the 5-min aggregated embedding
                embeddings_agg = [e[0].unsqueeze(1) for e in embeddings]
                
                # Extract granular embeddings if needed: e[1] is the granular 5-sec embedding
                if save_granular:
                    embeddings_granular = [e[1] for e in embeddings]
                
                # Save AGGREGATED embeddings (5-min level) - matching demo exactly
                for i in range(len(file_paths)):
                    file_path = file_paths[i]
                    subject_id = Path(file_path).stem
                    chunk_start = chunk_starts[i]
                    
                    output_file = output_dir / f"{subject_id}.hdf5"
                    
                    try:
                        # Save aggregated embeddings (main embeddings for downstream tasks)
                        with h5py.File(output_file, 'a') as hdf5_file:
                            for modality_idx, modality_type in enumerate(modality_types):
                                agg_data = embeddings_agg[modality_idx][i].cpu().numpy()  # [seq_len, 1, embed_dim]
                                
                                if modality_type in hdf5_file:
                                    dset = hdf5_file[modality_type]
                                    # Calculate position: chunk_start is in samples, convert to 5-min chunks
                                    chunk_start_correct = chunk_start // (embed_dim * 5 * 60)
                                    chunk_end = chunk_start_correct + agg_data.shape[0]
                                    if dset.shape[0] < chunk_end:
                                        dset.resize((chunk_end,) + tuple(agg_data.shape[1:]))
                                    dset[chunk_start_correct:chunk_end] = agg_data
                                else:
                                    hdf5_file.create_dataset(
                                        modality_type,
                                        data=agg_data,
                                        chunks=(embed_dim,) + tuple(agg_data.shape[1:]),
                                        maxshape=(None,) + tuple(agg_data.shape[1:])
                                    )
                        
                        processed_subjects.add(subject_id)
                    
                    except OSError as e:
                        if "truncated file" in str(e) or "Unable to synchronously open file" in str(e):
                            # Corrupted HDF5 file - remove it and recreate
                            logger.warning(f"Corrupted HDF5 file detected for {subject_id}, removing and recreating...")
                            if output_file.exists():
                                output_file.unlink()
                            
                            # Recreate the file
                            try:
                                with h5py.File(output_file, 'w') as hdf5_file:
                                    for modality_idx, modality_type in enumerate(modality_types):
                                        agg_data = embeddings_agg[modality_idx][i].cpu().numpy()
                                        hdf5_file.create_dataset(
                                            modality_type,
                                            data=agg_data,
                                            chunks=(embed_dim,) + tuple(agg_data.shape[1:]),
                                            maxshape=(None,) + tuple(agg_data.shape[1:])
                                        )
                                processed_subjects.add(subject_id)
                                logger.info(f"Successfully recreated embeddings for {subject_id}")
                            except Exception as e2:
                                logger.error(f"Failed to recreate embeddings for {subject_id}: {e2}")
                                error_count += 1
                        else:
                            logger.error(f"Error saving embeddings for {subject_id}: {e}")
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Unexpected error saving embeddings for {subject_id}: {e}")
                        error_count += 1
                
                # Save GRANULAR embeddings (5-sec level) if requested
                if save_granular:
                    for i in range(len(file_paths)):
                        file_path = file_paths[i]
                        subject_id = Path(file_path).stem
                        chunk_start = chunk_starts[i]
                        
                        output_file_granular = output_dir_granular / f"{subject_id}.hdf5"
                        
                        try:
                            with h5py.File(output_file_granular, 'a') as hdf5_file:
                                for modality_idx, modality_type in enumerate(modality_types):
                                    granular_data = embeddings_granular[modality_idx][i].cpu().numpy()
                                    
                                    if modality_type in hdf5_file:
                                        dset = hdf5_file[modality_type]
                                        # Granular: chunk_start is in samples, convert to 5-sec chunks
                                        chunk_start_correct = chunk_start // (embed_dim * 5)
                                        chunk_end = chunk_start_correct + granular_data.shape[0]
                                        if dset.shape[0] < chunk_end:
                                            dset.resize((chunk_end,) + tuple(granular_data.shape[1:]))
                                        dset[chunk_start_correct:chunk_end] = granular_data
                                    else:
                                        hdf5_file.create_dataset(
                                            modality_type,
                                            data=granular_data,
                                            chunks=(embed_dim,) + tuple(granular_data.shape[1:]),
                                            maxshape=(None,) + tuple(granular_data.shape[1:])
                                        )
                        except OSError as e:
                            if "truncated file" in str(e) or "Unable to synchronously open file" in str(e):
                                logger.warning(f"Corrupted granular HDF5 file for {subject_id}, removing and recreating...")
                                if output_file_granular.exists():
                                    output_file_granular.unlink()
                                
                                try:
                                    with h5py.File(output_file_granular, 'w') as hdf5_file:
                                        for modality_idx, modality_type in enumerate(modality_types):
                                            granular_data = embeddings_granular[modality_idx][i].cpu().numpy()
                                            hdf5_file.create_dataset(
                                                modality_type,
                                                data=granular_data,
                                                chunks=(embed_dim,) + tuple(granular_data.shape[1:]),
                                                maxshape=(None,) + tuple(granular_data.shape[1:])
                                            )
                                    logger.info(f"Successfully recreated granular embeddings for {subject_id}")
                                except Exception as e2:
                                    logger.error(f"Failed to recreate granular embeddings for {subject_id}: {e2}")
                            else:
                                logger.error(f"Error saving granular embeddings for {subject_id}: {e}")
                        except Exception as e:
                            logger.error(f"Unexpected error saving granular embeddings for {subject_id}: {e}")
                
                if len(processed_subjects) % 100 == 0:
                    logger.info(f"Processed {len(processed_subjects)} subjects...")
            
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                error_count += 1
                continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Embedding Generation Summary")
    logger.info("="*80)
    logger.info(f"Total files: {len(all_files)}")
    logger.info(f"Processed subjects: {len(processed_subjects)}")
    logger.info(f"Errors: {error_count}")
    if artifact_filter is not None and artifact_filter.enabled:
        logger.info(f"Filtered all-artifact chunks: {filtered_chunks_count}")
        logger.info(f"Artifact filtering was enabled")
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
