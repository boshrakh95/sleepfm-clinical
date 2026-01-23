#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo Script for SleepFM Clinical
Converted from demo.ipynb

Note: All data in this repo is synthetically made including sleep stage 
annotations, demographics or diseases. The data is for demo purposes only.
"""

# ============================================================================
# Preprocessing Details
#
# Before running this script, please preprocess your PSG files using the scripts
# provided in `sleepfm/preprocessing`. Note that PSG recordings may contain 
# different sets of channels across datasets. The predefined channel–modality 
# mappings used in this project are specified in `sleepfm/configs/channel_groups.json`.
#
# Although we have attempted to make this mapping as comprehensive as possible, 
# we strongly recommend reviewing the channels present in your specific PSG data. 
# In consultation with domain experts, you should group any additional or 
# dataset-specific channels into the appropriate modality categories and update 
# `channel_groups.json` accordingly. This step is critical to ensure that all 
# channels are correctly aligned with their intended modalities during 
# preprocessing and downstream modeling.
# ============================================================================

print("Starting SleepFM Clinical Demo Script")

# Imports
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import os
import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import pandas as pd

# Set up paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # sleepfm-clinical directory
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'sleepfm'))

# Change to the notebooks directory so relative paths work
os.chdir(script_dir)

from preprocessing.preprocessing import EDFToHDF5Converter
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer, SleepEventLSTMClassifier, DiagnosisFinetuneFullLSTMCOXPHWithDemo
import h5py
from utils import load_config, load_data, save_data, count_parameters
from torch.utils.data import Dataset, DataLoader

print("Imports done.")

embedding_extraction, sleep_staging, disease_prediction = False, False, True

# ============================================================================
# Part 0: Preprocessing EDF files
#
# Note: This is just a demo script that preprocesses a single, specific file. 
# run `sleepfm/preprocessing/preprocessing.sh` with appropriate folders to 
# generate multiple preprocessed files.
# ============================================================================

base_save_path = "demo_data"
os.makedirs(base_save_path, exist_ok=True)

root_dir = "/edf_root"      # dummy root not used for a single file conversion
target_dir = "/note"    # dummy target not used for a single file conversion

edf_path = "demo_data/demo_psg.edf"
hdf5_path = os.path.join(base_save_path, "demo_psg.hdf5")

print("Converting EDF to HDF5:", edf_path, "->", hdf5_path)
converter = EDFToHDF5Converter(
    root_dir=root_dir,
    target_dir=target_dir,
    resample_rate=128
)

# run for single file conversion
converter.convert(edf_path, hdf5_path)

# Use CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Part 1: Generating embeddings from SleepFM pretrained model
#
# Here we show generating embedding for 1 demo PSG. To see full script, 
# please check `sleepfm/pipeline/generate_embeddings.py`.
# ============================================================================

if embedding_extraction:

    model_path = "../sleepfm/checkpoints/model_base"
    channel_groups_path = "../sleepfm/configs/channel_groups.json"
    config_path = os.path.join(model_path, "config.json")

    print("Loading config and channel groups")

    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    modality_types = config["modality_types"]
    in_channels = config["in_channels"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = 0.0

    model_class = getattr(sys.modules[__name__], config['model'])
    model = model_class(in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head=pooling_head, dropout=dropout)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    model.to(device)
    total_layers, total_params = count_parameters(model)
    print(f'Trainable parameters: {total_params / 1e6:.2f} million')
    print(f'Number of layers: {total_layers}')

    checkpoint = torch.load(os.path.join(model_path, "best.pt"), map_location=device)

    # Handle DataParallel state dict: remove 'module.' prefix if present
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print("Config:", config)

    hdf5_paths = [os.path.join(base_save_path, "demo_psg.hdf5")]
    dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=hdf5_paths, split="test")

    dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=16, 
                                                num_workers=1, 
                                                shuffle=False, 
                                                collate_fn=collate_fn)

    output = os.path.join(base_save_path, "demo_emb")
    output_5min_agg = os.path.join(base_save_path, "demo_5min_agg_emb")
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_5min_agg, exist_ok=True)
    print("Saving embeddings to:", output)
    print("Saving 5-minute aggregated embeddings to:", output_5min_agg)

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
                (bas, resp, ekg, emg) = batch_data
                (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list

                bas = bas.to(device, dtype=torch.float)
                resp = resp.to(device, dtype=torch.float)
                ekg = ekg.to(device, dtype=torch.float)
                emg = emg.to(device, dtype=torch.float)

                mask_bas = mask_bas.to(device, dtype=torch.bool)
                mask_resp = mask_resp.to(device, dtype=torch.bool)
                mask_ekg = mask_ekg.to(device, dtype=torch.bool)
                mask_emg = mask_emg.to(device, dtype=torch.bool)

                embeddings = [
                    model(bas, mask_bas),
                    model(resp, mask_resp),
                    model(ekg, mask_ekg),
                    model(emg, mask_emg),
                ]

                # Model gives two kinds of embeddings. Granular 5 second-level embeddings and aggregated 5 minute-level embeddings. We save both of them below. 

                embeddings_new = [e[0].unsqueeze(1) for e in embeddings]

                for i in range(len(file_paths)):
                    print("processing file:", file_paths[i])

                    file_path = file_paths[i]
                    chunk_start = chunk_starts[i]
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output_5min_agg, f"{subject_id}.hdf5")

                    with h5py.File(output_path, 'a') as hdf5_file:
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            if modality_type in hdf5_file:
                                dset = hdf5_file[modality_type]
                                chunk_start_correct = chunk_start // (embed_dim * 5 * 60)
                                chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                                if dset.shape[0] < chunk_end:
                                    dset.resize((chunk_end,) + tuple(embeddings_new[modality_idx][i].shape[1:]))
                                dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                            else:
                                hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + tuple(embeddings_new[modality_idx][i].shape[1:]), maxshape=(None,) + tuple(embeddings_new[modality_idx][i].shape[1:]))

                embeddings_new = [e[1] for e in embeddings]

                for i in range(len(file_paths)):
                    print("processing file for granular embeddings:", file_paths[i])
                    file_path = file_paths[i]
                    chunk_start = chunk_starts[i]
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output, f"{subject_id}.hdf5")

                    with h5py.File(output_path, 'a') as hdf5_file:
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            if modality_type in hdf5_file:
                                dset = hdf5_file[modality_type]
                                chunk_start_correct = chunk_start // (embed_dim * 5)
                                chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                                if dset.shape[0] < chunk_end:
                                    dset.resize((chunk_end,) + tuple(embeddings_new[modality_idx][i].shape[1:]))
                                dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                            else:
                                hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + tuple(embeddings_new[modality_idx][i].shape[1:]), maxshape=(None,) + tuple(embeddings_new[modality_idx][i].shape[1:]))
                pbar.update()






# ============================================================================
# Part 2: Sleep Staging
#
# Note that below, we are using our finetuned sleep staging model. It is 
# always a good idea to finetune our model on your specific data, even if 
# you only have a handful of sample, so that the model can adapt to your 
# specific data distribution. Script to finetune your sleep staging model 
# head is given in `sleepfm/pipeline/finetune_sleep_staging.py`.
# ============================================================================

if sleep_staging:
    # Initialize variables needed for Part 2
    base_save_path = "demo_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Part 2 - Using device: {device}")
    
    # Load channel groups for Part 2
    channel_groups_path = "../sleepfm/configs/channel_groups.json"
    channel_groups = load_data(channel_groups_path)
    
    sleep_staging_model_path = "../sleepfm/checkpoints/model_sleep_staging"
    sleep_staging_config = load_data(os.path.join(sleep_staging_model_path, "config.json"))

    sleep_staging_model_params = sleep_staging_config['model_params']
    sleep_staging_model_class = getattr(sys.modules[__name__], sleep_staging_config['model'])

    sleep_staging_model = sleep_staging_model_class(**sleep_staging_model_params).to(device)
    sleep_staging_model_name = type(sleep_staging_model).__name__

    if device.type == "cuda":
        sleep_staging_model = nn.DataParallel(sleep_staging_model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print("Using CPU (no DataParallel)")

    print(f"Model initialized: {sleep_staging_model_name}")
    total_layers, total_params = count_parameters(sleep_staging_model)
    print(f'Trainable parameters: {total_params / 1e6:.2f} million')
    print(f'Number of layers: {total_layers}')

    sleep_staging_checkpoint_path = os.path.join(sleep_staging_model_path, "best.pth")
    checkpoint = torch.load(sleep_staging_checkpoint_path, map_location=device)

    # Handle DataParallel state dict: remove 'module.' prefix if present
    if list(checkpoint.keys())[0].startswith('module.') and not isinstance(sleep_staging_model, nn.DataParallel):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    sleep_staging_model.load_state_dict(checkpoint)


    # ============================================================================
    # Helper functions for loading data for sleep staging
    #
    # You can find similar functions within `sleepfm/models/dataset.py`. 
    # You may need to modify it slightly based on your usecase.
    # ============================================================================

    class SleepEventClassificationDataset(Dataset):
        def __init__(
            self,
            config,
            channel_groups,
            hdf5_paths,
            label_files,
            split="train",
        ):
            self.config = config
            self.max_channels = self.config["max_channels"]
            self.context = int(self.config["context"])
            self.channel_like = self.config["channel_like"]

            self.max_seq_len = config["model_params"]["max_seq_length"]

            # --- Build label lookup: {study_id: label_csv_path} ---
            # study_id = filename without extension, e.g. "SSC_12345"
            labels_dict = {
                os.path.basename(p).rsplit(".", 1)[0]: p
                for p in label_files
                if os.path.exists(p)
            }

            # --- Filter to HDF5s that exist and have a matching label file ---
            hdf5_paths = [p for p in hdf5_paths if os.path.exists(p)]
            hdf5_paths = [
                p for p in hdf5_paths
                if os.path.basename(p).rsplit(".", 1)[0] in labels_dict
            ]

            if config.get("max_files"):
                hdf5_paths = hdf5_paths[: config["max_files"]]

            self.hdf5_paths = hdf5_paths
            self.labels_dict = labels_dict

            # --- Build index map ---
            # Each item is (hdf5_path, label_path, start_index)
            if self.context == -1:
                self.index_map = [
                    (p, labels_dict[os.path.basename(p).rsplit(".", 1)[0]], -1)
                    for p in self.hdf5_paths
                ]
            else:
                self.index_map = []
                loop = tqdm.tqdm(self.hdf5_paths, total=len(self.hdf5_paths), desc=f"Indexing {split} data")
                for hdf5_file_path in loop:
                    file_prefix = os.path.basename(hdf5_file_path).rsplit(".", 1)[0]
                    label_path = labels_dict[file_prefix]

                    with h5py.File(hdf5_file_path, "r") as hf:
                        dset_names = list(hf.keys())
                        if len(dset_names) == 0:
                            continue

                        # Use first dataset to define length (same as your original behavior)
                        first_name = dset_names[0]
                        dataset_length = hf[first_name].shape[0]

                    for i in range(0, dataset_length, self.context):
                        self.index_map.append((hdf5_file_path, label_path, i))

            self.total_len = len(self.index_map)

        def __len__(self):
            return self.total_len

        def get_index_map(self):
            return self.index_map

        def __getitem__(self, idx):
            hdf5_path, label_path, start_index = self.index_map[idx]

            labels_df = pd.read_csv(label_path)
            labels_df["StageNumber"] = labels_df["StageNumber"].replace(-1, 0)

            y_data = labels_df["StageNumber"].to_numpy()
            if self.context != -1:
                y_data = y_data[start_index : start_index + self.context]

            x_data = []
            with h5py.File(hdf5_path, "r") as hf:
                dset_names = list(hf.keys())

                for dataset_name in dset_names:
                    if dataset_name in self.channel_like:
                        if self.context == -1:
                            x_data.append(hf[dataset_name][:])
                        else:
                            x_data.append(hf[dataset_name][start_index : start_index + self.context])

            if not x_data:
                # Skip this data point if x_data is empty
                return self.__getitem__((idx + 1) % self.total_len)

            x_data = np.array(x_data)  # (C, T, F) assuming each channel returns (T, F)
            x_data = torch.tensor(x_data, dtype=torch.float32)
            y_data = torch.tensor(y_data, dtype=torch.float32)

            min_length = min(x_data.shape[1], len(y_data))
            x_data = x_data[:, :min_length, :]
            y_data = y_data[:min_length]

            return x_data, y_data, self.max_channels, self.max_seq_len, hdf5_path


    def sleep_event_finetune_full_collate_fn(batch):
        x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

        num_channels = max(max_channels_list)

        max_seq_len_temp = max([item.size(1) for item in x_data])
        # Determine the max sequence length for padding
        if max_seq_len_list[0] is None:
            max_seq_len = max_seq_len_temp
        else:
            max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

        padded_x_data = []
        padded_y_data = []
        padded_mask = []

        for x_item, y_item in zip(x_data, y_data):
            tgt_sleep_no_sleep = np.where(y_item > 0, 1, 0)
            moving_avg_tgt_sleep_no_sleep = np.convolve(tgt_sleep_no_sleep, np.ones(1080)/1080, mode='valid')
            try:
                first_non_zero_index = np.where(moving_avg_tgt_sleep_no_sleep > 0.5)[0][0]
            except IndexError:
                first_non_zero_index = 0

            if first_non_zero_index < 0:
                first_non_zero_index = 0

            # Get the shape of x_item
            c, s, e = x_item.size()
            c = min(c, num_channels)
            s = min(s, max_seq_len + first_non_zero_index)  # Ensure the sequence length doesn't exceed max_seq_len

            # Create a padded tensor and a mask tensor for x_data
            padded_x_item = torch.zeros((num_channels, max_seq_len, e))
            mask = torch.ones((num_channels, max_seq_len))

            # Copy the actual data to the padded tensor and set the mask for real data
            padded_x_item[:c, :s-first_non_zero_index, :e] = x_item[:c, first_non_zero_index:s, :e]
            mask[:c, :s-first_non_zero_index] = 0  # 0 for real data, 1 for padding

            # Pad y_data with zeros to match max_seq_len
            padded_y_item = torch.zeros(max_seq_len)
            padded_y_item[:s-first_non_zero_index] = y_item[first_non_zero_index:s]

            # Append padded items to lists
            padded_x_data.append(padded_x_item)
            padded_y_data.append(padded_y_item)
            padded_mask.append(mask)

        # Stack all tensors into a batch
        x_data = torch.stack(padded_x_data)
        y_data = torch.stack(padded_y_data)
        padded_mask = torch.stack(padded_mask)

        return x_data, y_data, padded_mask, hdf5_path_list


    hdf5_paths = [os.path.join(base_save_path, "demo_emb/demo_psg.hdf5")]
    label_files = [os.path.join(base_save_path, "demo_psg.csv")]
    test_dataset = SleepEventClassificationDataset(sleep_staging_config, channel_groups, split="test", hdf5_paths=hdf5_paths, label_files=label_files)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=sleep_event_finetune_full_collate_fn)

    # Validation loop at the end of each epoch
    model.eval()
    all_targets = []
    all_logits = []
    all_outputs = []
    all_masks = []
    all_paths = []

    count = 0
    with torch.no_grad():
        for (x_data, y_data, padded_matrix, hdf5_path_list) in tqdm.tqdm(test_loader, desc="Evaluating"):
            x_data, y_data, padded_matrix, hdf5_path_list = x_data.to(device), y_data.to(device), padded_matrix.to(device), list(hdf5_path_list)
            outputs, mask = sleep_staging_model(x_data, padded_matrix)
            all_targets.append(y_data.cpu().numpy())
            all_outputs.append(torch.softmax(outputs, dim=-1).cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_paths.append(hdf5_path_list)


    save_path = os.path.join(base_save_path, "demo_sleep_staging")
    os.makedirs(save_path, exist_ok=True)

    targets_path = os.path.join(save_path, "all_targets.pickle")
    outputs_path = os.path.join(save_path, "all_outputs.pickle")
    logits_path = os.path.join(save_path, "all_logits.pickle")
    mask_path = os.path.join(save_path, "all_masks.pickle")
    file_paths = os.path.join(save_path, "all_paths.pickle")

    save_data(all_targets, targets_path)
    save_data(all_outputs, outputs_path)
    save_data(all_logits, logits_path)
    save_data(all_masks, mask_path)
    save_data(all_paths, file_paths)

    print(f"Sleep staging outputs shape: {all_outputs[0].shape}, targets shape: {all_targets[0].shape}")
    print(f"Number of batches: logits={len(all_logits)}, outputs={len(all_outputs)}, targets={len(all_targets)}, masks={len(all_masks)}")
    print(f"First batch shapes: logits={all_logits[0].shape}, outputs={all_outputs[0].shape}, targets={all_targets[0].shape}, masks={all_masks[0].shape}")

    all_logits_flat = [logits.reshape(-1, logits.shape[-1]) for logits in all_logits]
    all_outputs_flat = [outputs.reshape(-1, outputs.shape[-1]) for outputs in all_outputs]
    all_targets_flat = [targets.reshape(-1) for targets in all_targets]
    all_masks_flat = [mask.reshape(-1) for mask in all_masks]

    # Convert lists of flattened arrays to single concatenated arrays if desired
    all_logits_flat = np.concatenate(all_logits_flat, axis=0)
    all_outputs_flat = np.concatenate(all_outputs_flat, axis=0)
    all_targets_flat = np.concatenate(all_targets_flat, axis=0)
    all_masks_flat = np.concatenate(all_masks_flat, axis=0)

    print(f"Flattened shapes: logits={all_logits_flat.shape}, outputs={all_outputs_flat.shape}, targets={all_targets_flat.shape}, masks={all_masks_flat.shape}")

    mask_filter = all_masks_flat == 0

    # Apply the mask to each flattened array
    all_logits_filtered = all_logits_flat[mask_filter]
    all_outputs_filtered = all_outputs_flat[mask_filter]
    all_targets_filtered = all_targets_flat[mask_filter]

    counts = Counter(all_targets_filtered)
    total = sum(counts.values())
    prevalence_dict = {cls: count / total for cls, count in counts.items()}
    print(f"Class prevalence: {prevalence_dict}")

    class_labels = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}

    # Step 1: Get predicted labels (argmax on probabilities)
    predicted_labels = np.argmax(all_outputs_filtered, axis=1)

    fontsize = 12

    # Step 2: Compute F1 score for each class
    f1_scores = f1_score(all_targets_filtered, predicted_labels, average=None, labels=range(len(class_labels)))
    for idx, label in enumerate(class_labels):
        print(f"F1 Score for {label}: {f1_scores[idx]:.3f}")

    # Step 3: Create a confusion matrix and normalize it by row to get percentages
    conf_matrix = confusion_matrix(all_targets_filtered, predicted_labels, labels=range(len(class_labels)))
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

    # Plotting the confusion matrix with percentages
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"size": fontsize},  # Font size for numbers inside the heatmap
        cbar_kws={"shrink": 1},  # Adjust colorbar size
    )

    # Customizing axis labels and ticks
    plt.xlabel("Predicted Labels", fontsize=fontsize)
    plt.ylabel("True Labels", fontsize=fontsize)
    plt.xticks(fontsize=12, ha="center")  # Font size for x-axis tick labels with rotation
    plt.yticks(fontsize=12)  # Font size for y-axis tick labels

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(save_path, 'confusion_matrix.png')}")






# # ============================================================================
# # Part 3: Disease Prediction
# # ============================================================================

if disease_prediction:
    # Initialize variables needed for Part 3
    base_save_path = "demo_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Part 3 - Using device: {device}")
    
    # Load channel groups for Part 3
    channel_groups_path = "../sleepfm/configs/channel_groups.json"
    channel_groups = load_data(channel_groups_path)

    disease_model_path = "../sleepfm/checkpoints/model_diagnosis"
    config = load_data(os.path.join(disease_model_path, "config.json"))

    config["model_params"]["dropout"] = 0.0
    model_params = config['model_params']
    model_class = getattr(sys.modules[__name__], config['model'])
    model = model_class(**model_params).to(device)
    model_name = type(model).__name__

    if device.type == "cuda":
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print("Using CPU (no DataParallel)")

    print(f"Model initialized: {model_name}")
    total_layers, total_params = count_parameters(model)
    print(f'Trainable parameters: {total_params / 1e6:.2f} million')
    print(f'Number of layers: {total_layers}')

    checkpoint_path = os.path.join(disease_model_path, "best.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel state dict: remove 'module.' prefix if present
    if list(checkpoint.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)


    class DiagnosisFinetuneFullCOXPHWithDemoDataset(Dataset):
        def __init__(self, 
                    config,
                    channel_groups,
                    hdf5_paths=None,
                    demo_labels_path=None,
                    split="train"):

            self.config = config
            self.channel_groups = channel_groups
            self.max_channels = self.config["max_channels"]

            # --- Load demographic features ---
            if not demo_labels_path:
                demo_labels_path = config["demo_labels_path"]

            demo_labels_df = pd.read_csv(demo_labels_path)
            demo_labels_df = demo_labels_df.set_index("Study ID")
            study_ids = set(demo_labels_df.index)

            is_event_df = pd.read_csv(os.path.join(self.config["labels_path"], "is_event.csv"))
            event_time_df = pd.read_csv(os.path.join(self.config["labels_path"], "time_to_event.csv"))

            is_event_df = is_event_df.set_index('Study ID')
            event_time_df = event_time_df.set_index('Study ID')

            # --- Resolve HDF5 paths (explicit precedence) ---
            if hdf5_paths:
                # Use provided paths directly
                hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
            else:
                # Load from split file
                split_paths = load_data(config["split_path"])[split]
                hdf5_paths = [f for f in split_paths if os.path.exists(f)]

            # Filter by available demo labels
            hdf5_paths = [
                f for f in hdf5_paths
                if os.path.basename(f).split(".")[0] in study_ids
            ]

            # Optional truncation
            if config.get("max_files"):
                hdf5_paths = hdf5_paths[:config["max_files"]]

            labels_dict = {}
            # Loop over each study_id
            for study_id in tqdm.tqdm(study_ids):
                # Extract the row as a whole for both dataframes (faster than iterating over columns)
                is_event_row = list(is_event_df.loc[study_id].values)
                event_time_row = list(event_time_df.loc[study_id].values)
                demo_feats = list(demo_labels_df.loc[study_id].values)

                labels_dict[study_id] = {
                    "is_event": is_event_row,
                    "event_time": event_time_row, 
                    "demo_feats": demo_feats
                }

            # --- Build index map ---
            self.index_map = [
                (path, labels_dict[os.path.basename(path).split(".")[0]])
                for path in hdf5_paths
            ]

            print(f"Number of files in {split} set: {len(hdf5_paths)}")
            print(f"Number of files to be processed in {split} set: {len(self.index_map)}")

            self.total_len = len(self.index_map)
            self.max_seq_len = config["model_params"]["max_seq_length"]

            if self.total_len == 0:
                raise ValueError(f"No valid HDF5 files found for split='{split}'.")

        def __len__(self):
            return self.total_len

        def __getitem__(self, idx):
            hdf5_path, tte_event = self.index_map[idx]

            event_time = tte_event["event_time"]
            is_event = tte_event["is_event"]
            demo_feats = tte_event["demo_feats"]

            x_data = []
            with h5py.File(hdf5_path, 'r') as hf:
                dset_names = []
                for dset_name in hf.keys():
                    if isinstance(hf[dset_name], h5py.Dataset) and dset_name in self.config["modality_types"]:
                        dset_names.append(dset_name)
                
                random.shuffle(dset_names)
                for dataset_name in dset_names:
                    x_data.append(hf[dataset_name][:])

            if not x_data:
                # Skip this data point if x_data is empty
                return self.__getitem__((idx + 1) % self.total_len)

            # Convert x_data list to a single numpy array
            x_data = np.array(x_data)

            # Convert x_data to tensor
            x_data = torch.tensor(x_data, dtype=torch.float32)

            event_time = torch.tensor(event_time, dtype=torch.float32)
            is_event = torch.tensor(is_event) 

            demo_feats = torch.tensor(demo_feats, dtype=torch.float32)

            return x_data, event_time, is_event, demo_feats, self.max_channels, self.max_seq_len, hdf5_path


    def diagnosis_finetune_full_coxph_with_demo_collate_fn(batch):
        x_data, event_time, is_event, demo_feats, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

        num_channels = max(max_channels_list)

        if max_seq_len_list[0] == None:
            max_seq_len = max([item.size(1) for item in x_data])
        else:
            max_seq_len = max_seq_len_list[0]

        padded_x_data = []
        padded_mask = []
        for item in x_data:
            c, s, e = item.size()
            c = min(c, num_channels)
            s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

            # Create a padded tensor and a mask tensor
            padded_item = torch.zeros((num_channels, max_seq_len, e))
            mask = torch.ones((num_channels, max_seq_len))

            # Copy the actual data to the padded tensor and set the mask for real data
            padded_item[:c, :s, :e] = item[:c, :s, :e]
            mask[:c, :s] = 0  # 0 for real data, 1 for padding

            padded_x_data.append(padded_item)
            padded_mask.append(mask)
        
        # Stack all tensors into a batch
        x_data = torch.stack(padded_x_data)
        event_time = torch.stack(event_time)
        is_event = torch.stack(is_event)
        demo_feats = torch.stack(demo_feats)
        padded_mask = torch.stack(padded_mask)
        
        return x_data, event_time, is_event, demo_feats, padded_mask, hdf5_path_list


    save_path = os.path.join(base_save_path, "demo_diagnosis")
    os.makedirs(save_path, exist_ok=True)

    hdf5_paths = [os.path.join(base_save_path, "demo_emb/demo_psg.hdf5")]
    demo_labels_path = os.path.join(base_save_path, "demo_age_gender.csv")
    config["labels_path"] = base_save_path

    test_dataset = DiagnosisFinetuneFullCOXPHWithDemoDataset(config, channel_groups, split="test", hdf5_paths=hdf5_paths, demo_labels_path=demo_labels_path)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=diagnosis_finetune_full_coxph_with_demo_collate_fn)

    model.eval()
    all_event_times = []
    all_is_event = []
    all_outputs = []
    all_paths = []

    with torch.no_grad():
        for item in tqdm.tqdm(test_loader, desc="Evaluating"):
            x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = item
            x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), demo_feats.to(device), padded_matrix.to(device), list(hdf5_path_list)
            outputs = model(x_data, padded_matrix, demo_feats)
        
            logits = outputs.cpu().numpy()
            all_outputs.append(logits)
            all_event_times.append(event_times.cpu().numpy())
            all_is_event.append(is_event.cpu().numpy())
            all_paths.append(hdf5_path_list)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_is_event = np.concatenate(all_is_event, axis=0)
    all_paths = np.concatenate(all_paths)

    outputs_path = os.path.join(save_path, "all_outputs.pickle")
    event_times_path = os.path.join(save_path, "all_event_times.pickle")
    is_event_path = os.path.join(save_path, "all_is_event.pickle")
    file_paths = os.path.join(save_path, "all_paths.pickle")

    save_data(all_outputs, outputs_path)
    save_data(all_event_times, event_times_path)
    save_data(all_is_event, is_event_path)
    save_data(all_paths, file_paths)

    print(f"Disease prediction shapes: outputs={all_outputs.shape}, event_times={all_event_times.shape}, is_event={all_is_event.shape}")

    # ============================================================================
    # Disease Mapping and Results
    #
    # Above, you get the model outputs, which you can then use to look for specific 
    # disease diagnosis. Note that the shape of the output above is 1065, meaning, 
    # this model gives logprobs for 1065 conditions. We provide information about 
    # each disease index and its corresponding phecode here 
    # `sleepfm/configs/label_mapping.csv`. You can map it as follows.
    # ============================================================================

    labels_df = pd.read_csv("../sleepfm/configs/label_mapping.csv")

    labels_df["output"] = all_outputs[0]
    labels_df["is_event"] = all_is_event[0]
    labels_df["event_time"] = all_event_times[0]

    print("\nFirst few rows of disease predictions:")
    print(labels_df.head())

    # Above, you get the output hazards from our model, and also your labels for 
    # is_event and event_times. Is_event is an indicator for if the event occurred 
    # and event_time is the time to event

    print("\n=== Demo script completed successfully! ===")
