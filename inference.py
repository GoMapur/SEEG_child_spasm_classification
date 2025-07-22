#!/usr/bin/env python
"""
EEG Inference Script (metadata-driven, portable)

Usage:
  python inference.py --fold_id 0 --metadata_path path/to/metadata.json --data_dir path/to/npz_data --checkpoint_path path/to/model_checkpoint.pth --output_dir path/to/output_dir

- Loads model and checkpoint
- Loads test samples from metadata
- Runs inference and saves outputs/features to CSV/NPZ
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pylab as pylab
from src.utilities import (
    transform_waveform_to_model_input_gpu,
    butter_bandpass_filter_channel_wise_all_channel,
    min_max_normalization_channel_wise,
    load_data_by_recording_id
)
from src.models import Neural_CNN
from torch.utils.data import DataLoader
from src.dataloaders.scalp_eeg_fft_metadata import ScalpEEG_FFT_Metadata_Dataset
import argparse

def load_model(checkpoint_path, device):
    model = Neural_CNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def run_inference(args):
    device = args.device
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Created folder:", args.output_dir)

    print(f"Fold {args.fold_id}")
    print("Using checkpoint:", args.checkpoint_path)
    model = load_model(args.checkpoint_path, device)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Using model without compilation.")

    # Load all test samples from metadata
    test_dataset = ScalpEEG_FFT_Metadata_Dataset(
        args.metadata_path, args.data_dir, split='test', window_size=args.window_size, mode="continuous", step_size=args.step_size, pass_band=args.pass_band, stop_band=args.stop_band)
    dloader = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=4
    )

    df_rows = []
    cnn_features = []
    classifier_features_1 = []
    classifier_features_2 = []
    mean_voltage_rows = []
    variance_voltage_rows = []

    with torch.no_grad():
        for sample in tqdm(dloader):
            batch_waveform = sample["waveform"].to(device).float()
            batch_mean = sample["mean_voltage"]
            batch_var = sample["variance_voltage"]

            input_fft = transform_waveform_to_model_input_gpu(
                batch_waveform,
                pass_band=args.pass_band,
                stop_band=args.stop_band
            )
            preds, cnn_f, c1_f, c2_f = model(input_fft, return_features=True)
            preds = preds.squeeze(1).detach().cpu().numpy()

            cnn_f = cnn_f.detach().cpu().numpy()
            c1_f = c1_f.detach().cpu().numpy()
            c2_f = c2_f.detach().cpu().numpy()

            s_inds = sample["start_ind"].detach().cpu().numpy()
            e_inds = sample["end_ind"].detach().cpu().numpy()
            labels = sample["label"].detach().cpu().numpy()
            patient_ids = sample["patient_id"].detach().cpu().numpy()
            recording_ids = np.array(sample["recording_id"])

            if isinstance(batch_mean, torch.Tensor):
                batch_mean = batch_mean.detach().cpu().numpy()
            if isinstance(batch_var, torch.Tensor):
                batch_var = batch_var.detach().cpu().numpy()

            for irow in range(len(s_inds)):
                df_rows.append([
                    args.fold_id,
                    patient_ids[irow].item() if hasattr(patient_ids[irow], 'item') else patient_ids[irow],
                    recording_ids[irow],
                    s_inds[irow],
                    e_inds[irow],
                    preds[irow],
                    labels[irow]
                ])
                cnn_features.append(cnn_f[irow])
                classifier_features_1.append(c1_f[irow])
                classifier_features_2.append(c2_f[irow])
                mean_voltage_rows.append(batch_mean[irow])
                variance_voltage_rows.append(batch_var[irow])

    # Build final
    df_infer = pd.DataFrame(df_rows, columns=[
        "model_id", "patient_id", "recording_id", 
        "start_ind", "end_ind", "outputs", "label"
    ])
    cnn_arr = np.vstack(cnn_features)
    c1_arr = np.vstack(classifier_features_1)
    c2_arr = np.vstack(classifier_features_2)
    mean_v_arr = np.vstack(mean_voltage_rows)
    var_v_arr = np.vstack(variance_voltage_rows)

    csv_path = os.path.join(args.output_dir, f"fold_{args.fold_id}_inference_outputs.csv")
    npz_path = os.path.join(args.output_dir, f"fold_{args.fold_id}_inference_features.npz")
    df_infer.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        CNN_features=cnn_arr,
        classifier_features_1=c1_arr,
        classifier_features_2=c2_arr,
        voltage_mean=mean_v_arr,
        voltage_variance=var_v_arr
    )
    print(f"[DONE] Inference complete. Results saved to {csv_path} and {npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Inference Script (metadata-driven, portable)")
    parser.add_argument('--fold_id', type=int, required=True, help='Fold index')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata.json')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with .npz EEG data')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--window_size', type=int, default=3000)
    parser.add_argument('--step_size', type=int, default=600)
    parser.add_argument('--pass_band', type=float, default=1.0)
    parser.add_argument('--stop_band', type=float, default=50.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    run_inference(args) 