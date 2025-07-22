import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.Neural_CNN import Neural_CNN
from src.dataloaders.scalp_eeg_fft_metadata import ScalpEEG_FFT_Metadata_Dataset
from src.utilities import (
    butter_bandpass_filter_channel_wise_all_channel,
    min_max_normalization_channel_wise,
    load_data_by_recording_id,
    find_occlusion_channel_wise_band_with_threshold,
    apply_bandstop_to_waveform,
    transform_waveform_to_model_input_gpu,
    # Add other needed utilities here
)
from captum.attr import Occlusion
import matplotlib.pylab as pylab
import argparse

cm = 1/2.54
params = {'legend.fontsize': 6,
          'figure.figsize': (15*cm, 8*cm),
         'axes.labelsize': 6,
         'axes.titlesize':6,
         'xtick.labelsize':6,
         'ytick.labelsize':6}
pylab.rcParams.update(params)

def parse_args():
    parser = argparse.ArgumentParser(description="Frequency Patient-wise Model Interpretation (Occlusion Analysis)")
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata JSON file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to EEG data directory')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save results')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing model checkpoints (by fold)')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing log files (by fold)')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name (used for ckpt/log subfolders)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation (e.g., cuda:0 or cpu)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for occlusion analysis')
    parser.add_argument('--wanted_test_set', type=str, default='0-49', help='Test set patient IDs (e.g., "0-49" or "1,2,3")')
    parser.add_argument('--split', type=str, default='test', help='Data split to use (default: test)')
    return parser.parse_args()

def parse_wanted_test_set(s):
    if '-' in s:
        start, end = map(int, s.split('-'))
        return list(range(start, end+1))
    return [int(x) for x in s.split(',') if x.strip()]

def load_model(ckpt_path, device):
    model = Neural_CNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def get_fold_ckpt_and_log(experiment_name, fold_id):
    import glob
    from natsort import natsorted
    ckpt_folder = os.path.join(args.ckpt_dir, experiment_name, f'fold{fold_id}', '*')
    ckpt_path = natsorted(glob.glob(ckpt_folder))[-1]
    log_path = os.path.join(args.log_dir, experiment_name, f'fold{fold_id}', 'epoch_final_test_val.csv')
    return ckpt_path, log_path

def plot_model_input(fig, ax, data, cmap, channels, title):
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

def plot_occlusion_group_wise_mean_std(save_path, occlusion_preds, channels):
    # occlusion_preds: (N, C, F)
    occlusion_preds = np.array(occlusion_preds)
    mean = np.mean(occlusion_preds, axis=0)
    std = np.std(occlusion_preds, axis=0)
    cmap = plt.get_cmap("gist_ncar")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    plot_model_input(fig, ax, mean, cmap, channels, "Group-wise Occlusion Mean")
    fig.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return mean, std

if __name__ == '__main__':
    args = parse_args()
    args.wanted_test_set = parse_wanted_test_set(args.wanted_test_set)
    print("[INFO] Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Load metadata and dataset
    print(f"Loading dataset from {args.metadata_path} ...")
    dataset = ScalpEEG_FFT_Metadata_Dataset(
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        split=args.split,
        window_size=3000,
        n_channels=19,
        sample_rate=200,
        mode="continuous",
        step_size=3000,
        pass_band=1,
        stop_band=50
    )

    # Extract patient IDs from metadata
    patient_ids = sorted(set([s['patient_id'] for s in dataset.samples]))
    channels = None
    if len(dataset.samples) > 0:
        first_recording_id = dataset.samples[0]['recording_id']
        case_data_loaded = np.load(os.path.join(args.data_dir, f"{first_recording_id}.npz"), allow_pickle=True)
        channels = case_data_loaded['channel'][:19]

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    ckpt_exist = False
    if Path(os.path.join(args.save_path, "occlusion_contrast_data.npz")).exists():
        print("The save path already exists. Replotting...")
        ckpt_exist = True
        occlusion_contrast_data_loaded = np.load(os.path.join(args.save_path, "occlusion_contrast_data.npz"), allow_pickle=True)
        save_patient_ids = occlusion_contrast_data_loaded["patient_ids"]
        save_recording_ids = occlusion_contrast_data_loaded["recording_ids"]
        save_start_ids = occlusion_contrast_data_loaded["start_ids"]
        save_occlusions = occlusion_contrast_data_loaded["occlusions"]
        save_dataset_ids_in_datasets = occlusion_contrast_data_loaded["dataset_ids_in_datasets"]
        save_preds = []
    else:
        save_patient_ids = []
        save_recording_ids = []
        save_start_ids = []
        save_occlusions = []
        save_dataset_ids_in_datasets = []
        save_preds = []

    def get_fold_ckpt_and_log(experiment_name, fold_id):
        import glob
        from natsort import natsorted
        ckpt_folder = os.path.join(args.ckpt_dir, experiment_name, f'fold{fold_id}', '*')
        ckpt_path = natsorted(glob.glob(ckpt_folder))[-1]
        log_path = os.path.join(args.log_dir, experiment_name, f'fold{fold_id}', 'epoch_final_test_val.csv')
        return ckpt_path, log_path

    for fold_id in range(5):
        print(f"Fold {fold_id}")
        ckpt_path, log_path = get_fold_ckpt_and_log(args.experiment_name, fold_id)
        model = load_model(ckpt_path, args.device)
        log_df = pd.read_csv(log_path)
        test_set = log_df["patient_names"].unique()
        for test_id in test_set:
            if test_id not in args.wanted_test_set:
                continue
            print(f"occlusion on patient {test_id}")
            patient_samples = [i for i, s in enumerate(dataset.samples) if s['patient_id'] == test_id]
            if not patient_samples:
                continue
            patient_subset = torch.utils.data.Subset(dataset, patient_samples)
            dataloader = DataLoader(patient_subset, batch_size=64, num_workers=1, pin_memory=True, shuffle=True)
            all_model_preds = []
            for batch in dataloader:
                waveforms = batch['waveform'].float().to(args.device)
                input_imgs = transform_waveform_to_model_input_gpu(waveforms, 1, 50)
                model_preds = model(input_imgs).detach().cpu().numpy().flatten()
                all_model_preds += model_preds.tolist()
            all_model_preds = np.array(all_model_preds)
            if not ckpt_exist:
                occlusion = Occlusion(model)
                strides = (1, 2)
                sliding_window_shapes = (19, 75)
                baselines = 0.0  # TODO: Use average control if available
                attribution_preds_cpu = []
                start_ids_list = []
                for batch in dataloader:
                    waveforms = batch['waveform'].float().to(args.device)
                    input_imgs = transform_waveform_to_model_input_gpu(waveforms, 1, 50)
                    start_inds = batch['start_ind']
                    attribution_preds_cuda = occlusion.attribute(
                        input_imgs,
                        strides=strides,
                        sliding_window_shapes=sliding_window_shapes,
                        baselines=baselines
                    )
                    attribution_preds_cpu_single = attribution_preds_cuda.squeeze().cpu().detach().numpy()
                    if len(attribution_preds_cpu_single.shape) != 3:
                        attribution_preds_cpu_single = attribution_preds_cpu_single[None, :, :]
                    attribution_preds_cpu += attribution_preds_cpu_single.tolist()
                    start_ids_list += start_inds.tolist()
                attribution_preds_cpu = np.array(attribution_preds_cpu)
                start_ids_list = np.array(start_ids_list)
            else:
                continue
            for i in range(attribution_preds_cpu.shape[0]):
                start_id = start_ids_list[i]
                model_pred = all_model_preds[i]
                save_preds = np.concatenate((save_preds, np.array([model_pred])))
                if not ckpt_exist:
                    save_patient_ids += [test_id]
                    save_recording_ids += [dataset.samples[patient_samples[i]]['recording_id']]
                    save_start_ids += [start_id]
                    save_occlusions += [attribution_preds_cpu[i]]
                    save_dataset_ids_in_datasets += [0]
    if not ckpt_exist:
        save_patient_ids = np.array(save_patient_ids)
        save_recording_ids = np.array(save_recording_ids)
        save_start_ids = np.array(save_start_ids)
        save_occlusions = np.array(save_occlusions)
        save_dataset_ids_in_datasets = np.array(save_dataset_ids_in_datasets)
        np.savez_compressed(os.path.join(args.save_path, "occlusion_contrast_data.npz"),
            patient_ids=save_patient_ids,
            recording_ids=save_recording_ids,
            start_ids=save_start_ids,
            occlusions=save_occlusions,
            dataset_ids_in_datasets=save_dataset_ids_in_datasets)
    # Group-wise mean/std plotting (already present)
    if not ckpt_exist:
        print("Generating group-wise mean/std occlusion plots...")
        for unique_recording_id in np.unique(save_recording_ids):
            selected_ids = (save_recording_ids == unique_recording_id) & (save_preds >= 0.5)
            if np.sum(selected_ids) == 0:
                print(f"No positive predictions for recording {unique_recording_id}")
                continue
            patient_id = save_patient_ids[selected_ids][0]
            occlusion_preds = save_occlusions[selected_ids]
            start_ids = save_start_ids[selected_ids]
            groupwise_path = os.path.join(args.save_path, f"{patient_id}/{patient_id}_{unique_recording_id}_groupwise.png")
            mean, std = plot_occlusion_group_wise_mean_std(groupwise_path, occlusion_preds, channels)
            max_change = np.max(mean)
            plt.figure()
            plt.hist([max_change])
            plt.title(f"Max occlusion change for recording {unique_recording_id}")
            plt.savefig(os.path.join(args.save_path, f"{patient_id}/{unique_recording_id}_occlusion_max_changes.png"))
            plt.close()
    print("Done.") 