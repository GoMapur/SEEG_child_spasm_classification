import json
import numpy as np
import os
from torch.utils.data import Dataset
from src.utilities import load_data_by_recording_id, butter_bandpass_filter_channel_wise_all_channel, min_max_normalization_channel_wise

class ScalpEEG_FFT_Metadata_Dataset(Dataset):
    """
    Metadata-driven version of ScalpEEG_FFT_Dataset.
    Loads sample info from a metadata JSON file, but retains all original logic:
    windowing, filtering, normalization, and returns a dictionary with all metadata for each sample.
    """
    def __init__(self, metadata_path, data_dir, split='train', window_size=3000, n_channels=19, sample_rate=200, mode="continuous", step_size=3000, random_sample_num=8, pass_band=0.1, stop_band=60):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.samples = [
            dict(patient_id=patient['patient_id'], recording_id=rec['recording_id'], label=rec['label'])
            for patient in metadata['patients']
            for rec in patient['recordings']
            if rec['split'] == split
        ]
        self.data_dir = data_dir
        self.window_size = window_size
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.mode = mode
        self.step_size = step_size
        self.random_sample_num = random_sample_num
        self.pass_band = pass_band
        self.stop_band = stop_band
        self._precompute_lengths()

    def _precompute_lengths(self):
        self.lengths = []
        for sample in self.samples:
            original_waveform, _ = load_data_by_recording_id(self.data_dir, sample['recording_id'])
            if self.mode == "random":
                self.lengths.append(self.random_sample_num)
            else:
                max_start = original_waveform.shape[1] - self.window_size
                self.lengths.append(max_start // self.step_size + 1)
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _find_sample(self, idx):
        # Find which sample this idx belongs to
        sample_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[sample_idx]
        return sample_idx, local_idx

    def __getitem__(self, idx):
        sample_idx, local_idx = self._find_sample(idx)
        sample = self.samples[sample_idx]
        patient_id = sample['patient_id']
        recording_id = sample['recording_id']
        label = sample['label']
        original_waveform, eeg_channel_load = load_data_by_recording_id(self.data_dir, recording_id)
        processed_original_waveform = butter_bandpass_filter_channel_wise_all_channel(
            original_waveform, self.pass_band, self.stop_band, self.sample_rate, order=5
        )
        processed_original_waveform -= np.mean(processed_original_waveform, axis=0)
        processed_original_waveform = min_max_normalization_channel_wise(processed_original_waveform)
        if self.mode == "random":
            start_ind = np.random.randint(0, processed_original_waveform.shape[1] - self.window_size)
        else:
            start_ind = local_idx * self.step_size
        end_ind = start_ind + self.window_size
        waveform_window = processed_original_waveform[:, start_ind:end_ind]
        original_waveform_window = original_waveform[:, start_ind:end_ind]
        mean_voltage = np.mean(original_waveform_window, axis=1)
        variance_voltage = np.var(original_waveform_window, axis=1)
        sample_dict = {
            "patient_id": patient_id,
            "recording_id": recording_id,
            "start_ind": start_ind,
            "end_ind": end_ind,
            "power_spectrum": 1,  # placeholder
            "label": label,
            "waveform": waveform_window,
            "original_waveform": original_waveform_window,
            "mean_voltage": mean_voltage,
            "variance_voltage": variance_voltage,
        }
        return sample_dict 