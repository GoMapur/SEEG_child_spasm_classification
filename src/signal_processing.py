# signal_processing.py
"""Signal processing utilities for EEG data."""
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass_filter_channel_wise_all_channel(
    waveform: np.ndarray, lowcut: float, highcut: float, fs: float, order: int
) -> np.ndarray:
    """Apply a bandpass filter to all channels of the waveform."""
    # ... (implementation from utilities.py) ...
    pass
// ... add other signal processing functions as needed ... 