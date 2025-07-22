# from pyedflib import highlevel
import numpy as np
import os
import pickle
import scipy.io as sio
from scipy import signal
from sklearn.preprocessing import normalize
import shutil
import matplotlib.pylab as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
from scipy.signal import butter, lfilter, freqz
from multiprocessing import Process
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

### measurement: sensitivity, specificity, accuracy
def compute_metric(labels, predictions):
    
    specificity, recallTP = recall_score(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)

    res_dict = {"recall": recallTP, "specificity": specificity, "accuracy": accuracy}
    return res_dict

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

def read_edf(fn):
    print(fn)
    signals, signal_headers, header = highlevel.read_edf(fn)
    return signals, signal_headers, header

def parse_txt(fn):
    res = []
    with open(fn, "r") as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(",")
        temp = []
        for ll in l:
            temp.append(ll.strip())
        res.append(temp)
    return res

from PIL import Image

def save_imgs2dir(img_list, labels, name, output_dir):
    number_img = len(img_list)
    tmp_image_dir = os.path.join(output_dir, "tmp_imgs")
    if not os.path.exists(tmp_image_dir):
        os.makedirs(tmp_image_dir)
    for i in range(number_img):
        plt.figure(figsize=(10,10))
        #im = Image.fromarray((img_list[i]* 255).astype('uint8'), mode='L')
        #im = Image.fromarray((np.tile(img_list[i],3)* 255).astype('uint8'))
        #plt.imshow(img_list[i], vmin=0, vmax=1)
        sn.heatmap(img_list[i],cmap="viridis", annot = labels, vmin=0, vmax=1,fmt = '')

        plt.savefig(os.path.join(tmp_image_dir, f'{name}_{i}.jpg'))
        
        # if im.mode != 'RGB':
        #     im = im.convert('RGB')
        #im.save(os.path.join(tmp_image_dir, f'{name}_{i}.jpg'))
    print(f"generated images to {tmp_image_dir}")

def dict2array(dic):
    res = []
    print(sorted(dic.keys()))
    for key in sorted(dic.keys()):
        res.append(dic[key])
    return np.concatenate(res)

def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        #print(fn)
    return lookup

##for each channel, specify the width of the image
def construct_win_image(data, num_win, stride_size, mode="log_norm"):
    num_index = data.shape[1]
    new_segments = [] 
    if mode == "log_norm":
        data = np.log(data + 1e-8)
    for win_idx in range(0, num_index - num_win, stride_size):
        new_segments.append(data[:, win_idx: (win_idx + num_win)])
    #print(win_idx)
    #print(f"there are {len(new_segments)} segments with win_size {num_win}")
    return new_segments

def construct_window_seg(fn, patient_name):
    loaded_data = load_pickle(fn)
    all_segments = {}
    #channel_name
    num_win = 10
    stride_size = 5
    keys = list(loaded_data.keys())
    for key in keys:
        new_segments = construct_win_image(loaded_data[key], num_win, stride_size)
        all_segments[key] = new_segments
    return all_segments


def standardize_length(signals, length, shift_mean=False, magnitute_normalize=False):
    signal_standard = np.zeros((len(signals), length))
    for cnt, sig in enumerate(signals):
        signal_resample = signal.resample(sig, length)
        if shift_mean:
            signal_resample = signal_resample - np.mean(signal_resample)
       
        signal_standard[cnt, :] = signal_resample
    if magnitute_normalize:
        signal_standard = normalize(signal_standard, norm='l2')
    return signal_standard

def plot_heatmap(data, hori_label, verti_label, title, loc):
    # Create a dataset (fake)
    plt.close("all")
    plt.figure(figsize=(20,20))
    df = pd.DataFrame(data, index= verti_label, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    #p1 = sn.heatmap(df, cmap=plt.cm.Blues, annot=True, fmt='0.2f')
    p1 = sn.heatmap(df, cmap=plt.cm.Blues, annot=True )
    plt.title(title)
    fn = os.path.join(loc, title+".jpg")
    plt.savefig(fn)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def expand_dim(value, dim):
    if len(value.shape) != dim:
        value = value.unsqueeze(0)
    return value

def normalize_img(a):
    batch_num = a.shape[0]
    h = a.shape[1]
    w = a.shape[2]
    a_reshape = a.reshape(batch_num, -1)
    a_min = torch.min(a_reshape, -1)[0].unsqueeze(1)
    a_max = torch.max(a_reshape, -1)[0].unsqueeze(1)
    c = 255.0 * (a_reshape - a_min)/(a_max - a_min)
    c = c.reshape(batch_num,h, w)
    return c

def train_test_val_split(dataX, dataY, train_ratio, test_ratio, validation_ratio):
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    return x_train, x_val, x_test, y_train, y_val, y_test

def train_test_val_id_split(ids, train_ratio, test_ratio, validation_ratio, random_state=42):
    ids_train, ids_test = train_test_split(ids, test_size=1 - train_ratio, random_state=random_state)
    ids_val, ids_test = train_test_split(ids_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state+1)
    return ids_train, ids_val, ids_test

def notch_filtering(wav, fs, w0, Q):
    """ Apply a notch (band-stop) filter to the audio signal.
    
    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: The frequency to filter. See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.
        
    Returns:
        wav: Filtered waveform.
    """
    b, a = signal.iirnotch(2 * w0/fs, Q)
    wav = signal.lfilter(b, a, wav)
    return wav

def find_file_location_by_recording_id(data_dir, eeg_recording_id):
    data_path1 = os.path.join(data_dir, f"{eeg_recording_id.upper()}.npz")
    data_path2 = os.path.join(data_dir, f"{eeg_recording_id.upper()}A.npz")
    if os.path.exists(data_path1):
        data_path = data_path1
    elif os.path.exists(data_path2):
        data_path = data_path2
    elif eeg_recording_id.upper().endswith("A") and os.path.exists(os.path.join(data_dir, f"{eeg_recording_id.upper()[:-1]}.npz")):
        data_path = os.path.join(data_dir, f"{eeg_recording_id.upper()[:-1]}.npz")
    else:
        raise FileNotFoundError(f"{eeg_recording_id} not found in {data_path1} or {data_path2}")
    return data_path

def load_data_by_recording_id(data_dir, eeg_recording_id, n_channels=19, scale_factor=1e6, verbose=False):
    data_path = find_file_location_by_recording_id(data_dir, eeg_recording_id)
    loaded_npz = np.load(data_path, allow_pickle=True)
    eeg_channel_load = loaded_npz['channel'][:n_channels]
    eeg_data = np.vstack(loaded_npz['data'])[:n_channels,:] * scale_factor
    if verbose:
        print(f"{eeg_recording_id} loaded from {data_path}")
    return eeg_data, eeg_channel_load

def butter_bandpass_filter_channel_wise_all_channel(data, lowcut, highcut, sample_rate, order=5):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    data = data.copy()
    for i in range(data.shape[0]):
        data[i, :] = butter_bandpass_filter(data[i, :], lowcut, highcut, sample_rate, order)
    return data

def min_max_normalization_channel_wise(data):
    data = data.copy()
    enumerator = (data.max(axis=1)[:, None] - data.min(axis=1)[:, None])
    enumerator[enumerator==0.0] = 1.0
    data = (data - data.min(axis=1)[:, None]) / enumerator
    return data

def replace_zeroes_channel_wise(data, default_value=1e-12):
    ret =  []
    for i in range(data.shape[0]):
        row = data[i, :]
        if np.min(row) == 0.0:
            min_nonzero = default_value
        else:
            min_nonzero = np.min(row[np.nonzero(row)])
        row[row == 0] = min_nonzero
        ret += [row]
    return np.vstack(ret)

def plot_waveform(plot_data, channels):
    plt.figure(figsize=(32,16))

    plt.plot(plot_data.T * 2e5 + 80 * np.arange(18,-1,-1))
    plt.plot(np.zeros_like(plot_data.T) + 80*np.arange(18,-1,-1),'--',color='gray')
    plt.yticks([])
    plt.axis('tight')
    plt.legend(channels, loc='upper right')
    plt.show()


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def apply_bandstop_to_waveform(original_waveform, sampling_freq, rectangles, pass_band, stop_band):
    # print("before applying filter:", original_waveform.max(), original_waveform.min(), rectangles)
    original_waveform = original_waveform.copy()
    for rectangle in rectangles:
        (leftmost, rightmost, topmost, bottommost) = rectangle
        original_waveform = butter_bandreject_filter_channel_wise(original_waveform, pass_band + leftmost/15, pass_band + (rightmost+1)/15, sampling_freq, np.arange(topmost, bottommost+1))
        # print("removing band:", pass_band + leftmost/15, pass_band + (rightmost+1)/15, "from", topmost, bottommost)
    # print("after applying filter:", original_waveform.max(), original_waveform.min(), rectangles)
    return original_waveform

def apply_band_pass_to_waveform(original_waveform, sampling_freq, rectangles, pass_band, stop_band):
    # print("before applying filter:", original_waveform.max(), original_waveform.min(), rectangles)
    original_waveform = original_waveform.copy()
    ret_waveform = np.zeros_like(original_waveform)
    for rectangle in rectangles:
        (leftmost, rightmost, topmost, bottommost) = rectangle
        band_passed_waveform = butter_bandpass_filter_channel_wise(original_waveform, pass_band + leftmost/15, pass_band + (rightmost+1)/15, sampling_freq, np.arange(topmost, bottommost+1))
        for i in np.arange(topmost, bottommost+1):
            ret_waveform[i, :] += band_passed_waveform[i, :].copy()
    # print("after applying filter:", original_waveform.max(), original_waveform.min(), rectangles)

    return ret_waveform

def butter_bandreject_filter_channel_wise(data, lowcut, highcut, sample_rate, application_channel_indeces, order=3):
    data = data.copy()

    for i in application_channel_indeces:
        data[i, :] = butter_bandstop_filter(data[i, :], lowcut, highcut, sample_rate, order)
    return data

def butter_bandpass_filter_channel_wise(data, lowcut, highcut, sample_rate, application_channel_indeces, order=3):
    data = data.copy()

    for i in application_channel_indeces:
        data[i, :] = butter_bandpass_filter(data[i, :], lowcut, highcut, sample_rate, order)
    return data

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-3)
    high = min(highcut / nyq, 0.99)

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-3)
    high = min(highcut / nyq, 0.99)

    i, u = butter(order, [low, high], btype='bandpass')
    y = lfilter(i, u, data)
    return y

def find_occlusion_rectangle_with_threshold(data, threshold):
    data = (data>=threshold).astype(np.uint8)
    if data.sum() == 0:
        return []
    ret, labels = cv2.connectedComponents(data)
    rectangles = []
    for label in range(1, ret):
        idx = np.where(labels==label)
        leftmost = np.min(idx[1])
        rightmost = np.max(idx[1])
        topmost = np.min(idx[0])
        bottommost = np.max(idx[0])
        rectangles += [[leftmost, rightmost, topmost, bottommost]]
    return rectangles

def find_occlusion_channel_wise_band_with_threshold(data, threshold):
    data = (data>=threshold).astype(np.uint8)
    if data.sum() == 0:
        return []
    
    rectangles = []
    for i in range(data.shape[0]):
        ret, labels = cv2.connectedComponents(data[i, :][None, :])
        for label in range(1, ret):
            idx = np.where(labels==label)
            leftmost = np.min(idx[1])
            rightmost = np.max(idx[1])
            topmost = i
            bottommost = i
            rectangles += [[leftmost, rightmost, topmost, bottommost]]
    return rectangles

    
def min_max_normalization_channel_wise_gpu(data):
    data = data.clone()
    enumerator = (data.max(dim=2, keepdim=True)[0] - data.min(dim=2, keepdim=True)[0])
    enumerator[enumerator==0.0] = 1.0
    data = (data - data.min(dim=2, keepdim=True)[0]) / enumerator
    return data

# def replace_zeroes_channel_wise_gpu(data, default_value=1e-12):
#     ret =  []
#     for i in range(data.shape[0]):
#         row = data[i, :, :]
#         if torch.min(row) == 0.0:
#             min_nonzero = default_value
#         else:
#             min_nonzero = torch.min(row[torch.nonzero(row)])
#         row[row == 0] = min_nonzero
#         ret += [row]
#     return torch.vstack(ret)

def transform_waveform_to_model_input_gpu(waveform, pass_band, stop_band): 
    fourier_transform = torch.fft.rfft(waveform, dim=-1)
    abs_fourier_transform = torch.abs(fourier_transform)
    power_spectrum = abs_fourier_transform[:, :, int(pass_band*15+1):int(15*stop_band)+1]
    
    # power_spectrum = replace_zeroes_channel_wise_gpu(power_spectrum)
    power_spectrum_min_max_normalized = min_max_normalization_channel_wise_gpu(power_spectrum)
    # power_spectrum_addition = torch.log(power_spectrum + 1)

    # power_spectrum = torch.stack([power_spectrum_min_max_normalized, power_spectrum_addition])

    return power_spectrum_min_max_normalized
