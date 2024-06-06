# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 17:06:11 2023

Author: Fatemeh Dalilian
"""

import numpy as np
import scipy.io
import pandas as pd
from scipy import signal
import os

# Load participant data
participants = scipy.io.loadmat('/Matlab/Mat/participant_timeseries.mat')['participant']
overlap_percent = 0.2
window_step_length = 0.25

def compute_base_powers(window_step_length, overlap, open_eyes=True):
    """
    Compute baseline powers for EEG and heart rate signals.

    Parameters:
    window_step_length (float): Length of the window step in seconds.
    overlap (float): Percentage overlap between windows.
    open_eyes (bool): Whether to use open eyes condition.

    Returns:
    tuple: Frequencies, baseline powers for EEG and heart rate signals.
    """
    num_channels = 9
    EEGsrate = 256
    window_size = int(window_step_length * EEGsrate)
    num_frequencies = int((window_size / 2) + 1)
    baseline_power = np.zeros((49, num_frequencies, 1, num_channels))
    HR_baseline_power = np.zeros((49, num_frequencies, 1, 1))

    # Ensure overlap is less than window size
    noverlap = int(overlap * window_size)
  

    for par in range(1, 50):
        base_data = pd.read_csv(f"/Matfiles/base/{par}.csv")
        if open_eyes:
            rest_epoch = base_data[base_data['Annotations_Rest_Open_Active'] == 1]
        else:
            rest_epoch = base_data[base_data['Annotations_Rest_Closed_Active'] == 1]

        eeg_columns = rest_epoch.columns[31:40]
        eeg_signals = rest_epoch[eeg_columns].dropna(subset=eeg_columns)
        EEG_rest_signals = eeg_signals.tail(256 * 60)

        HR_column = rest_epoch.columns[30:31]
        HR_signal = rest_epoch[HR_column].dropna(subset=HR_column)
        HR_rest_signal = HR_signal.tail(256 * 60)

        HR_baseline_f, HR_baseline_t, HR_Zxx = signal.stft(HR_rest_signal.iloc[:, 0], fs=EEGsrate, nperseg=window_size, noverlap=noverlap)
        HR_powers = np.abs(HR_Zxx) ** 2
        HR_baseline_power[par - 1, :, 0, 0] = np.mean(HR_powers, 1)

        for channel in range(num_channels):
            EEG_rest_channel = EEG_rest_signals.iloc[:, channel]
            baseline_f, baseline_t, Zxx = signal.stft(EEG_rest_channel, fs=EEGsrate, nperseg=window_size, noverlap=noverlap)
            powers = np.abs(Zxx) ** 2
            baseline_power[par - 1, :, 0, channel] = np.mean(powers, 1)

    return baseline_f, baseline_power, HR_baseline_f, HR_baseline_power

# Compute baseline powers
baseline_f, baseline_power, HR_baseline_f, HR_baseline_power = compute_base_powers(window_step_length, overlap_percent, open_eyes=True)

# Save results
os.makedirs('baseline', exist_ok=True)
np.save(os.path.join('baseline', f'baseline_f_{overlap_percent}_{window_step_length}.npy'), baseline_f)
np.save(os.path.join('baseline', f'baseline_power_{overlap_percent}_{window_step_length}.npy'), baseline_power)
np.save(os.path.join('baseline', f'HR_baseline_f_{overlap_percent}_{window_step_length}.npy'), HR_baseline_f)
np.save(os.path.join('baseline', f'HR_baseline_power_{overlap_percent}_{window_step_length}.npy'), HR_baseline_power)
