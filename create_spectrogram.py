# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:30:16 2023

Author: Fatemeh Dalilian
"""

import numpy as np
import pywt
import os
import scipy.io
import h5py
import pandas as pd
from scipy.signal import stft
from scipy import signal

def create_spectrogram(data, srate, window_step_length, time_length, overlap_percent):
    """
    Create a spectrogram from a 9-channel EEG signal using the Short-Time Fourier Transform (STFT).
    
    Args:
        data (ndarray): 3D NumPy array of shape (sample_size, data_length, num_channels)
        srate (int): Sampling rate of the data
        window_step_length (float): Length of the window step during spectrogram creation
        time_length (int): Length of the EEG data to consider in seconds
        overlap_percent (float): Percentage overlap between windows
    
    Returns:
        tuple: Frequencies, times, and spectrograms as 4D NumPy array
    """
    data = data[:, -srate*time_length:, :]
    sample_size, data_size, num_channels = data.shape
    
    window_size = int(window_step_length * srate)
    overlap_size = int(window_size * overlap_percent)
    
    if overlap_size >= window_size:
        raise ValueError('Overlap must be less than the window size.')

    window_number = int((data_size - overlap_size) / (window_size - overlap_size)) + 2
    frequency_number = int((window_size / 2) + 1)
    
    # Initialize an empty 4D array to store the spectrograms
    spectrograms = np.zeros((sample_size, frequency_number, window_number, num_channels))
    
    for sample in range(sample_size):
        for channel in range(num_channels):
            sample_data = data[sample, :, channel]
            f, t, Zxx = signal.stft(sample_data, fs=srate, nperseg=window_size, noverlap=overlap_size)
            powers = np.abs(Zxx) ** 2
            spectrograms[sample, :, :, channel] = powers
    
    return f, t, spectrograms

# Example usage (commented out)
# EEG_data = scipy.io.loadmat('path_to_EEG_data.mat')['EEG_timeseries']
# srate = 256
# window_step_length = 0.5
# time_length = 6
# overlap_percent = 0.5
# freq, time, spectrograms = create_spectrogram(EEG_data, srate, window_step_length, time_length, overlap_percent)
