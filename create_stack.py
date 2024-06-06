# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:18:36 2024

@author: Fatemeh Dalilian
"""


""" This is a module to create a stack of time-frequency representations for EEG, ET, HR, and controller data. """

import numpy as np
import os
from skimage.transform import resize
import create_spectrogram
import baseline_normalization

def resize_spectrograms(spectrograms, target_height, target_width):
    """
    Resize and reshape spectrograms to match target dimensions.

    Args:
        spectrograms (ndarray): 4D NumPy array of spectrograms
        target_height (int): Target height of spectrograms
        target_width (int): Target width of spectrograms

    Returns:
        ndarray: Resized and reshaped spectrograms
    """
    num_spectrograms, height, width, num_channels = spectrograms.shape
    reshaped_spectrograms = np.reshape(spectrograms, (num_spectrograms * height, width, num_channels))
    resized_spectrograms = resize(reshaped_spectrograms, (num_spectrograms * target_height, target_width, num_channels))
    resized_spectrograms = np.reshape(resized_spectrograms, (num_spectrograms, target_height, target_width, num_channels))

    return resized_spectrograms

def create_stack(overlap_percent, window_step_length, time_length, EEG_data, HR_data, ET_data, controller_data, participants):
    """
    Generate a stacked array of time-frequency representations for EEG, ET, HR, and controller data.

    Args:
        overlap_percent (float): Overlap percentage for STFT
        window_step_length (float): Window step length for STFT
        time_length (int): Time length of data to consider
        EEG_data (ndarray): EEG data array
        HR_data (ndarray): Heart rate data array
        ET_data (ndarray): Eye-tracking data array
        controller_data (ndarray): Controller input data array
        participants (ndarray): Participant data array

    Returns:
        ndarray: Stacked array of normalized spectrograms
    """
    ET_sampling_rate = 50
    ET_freq, ET_time, ET_spectrograms = create_spectrogram.create_spectrogram(ET_data, ET_sampling_rate, window_step_length, time_length, overlap_percent)
    
    EEG_sampling_rate = 256
    EEG_freq, EEG_time, EEG_spectrograms = create_spectrogram.create_spectrogram(EEG_data, EEG_sampling_rate, window_step_length, time_length, overlap_percent)
    
    controller_sampling_rate = 10
    controller_freq, controller_time, controller_spectrograms = create_spectrogram.create_spectrogram(controller_data, controller_sampling_rate, window_step_length, time_length, overlap_percent)
    

    #baseline_frequencies = np.load(os.path.join('baseline', f'baseline_f_{overlap_percent}_{window_step_length}.npy'))
    baseline_power = np.load(os.path.join('baseline', f'baseline_power_{overlap_percent}_{window_step_length}.npy'))

    normalized_EEG_spectrograms = baseline_normalization.baseline_normalize_spectrograms(EEG_spectrograms, baseline_power, participants)
    
    
    truncated_EEG_spectrograms = normalized_EEG_spectrograms[:, 0:40, :, :]
    _, target_height, target_width, _ = truncated_EEG_spectrograms.shape
    
    resized_ET_spectrograms = resize_spectrograms(ET_spectrograms, target_height, target_width)
    
    resized_controller_spectrograms = resize_spectrograms(controller_spectrograms, target_height, target_width)
    
    stacked_array = np.concatenate((truncated_EEG_spectrograms, resized_ET_spectrograms, resized_controller_spectrograms), axis=-1)

    return stacked_array

# Example usage (commented out)
# overlap_percent = 0.2
# window_step_length = 0.5
# time_length = 6
# EEG_data = scipy.io.loadmat('path_to_EEG_data.mat')['EEG_timeseries']
# HR_data = scipy.io.loadmat('path_to_HR_data.mat')['HR_timeseries']
# ET_data = scipy.io.loadmat('path_to_ET_data.mat')['ET_timeseries']
# controller_data = scipy.io.loadmat('path_to_controller_data.mat')['controller_timeseries']
# participants = scipy.io.loadmat('path_to_participants.mat')['participant']
# stacked_array = create_stack(overlap_percent, window_step_length, time_length, EEG_data, HR_data, ET_data, controller_data, participants)
