# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:49:52 2023

Author: Fatemeh Dalilian
"""

import numpy as np

def baseline_normalize_spectrograms(spectrograms, baseline_power, participants):
    """
    Baseline normalize the spectrograms using the baselines computed from compute_base_powers.

    Args:
        spectrograms (ndarray): 4D NumPy array of shape (sample_size, num_frequencies, window_steps, num_channels)
        baseline_power (ndarray): 4D NumPy array of shape (num_participants, num_frequencies, 1, num_channels) containing the baselines
        participants (ndarray): 1D NumPy array of shape (sample_size,) indicating which sample belongs to which participant

    Returns:
        ndarray: Baseline normalized spectrograms of the same shape as input spectrograms
    """
    num_participants = baseline_power.shape[0]
    normalized_spectrograms = spectrograms.copy()

    for par in range(num_participants):
        participant_mask = np.squeeze(participants == par)
        base = baseline_power[par]
        #indices_of_zeros = np.argwhere(baseline_power == 0)
        normalized_spectrograms[np.squeeze(participant_mask)] /= np.expand_dims(base, axis=0)

    return normalized_spectrograms

# Example usage (commented out)
# import numpy as np
# spectrograms = np.load('path_to_spectrograms.npy')
# baseline_power = np.load('path_to_baseline_power.npy')
# participants = np.load('path_to_participants.npy')
# normalized_spectrograms = baseline_normalize_spectrograms(spectrograms, baseline_power, participants)
