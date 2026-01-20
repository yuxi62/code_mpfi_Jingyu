# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:40:03 2026

@author: Jingyu Cao
"""
import numpy as np
import cupy as cp
from tqdm import tqdm
import random

def circular_shuffle_ratio_session(session_array, times, gpu=0): #array: row, col = trials, F
    global bef, aft
    tot_roi, tot_trial, tot_t = session_array.shape
    # tot_trial = roi_array.shape[0]
    # tot_t = roi_array.shape[1]  # total time bins of input array    
    shuffle_ratios = []
    if gpu:
        session_array = cp.array(session_array)
    for i in tqdm(range(times), desc='shuffling...'):
        if gpu:
            curr_shuffle = cp.zeros(session_array.shape)
            for t in range(tot_trial):
                curr_shuffle[:, t, :]=cp.roll(session_array[:, t, :], -random.randint(1, tot_t), axis=-1)
                    
            # curr_shuffle = robust_sd_filter(curr_shuffle, gpu=1)
            curr_shuffle_mean = cp.nanmean(curr_shuffle, axis=1)
            curr_shuffle_ratio = cp.mean(curr_shuffle_mean[:, int(bef*30)+15:int(bef*30)+45], axis=-1)/\
                                 cp.mean(curr_shuffle_mean[:, int(bef*30)-45:int(bef*30)-15], axis=-1)
            shuffle_ratios.append(curr_shuffle_ratio.get())                           
    return np.vstack(shuffle_ratios).T

def circular_shuffle_mean_session(session_array, shuff_times, gpu=1):
    """
    Perform circular shuffle of trial-aligned traces for each ROI and compute mean trace across trials.

    Parameters
    ----------
    session_array : np.ndarray or cp.ndarray
        3D array of shape [n_rois, n_trials, trial_frames]
    shuff_times : int
        Number of circular shuffles to perform
    gpu : bool or int, optional
        If True (or 1), use GPU via CuPy. Otherwise use CPU.

    Returns
    -------
    shuffle_means : np.ndarray
        3D array of shape [n_rois, shuff_times, trial_frames]
    """

    # Choose backend
    xp = cp if gpu else np
    session_array = xp.array(session_array)

    n_rois, n_trials, n_frames = session_array.shape
    shuffle_means = xp.zeros((n_rois, shuff_times, n_frames), dtype=session_array.dtype)

    for i in tqdm(range(shuff_times), desc='shuffling...'):
        # Generate random shift amount for each trial (shared across ROIs)
        # shifts = xp.random.randint(1, n_frames, size=n_trials)
        # Initialize array to hold current shuffle
        shuffled = xp.empty_like(session_array)

        for t in range(n_trials):
            shuffled[:, t, :] = xp.roll(session_array[:, t, :], -random.randint(1, n_frames), axis=-1)

        # Apply optional post-shuffle processing
        # You can replace this with your own filter function if needed
        # shuffled = robust_sd_filter(shuffled, gpu=gpu)  # Optional

        # Mean over trials (axis=1)
        shuffle_means[:, i, :] = xp.nanmean(shuffled, axis=1)

    if gpu:
        return shuffle_means.get()
    else:
        return shuffle_means