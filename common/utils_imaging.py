# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:31:53 2026

@author: Jingyu Cao
"""
import numpy as np
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d, percentile_filter


def rolling_min(array,win):
    length = array.shape[-1]
    half_win = int(np.ceil(win/2))
    array_padding = np.hstack((array[:,:half_win], array, array[:,-half_win:length]))
    output = np.array([np.min(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
    return output

def rolling_max(array,win):
    length = array.shape[-1]
    half_win = int(np.ceil(win/2))
    array_padding = np.hstack((array[:,:half_win], array, array[:,-half_win:length]))
    output = np.array([np.max(array_padding[:,i:i+win], axis=1) for i in range(half_win,length+half_win)]).T
    return output

def get_dff(F_array, win=1800, sigma=300, gpu=0, return_baseline=0, time_axis=-1): # by default (what has been used in matlab pipeline, window=60s, sigma=10s)
    '''
    calculate dFF for each ROI
    
    F: raw F signal, whole session
    win: window used for calculate baseline
    sigma: for gaussian filtering
    
    return dFF
    
    '''
    # F_array = np.asarray(F_array)
    if F_array.ndim == 1:
        F_array = F_array[np.newaxis, :] 
        
    if gpu ==0:
        baseline = sci_gaussian_filter(F_array, sigma, axes=time_axis)
        # baseline = gaussian_filter1d(F, sigma)
        baseline = rolling_min(baseline,win)
        baseline = rolling_max(baseline,win)
    elif gpu ==1:
        F_array = cp.array(F_array)
        # baseline = gaussian_filter(F_array, sigma, axis=time_axis, gpu=1)
        # baseline = rolling_filter(baseline, 'min', win, axis=time_axis, gpu=1)
        # baseline = rolling_filter(baseline, 'max', win, axis=time_axis, gpu=1)
        baseline = gaussian_filter1d(F_array, sigma=sigma, axis=time_axis)
        baseline = minimum_filter1d(baseline, size=win, axis=time_axis)
        baseline = maximum_filter1d(baseline, size=win, axis=time_axis)
    dff = (F_array-baseline)/baseline
    
    if return_baseline:
        return dff, baseline
    else:    
        return dff

def percentile_dff_abs(F_array, window_size=1800, q=10, t_axis=-1, return_baseline=0):
    """
    Calculate dF/F using percentile baseline with absolute value normalization.

    Parameters
    ----------
    F_array : array_like
        Raw fluorescence array, shape (n_rois, n_frames) or (n_frames,)
    window_size : int
        Window size for percentile filter (default: 1800 frames = 60s at 30Hz)
    q : float
        Percentile for baseline (default: 10)
    t_axis : int
        Time axis (default: -1, last axis)
    return_baseline : bool
        Whether to return baseline along with dF/F

    Returns
    -------
    dff : ndarray
        Delta F over F (normalized by absolute baseline)
    baseline : ndarray (optional)
        Baseline fluorescence if return_baseline=True
    """
    array = cp.asarray(F_array)

    # For 2D arrays, size must match array dimensions
    # size=(1, window_size) applies filter only along time axis
    if array.ndim == 2:
        if t_axis == -1 or t_axis == 1:
            filter_size = (1, window_size)
        else:
            filter_size = (window_size, 1)
    else:
        filter_size = window_size

    baseline = percentile_filter(array, q, size=filter_size)
    epsilon = 1e-9

    dff = (array - baseline) / (cp.abs(baseline) + epsilon)

    return (dff.get(), baseline.get()) if return_baseline else dff.get()


def percentile_dff(F_array, window_size=1800, q=10, t_axis=-1, return_baseline=0):
    """
    Calculate dF/F using percentile baseline.

    Parameters
    ----------
    F_array : array_like
        Raw fluorescence array, shape (n_rois, n_frames) or (n_frames,)
    window_size : int
        Window size for percentile filter (default: 1800 frames = 60s at 30Hz)
    q : float
        Percentile for baseline (default: 10)
    t_axis : int
        Time axis (default: -1, last axis)
    return_baseline : bool
        Whether to return baseline along with dF/F

    Returns
    -------
    dff : ndarray
        Delta F over F
    baseline : ndarray (optional)
        Baseline fluorescence if return_baseline=True
    """
    array = cp.asarray(F_array)

    # For 2D arrays, size must match array dimensions
    # size=(1, window_size) applies filter only along time axis
    if array.ndim == 2:
        if t_axis == -1 or t_axis == 1:
            filter_size = (1, window_size)
        else:
            filter_size = (window_size, 1)
    else:
        filter_size = window_size

    baseline = percentile_filter(array, q, size=filter_size)
    epsilon = 1e-9

    dff = (array - baseline) / (baseline + epsilon)

    return (dff.get(), baseline.get()) if return_baseline else dff.get()