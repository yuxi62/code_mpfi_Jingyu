# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:31:53 2026

@author: Jingyu Cao
"""
import numpy as np

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d, percentile_filter


def percentile_dff_abs(F_array, window_size, q=10, t_axis=-1, return_baseline=0):
    array = cp.asarray(F_array)
    
    baseline = percentile_filter(array, q, size=window_size)
    # Add a small epsilon to the baseline to prevent division by zero in dark regions.
    epsilon = 1e-9
    
    dff = (array-baseline)/(cp.abs(baseline)+epsilon)
    
    return (dff, baseline) if return_baseline else dff

def percentile_dff(F_array, window_size, q=10, t_axis=-1, return_baseline=0):
    array = cp.asarray(F_array)
    
    baseline = percentile_filter(array, q, size=window_size)
    # Add a small epsilon to the baseline to prevent division by zero in dark regions.
    epsilon = 1e-9
    
    dff = (array-baseline)/(baseline+epsilon)
    
    return (dff, baseline) if return_baseline else dff