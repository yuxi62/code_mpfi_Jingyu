# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:02:17 2025

@author: Jingyu Cao
"""
#%%
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import cupy as cp
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.stats import sem, median_abs_deviation
from cupyx.scipy.ndimage import median_filter, gaussian_filter1d
if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
if ("Z:\Yingxue\code\PythonMotionCorrectionGECO" in sys.path) == False:
    sys.path.append("Z:\Yingxue\code\PythonMotionCorrectionGECO")
import Select_GCaMP_ROIs_Jingyu, Generate_masks, Select_GECO_ROIs
# importlib.reload(Select_GCaMP_ROIs_Jingyu)

import anm_list_running as anm
import utils_Jingyu as utl
from robust_sd_filter import robust_filter_along_axis, detect_active_trial
#%% functions
def robust_zscore(data, axis=-1, keepdims=False, gpu=1):
    """
    Compute robust z-scores using MAD along an axis.

    Parameters
    ----------
    data : array-like (NumPy or CuPy)
    axis : int or tuple, default None
        Axis along which to compute z-scores.
    keepdims : bool
        If True, retains reduced dimensions in the result.
    gpu : 1 = use CuPy, 0 = use NumPy

    Returns
    -------
    z : NumPy or CuPy array
        Robust z-score normalized array.
    """
    if gpu == 1:
        xp = cp
        data = xp.asarray(data)

        # always keepdims=True internally so broadcasting works
        median = xp.nanmedian(data, axis=axis, keepdims=True)

        mad = xp.nanmedian(xp.abs(data - median), axis=axis, keepdims=True)
        # scale='normal' → std ≈ MAD / 0.67449
        mad = mad / 0.67449
    else:
        xp = np
        data = xp.asarray(data)

        # same idea: keepdims=True internally
        median = xp.nanmedian(data, axis=axis, keepdims=True)
        mad = median_abs_deviation(
            data, axis=axis, scale='normal', nan_policy='omit', keepdims=True
        )

    # avoid division by zero *per slice*
    mad_safe = xp.where(mad == 0, xp.nan, mad)
    z = (data - median) / mad_safe
    # replace NaNs (where mad==0) with 0
    # z = xp.nan_to_num(z, nan=0.0)

    # now handle keepdims for the *output*
    # if not keepdims and axis is not None:
    #     z = xp.squeeze(z, axis=axis)

    return z

def cp_sem(arr, axis=1, nan_policy="propagate"):
    """
    Standard error of the mean along a given axis for a CuPy array.

    Parameters
    ----------
    arr : cp.ndarray
    axis : int
        Axis along which to compute the SEM (default 1).
    nan_policy : {'propagate', 'omit'}
        - 'propagate': return NaN if any NaNs are present in the slice
        - 'omit'     : ignore NaNs (matches scipy.stats.sem)

    Returns
    -------
    cp.ndarray
        SEM values with the specified axis collapsed.
    """
    if nan_policy == "omit":
        # sample s.d. with ddof=1, ignoring NaNs
        sd   = cp.nanstd(arr, axis=axis, ddof=1)
        n    = cp.count_nonzero(~cp.isnan(arr), axis=axis)
    else:  # 'propagate'
        sd   = cp.std(arr, axis=axis, ddof=1)
        n    = arr.shape[axis]

    return sd / cp.sqrt(n)

# def robust_std_filter(vector):
#     # thresh = np.nanmean(vector)+1*np.nanstd(vector)
#     # thresh = median_abs_deviation(vector)/0.6745
#     thresh = median_abs_deviation(vector)*4
#     vector[vector<thresh]=0
#     return vector

# def nd_to_list(arr):
#     """Return nested Python lists or None."""
#     if arr is None:
#         return None
#     # up-cast to float32 only to keep Parquet files small
#     return arr.astype("float32").tolist()
def nd_to_list(arr):
    """
    Convert a numpy / cupy / xarray array to nested Python lists of float32.

    - If `arr` is None (or you deliberately pass something falsey), return [].
    - Otherwise, cast to float32 to keep Parquet footprint small and call .tolist().
    """
    if arr is None:
        return []          # no data → empty list
    return arr.astype("float32").tolist()

def roi_map_to_list(roi_map):
    """
    Convert ROI map of shape (n_roi, H, W) into a list of dicts
    [{'npix': int, 'ypix': array, 'xpix': array}, ...]
    """
    roi_list = []
    n_roi = roi_map.shape[0]

    for i in range(n_roi):
        ypix, xpix = np.where(roi_map[i] > 0)  # coordinates where roi is active
        roi_list.append({
            'npix': len(ypix),
            'ypix': ypix,
            'xpix': xpix
        })
    return roi_list


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

# main function
def concat_F_to_df(rec, concat_path, active_soma_only=True):
    
    INI_PATH = concat_path
    df_cal_profile_all = pd.DataFrame(
        {'anm_id': str,
         'date': str,
         'unit_id': int,
         'map': [],
         'npix': int,
         "mean_chan2": float,
         "active_soma": bool,
         # 'is_cell': bool,
         
         'dff_baseline_ss1': [],
         'dff_baseline_ss2': [],
         'dff_baseline_ss3': [],
         
         'full_dff_trace_raw_ss1': [],
         'full_dff_trace_raw_ss2': [],
         'full_dff_trace_raw_ss3': [],
         
         'full_dff_trace_rsd_ss1': [],
         'full_dff_trace_rsd_ss2': [],
         'full_dff_trace_rsd_ss3': [],

         'full_dff_trace_zscore_ss1': [],
         'full_dff_trace_zscore_ss2': [],
         'full_dff_trace_zscore_ss3': [],

         'full_dff_trace_zscore_rsd_ss1': [],
         'full_dff_trace_zscore_rsd_ss2': [],
         'full_dff_trace_zscore_rsd_ss3': [],

         # 'run_ratio_ss1': float,
         # 'run_cal_profile_all_ss1': [],
         # 'run_cal_profile_ss1': [],
         # 'sem_ss1': [],
         
         # 'run_ratio_ss2': float,
         # 'run_cal_profile_all_ss2': [],
         # 'run_cal_profile_ss2': [],
         # 'sem_ss2':[],
         
         # 'run_ratio_ss3': float,
         # 'run_cal_profile_all_ss3': [],
         # 'run_cal_profile_ss3': [],
         # 'sem_ss3':[],
        }
        )
    
    sig_master = xr.open_dataarray(INT_PATH+r"\sig_master.nc").rename(
            {"master_uid": "unit_id"}
        )
    A_master = xr.open_dataarray(INT_PATH+r"\A_master.nc").rename(
        {"master_uid": "unit_id"}
        )
    
    F_all = np.array(sig_master).squeeze()
    roi_map = np.array(A_master).squeeze()
    
    gcamp_stats = roi_map_to_list(roi_map)
    np.save(INT_PATH + r'\unit_id_stat.npy', np.asarray(gcamp_stats, dtype='object'))
    
    path_result = Path(r"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\results_masks\{}".format(
                                                        anm_id+'-'+date))
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    
    p_suite2p_ss1 = r"Z:\Jingyu\2P_Recording\{}\{}\02\suite2p\plane0".format(anm_id, 
                                                        anm_id+'-'+date)
    ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
    mean_img = ops['meanImg']
    global_gcamp_mask = Generate_masks.generate_and_save_dlight_mask(
                        mean_img=mean_img, 
                        output_dir=path_result,
                        output_filename_base="global_gcamp_mask",
                        gaussian_sigma=1.5,
                        peak_min_distance=5,
                        adaptive_block_size=5
                        )
    # classification_thresholds = {
    #     'min_soma_roi_npix': 100, # number of pixels in each ROI
    #     'max_soma_roi_npix': 600,
    #     'compactness_threshold': 0.3, # compactness calculated by 4 * pi * npix / (perimeter**2)
    #     'hollowness_threshold': 0.25, # whether the ROI & global_geco_mask is hollow
    #     'area_threshold': 1000, # area of the square box around the ROI
    #     'skewness_threshold': 0.8 # skewness of F-0.7*Fneu
    # }
    
    classification_thresholds = {
            'min_soma_roi_npix': 100, # number of pixels in each ROI
            'max_soma_roi_npix': 600,
            'compactness_threshold_min': 0.1, # compactness calculated by 4 * pi * npix / (perimeter**2) default: 0.2
            'hollowness_threshold_min': 0.25, # whether the ROI & global_geco_mask is hollow default: 0.25
            'area_threshold': 2000, # area of the square box around the ROI
            'skewness_threshold': 0.8, # skewness of F-0.7*Fneu,
            'aspect_ratio_max': 3,
        }

    
    active_soma = Select_GCaMP_ROIs_Jingyu.classify_and_save_somas(
        stat_array=gcamp_stats,
        f = F_all,
        # fneu_path=fneu_path,
        mean_img=mean_img,
        soma_mask=global_gcamp_mask,
        output_dir=path_result,
        thresholds=classification_thresholds,
        figure_on=True
     )
    
    dff_all, baseline_all = utl.get_dff(F_all, gpu=1, return_baseline=1)
    dff_all = dff_all.get(); baseline_all = baseline_all.get()
    dff_all = utl.trace_filter(dff_all)
    # dff_all_sm = gaussian_filter1d(cp.array(dff_all), sigma=1).get()
    
    nframes = []
    # behs = []
    for ss in sessions:
        session = anm_id+'-'+date+'-'+ss
        p_suite2p = r"Z:\Jingyu\2P_Recording\{}\{}\{}\suite2p\plane0".format(anm_id, 
                                                            anm_id+'-'+date, 
                                                            ss,)    
        # p_beh = r"Z:\Jingyu\2P_Recording\all_session_info\{}\all_anm_behaviour\{}.pkl".format(exp, session)
        
        # behs.append(pd.read_pickle(p_beh))
        
        ops = np.load(p_suite2p+r'\ops.npy', allow_pickle=True).item()
        nframes.append(ops['nframes'])
    
        ref_chan2 = ops['meanImg_chan2_corrected']
    
    # Flatten rec_chan2 and roi_map for vectorized computation
    ref_flat = ref_chan2.ravel()  # shape (512*512,)
    roi_map_tmp = roi_map.copy()
    roi_map_tmp[roi_map_tmp>0] = 1
    roi_flat = roi_map_tmp.reshape(roi_map_tmp.shape[0], -1)  # shape (n_rois, 512*512)
    
    # Count pixels per ROI
    roi_pixel_counts = roi_flat.sum(axis=1)  # shape (n_rois,)
    
    # Multiply and sum pixel values for each ROI
    roi_sums = roi_flat @ ref_flat  # shape (n_rois,)
    
    # Compute mean by dividing sum by pixel count
    roi_means = roi_sums / roi_pixel_counts
        
    dff_all_ss1 = dff_all[:, :nframes[0]]
    dff_all_ss2 = dff_all[:, nframes[0]:nframes[0]+nframes[1]]
    baseline_ss1 = baseline_all[:, :nframes[0]]
    baseline_ss2 = baseline_all[:, nframes[0]:nframes[0]+nframes[1]]
    
    # actitvify filter using roubust sd
    dff_all_rsd_ss1 = robust_filter_along_axis(dff_all_ss1, factor=2, gpu=1).get()
    dff_all_rsd_ss2 = robust_filter_along_axis(dff_all_ss2, factor=2, gpu=1).get()
    
    # robust zscore for dff trace
    dff_all_zscore_ss1 = robust_zscore(dff_all_ss1, gpu=1).get()
    dff_all_zscore_ss2 = robust_zscore(dff_all_ss2, gpu=1).get()
    
    # filtered zscored dff trace
    dff_all_zscore_rsd_ss1 = dff_all_zscore_ss1*(dff_all_rsd_ss1>0)
    dff_all_zscore_rsd_ss2 = dff_all_zscore_ss2*(dff_all_rsd_ss2>0)
    
    # plot: validate zcore
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    win = slice(4000, 5500)
    ax.plot(dff_all_zscore_ss2[27, win], color='green', lw=.8, label='zscored_raw_dff')
    ax.plot(dff_all_ss2[27, win], color='grey', lw=.8, label='raw_dff')
    ax.set(ylabel='dF/F', ylim=(-3, 12))
    ax = ax.twinx()
    ax.plot(dff_all_zscore_rsd_ss2[27, win], color='tab:red', lw=.8, 
            label='zscored_filterd_dF/F', alpha=.6)
    ax.plot(dff_all_rsd_ss2[27, win], color='tab:blue', lw=.8,
            label='filtered_dff', alpha=.7)
    ax.set(ylabel='robust_sd_filterd_dF/F', ylim=(-3, 12))
    fig.legend(frameon=False, prop={'size': 8})
    plt.show()
    
    # # thresholding test
    # dff_test = dff_all_ss1[15 :]
    # dff_test_thres = dff_test.copy()
    # thresh = median_abs_deviation(dff_test)/0.6745
    # dff_test_thres[dff_test<thresh]=0
    # plt.plot(dff_test[:3000], lw=.6)
    # plt.plot(dff_test_thres[:3000], lw=.6)
    
    # dff_all_ss1_thresh = np.apply_along_axis(robust_std_filter, 1, dff_all_ss1)
    # dff_all_ss2_thresh = np.apply_along_axis(robust_std_filter, 1, dff_all_ss2)
    
    # dff_all_ss1_thresh = cp.array(dff_all_ss1_thresh)
    # dff_all_ss2_thresh = cp.array(dff_all_ss2_thresh)
    
    # run_aligned_ss1 = utl.align_trials(dff_all_ss1, 'run', behs[0], bef, aft, gpu=1)
    # run_aligned_ss2 = utl.align_trials(dff_all_ss2, 'run', behs[1], bef, aft, gpu=1)
    
    # ss1_avg = cp.nanmean(run_aligned_ss1, axis=1)
    # ss1_sem = cp_sem(run_aligned_ss1, axis=1, nan_policy='omit')
    # ss1_avg_nr = utl.normalize(ss1_avg, gpu=1)
    # ro_ratio_ss1 = cp.nanmean(ss1_avg_nr[:, int(bef*30)+15:int(bef*30)+45], axis=-1)/\
    #                   cp.nanmean(ss1_avg_nr[:, int(bef*30)-45:int(bef*30)-15], axis=-1)  
    
    # ss2_avg = cp.nanmean(run_aligned_ss2, axis=1)
    # ss2_sem = cp_sem(run_aligned_ss2, axis=1, nan_policy='omit')
    # ss2_avg_nr = utl.normalize(ss2_avg, gpu=1)
    # ro_ratio_ss2 = cp.nanmean(ss2_avg_nr[:, int(bef*30)+15:int(bef*30)+45], axis=-1)/\
    #                   cp.nanmean(ss2_avg_nr[:, int(bef*30)-45:int(bef*30)-15], axis=-1)
    
    # (ss1_avg, ss1_sem, ro_ratio_ss1, ss2_avg, ss2_sem, ro_ratio_ss2,
    #  run_aligned_ss1, run_aligned_ss2) = [
    #      arr.get() for arr in
    #      (ss1_avg, ss1_sem, ro_ratio_ss1, ss2_avg, ss2_sem,
    #       ro_ratio_ss2,
    #       run_aligned_ss1, run_aligned_ss2)
    #      ]      
         
    if len(sessions) ==3:
        dff_all_ss3 = dff_all[:, nframes[0]+nframes[1]:]
        baseline_ss3 = baseline_all[:, nframes[0]+nframes[1]:]
        dff_all_rsd_ss3 = robust_filter_along_axis(dff_all_ss3, factor=2, gpu=1).get()
        dff_all_zscore_ss3 = robust_zscore(dff_all_ss3, gpu=1).get()
        dff_all_zscore_rsd_ss3 = dff_all_zscore_ss3[dff_all_rsd_ss3>0]
        # run_aligned_ss3 = utl.align_trials(dff_all_ss3, 'run', behs[2], bef, aft, gpu=1)
        # ss3_avg = cp.nanmean(run_aligned_ss3, axis=1)
        # ss3_sem = cp_sem(run_aligned_ss3, axis=1, nan_policy='omit')
        # ss3_avg_nr = utl.normalize(ss3_avg, gpu=1)
        # ro_ratio_ss3 = cp.nanmean(ss3_avg_nr[:, int(bef*30)+15:int(bef*30)+45], axis=-1)/\
        #                   cp.nanmean(ss3_avg_nr[:, int(bef*30)-45:int(bef*30)-15], axis=-1)         
    
        # (ss3_avg, ss3_sem, ro_ratio_ss3, run_aligned_ss3) = [
        #      arr.get() for arr in
        #      (ss3_avg, ss3_sem, ro_ratio_ss3, run_aligned_ss3)
        #      ]     
 
    
    for roi in tqdm(range(F_all.shape[0]), desc='extracting roi info...'):
        # if len(sessions) ==3:
        #     roi_ratio_ss3 = ro_ratio_ss3[roi]
        #     roi_avg_ss3 = ss3_avg[roi,:],
        #     roi_sem_ss3 = ss3_sem[roi,:]
        # else:
        #     roi_ratio_ss3, roi_avg_ss3, roi_sem_ss3 = [], [], []
    
        
        row = {
            "anm_id": anm_id,
            "date": date,
            "unit_id": roi,
            "map": nd_to_list(roi_map[roi]), # 512×512 → list-of-list
            "npix": np.sum(roi_map[roi]>0),
            "mean_chan2": roi_means[roi],
            "active_soma": active_soma[roi],
            
            'baseline_ss1': nd_to_list(baseline_ss1[roi]),
            'baseline_ss2': nd_to_list(baseline_ss2[roi]),
            'baseline_ss3': nd_to_list(baseline_ss3[roi]) if locals().get("baseline_ss3") is not None else [],
            
            'full_dff_trace_raw_ss1': dff_all_ss1[roi],
            'full_dff_trace_raw_ss2': dff_all_ss2[roi],
            'full_dff_trace_raw_ss3': dff_all_ss3[roi] if locals().get("dff_all_ss3") is not None else [],
            
            'full_dff_trace_rsd_ss1': dff_all_rsd_ss1[roi],
            'full_dff_trace_rsd_ss2': dff_all_rsd_ss2[roi],
            'full_dff_trace_rsd_ss3': dff_all_rsd_ss3[roi] if locals().get("dff_all_rsd_ss3") is not None else [],
            
            'full_dff_trace_zscore_ss1': dff_all_zscore_ss1[roi],
            'full_dff_trace_zscore_ss2': dff_all_zscore_ss2[roi],
            'full_dff_trace_zscore_ss3': dff_all_zscore_ss3[roi] if locals().get("dff_all_zcore_ss3") is not None else [],

            'full_dff_trace_zscore_rsd_ss1': dff_all_zscore_rsd_ss1[roi],
            'full_dff_trace_zscore_rsd_ss2': dff_all_zscore_rsd_ss2[roi],
            'full_dff_trace_zscore_rsd_ss3': dff_all_zscore_rsd_ss3[roi] if locals().get("dff_all_zscore_rsd_ss3") is not None else [],

            # "run_ratio_ss1": float(ro_ratio_ss1[roi]),
            # "run_cal_profile_all_ss1": nd_to_list(run_aligned_ss1[roi]),
            # "run_cal_profile_ss1":   nd_to_list(ss1_avg[roi]),
            # "sem_ss1":               nd_to_list(ss1_sem[roi]),
            
            # "run_ratio_ss2": float(ro_ratio_ss2[roi]),
            # "run_cal_profile_all_ss2": nd_to_list(run_aligned_ss2[roi]),
            # "run_cal_profile_ss2":   nd_to_list(ss2_avg[roi]),
            # "sem_ss2":               nd_to_list(ss2_sem[roi]),
            
            # "run_ratio_ss3": float(ro_ratio_ss3[roi]) if locals().get("ro_ratio_ss3") is not None else np.nan,
            # "run_cal_profile_all_ss3": nd_to_list(run_aligned_ss3[roi]) if locals().get("run_aligned_ss3") is not None else [],
            # "run_cal_profile_ss3":     nd_to_list(ss3_avg[roi]) if locals().get("ss3_avg") is not None else [],
            # "sem_ss3":                 nd_to_list(ss3_sem[roi]) if locals().get("ss3_sem") is not None else [],
        }
        # if len(sessions) == 3:
        #     row.update({
        #         "run_ratio_ss3": float(ro_ratio_ss3[roi]),
        #         "run_cal_profile_all_ss3": nd_to_list(run_aligned_ss3[roi]),
        #         "run_cal_profile_ss3":     nd_to_list(ss3_avg[roi]),
        #         "sem_ss3":                 nd_to_list(ss3_sem[roi]),
        #     })
        
        df_cal_profile_all.loc[len(df_cal_profile_all)] = row #!!!! not sure what was this used for
        
        # plot single ROIs avg profile
        # !to do
        # if plot_rois:
    if active_soma_only:
        return df_cal_profile_all.loc[df_cal_profile_all['active_soma']]
    else:
        return df_cal_profile_all    
#%%
exp = r'GCaMP8s_infusion'
p_session_info = r'Z:\Jingyu\Code\Python\2p_SCH23390_infusion\session_info_new.pkl'
if not os.path.exists(p_session_info):
    df_session_info = []
else:
    df_session_info = pd.read_pickle(p_session_info)
# rec_lst = df_session_info.loc[(df_session_info['n_good_trials_ss1']>60)&
#                               (df_session_info['n_good_trials_ss2']>60)&
#                               (df_session_info['latency']==20)]
rec_lst = df_session_info.loc[
                            (df_session_info['perc_valid_trials_ss1']>0.6)&
                            (df_session_info['perc_valid_trials_ss2']>0.6)&
                            (df_session_info['latency']==20)
                            # (df_session_info['anm']=='AC316')
                            # &(df_session_info['date']=='20251104'),
                            ]
  
for rec_idx, rec in rec_lst.iterrows():
# for rec in rec_lst:    
    anm_id = rec['anm']
    sessions = rec['session']
    ss = rec['session']
    n_sessions = len(ss)
    date = rec['date']
    
    # df_session_info = update_session_info(df_session_info, rec)
#%%    
    print('{}_{}_processing...--------------------------'.format(anm_id, date))
    # INT_PATH = r"Z:\Jingyu\2P_Recording\{}\{}\concat".format(anm_id, anm_id+'-'+date)
    INT_PATH = r"Z:\Jingyu\2P_Recording\{}\{}\concat".format(anm_id, anm_id+'-'+date)
    p_df_profile = INT_PATH+r"\{}_{}_df_cal_profile_all_valid_trials_new.parquet".format(anm_id, date)
    
    # process_df = 1
    if not os.path.exists(p_df_profile) and ('muscimol' not in rec['label']):
    # if process_df:
        bef, aft = 2, 4
        df_cal_profile_all = concat_F_to_df(rec, INT_PATH,
                                            active_soma_only = True)
        session_info = df_session_info.loc[anm_id+'-'+date]
        df_cal_profile_all = identify_cell_response(rec, df_cal_profile_all, session_info,
                                                    trial_bef=2, trial_aft=4, pre_window=(-1, 0), post_window=(0, 1.5),
                                                    )
        
        # df_cal_profile_all = df_cal_profile_all.loc[df_cal_profile_all['active_soma']]
        df_cal_profile_all.drop(columns=[c for c in 
                                         [
                                         'run_cal_profile_all_raw_ss1',
                                         'run_cal_profile_all_raw_ss2',
                                         'run_cal_profile_all_raw_ss3',
                                         'run_cal_profile_all_rsd_ss1',
                                         'run_cal_profile_all_rsd_ss2',
                                         'run_cal_profile_all_rsd_ss3',
                                         ] 
                                         if c in df_cal_profile_all.columns]).to_parquet(
            p_df_profile,
            engine="pyarrow",
            compression="zstd",        # zstd is ~20-30 % smaller than snappy with similar speed
            compression_level=3,       # 1-3 keeps it CPU-cheap
            # row_group_size=1_000_000,  # tweak if columns are very wide/narrow
            index=False                # skip the RangeIndex column
        )
        
        del df_cal_profile_all
    else:
        print(anm_id+'-'+date+'processed already')

# df_session_info.to_pickle(p_session_info)
