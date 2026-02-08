# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 23:35:49 2025

@author: Jingyu Cao
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
import utils_Jingyu as utl
import plotting_functions_Jingyu as pf
#%%
exp = 'dlight_Ai14_Dbh'
f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_new.pkl".format(exp)
df_selected = pd.read_pickle(f_out_df_selected)
rec_lst = df_selected.index.tolist()
ex_lst = [
'AC963-20250218-04',
'AC964-20250203-04',
'AC964-20250205-04',
'AC964-20250218-02',
'AC966-20250217-04',
'AC966-20250219-02',
]
error_lst = ['AC966-20250226-02', ]
# error_lst = ['AC964-20250120-04', 'AC964-20250121-02', 'AC964-20250121-04', 'AC964-20250121-06', 'AC964-20250131-02', 'AC964-20250131-04', 'AC964-20250204-02', 'AC964-20250204-04', 'AC964-20250205-02', 'AC964-20250222-02', 'AC964-20250314-02', 'AC964-20250411-02', 'AC966-20250217-02', 'AC966-20250219-04', 'AC966-20250226-02', 'AC967-20250225-04']
rec_lst = [i for i in rec_lst if i not in ex_lst and i not in error_lst]

for rec in tqdm(rec_lst):
    print(rec)
    # rec = 'AC969-20250319-04'
    beh = pd.read_pickle(r"Z:\Jingyu\Code\dlight_imgaing\dlight_Ai14_Dbh\behaviour_profile\{}.pkl".format(rec))
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    p_regression = p_data+r"\regression_result_dilation_new_fibermask_dlightmask"
    p_green_fov_trace = p_regression + r'\fov_trace_dlight.npy'
    p_red_fov_trace = p_regression + r'\fov_trace_red.npy'
    
    raw_dlight_traces = np.load(p_green_fov_trace)
    red_traces = np.load(p_red_fov_trace)

    # smoothing
    raw_dlight_traces = gaussian_filter1d(raw_dlight_traces, sigma=1, axis=-1)
    red_traces = gaussian_filter1d(red_traces, sigma=1, axis=-1)

    # align to run-onset
    alignment = 'run'; bef, aft = 1, 4
    raw_dlight_traces_aligned = utl.align_trials(raw_dlight_traces, alignment, beh, bef, aft)
    red_traces_aligned = utl.align_trials(red_traces, alignment, beh, bef, aft)

    # raw_dlight_mean = np.nanmean(raw_dlight_traces_aligned, axis=0)
    # red_traces_mean = np.nanmean(red_traces_aligned, axis=0)
    
    # calculate correlation on full trace
    df_selected['rg_corr_full_trial_single_trial_r'] = ''
    df_selected['rg_corr_full_trial_single_trial_p'] = ''

    corr_window = [-bef, aft]
    raw_dlight_tarces_window = raw_dlight_traces_aligned[:, int((bef+corr_window[0])*30):int((bef+corr_window[1])*30)]
    red_dlight_tarces_window = red_traces_aligned[:, int((bef+corr_window[0])*30):int((bef+corr_window[1])*30)]
    
    # full_trial_mean_trace correlation
    raw_dlight_window_mean = np.nanmean(raw_dlight_tarces_window, axis=0)
    red_traces_window_mean = np.nanmean(red_dlight_tarces_window, axis=0)
    # normalizatiion
    raw_dlight_window_mean_nor = utl.normalize(raw_dlight_window_mean)
    red_traces_window_mean_nor = utl.normalize(red_traces_window_mean)
    corr = pearsonr(raw_dlight_window_mean_nor, red_traces_window_mean_nor)
    df_selected.loc[rec, 'rg_corr_full_trial_mean_trace_r'] = corr[0]
    df_selected.loc[rec, 'rg_corr_full_trial_mean_trace_p'] = corr[1]
    
    # full_trial_trace single trial correlation
    # normalization
    raw_dlight_tarces_window_nor = utl.normalize(raw_dlight_tarces_window)
    red_tarces_window_nor = utl.normalize(red_dlight_tarces_window)
    corr_single_trial = pearsonr(raw_dlight_tarces_window_nor, red_tarces_window_nor, axis=-1)
    df_selected.at[rec, 'rg_corr_full_trial_single_trial_r'] = corr_single_trial[0]
    df_selected.at[rec, 'rg_corr_full_trial_single_trial_p'] = corr_single_trial[1]
    single_trial_corr_r_median = np.nanmedian(corr_single_trial[0])
    single_trial_corr_p_median = np.nanmedian(corr_single_trial[1])
    df_selected.at[rec, 'rg_corr_full_trial_single_trial_r_median'] = single_trial_corr_r_median
    df_selected.at[rec, 'rg_corr_full_trial_single_trial_p_median'] = single_trial_corr_p_median
    

#%%
    # f_out = "Z:\Jingyu\Code\dlight_imgaing\dlight_Ai14_Dbh\plot_wholeFOV_correlation"
    # if not os.path.exists(f_out):
    #     os.makedirs(f_out)
    # fig, ax = plt.subplots(dpi=200)
    # xaxis = np.arange(30*(bef+aft))/30-1
    # pf.plot_mean_trace(raw_dlight_traces_aligned,
    #                    xaxis, ax, color='green', label='original_dlight')
    # ax.legend(frameon=False)
    
    # ax = ax.twinx()
    # pf.plot_mean_trace(red_traces_aligned,
    #                    xaxis, ax, color='red', label='Ai14:Dbh')
    # ax.legend(frameon=False)
    
    # ax.set(title=f'{rec}_full_trial_mean_corr={corr[0]:.2f}_p={corr[1]:.4f}\n\
    #                single_trial_corr_median={single_trial_corr_r_median:.2f}')
    
    # plt.tight_layout()
    # plt.savefig(f_out+r'\{}.png'.format(rec), dpi=200)
    # # plt.show()
    # plt.close()
    
    

# df_selected.to_pickle(r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp))

#%% correlation with speed
    # upsample dlight trace to 1000 hz
    fs_old = 30
    fs_new = 1000
    n_old = raw_dlight_traces.shape[-1]
    duration = n_old / fs_old
    # time vectors
    t_old = np.arange(n_old) / fs_old
    t_new = np.arange(round(n_old * fs_new / fs_old)) / fs_new
    # dlight trace
    raw_dlight_tarces_1000 = np.interp(t_new, t_old, raw_dlight_traces) # linear interpolation
    # upsample frame times
    frame_times_1000 = np.interp(t_new, t_old, beh['frame_times'][:n_old])
    # find new run_onset index
    run_onsets_times = beh['run_onsets']
    
    run_frames_1000 = []
    for run in run_onsets_times:
        if np.isnan(run):
            run_frames_1000.append(0)
        else:
            nearest_frame_idx = np.argmin(np.abs(frame_times_1000 - run))
            run_frames_1000.append(nearest_frame_idx)
    
    # align upsampled trace to run
    bef, aft = 0, 4
    win_frames = int((bef+aft)*fs_new)
    tot_trial = len(run_onsets_times)
    aligned_signal = np.zeros((tot_trial, win_frames))
    for t in range(tot_trial):
        curr_trace = raw_dlight_tarces_1000[run_frames_1000[t]-int(bef*fs_new):run_frames_1000[t]+int(aft*fs_new)]
        if curr_trace.shape[0]<win_frames or run_frames_1000[t]==0:
            aligned_signal[t,:]=np.nan
        else:
            aligned_signal[t,:]=curr_trace
    raw_dlight_tarces_1000_aligned = aligned_signal     
    # raw_dlight_tarces_1000_aligned = utl.align_trials(raw_dlight_tarces_1000, alignment, beh, bef, aft, fs=fs_new)
    
    aligned_trials = ~np.isnan(raw_dlight_tarces_1000_aligned).any(axis=1)
    aligned_trials_idx = np.where(aligned_trials)[0]
    raw_dlight_traces_1000_mean = np.nanmean(raw_dlight_tarces_1000_aligned[aligned_trials], axis=0)
    # calculate spead trial mean
    speeds = [np.vstack(beh['speed_times_aligned'][i])[:4000, 1] for i in aligned_trials_idx]
    speeds = np.stack([utl.zero_padding(speed, 4000) for speed in speeds])
    speed_mean = np.nanmean(speeds, axis=0)
    
    #%% single trial correlation speed-dlight
    x = raw_dlight_tarces_1000_aligned[aligned_trials]
    y = speeds
    single_trial_corr = pearsonr(x, y, axis=-1)
    print(np.nanmedian(single_trial_corr[0]))
    df_selected.at[rec, 'speed_corr_single_trial_r_median'] = np.nanmedian(single_trial_corr[0])
    #%%
    x = speed_mean
    y = raw_dlight_traces_1000_mean
    # normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # compute cross-correlation
    corr = np.correlate(x, y, mode='full')
    lags = np.arange(-len(x)+1, len(x))
    
    # normalize to correlation coefficient
    corr /= len(x)
    
    df_selected.at[rec, 'speed_corr_peak_lag'] = lags[np.argmax(corr)]/1000
    df_selected.at[rec, 'speed_lag0_corr='] = corr[lags == 0][0]
    
    #%% plot
    f_out = r"Z:\Jingyu\Code\dlight_imgaing\{}\plot_wholeFOV_speed_correlation".format(exp)

    xaxis = np.arange(fs_new*(bef+aft))/fs_new-bef
    
    fig, axs = plt.subplots(1, 2, figsize=(5,2.5), dpi=200)
    ax=axs[0]
    pf.plot_mean_trace(raw_dlight_tarces_1000_aligned, xaxis, ax, label='raw_dlight')
    ax.legend(frameon=False, prop={'size': 6}, loc=(0.65, 0.85))
    ax=ax.twinx()
    pf.plot_mean_trace(speeds, xaxis, ax,  color='tab:blue', label='speed')
    ax.set(xlabel='time from run (s)')
    ax.set_title('{}\nmean speed and dlight trace\nsingle_trial_corr_median={:.3f}'
                 .format(rec, np.median(single_trial_corr[0])), size=8)
    ax.legend(frameon=False, prop={'size': 6}, loc=(0.65, 0.9))

    ax = axs[1]
    ax.plot(lags, corr)
    ax.set(xlabel='Lag (samples)', ylabel='Correlation')
    ax.set_title('Correlogram (1000 hz)\nspeed_mean vs raw_dlight_mean\npeak_lag={}s, lag0_corr={:.3f}'
              .format(lags[np.argmax(corr)]/1000, corr[lags == 0][0]),
              size=8)
    
    fig.tight_layout()
    plt.savefig(f_out+r'\{}.png'.format(rec), dpi=200)
    # plt.show()
    plt.close()

df_selected.to_pickle(r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp))    
    
