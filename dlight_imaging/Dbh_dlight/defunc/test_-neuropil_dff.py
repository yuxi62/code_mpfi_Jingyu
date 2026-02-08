# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:37:16 2026

@author: Jingyu Cao
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

# sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
# sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu")
from common.trial_selection import seperate_valid_trial
from common.utils_imaging import percentile_dff, align_trials
from common.utils_basic import trace_filter
from common.event_response_quantification import quantify_event_response
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils as utl
from common import plotting_functions_Jingyu as pf
from common.plot_single_trial_function import align_frame_behaviour, plot_single_trial_html_roi
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\dff_validation_-neu_AC964-20250131")
regression_name ='single_trial_regression'
# DILATION_STEPS = (2, 4, 6, 8, 10)
DILATION_STEPS = (0, ) # for testing


rec_lst = ['AC964-20250131-02', ] # for testing
for rec in tqdm(rec_lst):
    print(f'\nprocessing {rec}...')
    # load run-onset event frames
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    beh_aligned = align_frame_behaviour(beh)
    run_onset_frames = np.array(beh['run_onset_frames'])
    valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
    run_onset_frames_valid = run_onset_frames[valid_trials]
    
    for k_size in tqdm(DILATION_STEPS):
        print(f'\ndilation={k_size}...')
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / r'dilation_k={}'.format(k_size))
        # if not (p_regression / f'{rec}_profile_stat.parquet').exists():
            
        # load raw traces 
        raw_traces = np.load(OUR_DIR_REGRESS/rec/f'{rec}_raw_traces_k={k_size}.npz')
        dlight_raw = raw_traces['original_dlight']
        red_trace  = raw_traces['red_trace']
        dlight_neu_trace = raw_traces['neuropil_dlight']
        raw_traces.close()
        # lode regression result traces
        regressed_dlight = np.load(p_regression/'corrected_dlight_trace_-p2neu.npy')
        # dlight_corr = regressed_dlight - 0.6*dlight_neu_trace
        dlight_corr = regressed_dlight
        x, y, T = dlight_raw.shape
        dff_dlight_corr, baseline_all = percentile_dff(dlight_corr.reshape(x*y, T), q=20, return_baseline=True)
        dff_dlight_corr = dff_dlight_corr.reshape(x, y, T)
        baseline_all =  baseline_all.reshape(x, y, T)
        dff_dlight_raw, baseline_dlight_raw = percentile_dff(dlight_raw.reshape(x*y, T), q=15, return_baseline=True)
        dff_dlight_raw = dff_dlight_raw.reshape(x, y, T)
        baseline_dlight_raw =  baseline_dlight_raw.reshape(x, y, T)
        dff_dlight_corr_10, baseline_all_10 = percentile_dff(dlight_corr.reshape(x*y, T), q=10, return_baseline=True)
        dff_dlight_corr_10 = dff_dlight_corr_10.reshape(x, y, T)
        baseline_all_10 =  baseline_all_10.reshape(x, y, T)
        # list of grids with signals extracted
        valid_grids = utl.get_valid_grids(regressed_dlight)
        
        
        for i in range(10):
            roi = valid_grids[i]
            roi_dlight_raw = dlight_raw[roi]
            roi_neu = dlight_neu_trace[roi]
            roi_red = red_trace [roi]
            roi_dlight_corr = dlight_corr[roi]
            roi_dff = dff_dlight_corr[roi]
            roi_baseline = baseline_all[roi]
            roi_dff_raw = dff_dlight_raw[roi]
            roi_baseline_raw = baseline_dlight_raw[roi]
            roi_dff_10 = dff_dlight_corr_10[roi]
            roi_baseline_10 = baseline_all_10[roi]


            # roi_dff, roi_baseline = percentile_dff(roi_dlight_corr, q=20, return_baseline=True)
            roi_dlight_corr_aligned = align_trials(roi_dff, 'run', beh, 1, 4)
            
            fig = plot_single_trial_html_roi(beh_aligned, roi_dlight_corr, roi_dff_10, 
                                             roi_baseline_10,
                                             labels = {
                                                'ch1': 'F_corr',
                                                'ch2': 'dff',
                                                'ch3': '10% baseline'
                                                }, 
                                             colors ={
                                                 'ch1': 'grey',
                                                 'ch2': 'green',
                                                 'ch3': 'blue'},
                                            shared_yaxis=['ch1', 'ch3']
                                            )
            fig.write_html(OUT_DIR_FIG / f'example_dlight_p2neu_10perc_dff_{roi}.html')
            # fig.show()
            
            # fig, ax = plt.subplots(dpi=300)
            # ax.plot(roi_dlight[:1000], color='green', label='raw dlight', alpha=.7)
            # ax.plot(roi_neu[:1000], color='pink', label='dlight_neu')
            # ax.plot(roi_dlight_corr[:1000], color='blue', label='dlight_corr', alpha=.7)
            # ax.legend(frameon=False)
            # plt.show()
            
            # fig, ax = plt.subplots(dpi=300)
            # ax.plot(roi_dlight_corr[:1000], color='blue', label='dlight_corr', alpha=.7)
            # ax.plot(roi_baseline[:1000], color='orange', label='20% baseline', alpha=1)
            # ax.set(ylabel= 'F (a.u.)')
            # tax = ax.twinx()
            # tax.plot(100*roi_dff[:1000], color='darkred', label='dlight_corr_dff', alpha=.7)
            # tax.set(ylabel='% dFF')
            # ax.legend(frameon=False)
            # tax.legend(frameon=False)
            # plt.show()
            
            # fig, ax = plt.subplots(dpi=300)
            # ax.hist(roi_baseline, bins=50)
            # ax.set(xlabel='baseline F', ylabel='frame count')
            # plt.show()
        
        neu_mean = np.nanmean(dlight_neu_trace, axis=-1)
        baseline_mean = np.nanmean(baseline_all_10, axis=-1)
        baseline_meidan = np.nanmedian(baseline_all_10, axis=-1)
        fig, ax = plt.subplots(dpi=300)
        ax.hist(baseline_mean.ravel(), bins=30)
        ax.set(xlabel='baseline mean F', ylabel='roi count')
        plt.show()
        
        x = neu_mean.ravel()
        y = baseline_mean.ravel()
        
        # IMPORTANT: keep x/y paired when dropping NaNs
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        
        res = scipy.stats.linregress(x, y)
        
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, alpha=0.6)
        ax.set(ylim=(0,25))
        
        # fitting line
        xx = np.linspace(x.min(), x.max(), 200)
        yy = res.slope * xx + res.intercept
        ax.plot(xx, yy, lw=2, label=f"fit: y = {res.slope:.3g}x + {res.intercept:.3g}")
        
        # slope text (plus optional stats)
        ax.text(
            0.02, 0.98,
            f"slope = {res.slope:.4g}\n$R^2$ = {res.rvalue**2:.4g}\np = {res.pvalue:.3g}",
            transform=ax.transAxes, va="top", ha="left",
            # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        )
        
        ax.set(xlabel="neu_F_mean", ylabel="dlight_20% baseline mean")
        ax.legend(frameon=False)
        plt.show()
        
        
        ## 10% baseline
        neu_mean = np.nanmean(dlight_neu_trace, axis=-1)
        baseline_mean = np.nanmean(baseline_all_10, axis=-1)
        baseline_meidan = np.nanmedian(baseline_all_10, axis=-1)
        fig, ax = plt.subplots(dpi=300)
        ax.hist(baseline_mean.ravel(), bins=30)
        ax.set(xlabel='baseline mean F', ylabel='roi count')
        plt.show()
        
        x = neu_mean.ravel()
        y = baseline_mean.ravel()
        
        # IMPORTANT: keep x/y paired when dropping NaNs
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        
        res = scipy.stats.linregress(x, y)
        
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, alpha=0.6)
        
        # fitting line
        xx = np.linspace(x.min(), x.max(), 200)
        yy = res.slope * xx + res.intercept
        ax.plot(xx, yy, lw=2, label=f"fit: y = {res.slope:.3g}x + {res.intercept:.3g}")
        
        # slope text (plus optional stats)
        ax.text(
            0.02, 0.98,
            f"slope = {res.slope:.4g}\n$R^2$ = {res.rvalue**2:.4g}\np = {res.pvalue:.3g}",
            transform=ax.transAxes, va="top", ha="left",
            # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        )
        
        ax.set(xlabel="neu_F_mean", ylabel="dlight_10% baseline mean")
        ax.legend(frameon=False)
        plt.show()
        
        
        
        #%% test neuropil factor
        # import scipy
        # low_raw = np.min(dlight_raw, axis=-1)
        # idx = np.argmin(dlight_raw, axis=-1)
        # low_neu = dlight_neu_trace[np.arange(32)[:, None], np.arange(32)[None, :], idx]
        
        # x = low_neu.ravel()
        # y = low_raw.ravel()
        
        # # IMPORTANT: keep x/y paired when dropping NaNs
        # m = np.isfinite(x) & np.isfinite(y)
        # x = x[m]
        # y = y[m]
        
        # res = scipy.stats.linregress(x, y)
        
        # fig, ax = plt.subplots()
        # ax.scatter(x, y, s=10, alpha=0.6)
        
        # # fitting line
        # xx = np.linspace(x.min(), x.max(), 200)
        # yy = res.slope * xx + res.intercept
        # ax.plot(xx, yy, lw=2, label=f"fit: y = {res.slope:.3g}x + {res.intercept:.3g}")
        
        # # slope text (plus optional stats)
        # ax.text(
        #     0.02, 0.98,
        #     f"slope = {res.slope:.4g}\n$R^2$ = {res.rvalue**2:.4g}\np = {res.pvalue:.3g}",
        #     transform=ax.transAxes, va="top", ha="left",
        #     # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        # )
        
        # ax.set(xlabel="neu_F", ylabel="dlight_F")
        # ax.legend(frameon=False)
        # plt.show()
        # #%%
        # # from cupyx.scipy.ndimage import percentile_filter
        # # x, y, T = dlight_raw.shape
        # # raw_10perc = percentile_filter(cp.array(dlight_raw).reshape(x*y, T), percentile=10, size=(1, 9000)).get()
        # plt.hist(raw_10perc.ravel(), bins=100)
        # plt.show()
        # #%%
        # raw_10perc_mean = np.nanmean(raw_10perc, axis=-1)
        # neu_mean = np.nanmean(dlight_neu_trace, axis=-1)
        
        # x = raw_10perc_mean.ravel()
        # y = neu_mean.ravel()
        
        # # IMPORTANT: keep x/y paired when dropping NaNs
        # m = np.isfinite(x) & np.isfinite(y)
        # x = x[m]
        # y = y[m]
        
        # res = scipy.stats.linregress(x, y)
        
        # fig, ax = plt.subplots()
        # ax.scatter(x, y, s=10, alpha=0.6)
        
        # # fitting line
        # xx = np.linspace(x.min(), x.max(), 200)
        # yy = res.slope * xx + res.intercept
        # ax.plot(xx, yy, lw=2, label=f"fit: y = {res.slope:.3g}x + {res.intercept:.3g}")
        
        # # slope text (plus optional stats)
        # ax.text(
        #     0.02, 0.98,
        #     f"slope = {res.slope:.4g}\n$R^2$ = {res.rvalue**2:.4g}\np = {res.pvalue:.3g}",
        #     transform=ax.transAxes, va="top", ha="left",
        #     # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8")
        # )
        
        # ax.set(xlabel="neu_F_mean", ylabel="dlight_10%_mean")
        # ax.legend(frameon=False)
        # plt.show()
        

        
        