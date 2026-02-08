# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:59:19 2026

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
from common.plot_single_trial_function import align_frame_behaviour, plot_single_trial_html_roi
from common import plotting_functions_Jingyu as pf

if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
from utils_Jingyu import percentile_dff_abs
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = ''
regression_name ='single_trial_regression'
DILATION_STEPS = (2, 4, 6, 8, 10)
baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5

p_pooled_df = OUT_DIR_RAW_DATA / rf"df_population_pooled_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.pkl"
df_pool = pd.read_pickle(p_pooled_df)

rec = 'AC964-20250131-02'
print(f'\nprocessing {rec}...')
# load run-onset event frames
p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
beh = pd.read_pickle(p_beh_file)
beh_aligned = align_frame_behaviour(beh)
run_onset_frames = np.array(beh['run_onset_frames'])
valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
run_onset_frames_valid = run_onset_frames[valid_trials]

k_size = 0 # DILATION_STEPS = (0, ) # for testing
print(f'\ndilation={k_size}...')
# lode regression result traces
p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                / r'dilation_k={}'.format(k_size))
# if not (p_regression / f'{rec}_profile_stat.parquet').exists():
regress_traces = np.load(p_regression / 'single_trial_regression_res_traces.npz')
corrected_dlight = regress_traces['corrected_dlight']
red_trace = regress_traces['red_trace']
roi_stats = df_pool.loc[df_pool['rec_id']==rec]
# list of grids with signals extracted
valid_grids = utl.get_valid_grids(corrected_dlight)
dlight_up_rois = roi_stats.loc[roi_stats['Up'], 'roi_id']

p_old_regress_trace = r"Z:\Jingyu\2P_Recording\AC964\AC964-20250131\02\RegOnly\regression_result_dilation_new_fibermask_dlightmask\dilation_k=0\regression_single_trial\G_dilated_axon_dlight_R_dilated_axon_dlight\all_corrected_traces_regress_with_template.npz"
old_corrected_dlight_all = np.load(p_old_regress_trace)['corrected_dlight']
p_old_dff = r"Z:\Jingyu\2P_Recording\AC964\AC964-20250131\02\RegOnly\regression_result_dilation_new_fibermask_dlightmask\dilation_k=0\corrected_dlight_dff.npy"
old_dff_dlight_all = np.load(p_old_dff)
#%%
for roi in dlight_up_rois.iloc[:20]:
    # roi = dlight_up_rois.iloc[2]
    F_raw = corrected_dlight[roi[0], roi[1]]
    old_F = old_corrected_dlight_all[roi[0], roi[1]]
    # old_dff = old_dff_dlight_all[roi[0], roi[1]]
    old_dff = percentile_dff_abs(old_F, q=10, window_size=1800).get()
    dff, baseline = percentile_dff(F_raw , q=15, return_baseline=True)
    # dff_10, baseline_10 = percentile_dff_abs(F_raw, q=10, window_size=1800, return_baseline=True)
    # dff_abs, baseline_abs = dff_abs.get(), baseline_abs.get()
    dff_10, baseline_10 = percentile_dff(F_raw , q=10, return_baseline=True)

    F_aligned = align_trials(F_raw, 'run', beh, 1, 4)
    dff_aligned = align_trials(dff, 'run', beh, 1, 4)
    dff_10_aligned = align_trials(dff_10, 'run', beh, 1, 4)
    old_dff_aligned = align_trials(old_dff, 'run', beh, 1, 4)
    
    fig, ax = plt.subplots(dpi=200, figsize=(3, 3))
    pf.plot_mean_trace(dff_aligned, ax, label='15% dFF')
    pf.plot_mean_trace(dff_10_aligned, ax, color='blue', label='10% dFF')
    pf.plot_mean_trace(old_dff_aligned, ax, color='pink', label='10% old dFF')
    plt.legend(frameon=False)
    plt.show()
#%%
fig = plot_single_trial_html_roi(beh_aligned, F_raw, dff, baseline,
                                labels = {
                                    'ch1': 'raw_F',
                                    'ch2': 'dFF',
                                    'ch3': 'baseline'
                                    }, 
                                shared_yaxis=['ch1', 'ch3']
                                )
# fig.write_html(r"Z:\Jingyu\example_dff.html")
fig.show()

fig = plot_single_trial_html_roi(beh_aligned, F_raw, dff_abs, baseline_abs,
                                labels = {
                                    'ch1': 'raw_F',
                                    'ch2': 'dFF',
                                    'ch3': 'baseline'
                                    }, 
                                shared_yaxis=['ch1', 'ch3']
                                )
# fig.write_html(r"Z:\Jingyu\example_dff.html")
fig.show()


#%%
# loading dFF traces
H, W, T = corrected_dlight.shape  # (32, 32, nframes)
if not (p_regression / 'dff_corrected_dlight.npy').exists():
    print('calculating dlight dFF trace...') 
    dff_dlight = percentile_dff(corrected_dlight.reshape(H * W, T) , q=15).reshape(H, W, T)
    np.save(p_regression / 'dff_corrected_dlight.npy', dff_dlight.reshape(H, W, T))
else:
    print('loading dlight dFF trace...') 
    dff_dlight = np.load(p_regression / 'dff_corrected_dlight.npy')
    
if not (p_regression / 'dff_red.npy').exists():
    print('calculating red dFF trace...') 
    dff_red = percentile_dff(red_trace.reshape(H * W, T) , q=15).reshape(H, W, T) 
    np.save(p_regression / 'dff_red.npy', dff_red.reshape(H, W, T))
else:
    print('loading red dFF trace...')
    dff_red = np.load(p_regression / 'dff_red.npy')

