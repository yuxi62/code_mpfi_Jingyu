# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:38:52 2026

@author: Jingyu Cao
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
# import sys
# sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
# sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu")
from common.utils_basic import trace_filter
from common.trial_selection import seperate_valid_trial
from common.utils_imaging import percentile_dff
from common.event_response_quantification import quantify_event_response
import dlight_imaging.regression.utils as utl
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res_defunc'
OUT_DIR_FIG = ''
regression_name = 'single_trial_regression_anat_roi'
#%%
exp = r'dlight_GECO_Ai14_Dbh'
f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp)
df_selected = pd.read_pickle(f_out_df_selected)
df_selected = df_selected.loc[(df_selected['speed_corr_single_trial_r_median']<0.3)
                              # &(~df_selected.index.str.contains('AC991'))
                              ]
rec_lst = df_selected.index.tolist()
#%%
# test session
# rec_lst = ['AC953-20240919-02', ]
for rec in tqdm(rec_lst):
    print(f'\nprocessing {rec}...')
    # load run-onset event frames
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    run_onset_frames = np.array(beh['run_onset_frames'])
    valid_trials = (seperate_valid_trial(beh))&(run_onset_frames!=-1)
    run_onset_frames_valid = run_onset_frames[valid_trials]
    
    # lode regression result traces
    p_regression = (OUR_DIR_REGRESS / rec / regression_name)
    # if not (p_regression / f'{rec}_profile_stat.parquet').exists():
    regress_traces = np.load(p_regression / regression_name / f'{regression_name}_res_trace_-neu.npz')
    corrected_dlight = regress_traces['corrected_dlight']
    # loading dFF traces
    if not (p_regression / 'dff_corrected_dlight.npy').exists():
        print('calculating dlight dFF trace...') 
        dff_dlight = percentile_dff(corrected_dlight, q=15)
        np.save(p_regression / 'dff_corrected_dlight.npy', dff_dlight)
    else:
        print('loading dlight dFF trace...') 
        dff_dlight = np.load(p_regression / 'dff_corrected_dlight.npy')
        
    if not (p_regression / 'dff_geco_soma.npy').exists():
        print('calculating soma geco dFF trace...') 
        anm, date, ss = rec.split('-')
        p_suite2p_geco = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p_anat_detec\plane0")
        geco_trace = np.load(p_suite2p_geco / 'F.npy')
        geco_trace_neu = np.load(p_suite2p_geco / 'Fneu.npy')
        p_masks = utl.load_masks_geco_dlight(rec, OUR_DIR_REGRESS)
        is_soma =  np.load(p_masks/r'soma_class.npz')['is_soma']
        soma_indices = np.where(is_soma>0)[0]
        # correct calcium trace with neuropil signal
        geco_trace_corr = geco_trace - 0.7*geco_trace_neu
        geco_trace_corr_soma = geco_trace_corr[soma_indices]
        dff_geco_soma = percentile_dff(geco_trace_corr_soma, q=20) 
        np.save(p_regression / 'dff_geco_soma.npy', dff_geco_soma)
    else:
        print('loading soam geco dFF trace...')
        dff_geco_soma = np.load(p_regression / 'dff_geco_soma.npy')
    
    dff_dlight_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_dlight, n_sd=5)
    dff_geco_safe   = np.apply_along_axis(trace_filter, axis=-1, arr=dff_geco_soma, n_sd=5)

    dff_dlight_sm = cp_gaussian_filter1d(cp.array(dff_dlight_safe), 
                                               sigma=1).get()
    dff_geco_sm = cp_gaussian_filter1d(cp.array(dff_geco_safe), 
                                               sigma=1).get()
    
    df_roi_stats = quantify_event_response(corrected_traces = dff_dlight_sm, 
                                        event_frames=run_onset_frames_valid,
                                        baseline_window=(-1, 0), 
                                        response_window=(0, 1.5), # seconds
                                        dilation_k = 0,
                                        imaging_rate=30.0, shuffle_test=True,
                                        shuffle_params={'times': 1000,
                                                        'pre_event_window':  2, # seconds
                                                        'post_event_window': 4 }
                                        )
    df_roi_stats_red = quantify_event_response(corrected_traces = dff_geco_sm, 
                                        event_frames=run_onset_frames_valid,
                                        baseline_window=(-1, 0), 
                                        response_window=(0, 1.5), # seconds
                                        dilation_k = 0,
                                        imaging_rate=30.0, shuffle_test=True,
                                        shuffle_params={'times': 1000,
                                                        'pre_event_window':  2, # seconds
                                                        'post_event_window': 4 }
                                        )
    df_roi_stats.to_parquet(p_regression / f'{rec}_profile_stat_dlight.parquet' )
    df_roi_stats_red.to_parquet(p_regression / f'{rec}_profile_stat_geco.parquet' )
    
        
        
       

        
