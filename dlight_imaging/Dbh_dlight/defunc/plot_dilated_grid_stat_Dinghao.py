# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:24:42 2026

plot dilation statistics for dLight (spont.)

@author: Jingyu Cao
@modifer: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np
import pandas as pd

from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()

# Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst


#%%
def classify_roi(df_stat):
    df_stat = df_stat.copy()
    df_stat['dlight_valid'] = df_stat['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    df_stat['red_valid'] = df_stat['mean_profile_red'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    df_stat = df_stat.loc[(df_stat['dlight_valid'])&(df_stat['red_valid'])]

    edges = [0, 31]
    df_stat['edge'] = df_stat['roi_id'].apply(lambda rc: any(v in edges for v in rc))
    df_stat['shuffle_amps_thresh_up'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_stat['shuffle_amps_thresh_down'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))
    
    df_stat['Up'] = np.where(
                                #(geco_stats['max_sustained_sec']>0.09)
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']>df_stat['shuffle_amps_thresh_up'])&
                                (df_stat['effect_size']>0.05),
                                True, False)

    df_stat['Down'] = np.where(
                                #(geco_stats['max_sustained_sec']>0.09)
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']<df_stat['shuffle_amps_thresh_down'])&
                                (df_stat['effect_size']< -0.05),
                                True, False)
    return df_stat


#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")

baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'
DILATION_STEPS = (0, 2, 4, 6, 8, 10)

df_stats_all = pd.DataFrame()
for rec in rec_lst:
    print(f'loading: {rec}-------------------------------------------')
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    for k in DILATION_STEPS:
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / f'dilation_k={k}')
        p_stats = p_regression / f'{rec}_profile_stat.parquet'
        p_stats_red = p_regression / f'{rec}_profile_stat_red.parquet'
        roi_stats = pd.read_parquet(p_stats)
        mean_profile_red = pd.read_parquet(p_stats_red)['mean_profile']
        roi_stats['mean_profile_red'] = mean_profile_red
        roi_stats['rec_id'] = rec
        roi_stats = classify_roi(roi_stats)
        df_stats_all = pd.concat((df_stats_all, roi_stats))
        

#%% plot session mean +/- sem for different dialtion steps
df_dlightUp_all  = pd.DataFrame()
df_dilation_stat = pd.DataFrame()
for g_idx, df_rec in df_stats_all.groupby('rec_id'):
    # start with rois that are dLightUp without dilation
    dlightUp_mask = (df_rec['dilation_k'] == 0) & (df_rec['Up'])
    dlightUp_rois = df_rec.loc[dlightUp_mask, 'roi_id'].apply(tuple).values
    df_dlightUp_dilations = df_rec[
        df_rec['roi_id'].apply(tuple).isin(dlightUp_rois)
    ]
    df_dlightUp_all = pd.concat((df_dlightUp_all, df_dlightUp_dilations))
    
df_dilation_stat = df_dlightUp_all.groupby(['rec_id', 'dilation_k']
                                           )['response_amplitude'].agg(list).reset_index()

df_dilation_stat['response_amplitude_mean'] = df_dilation_stat['response_amplitude'].apply(lambda x: np.nanmean(x))
# df_dilation_stat['response_amplitude_sem'] = df_dilation_stat['response_amplitude'].apply(lambda x: sem(x, nan_policy='omit'))
    

