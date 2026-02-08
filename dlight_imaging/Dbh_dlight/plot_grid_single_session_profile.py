# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:06:40 2025

@author: Jingyu Cao
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.utils_basic import normalize
# Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst
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
save_plot = False
#%% MAIN
rec_lst = ['AC964-20250131-02', ] # for testing
# rec_lst = ['AC969-20250319-04', ] # for testing
df_pool_sorted = pd.DataFrame()
for rec in rec_lst:
    print(f'loading: {rec}--------------------------------------------------')
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                    / r'dilation_k=0')
    p_stats = p_regression / f'{rec}_profile_stat_20dff.parquet'
    p_stats_red = p_regression / f'{rec}_profile_stat_red_20dff.parquet'
    roi_stats = pd.read_parquet(p_stats)
    mean_profile_red = pd.read_parquet(p_stats_red)['mean_profile']
    roi_stats['mean_profile_red'] = mean_profile_red
    roi_stats['rec_id'] = rec
    df_pool_sorted = pd.concat((df_pool_sorted, roi_stats))

#%%
df_pool_sorted['dlight_valid'] = df_pool_sorted['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
df_pool_sorted['red_valid'] = df_pool_sorted['mean_profile_red'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
df_pool_sorted = df_pool_sorted.loc[(df_pool_sorted['dlight_valid'])&(df_pool_sorted['red_valid'])]
#%%
edges = [0, 31]
df_pool_sorted['edge'] = df_pool_sorted['roi_id'].apply(lambda rc: any(v in edges for v in rc))
df_pool_sorted['shuffle_amps_thresh_up'] = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
df_pool_sorted['shuffle_amps_thresh_down'] = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))
#%% reassign Up and Down using chosen thresholding
df_pool_sorted['Up'] = np.where(
                            #(geco_stats['max_sustained_sec']>0.09)
                            ~(df_pool_sorted['edge'])&
                            (df_pool_sorted['response_amplitude']>df_pool_sorted['shuffle_amps_thresh_up'])&
                            (df_pool_sorted['effect_size']>0.05),
                            True, False)

df_pool_sorted['Down'] = np.where(
                            #(geco_stats['max_sustained_sec']>0.09)
                            ~(df_pool_sorted['edge'])&
                            (df_pool_sorted['response_amplitude']<df_pool_sorted['shuffle_amps_thresh_down'])&
                            (df_pool_sorted['effect_size']< -0.05),
                            True, False)

df_pool_sorted.loc[df_pool_sorted['Up'], 'roi_type'] = 'Up'
df_pool_sorted.loc[df_pool_sorted['Down'], 'roi_type'] = 'Down'
df_pool_sorted.loc[(df_pool_sorted['Up']==0)&
                   (df_pool_sorted['Down']==0)
                   , 'roi_type'] = 'Stable'

# p_pooled_df = OUT_DIR_RAW_DATA / rf"df_population_pooled_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.pkl"
# df_pool_sorted.to_pickle(p_pooled_df)
#%% plot heatmap

# df_pool_sorted = df_pool_sorted.sort_values(by=['roi_type', 'response_amplitude'], ascending=[False, False])
df_pool_sorted = df_pool_sorted.sort_values(by=['roi_type', 'effect_size'], ascending=[False, False])

# get the 'dlight_mean_trace' column in that sorted order
traces = np.stack(df_pool_sorted['mean_profile'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
          aspect='auto', interpolation='none',
          extent=[-2, 4, traces.shape[0], 0],
          # cmap='YlGnBu_r',
          cmap='Greys')
ax.set(xlim=(-1, 4))
roi_types = df_pool_sorted['roi_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes

for idx in change_idx:
    ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like

save_fig(fig, OUT_DIR_FIG, r'dlight_pupulation_heatmap_greys_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), save=save_plot)

#%% plot population mean trace
dlightUp_traces_dlight = 100*np.stack(df_pool_sorted.loc[df_pool_sorted['Up'], 'mean_profile'])
dlightUp_traces_red = 100*np.stack(df_pool_sorted.loc[df_pool_sorted['Up'], 'mean_profile_red'])


bef, aft = 2, 4
xaxis = np.arange(30*(bef+aft))/30-bef    
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars(dlightUp_traces_dlight, dlightUp_traces_red, xaxis, ax,
                                  colors = ("tab:green", "tab:red"),
                                  timebar=0.5, dffbar=1, 
                                  show_xaxis=1, xlabel='time from run (s)')

save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), save=save_plot)



#%% plot roi mean trace       
out_dir = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\TEST_PLOTS\single_rois_test_AC964-20250131-02\single_rois_-p2neu_10dff")
for idx, df_up_rois in df_pool_sorted.loc[df_pool_sorted['Up']].iterrows():
    dlight_trace = 100*np.stack(df_up_rois['mean_profile'])
    red_trace = 100*np.stack(df_up_rois['mean_profile_red'])


    bef, aft = 2, 4
    xaxis = np.arange(30*(bef+aft))/30-bef    
    fig, ax = plt.subplots(dpi=300, figsize=(2,2))
    ax.plot(dlight_trace, color='green', label='dlight')
    ax.plot(red_trace, color='tab:red', label='red')
    ax.set(ylabel='% dF/F')
    save_fig(fig, out_dir, f'DA_up_roi_mean_trace_{idx}', save=1)