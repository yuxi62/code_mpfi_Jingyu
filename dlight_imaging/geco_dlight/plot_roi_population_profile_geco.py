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
pf.mpl_formatting()
save_fig = pf.save_fig
from common.utils_basic import normalize

# HELP FUNCS
def division_helper(a, b):
    if b!=0:
        res = a/b
    else:
        res = np.nan
    return res

def calculate_percs(roi_stats):
    
    # roi_stats = roi_stats.set_index('roi_id', drop=False)
    
    dic_stats_session = {
    'perc_pyrUp_dlightUp':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightUp']])
        ),

    'perc_pyrUp_dlightStable':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightStable'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightStable']])
        ),

    'perc_pyrUp_dlightDown':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown'] & roi_stats['pyrUp']]),
            len(roi_stats.loc[roi_stats['dlightDown']])
        ),

    'perc_pyrUp_no_dlightUp':
        division_helper(
            len(roi_stats.loc[(~roi_stats['dlightUp']) & roi_stats['pyrUp']]),
            len(roi_stats.loc[(~roi_stats['dlightUp'])])
        ),

    'perc_pyrUp_all':
        division_helper(
            len(roi_stats.loc[roi_stats['pyrUp']]),
            len(roi_stats)
        ),

    'perc_pyrDown_dlightUp':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightUp']])
        ),

    'perc_pyrDown_dlightStable':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightStable'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightStable']])
        ),

    'perc_pyrDown_dlightDown':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown'] & roi_stats['pyrDown']]),
            len(roi_stats.loc[roi_stats['dlightDown']])
        ),

    'perc_pyrDown_no_dlightUp':
        division_helper(
            len(roi_stats.loc[(~roi_stats['dlightUp']) & roi_stats['pyrDown']]),
            len(roi_stats.loc[(~roi_stats['dlightUp'])])
        ),

    'perc_pyrDown_all':
        division_helper(
            len(roi_stats.loc[roi_stats['pyrDown']]),
            len(roi_stats)
        ),

    'perc_dlightUp_all':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightUp']]),
            len(roi_stats)
        ),

    'perc_dlightDown_all':
        division_helper(
            len(roi_stats.loc[roi_stats['dlightDown']]),
            len(roi_stats)
        )
        }
    return dic_stats_session
#%% Load recording list
exp = r'dlight_GECO_Ai14_Dbh'
f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp)
df_selected = pd.read_pickle(f_out_df_selected)
df_selected = df_selected.loc[(df_selected['speed_corr_single_trial_r_median']<0.3)
                              # &(~df_selected.index.str.contains('AC991'))
                              ]
rec_lst = df_selected.index.tolist()
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = r"Z:\Jingyu\LC_HPC_manuscript\fig_GECO_dlight"
save_plot=1

baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
pyrUp_thresh = 1.1
pyrDown_thresh = 1/pyrUp_thresh
regression_name = 'single_trial_regression_anat_roi'

#%% data pooling
df_pooled_profile = pd.DataFrame()
for rec in rec_lst:
    print(f'loading: {rec}--------------------------------------------------')
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    p_regression = (OUR_DIR_REGRESS / rec / regression_name )
    is_soma_idx = np.load(OUR_DIR_REGRESS / rec / 'masks' / 'soma_class.npz')['is_active_soma']
    p_stats      = p_regression / f'{rec}_profile_stat_dlight.parquet'
    p_stats_geco = p_regression / f'{rec}_profile_stat_geco.parquet'
    dlight_stats = pd.read_parquet(p_stats)
    geco_stats = pd.read_parquet(p_stats_geco)
    dlight_stats['rec_id'] = rec
    dlight_stats['mean_profile_geco'] = geco_stats['mean_profile']
    dlight_stats['geco_ratio'] = geco_stats['response_ratio']
    dlight_stats['is_soma'] = is_soma_idx
    dlight_stats = dlight_stats.loc[dlight_stats['is_soma']]
    df_pooled_profile = pd.concat((df_pooled_profile, dlight_stats))

df_pooled_profile = df_pooled_profile.loc[df_pooled_profile['n_keep_trial']>100]
#%% reassign Up and Down using chosen thresholding
df_pool_sorted = df_pooled_profile.copy() # withou modifying the original pooled data

df_pool_sorted['shuffle_amps_thresh_up']   = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
df_pool_sorted['shuffle_amps_thresh_down'] = df_pool_sorted['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))

# df_pool_sorted['dlight_valid'] = df_pool_sorted['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
# df_pool_sorted['geco_valid'] = df_pool_sorted['mean_profile_geco'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))

df_pool_sorted['mean_dlight'] = (df_pool_sorted['mean_profile'].apply(np.nanmean))
df_pool_sorted['mean_geco'] = (df_pool_sorted['mean_profile_geco'].apply(np.nanmean))
df_pool_sorted['dlight_valid'] = df_pool_sorted['mean_dlight'].apply(lambda x: 0<x<1.5)
df_pool_sorted['geco_valid'] = df_pool_sorted['mean_geco'].apply(lambda x: 0<x<1.5)
df_pool_sorted = df_pool_sorted.loc[(df_pool_sorted['dlight_valid'])&(df_pool_sorted['geco_valid'])]
# df_pool_sorted = df_pool_sorted.loc[(df_pool_sorted['dlight_valid'])]

df_pool_sorted['dlightUp'] = np.where(
                            (df_pool_sorted['response_amplitude']>df_pool_sorted['shuffle_amps_thresh_up'])&
                            (df_pool_sorted['effect_size']>0.05),
                            True, False)
df_pool_sorted['dlightDown'] = np.where(
                            (df_pool_sorted['response_amplitude']<df_pool_sorted['shuffle_amps_thresh_down'])&
                            (df_pool_sorted['effect_size']< -0.05),
                            True, False)
df_pool_sorted['dlightStable'] = (~df_pool_sorted['dlightUp'])&(~df_pool_sorted['dlightDown'])

df_pool_sorted['pyrUp'] = np.where(
                            (df_pool_sorted['geco_ratio']> pyrUp_thresh),
                            True, False)
df_pool_sorted['pyrDown'] = np.where(
                            (df_pool_sorted['geco_ratio']<pyrDown_thresh),
                            True, False)
df_pool_sorted['pyrStable'] = (~df_pool_sorted['pyrUp'])&(~df_pool_sorted['pyrDown'])


df_pool_sorted.loc[df_pool_sorted['dlightUp'],     'dlight_type'] = 'Up'
df_pool_sorted.loc[df_pool_sorted['dlightDown'],   'dlight_type'] = 'Down'
df_pool_sorted.loc[df_pool_sorted['dlightStable'], 'dlight_type'] = 'Stable'

df_pool_sorted.loc[df_pool_sorted['pyrUp'],     'geco_type'] = 'Up'
df_pool_sorted.loc[df_pool_sorted['pyrDown'],   'geco_type'] = 'Down'
df_pool_sorted.loc[df_pool_sorted['pyrStable'], 'geco_type'] = 'Stable'


p_pooled_df = OUT_DIR_RAW_DATA / rf"df_population_pooled_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_{amp_shuff_thresh_up}.pkl"
df_pool_sorted.to_pickle(p_pooled_df)

df_pool_dlight_up = df_pool_sorted.loc[df_pool_sorted['dlightUp']]
df_pool_non_dlight_up = df_pool_sorted.loc[~df_pool_sorted['dlightUp']]
#%% plot heatmap
# all soma rois dlight
df_pool_sorted = df_pool_sorted.sort_values(by=['dlight_type', 'effect_size'], ascending=[False, False])
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
ax.set(xlim=(-1, 4), title='dLight_mean_profile sort by dLight')
roi_types = df_pool_sorted['dlight_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
for idx in change_idx:
    ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like
save_fig(fig, OUT_DIR_FIG, r'all_dlight_population_heatmap_greys_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), 
            save=save_plot)

# all soma rois GECO
df_pool_sorted = df_pool_sorted.sort_values(by=['geco_type', 'geco_ratio'], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_pool_sorted['mean_profile_geco'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
          aspect='auto', interpolation='none',
          extent=[-2, 4, traces.shape[0], 0],
          # cmap='YlGnBu_r',
          cmap='Greys')
ax.set(xlim=(-1, 4), title='geco_mean_profile sort by geco')
roi_types = df_pool_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
for idx in change_idx:
    ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like
save_fig(fig, OUT_DIR_FIG, r'all_GECO_population_heatmap_greys_pyrUp_thresh={}'
            .format(pyrUp_thresh), 
            save=save_plot)

# DA-up rois GECO
df_sorted = df_pool_dlight_up.sort_values(by=['geco_type', 'geco_ratio'], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted['mean_profile_geco'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
          aspect='auto', interpolation='none',
          extent=[-2, 4, traces.shape[0], 0],
          # cmap='YlGnBu_r',
          cmap='Greys')
ax.set(xlim=(-1, 4), title='geco_mean_profile sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
for idx in change_idx:
    ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like
save_fig(fig, OUT_DIR_FIG, r'DA_up_GECO_pupulation_heatmap_greys_pyrUp_thresh={}'
            .format(pyrUp_thresh), 
            save=save_plot)

df_sorted = df_pool_non_dlight_up.sort_values(by=['geco_type', 'geco_ratio'], ascending=[False, False])
# get the 'geco_mean_trace' column in that sorted order
traces = np.stack(df_sorted['mean_profile_geco'])
traces = gaussian_filter1d(traces, sigma=1)
traces = normalize(traces)
fig, ax = plt.subplots(figsize=(3,3), dpi=300)
ax.imshow(traces,
          aspect='auto', interpolation='none',
          extent=[-2, 4, traces.shape[0], 0],
          # cmap='YlGnBu_r',
          cmap='Greys')
ax.set(xlim=(-1, 4), title='geco_mean_profile sort by geco')
roi_types = df_sorted['geco_type'].values
change_idx = np.where(roi_types[:-1] != roi_types[1:])[0] + 1  # row indices where type changes
for idx in change_idx:
    ax.axhline(idx, color='red', lw=0.8, ls='--')  # adjust style as you like
save_fig(fig, OUT_DIR_FIG, r'non_DA_up_GECO_pupulation_heatmap_greys_pyrUp_thresh={}'
            .format(pyrUp_thresh), 
            save=save_plot)
#%% plot population mean trace

dlightUp_pyrUp_dlight_traces       = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                         (df_pool_sorted['geco_type']=='Up'), 'mean_profile'])
dlightUp_pyrUp_geco_traces         = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                      (df_pool_sorted['geco_type']=='Up'), 'mean_profile_geco'])
dlightUp_pyrDown_dlight_traces     = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                         (df_pool_sorted['geco_type']=='Down'), 'mean_profile'])
dlightUp_pyrDown_geco_traces       = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']=='Up')&
                                                      (df_pool_sorted['geco_type']=='Down'), 'mean_profile_geco'])

non_dlightUp_pyrUp_dlight_traces   = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                             (df_pool_sorted['geco_type']=='Up'), 'mean_profile'])
non_dlightUp_pyrUp_geco_traces     = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                          (df_pool_sorted['geco_type']=='Up'), 'mean_profile_geco'])
non_dlightUp_pyrDown_dlight_traces = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                             (df_pool_sorted['geco_type']=='Down'), 'mean_profile'])
non_dlightUp_pyrDown_geco_traces   = 100*np.stack(df_pool_sorted.loc[(df_pool_sorted['dlight_type']!='Up')&
                                                          (df_pool_sorted['geco_type']=='Down'), 'mean_profile_geco'])

bef, aft = 2, 4
xaxis = np.arange(30*(bef+aft))/30-bef    

# plot DA-up dlight + GECO
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars(dlightUp_pyrUp_dlight_traces, dlightUp_pyrUp_geco_traces, xaxis, ax,
                                  colors = ("tab:green", "tab:red"),
                                  timebar=0.5, dffbar=1, 
                                  show_xaxis=1, xlabel='time from run (s)')
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_dlightUp_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), 
            save=save_plot)

# plot non DA-up dlight + GECO  
fig, ax = plt.subplots(dpi=300, figsize=(2,2))
pf.plot_two_traces_with_scalebars(non_dlightUp_pyrUp_dlight_traces, non_dlightUp_pyrUp_geco_traces, xaxis, ax,
                                  colors = ("tab:green", "tab:red"),
                                  timebar=0.5, dffbar=1, 
                                  show_xaxis=1, xlabel='time from run (s)')
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_non-dlightUp_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), 
            save=save_plot)

# plot DA-up vs non-DA-up GECO pyrUp
fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
pf.plot_two_traces_with_binned_stats(dlightUp_pyrUp_geco_traces, non_dlightUp_pyrUp_geco_traces,
                                     ax, bef=2, aft=4,
                                     colors = ['brown', 'indianred'],
                                     labels = ['dLight_Up_pyrUp', 'non_dLight_Up_pyrUp'],
                                     )
ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_dlightUp-pyrUp_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), 
            save=save_plot)

# plot DA-up vs non-DA-up GECO pyrDown 
fig, ax = plt.subplots(dpi=300, figsize=(2.5,2.5))
pf.plot_two_traces_with_binned_stats(dlightUp_pyrDown_geco_traces, non_dlightUp_pyrDown_geco_traces,
                                     ax, bef=2, aft=4,
                                     colors = ['indigo', 'blueviolet'],
                                     labels = ['dLight_Up_pyrDown', 'non_dLight_Up_pyrDown'],
                                     )
ax.set(xlim=(-1, 4))
save_fig(fig, OUT_DIR_FIG, r'pupulation_mean_trace_dlightUp-pyrDown_ES={}_amp={}'
            .format(effect_size_thresh, amp_shuff_thresh_up), 
            save=save_plot)


#%% plot %pyrUp and %pyrDown

df_perc = calculate_percs(df_pool_sorted)

# Calculate percentages per recording
df_perc_pool = df_pool_sorted.groupby('rec_id').apply(calculate_percs).apply(pd.Series)
df_perc_pool.index.name = 'rec_id'

# plot non-DA_up vs DA_up
# pyrUp
a = df_perc_pool['perc_pyrUp_no_dlightUp']
b = df_perc_pool['perc_pyrUp_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                              # ylim=(0, 85), 
                             ylabel='% PyrUp',
                             colors=('lightcoral', 'firebrick'),
                             xticklabels=('dlightStable+Down', 'dlightUp'))
save_fig(fig, OUT_DIR_FIG, r'%pyrUp_barplot', save=save_plot)
# pyrDown
a = df_perc_pool['perc_pyrDown_no_dlightUp']
b = df_perc_pool['perc_pyrDown_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                             # ylim=(0, 45), 
                             ylabel='% PyrDown',
                             colors=('violet', 'Purple'),
                             xticklabels=('dlightStable+Down', 'dlightUp'))
save_fig(fig, OUT_DIR_FIG, r'%pyrDown_barplot', save=save_plot)

# plot DA-stable vs DA_up
# pyrUp
a = df_perc_pool['perc_pyrUp_dlightStable']
b = df_perc_pool['perc_pyrUp_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                             # ylim=(0, 45), 
                             ylabel='% PyrUp',
                             colors=('lightcoral', 'firebrick'),
                             xticklabels=('dlightStable', 'dlightUp'))
plt.tight_layout()
plt.show()
# pyrDown
a = df_perc_pool['perc_pyrDown_dlightStable']
b = df_perc_pool['perc_pyrDown_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                             # ylim=(0, 45), 
                             ylabel='% PyrDown',
                             colors=('violet', 'Purple'),
                             xticklabels=('dlightStable', 'dlightUp'))
plt.tight_layout()
plt.show()


# plot DA_down vs DA_up
# pyrUp
a = df_perc_pool['perc_pyrDown_dlightDown']
b = df_perc_pool['perc_pyrDown_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                             # ylim=(0, 45), 
                             ylabel='% PyrDown',
                             colors=('violet', 'Purple'),
                             xticklabels=('dlightDown', 'dlightUp'))
fig.tight_layout()
plt.show()
# pyrDown
a = df_perc_pool['perc_pyrUp_dlightDown']
b = df_perc_pool['perc_pyrUp_dlightUp']
fig, ax = plt.subplots(dpi=300, figsize=(2,3))
pf.plot_bar_with_paired_scatter(ax, np.array(a)*100, np.array(b)*100,
                             # ylim=(0, 45), 
                             ylabel='% PyrUp',
                             colors=('lightcoral', 'firebrick'),
                             xticklabels=('dlightDown', 'dlightUp'))
fig.tight_layout()
plt.show()




       
