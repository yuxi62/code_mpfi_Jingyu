# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 19:16:29 2025

@author: Jingyu Cao
"""

#%% imports 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.ndimage import gaussian_filter1d

from drug_infusion.plot_functions import plot_population_heatmap
from drug_infusion.utils_infusion import sort_response

from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
from common.utils_basic import normalize
#%% PATHS AND PARAMS

# session list 
drug = 'SCH'
# drug = 'prazosin'
# drug = 'propranolol'

import rec_lst_infusion as recs
if drug=='SCH':
    rec_drug = recs.rec_SCH
    rec_ctrl = recs.rec_SCH_ctrl
elif drug=='prazosin':
    rec_drug = recs.rec_praz
    rec_ctrl = recs.rec_praz_ctrl
elif drug=='propranolol':
    rec_drug = recs.rec_prop
    rec_ctrl = recs.rec_prop_ctrl

# PARAMS
pre_window=(-1, -0.5)
post_window=(0.5, 1.5)
thresh_up = 1.1
thresh_down = 1/thresh_up
bef, aft = 2, 4

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUT_DIR_FIG = Path(rf"Z:\Jingyu\LC_HPC_manuscript\fig_infusion\{drug}")
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir()
save_plot = False
#%% data pooling
df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
# data pooling for drug sessions
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_drug), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_drug_pool = pd.concat((df_drug_pool, df_profile))
    
# data pooling for ctrl sessions
for rec_idx, rec in tqdm(rec_ctrl.iterrows(), total=len(rec_ctrl), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_ctrl_pool = pd.concat((df_ctrl_pool, df_profile))    
#%% selection recordings for statistics
trace_col = 'mean_profile_valid'
ratio_col = 'response_ratio_valid'

df_drug_pool_pyr = sort_response(df_drug_pool, thresh_up, thresh_down,
                                 ratio_type=ratio_col,
                                 trace_type=trace_col)
df_ctrl_pool_pyr = sort_response(df_ctrl_pool, thresh_up, thresh_down,
                                 ratio_type=ratio_col ,
                                 trace_type=trace_col)
 
# select recordings (rec_id) with n_keep_trials > 15 for both ss1 and ss2
valid_recs = df_drug_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trial_valid_ss1'].iloc[0] > 15) & (g['n_keep_trial_valid_ss2'].iloc[0] > 15),
    include_groups=False)
valid_recs = valid_recs[valid_recs].index
df_drug_pool_pyr = df_drug_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

# Select only the first 3 recording dates per animal
first3_dates = (df_drug_pool_pyr[['anm', 'date']].drop_duplicates()
                .sort_values(['anm', 'date'])
                .groupby('anm').head(3))
df_drug_pool_pyr_first3 = df_drug_pool_pyr.merge(first3_dates, on=['anm', 'date'], how='inner')
# df_drug_pool_pyr_first3.to_parquet(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe\df_drug_pool_pyr_first3_{drug}.parquet")

valid_recs = df_ctrl_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trial_valid_ss1'].iloc[0] > 15) & (g['n_keep_trial_valid_ss2'].iloc[0] > 15),
    include_groups=False)
valid_recs = valid_recs[valid_recs].index
df_ctrl_pool_pyr = df_ctrl_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

# Select only the first 3 recording dates per animal for ctrl
# Only include animals that are in df_drug_pool_pyr_first3
animals_in_drug = df_drug_pool_pyr_first3['anm'].unique()
first3_dates_ctrl = (df_ctrl_pool_pyr[df_ctrl_pool_pyr['anm'].isin(animals_in_drug)][['anm', 'date']].drop_duplicates()
                     .sort_values(['anm', 'date'])
                     .groupby('anm').head(3))
df_ctrl_pool_pyr_first3 = df_ctrl_pool_pyr.merge(first3_dates_ctrl, on=['anm', 'date'], how='inner')
# df_ctrl_pool_pyr_first3.to_parquet(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe\df_ctrl_pool_pyr_first3_{drug}.parquet")

df_drug_pool_pyr = df_drug_pool_pyr_first3
df_ctrl_pool_pyr = df_ctrl_pool_pyr_first3

#%% plot heatmap
# drug sessions
rec_id =  f'{drug}-drug'

prefix = 'ss1_baseline'
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix,
                            session_for_sorting='ss1', activity_profile=trace_col, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

prefix = f'ss2_{drug}'
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix,
                            session_for_sorting='ss2', activity_profile=trace_col, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)


# saline ctrl
rec_id = f'{drug}-ctrl'

prefix = 'ss1_baseline'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix,
                            session_for_sorting='ss1', activity_profile=trace_col, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

prefix = 'ss2_saline'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix,
                            session_for_sorting='ss2', activity_profile=trace_col, ratio=ratio_col,
                            plot_mean=0)
save_fig(fig, OUT_DIR_FIG, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)


#%% plot mean traces
for cell_type in ['pyrUp', 'pyrDown']:
    # drug ss1 vs ss2
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    profile_a = 100*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    profile_b = 100*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])      
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                          test='ranksum',
                                          time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                          baseline_window=(-0.5, 0.5),
                                          labels = ['baseline', f'{drug}'],
                                          colors = ['steelblue', 'orange'],
                                          scalebar_dff=1,
                                          bef=2, aft=4, sample_freq=30,
                                          show_scalebar=True,)
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{drug}_{cell_type}_mean_trace_ss1_ss2', save=save_plot)
    
    
    # plot ROI mean trace normalized trace
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    a = normalize(a)
    pf.plot_mean_trace(a, 
                       ax, color='steelblue', label='ss1')
    b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])
    b = normalize(b)
    pf.plot_mean_trace(b,
                       ax, color='orange', label=f'ss2_{drug}')
    ax.set(ylabel='norm. dF/F')
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name='', save=0)
    
    
    # plot session mean trace
    # fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    # cell_type = 'pyrUp_ss1'
    # grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
    # profile_a = 100*np.stack(grouped[f'{trace_col}_ss1'].mean())
    # cell_type = 'pyrUp_ss2'
    # grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
    # profile_b = 100*np.stack(grouped[f'{trace_col}_ss2'].mean())
    # pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
    #                                       test='ranksum',
    #                                       time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
    #                                       baseline_window=(-0.5, 0.5),
    #                                       labels = ['baseline', f'{drug}'],
    #                                       colors = ['steelblue', 'orange'],
    #                                       bef=2, aft=4, sample_freq=30,
    #                                       scalebar_dff=1,
    #                                       show_scalebar=True,
    #                                       )
    # ax.set(xlim=(-1, 4))
    # ax.legend(frameon=False)
    # save_fig(fig, OUT_DIR_FIG, fig_name='', save=save_plot)
    
    
    # plot saline ss1 vs ss2
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    profile_a = 100*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
    profile_b = 100*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])       
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                          test='ranksum',
                                          time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                          baseline_window=(-0.5, 0.5),
                                          labels = ['baseline', f'saline ({drug})'],
                                          colors = ['steelblue', 'grey'],
                                          bef=2, aft=4, sample_freq=30,
                                          scalebar_dff=1,
                                          show_scalebar=True,
                                          )
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'saline_{cell_type}_mean_trace_ss1_ss2', save=save_plot)
    
    # plot mean trace compare saline and SCH
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    profile_a = 100*np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])
    profile_b = 100*np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{trace_col}_ss2'])     
    pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                          test='ranksum',
                                          time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                          baseline_window=(-0.5, 0.5),
                                          labels = [f'saline ({drug})', f'{drug}'],
                                          colors = ['grey', 'orange'],
                                          bef=2, aft=4, sample_freq=30,
                                          scalebar_dff=1,
                                          show_scalebar=True,
                                          )
    ax.set(xlim=(-1, 4))
    ax.legend(frameon=False)
    save_fig(fig, OUT_DIR_FIG, fig_name=f'{cell_type}_mean_trace_saline_{drug}', save=save_plot)

#%% quantifying mean amplitude difference
amp_col = 'response_amplitude_valid'
for cell_type in ['pyrUp', 'pyrDown']:
    # drug ss1 vs ss2
    profile_a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1'], f'{amp_col}_ss1'])
    profile_b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2'], f'{amp_col}_ss2'])

    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_unpaired_violin(profile_a, profile_b,
                                    ylabel= f'{cell_type}_response_amp',
                                    colors=['steelblue', 'orange'],
                                    colname=['baseline', f'{drug}'],
                                    ax = ax,
                                    markersize = 1
                                    # ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=0)
    
    # Get session-mean amplitude for cells classified in ss1
    grouped_a = df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss1']].groupby(['anm', 'date'])
    profile_a = np.array([g[f'{amp_col}_ss1'].mean() for _, g in grouped_a])
    
    # Get session-mean amplitude for cells classified in ss2
    grouped_b = df_drug_pool_pyr.loc[df_drug_pool_pyr[f'{cell_type}_ss2']].groupby(['anm', 'date'])
    profile_b = np.array([g[f'{amp_col}_ss2'].mean() for _, g in grouped_b])


    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_bar_with_paired_scatter(ax, profile_a, profile_b,
                                    ylabel=f'{cell_type}_response_amp',
                                    colors=['steelblue', 'orange'],
                                    xticklabels=['baseline', f'{drug}'],
                                    # ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=0)           
#%% Bar plot
grouped = df_drug_pool_pyr.groupby(['anm', 'date'],)
recs = [name for name, g in grouped]
df_perc_drug = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / len(g),
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / len(g),
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum()  / len(g),
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum()  / len(g),
        'delta_perc_pyrUp':   g['pyrUp_ss2'].sum()   / len(g) - g['pyrUp_ss1'].sum()   / len(g),
        'delta_perc_pyrDown': g['pyrDown_ss2'].sum()  / len(g) - g['pyrDown_ss1'].sum()  / len(g),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

grouped = df_ctrl_pool_pyr.groupby(['anm', 'date'],)
df_perc_ctrl = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / len(g),
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / len(g),
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum()  / len(g),
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum()  / len(g),
        'delta_perc_pyrUp':   g['pyrUp_ss2'].sum()   / len(g) - g['pyrUp_ss1'].sum()   / len(g),
        'delta_perc_pyrDown': g['pyrDown_ss2'].sum()  / len(g) - g['pyrDown_ss1'].sum()  / len(g),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

for cell_type, ylim in zip(['pyrUp', 'pyrDown'], [(0, 0.6), (0, 0.5)]):
    
    # drug sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_bar_with_paired_scatter(ax, df_perc_drug[f'perc_{cell_type}_ss1'], df_perc_drug[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['steelblue', 'orange'],
                                    xticklabels=['baseline', f'{drug}'],
                                    ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_{drug}', save=save_plot)

    # ctrl sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_bar_with_paired_scatter(ax, df_perc_ctrl[f'perc_{cell_type}_ss1'], df_perc_ctrl[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['steelblue', 'grey'],
                                    xticklabels=['baseline', f'saline({drug})'],
                                    ylim=ylim,
                                    )
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_ctrl', save=save_plot)

    # saline vs drug sessions
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl[f'perc_{cell_type}_ss2'], df_perc_drug[f'perc_{cell_type}_ss2'],
                                    ylabel=f'% {cell_type}',
                                    colors=['grey', 'orange'],
                                    xticklabels=['saline', f'{drug}'])
    save_fig(fig, OUT_DIR_FIG, fig_name=f'perc_{cell_type}_saline_{drug}', save=save_plot)


    # Δ% to baseline
    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
    pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl[f'delta_perc_{cell_type}'], df_perc_drug[f'delta_perc_{cell_type}'],
                                    ylabel=f'Δ% {cell_type} (vs baseline)',
                                    colors=['grey', 'orange'],
                                    xticklabels=[f'saline ({drug})', f'{drug}'])
    save_fig(fig, OUT_DIR_FIG, fig_name=f'delt_perc_{cell_type}_saline_{drug}', save=save_plot)



    



