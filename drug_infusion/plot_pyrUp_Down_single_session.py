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

#%% func
def _plot_session(df_session, drug, out_dir=None, save_plot=0):
    
    # plot heatmap
    rec_id =  f'{drug}-drug'
    
    prefix = 'ss1_baseline'
    fig=plot_population_heatmap(df_session, rec_id, bef, aft, 'ss1', prefix=prefix,
                                session_for_sorting='ss1', activity_profile=trace_col, ratio=ratio_col,
                                plot_mean=0)
    save_fig(fig, out_dir, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

    prefix = f'ss2_{drug}'
    fig=plot_population_heatmap(df_session, rec_id, bef, aft, 'ss2', prefix=prefix,
                                session_for_sorting='ss2', activity_profile=trace_col, ratio=ratio_col,
                                plot_mean=0)
    save_fig(fig, out_dir, fig_name=f'heatmap_{prefix}_{rec_id}', save=save_plot)

    # plot mean trace
    for cell_type in ['pyrUp', 'pyrDown']:
        # drug ss1 vs ss2
        fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
        profile_a = 100*np.stack(df_session.loc[df_session[f'{cell_type}_ss1'], f'{trace_col}_ss1'])
        profile_b = 100*np.stack(df_session.loc[df_session[f'{cell_type}_ss2'], f'{trace_col}_ss2'])      
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
        save_fig(fig, out_dir, fig_name=f'{drug}_{cell_type}_mean_trace_ss1_ss2', save=save_plot)
        
        
        # plot ROI mean trace normalized trace
        # fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
        # a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss1'], f'{trace_col}_ss1'])
        # a = normalize(a)
        # pf.plot_mean_trace(a, 
        #                    ax, color='steelblue', label='ss1')
        # b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], f'{trace_col}_ss2'])
        # b = normalize(b)
        # pf.plot_mean_trace(b,
        #                    ax, color='orange', label=f'ss2_{drug}')
        # ax.set(ylabel='norm. dF/F')
        # ax.legend(frameon=False)
        # save_fig(fig, OUT_DIR_FIG, fig_name='', save=save_plot)
        


#%% PATHS AND PARAMS

# session list 
# drug = 'SCH'
drug = 'prazosin'
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
trace_col = 'mean_profile_good'
ratio_col = 'response_ratio_good'

# PATHS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS\single_sessions_good_trials_plot")
if not OUT_DIR_FIG.exists():
    OUT_DIR_FIG.mkdir()
save_plot = 1
#%% data pooling
df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
# data pooling for drug sessions
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_drug), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    out_dir = OUT_DIR_FIG/f'{rec_id}_{drug}'
    if not (out_dir).exists():
        out_dir.mkdir()
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_profile = sort_response(df_profile, thresh_up, thresh_down,
                                     ratio_type=ratio_col,
                                     trace_type=trace_col)
    try:
        _plot_session(df_profile, drug, out_dir=out_dir, save_plot=save_plot)
    except:
        print('error')
    
# data pooling for ctrl sessions
for rec_idx, rec in tqdm(rec_ctrl.iterrows(), total=len(rec_ctrl), desc="loading sessions"):
    anm    = rec['anm']
    date   = rec['date']
    rec_id = anm +'-'+date
    out_dir = OUT_DIR_FIG/f'{rec_id}_saline'
    if not (out_dir).exists():
        out_dir.mkdir()
    p_profile  = OUTPUT_RES/f'{anm}-{date}_raw_dff_profile.parquet'
    df_profile =  pd.read_parquet( p_profile)
    df_profile['anm'] = anm
    df_profile['date'] = date
    df_profile = sort_response(df_profile, thresh_up, thresh_down,
                                      ratio_type=ratio_col,
                                      trace_type=trace_col)
    try:
        _plot_session(df_profile, 'saline', out_dir=out_dir, save_plot=save_plot)
    except:
        print('error')
    


    
    

    
    
    




    



