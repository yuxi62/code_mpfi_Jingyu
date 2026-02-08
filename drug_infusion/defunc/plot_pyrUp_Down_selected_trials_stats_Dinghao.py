# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 19:16:29 2025

@author: Jingyu Cao
@modifier: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.ndimage import gaussian_filter1d

sys.path.extend(['Z:/Jingyu/code_mpfi_Jingyu/drug_infusion',
                 'Z:/Jingyu/code_mpfi_Jingyu/common'])
from plot_functions import plot_population_heatmap
from utils_infusion import load_profile, load_behaviour, calculate_ratio, sort_response
from rec_lst_infusion import rec_lst, df_session_info
from trial_selection import select_good_trials

from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()

from common.utils_basic import normalize


#%% paths and parameters 
save_stem = Path('Z:/Jingyu/LC_HPC_manuscript/figure6_drug_infusion_test')


#%% func
def _plot_session(df_session, drug, out_dir=None):
    cell_type = 'active_soma'
    prefix = 'ss1_baseline'
    fig=plot_population_heatmap(df_session, rec_id, bef, aft, 'ss1', prefix=prefix, suffix=f'{cell_type}',
                                session_for_sorting='ss1', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good')
    fig.tight_layout()
    if out_dir:
        plt.savefig(out_dir+r'\heatmap_ss1.png', dpi=200)
    else:
        plt.show()
    plt.close()
    
    prefix = f'ss2_{drug}'
    fig=plot_population_heatmap(df_session, rec_id, bef, aft, 'ss2', prefix=prefix, suffix=f'{cell_type}',
                                session_for_sorting='ss2', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good')
    fig.tight_layout()
    if out_dir:
        plt.savefig(out_dir+r'\heatmap_ss2.png', dpi=200)
    else:
        plt.show()
    plt.close()

    # plot mean trace compare ss1 and ss2
    fig, ax = plt.subplots(figsize=(3, 2), dpi=200)        
    pf.plot_mean_trace(np.stack(df_session.loc[df_session['pyrUp_ss1'], 'run_cal_profile_raw_good_ss1']), 
                       ax, label='ss1')
    pf.plot_mean_trace(np.stack(df_session.loc[df_session['pyrUp_ss2'], 'run_cal_profile_raw_good_ss2']),
                       ax, color='orange', label=f'ss2_{drug}')
    ax.legend(frameon=False)
    if out_dir:
        plt.savefig(out_dir+r'\mean_trace.png', dpi=200)
    else:
        plt.show()
    plt.close()

def process_profile(rec, plot_single_session=0):
    anm_id = rec['anm']
    date = rec['date']
    rec_id = anm_id+'-'+date
    drug = rec['label'][1]
    
    df_gcamp_profile = load_profile(rec)
    session_info = df_session_info.loc[rec_id]
    
    meta = rec.loc[['anm', 'date', 'session', 'label', 'conc', 'latency']] 
    meta_df = pd.DataFrame([meta] * len(df_gcamp_profile))  # repeat for each row
    
    for trace_type in ['raw', 'rsd']:
        beh_ss1 = load_behaviour(rec, 'ss1')
        selected_trials_ss1 = select_good_trials(beh_ss1)
        selected_trials_ss1[:10] = 0 # exclude first 10 trials
        df_gcamp_profile['n_keep_trials_ss1'] = np.sum(selected_trials_ss1)
        df_gcamp_profile = calculate_ratio(df_gcamp_profile, 'ss1', session_info, trace_type, beh_ss1,
                            selected_trials=selected_trials_ss1,
                            trial_bef=bef, trial_aft=aft, 
                            pre_window=pre_window, post_window=post_window)
        beh_ss2 = load_behaviour(rec, 'ss2')
        selected_trials_ss2 = select_good_trials(beh_ss2)
        selected_trials_ss2[:10] = 0 # exclude first 10 trials
        df_gcamp_profile['n_keep_trials_ss2'] = np.sum(selected_trials_ss2)
        df_gcamp_profile = calculate_ratio(df_gcamp_profile, 'ss2', session_info, trace_type, beh_ss2,
                            selected_trials=selected_trials_ss2,
                            trial_bef=bef, trial_aft=aft, 
                            pre_window=pre_window, post_window=post_window)  
    if plot_single_session:
        out_path = save_stem / drug / f'{drug}_{rec_id}'
        out_path.mkdir(exist_ok=True)
        
        df_gcamp_profile = sort_response(df_gcamp_profile, thresh_up, thresh_down, active_thresh, ratio_type='run_ratio_raw')
        _plot_session(df_gcamp_profile, drug, out_path)
    
    # Reset index of df_gcamp_profile to align meta info
    df_gcamp_profile = df_gcamp_profile.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    df_processed_profile = pd.concat([meta_df, df_gcamp_profile[[
         'unit_id',
         'active_soma',
         'n_keep_trials_ss1',
         'n_keep_trials_ss2', 

         'run_ratio_rsd_ss1',
         'run_ratio_rsd_ss2',
         'run_ratio_rsd_good_ss1',
         'run_ratio_rsd_good_ss2',
         
         'run_cal_profile_rsd_ss1',
         'run_cal_profile_rsd_ss2',
         'run_cal_profile_rsd_good_ss1',
         'run_cal_profile_rsd_good_ss2',
         
         'run_ratio_raw_ss1',
         'run_ratio_raw_ss2',
         'run_ratio_raw_good_ss1',
         'run_ratio_raw_good_ss2',
         
         'run_cal_profile_raw_ss1',
         'run_cal_profile_raw_ss2',
         'run_cal_profile_raw_good_ss1',
         'run_cal_profile_raw_good_ss2',
     ]]], axis=1)
    
    return df_processed_profile

#%% parameters 
# drug = 'prazosin'
# drug = 'propranolol'
drug = 'SCH'

# sub-select sessions with specified drugs 
rec_ctrl = rec_lst[rec_lst['label'].apply(lambda x: 'ctrl' in x)]

rec_drug = rec_lst[rec_lst['label'].apply(lambda x: f'{drug}' in x)]
rec_drug = rec_drug[rec_drug['conc'].apply(lambda x: np.nanmin(x)>=1)] # with concentration > 1 mg/ml

# only keep ctrl sessions from animals that have drug sessions
rec_ctrl = rec_ctrl[rec_ctrl['anm'].isin(rec_drug['anm'])]
rec_all = pd.concat((rec_drug, rec_ctrl))

# save stem initiation
save_path = save_stem / drug
save_path.mkdir(exist_ok=True)

df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()


#%% data pooling
pre_window=(-1, -0.5)
post_window=(0.5, 1.5)
thresh_up = 1.25
thresh_down = 1/thresh_up
active_thresh = 0
bef, aft = 2, 4
plot_single_session = 0
# trace_type = 'raw' # using raw_dff, ulternative is 'rsd'

errors = []
# data pooling for drug sessions
# for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_lst), desc="Processing sessions"):
for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC302')&(rec_all['date']=='20250729')].iterrows():
    anm_id = rec['anm']
    date = rec['date']
    rec_id = anm_id+'-'+date
    p_df_processed_profile = f'Z:/Jingyu/Code/2p_SCH23390_infusion/raw_data/processed_dataframe/{rec_id}_df_processed_profile_{pre_window}_{post_window}.parquet'
    
    if not os.path.exists(p_df_processed_profile):
        df_processed_profile = process_profile(rec, plot_single_session)     
        df_processed_profile.to_parquet(p_df_processed_profile)
    else:
        df_processed_profile = pd.read_parquet(p_df_processed_profile)
    
    df_drug_pool = pd.concat((df_drug_pool, df_processed_profile))
    
# df_drug_pool.to_parquet(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe\pooled_df_drug_pool_{}_{}_thresh={}.parquet"
#                     .format(pre_window, post_window, thresh_up))    
    
# data pooling for ctrl sessions
for rec_idx, rec in tqdm(rec_ctrl.iterrows(), total=len(rec_lst), desc="Processing sessions"):
# for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC989')&(rec_all['date']=='20250625')].iterrows():
    anm_id = rec['anm']
    date = rec['date']
    rec_id = anm_id+'-'+date
    p_df_processed_profile = r"Z:\Jingyu\Code\2p_SCH23390_infusion\raw_data\processed_dataframe\{}_df_processed_profile_{}_{}.parquet"\
                        .format(rec_id, pre_window, post_window)
    if not os.path.exists(p_df_processed_profile):
        df_processed_profile = process_profile(rec, plot_single_session)     
        df_processed_profile.to_parquet(p_df_processed_profile)
    else:
        df_processed_profile = pd.read_parquet(p_df_processed_profile)
    
    df_ctrl_pool = pd.concat((df_ctrl_pool, df_processed_profile))

#%% selection recordings for statistics
df_drug_pool_pyr = sort_response(df_drug_pool, thresh_up, thresh_down,
                                 active_thresh=0, ratio_type='run_ratio_raw')
df_ctrl_pool_pyr = sort_response(df_ctrl_pool, thresh_up, thresh_down,
                                 active_thresh=0, ratio_type='run_ratio_raw')
 
# select recordings (rec_id) with n_keep_trials > 15 for both ss1 and ss2
valid_recs = df_drug_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trials_ss1'].iloc[0] > 15) & (g['n_keep_trials_ss2'].iloc[0] > 15))
valid_recs = valid_recs[valid_recs].index
df_drug_pool_pyr = df_drug_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

# Select only the first 3 recording dates per animal
first3_dates = (df_drug_pool_pyr[['anm', 'date']].drop_duplicates()
                .sort_values(['anm', 'date'])
                .groupby('anm').head(3))
df_drug_pool_pyr_first3 = df_drug_pool_pyr.merge(first3_dates, on=['anm', 'date'], how='inner')
# df_drug_pool_pyr_first3.to_parquet(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe\df_drug_pool_pyr_first3_{drug}.parquet")

valid_recs = df_ctrl_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trials_ss1'].iloc[0] > 15) & (g['n_keep_trials_ss2'].iloc[0] > 15))
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

# df_drug_pool_pyr = df_drug_pool_pyr_first3
# df_ctrl_pool_pyr = df_ctrl_pool_pyr_first3
    
#%%
trace_type = 'raw_good'
# Apply trace_filter to all trace columns
trace_cols = [
              # 'run_cal_profile_rsd_good_ss1', 'run_cal_profile_rsd_good_ss2',
              # 'run_cal_profile_rsd_ss1', 'run_cal_profile_rsd_ss2',
              # 'run_cal_profile_raw_ss1', 'run_cal_profile_raw_ss2',
              f'run_cal_profile_{trace_type}_ss1', f'run_cal_profile_{trace_type}_ss2'
              ]

# for col in trace_cols:
#     if col in df_drug_pool_pyr.columns:
#         df_drug_pool_pyr[col] = df_drug_pool_pyr[col].apply(
#             lambda x: trace_filter_sd(np.array(x), n_sd=5) if x is not None and len(x) > 0 else x)
#     if col in df_ctrl_pool_pyr.columns:
#         df_ctrl_pool_pyr[col] = df_ctrl_pool_pyr[col].apply(
#             lambda x: trace_filter_sd(np.array(x), n_sd=5) if x is not None and len(x) > 0 else x)

# plot drug ss1 vs ss2
fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
profile_a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss1'], f'run_cal_profile_{trace_type}_ss1'])
profile_b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], f'run_cal_profile_{trace_type}_ss2'])      
pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                      test='ranksum',
                                      time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                      baseline_window=(-0.5, 0.5),
                                      labels = ['baseline', f'{drug}'],
                                      colors = ['steelblue', 'orange'],
                                      bef=2, aft=4, sample_freq=30,
                                      show_scalebar=True,)
ax.set(xlim=(-1, 4))
ax.legend(frameon=False)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_pyrUp_mean_trace_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)

plt.show()
plt.close()

# plot ROI mean trace normalized trace
fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss1'], f'run_cal_profile_{trace_type}_ss1'])
a = normalize(a)
pf.plot_mean_trace(a, 
                   ax, color='steelblue', label='ss1')
b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], f'run_cal_profile_{trace_type}_ss2'])
b = normalize(b)
pf.plot_mean_trace(b,
                   ax, color='orange', label=f'ss2_{drug}')
ax.set(ylabel='norm. dF/F')
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
plt.show()
plt.close()

# plot session mean trace
fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
cell_type = 'pyrUp_ss1'
grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
profile_a = np.stack(grouped[f'run_cal_profile_{trace_type}_ss1'].mean())
cell_type = 'pyrUp_ss2'
grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
profile_b = np.stack(grouped[f'run_cal_profile_{trace_type}_ss2'].mean())
pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                      test='ranksum',
                                      time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                      baseline_window=(-0.5, 0.5),
                                      labels = ['baseline', f'{drug}'],
                                      colors = ['steelblue', 'orange'],
                                      bef=2, aft=4, sample_freq=30,
                                      show_scalebar=True,
                                      )
ax.set(xlim=(-1, 4))
ax.legend(frameon=False)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_pyrUp_mean_trace_session_mean_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

# plot saline ss1 vs ss2
fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
profile_a = np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss1'], f'run_cal_profile_{trace_type}_ss1'])
profile_b = np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss2'], f'run_cal_profile_{trace_type}_ss2'])       
pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                      test='ranksum',
                                      time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                      baseline_window=(-0.5, 0.5),
                                      labels = ['baseline', 'saline'],
                                      colors = ['steelblue', 'grey'],
                                      bef=2, aft=4, sample_freq=30,
                                      show_scalebar=True,
                                      )
ax.set(xlim=(-1, 4))
ax.legend(frameon=False)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_pyrUp_mean_trace_{}_ctrl_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

# plot mean trace compare saline and SCH
fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
profile_a = np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss2'], f'run_cal_profile_{trace_type}_ss2'])
profile_b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], f'run_cal_profile_{trace_type}_ss2'])     
pf.plot_two_traces_with_binned_stats(profile_a, profile_b, ax,
                                      test='ranksum',
                                      time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                      baseline_window=(-0.5, 0.5),
                                      labels = ['saline', f'{drug}'],
                                      colors = ['grey', 'orange'],
                                      bef=2, aft=4, sample_freq=30,
                                      show_scalebar=True,
                                      )
ax.set(xlim=(-1, 4))
ax.legend(frameon=False)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_pyrUp_mean_trace_{}vs_saline_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()
#%% Bar plot
cell_type = 'active_soma'
grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'], )
recs = [name for name, g in grouped]
df_perc_drug = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / g['pyrAct_ss1'].sum() if g['pyrAct_ss1'].sum() > 0 else np.nan,
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / g['pyrAct_ss2'].sum() if g['pyrAct_ss2'].sum() > 0 else np.nan,
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum() / g['pyrAct_ss1'].sum() if g['pyrAct_ss1'].sum() > 0 else np.nan,
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum() / g['pyrAct_ss2'].sum() if g['pyrAct_ss2'].sum() > 0 else np.nan,
        'delta_perc_pyrUp': (g['pyrUp_ss2'].sum()  / g['pyrAct_ss2'].sum()) - (g['pyrUp_ss1'].sum()   / g['pyrAct_ss1'].sum()),
        'delta_perc_pyrDown': (g['pyrDown_ss2'].sum()  / g['pyrAct_ss2'].sum()) - (g['pyrDown_ss1'].sum()   / g['pyrAct_ss1'].sum()),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

grouped = df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr[cell_type]].groupby(['anm', 'date'], )
df_perc_ctrl = grouped.apply(
    lambda g: pd.Series({
        'perc_pyrUp_ss1':   g['pyrUp_ss1'].sum()   / g['pyrAct_ss1'].sum() if g['pyrAct_ss1'].sum() > 0 else np.nan,
        'perc_pyrUp_ss2':   g['pyrUp_ss2'].sum()   / g['pyrAct_ss2'].sum() if g['pyrAct_ss2'].sum() > 0 else np.nan,
        'perc_pyrDown_ss1': g['pyrDown_ss1'].sum() / g['pyrAct_ss1'].sum() if g['pyrAct_ss1'].sum() > 0 else np.nan,
        'perc_pyrDown_ss2': g['pyrDown_ss2'].sum() / g['pyrAct_ss2'].sum() if g['pyrAct_ss2'].sum() > 0 else np.nan,
        'delta_perc_pyrUp': (g['pyrUp_ss2'].sum()  / g['pyrAct_ss2'].sum()) - (g['pyrUp_ss1'].sum()   / g['pyrAct_ss1'].sum()),
        'delta_perc_pyrDown': (g['pyrDown_ss2'].sum()  / g['pyrAct_ss2'].sum()) - (g['pyrDown_ss1'].sum()   / g['pyrAct_ss1'].sum()),
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_paired_scatter(ax, df_perc_drug['perc_pyrUp_ss1'], df_perc_drug['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['steelblue', 'orange'],
                                xticklabels=['baseline', f'{drug}'],
                                ylim=(0, 0.7)
                                )
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrUp_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_paired_scatter(ax, df_perc_drug['perc_pyrDown_ss1'], df_perc_drug['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['steelblue', 'orange'],
                                xticklabels=['baseline', f'{drug}'],
                                ylim=(0, 0.12),
                                )
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrDown_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_paired_scatter(ax, df_perc_ctrl['perc_pyrUp_ss1'], df_perc_ctrl['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['steelblue', 'grey'],
                                xticklabels=['baseline', f'saline'],
                                ylim=(0, 0.7))
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrUp_{}_ctrl_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_paired_scatter(ax, df_perc_ctrl['perc_pyrDown_ss1'], df_perc_ctrl['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['steelblue', 'grey'],
                                xticklabels=['baseline', 'saline'],
                                ylim=(0, 0.12)
                                )
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrDown_{}_ctrl_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['perc_pyrUp_ss2'], df_perc_drug['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrUp_{}_vs_saline_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['perc_pyrDown_ss2'], df_perc_drug['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_perc_pyrDown_{}_vs_saline_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['delta_perc_pyrUp'], df_perc_drug['delta_perc_pyrUp'],
                                ylabel='Δ% PyrUp (vs baseline)',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_delta_perc_pyrUp_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['delta_perc_pyrDown'], df_perc_drug['delta_perc_pyrDown'],
                                ylabel='Δ% PyrDown (vs baseline)',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_delta_perc_pyrDown_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()
    
#%% plot heatmap
prefix = 'ss1_baseline'
rec_id = df_drug_pool_pyr.anm.iloc[0] + '-' + df_drug_pool_pyr.date.iloc[0]
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss1', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good',
                            plot_mean=0)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_heatmap_ss1_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

prefix = f'ss2_{drug}'
fig=plot_population_heatmap(df_drug_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss2', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good',
                            plot_mean=0)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_heatmap_ss2_{}_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

# saline ctrl
prefix = 'ss1_baseline'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss1', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss1', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good',
                            plot_mean=0)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_heatmap_ss1_{}_ctrl_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()

prefix = f'ss2_{drug}'
fig=plot_population_heatmap(df_ctrl_pool_pyr, rec_id, bef, aft, 'ss2', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss2', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good',
                            plot_mean=0)
fig.tight_layout()
for form in ['png', 'pdf',]:
    plt.savefig(out_dir_parent+r'\f6_heatmap_ss2_{}_ctrl_ratio-{}.{}'.format(drug, thresh_up, form),
                dpi=300)
plt.show()
plt.close()


