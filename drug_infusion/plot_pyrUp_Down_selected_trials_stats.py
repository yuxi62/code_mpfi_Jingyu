# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 19:16:29 2025

@author: Jingyu Cao
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
from drug_infusion.plot_functions import plot_population_heatmap
from drug_infusion.utils_infusion import load_profile, load_behaviour, calculate_ratio, sort_response
from drug_infusion.rec_lst_infusion import selected_rec_lst, df_session_info
from drug_infusion.trial_selection import select_good_trials

from common import plotting_functions_Jingyu as pf
from common.utils_basic import normalize
#%%
# selected subset of recordings
rec_lst = selected_rec_lst

# drug = 'prazosin'
# drug = 'propranolol'
drug = 'SCH'
rec_drug = rec_lst[rec_lst['label'].apply(lambda x: f'{drug}' in x)]
rec_drug = rec_drug[rec_drug['conc'].apply(lambda x: np.nanmin(x)>=1)] # with concentration > 1 mg/ml
rec_ctrl = rec_lst[rec_lst['label'].apply(lambda x: 'ctrl' in x)]
# only keep ctrl sessions from animals that have drug sessions
rec_ctrl = rec_ctrl[rec_ctrl['anm'].isin(rec_drug['anm'])]
rec_all = pd.concat((rec_drug, rec_ctrl))

out_dir_parent = r"Z:\Jingyu\Code\2p_SCH23390_infusion\session_selected_trials\{}".format(drug)
if not os.path.exists(out_dir_parent):
    os.makedirs(out_dir_parent)
    
df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
#%% data pooling
pre_window=(-1, 0)
post_window=(0, 1.5)
thresh_up = 1.5
thresh_down = 1/thresh_up
active_thresh = 0
bef, aft = 2, 4
reprocess_pooling = 1
trace_type = 'raw' # using raw_dff, ulternative is 'rsd'

errors = []
# data pooling for drug sessions
# for rec_idx, rec in rec_drug.iterrows():
for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC986')&(rec_all['date']=='20250624')].iterrows():
    anm_id = rec['anm']
    date = rec['date']
    rec_id = anm_id+'-'+date
    
    df_gcamp_profile = load_profile(rec)
    session_info = df_session_info.loc[rec_id]
    
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
    
    # df_drug_pool = pd.concat((df_drug_pool, df_to_pool))
    
    
# data pooling for ctrl sessions
# for rec_idx, rec in rec_ctrl.iterrows():
# for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC310')&(rec_all['date']=='20250820')].iterrows():
#     df_to_pool = extract_session_info(rec, trial_type, df_session_info, bef, aft,
#                                       pre_window, post_window)
#     df_ctrl_pool = pd.concat((df_ctrl_pool, df_to_pool))

# # !!!! exclude baseline drifting session
# df_filtered = df_drug_pool.loc[
#     ~((df_drug_pool['anm'] == 'AC310') & (df_drug_pool['date'] == '20250826'))
# ]   
#%%
df_drug_pool_pyr = sort_response(df_drug_pool, thresh_up, thresh_down, 
                                 active_thresh=0, ratio_type='run_ratio_rsd_good')       
df_ctrl_pool_pyr = sort_response(df_ctrl_pool, thresh_up, thresh_down, 
                                 active_thresh=0, ratio_type='run_ratio_rsd_good')
# #%% plot mean pyrUp traces for seleceted trials
# select sessions only with keep trials > 15 for both ss1 and ss2
df_drug_pool_pyr = df_drug_pool_pyr.loc[(df_drug_pool_pyr['n_keep_trials_ss1'] > 15) & 
                              (df_drug_pool_pyr['n_keep_trials_ss2'] > 15)]
# plot mean trace compare ss1 and ss2
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)        
pf.plot_mean_trace(np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss1'], 'run_cal_profile_rsd_good_ss1']), 
                    ax, color='steelblue', label='ss1')
pf.plot_mean_trace(np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], 'run_cal_profile_rsd_good_ss2']),
                    ax, color='orange', label=f'ss2_{drug}')
# ax.set(ylim=(0.12, 0.18))
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
plt.show()
plt.close()

# plot ROI mean trace normalized trace
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)
a = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss1'], 'run_cal_profile_rsd_good_ss1'])
a = normalize(a)
pf.plot_mean_trace(a, 
                   ax, color='steelblue', label='ss1')
b = np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], 'run_cal_profile_rsd_good_ss2'])
b = normalize(b)
pf.plot_mean_trace(b,
                   ax, color='orange', label=f'ss2_{drug}')
ax.set(ylabel='norm. dF/F')
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
plt.show()
plt.close()

# plot session mean trace
cell_type = 'pyrUp_ss1'
grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
a = np.stack(grouped['run_cal_profile_rsd_good_ss1'].mean())
# a = utl.normalize(a)
cell_type = 'pyrUp_ss2'
grouped = df_drug_pool_pyr.loc[df_drug_pool_pyr[cell_type]].groupby(['anm', 'date'])
b = np.stack(grouped['run_cal_profile_rsd_good_ss2'].mean())
# b = utl.normalize(b)
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)
pf.plot_mean_trace(a, 
                   ax, color='steelblue', label='ss1')
pf.plot_mean_trace(b,
                   ax, color='orange', label=f'ss2_{drug}')
ax.set(ylabel='dF/F')
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
plt.show()
plt.close()

df_ctrl_pool_pyr = df_ctrl_pool_pyr.loc[(df_ctrl_pool_pyr['n_keep_trials_ss1'] > 15) & 
                              (df_ctrl_pool_pyr['n_keep_trials_ss2'] > 15)]
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)        
pf.plot_mean_trace(np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss1'], 'run_cal_profile_rsd_good_ss1']), 
                    ax, color='steelblue', label='ss1')
pf.plot_mean_trace(np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss2'], 'run_cal_profile_rsd_good_ss2']),
                    ax, color='orange', label=f'ss2_ctrl')
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
# ax.set(ylim=(0.12, 0.18))
plt.show()
plt.close()


# plot mean trace compare saline and SCH
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)        
pf.plot_mean_trace(np.stack(df_ctrl_pool_pyr.loc[df_ctrl_pool_pyr['pyrUp_ss2'], 'run_cal_profile_rsd_good_ss2']), 
                    ax, color='grey', label='ss2_saline')
pf.plot_mean_trace(np.stack(df_drug_pool_pyr.loc[df_drug_pool_pyr['pyrUp_ss2'], 'run_cal_profile_rsd_good_ss2']),
                    ax, color='orange', label=f'ss2_{drug}')
# ax.set(ylim=(0.12, 0.18))
ax.legend(frameon=False)
# plt.savefig(out_dir+f'\pyrUp_mean_trace.png', dpi=200)
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
        'n_rois': len(g)
    }), include_groups=False
).reset_index()

fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_paired_scatter(ax, df_perc_drug['perc_pyrUp_ss1'], df_perc_drug['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['steelblue', 'orange'],
                                xticklabels=['baseline', f'{drug}'])
plt.show()

fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_paired_scatter(ax, df_perc_drug['perc_pyrDown_ss1'], df_perc_drug['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['steelblue', 'orange'],
                                xticklabels=['baseline', f'{drug}'])
plt.show()

fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_paired_scatter(ax, df_perc_ctrl['perc_pyrUp_ss1'], df_perc_ctrl['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['steelblue', 'grey'],
                                xticklabels=['baseline', f'saline'])
plt.show()

fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_paired_scatter(ax, df_perc_ctrl['perc_pyrDown_ss1'], df_perc_ctrl['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['steelblue', 'grey'],
                                xticklabels=['baseline', 'saline'])
plt.show()


fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['perc_pyrUp_ss2'], df_perc_drug['perc_pyrUp_ss2'],
                                ylabel='% pyrUp',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
plt.show()

fig, ax = plt.subplots(figsize=(1, 2), dpi=200)    
pf.plot_bar_with_unpaired_scatter(ax, df_perc_ctrl['perc_pyrDown_ss2'], df_perc_drug['perc_pyrDown_ss2'],
                                ylabel='% pyrDown',
                                colors=['grey', 'orange'],
                                xticklabels=['salien', f'{drug}'])
plt.show()


#%% plot heatmap of cells in selected trials
cell_type = ''
# df_rec = df_drug_pool_pyr.loc[(df_drug_pool_pyr['pyrUp_ss1'])
#                                   &(df_drug_pool_pyr['pyrUp_ss2']), 
#                                     ]
df_rec = df_ctrl_pool_pyr
colors = ['steelblue', 'gray']
rec_id = 'pool'

prefix='saline_ss1'
# prefix = f'{drug}_baseline'
fig=plot_population_heatmap(df_rec, rec_id, bef, aft, 'ss1', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss1', activity_profile='run_cal_profile_rsd_good')
fig.tight_layout()
# plt.savefig(out_dir+f'\heatmap_ss1.png', dpi=200) 
plt.show()
plt.close()

# prefix=f'{drug}_SCH'
prefix='saline_ss2'
fig=plot_population_heatmap(df_rec, rec_id, bef, aft, 'ss2', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss2', activity_profile='run_cal_profile_rsd_good')
fig.tight_layout()
# plt.savefig(out_dir+f'\heatmap_ss2.png', dpi=200) 
# for form in ['png', 'pdf',]:
#     plt.savefig(out_dir+r'\f6_population_heatmap_ctrl_cell={}_ss1_ratio={}_{:.2f}.{}'.format(cell_type, thresh_up, thresh_down, form),
#                 dpi=300)
plt.show()
plt.close()

#%%
# df_temp = df_drug_pool_pyr.loc[
# ((df_drug_pool_pyr['anm'] == 'AC310') & (df_drug_pool_pyr['date'] == '20250819'))
# ]   
cell_type = 'active_soma'

thresh_up = 1.2
thresh_down = 1/thresh_up
df_temp = df_gcamp_profile
df_temp = sort_response(df_temp, thresh_up, thresh_down, active_thresh, ratio_type='run_ratio_raw')

prefix='ss1'
# prefix = f'{drug}_baseline'
fig=plot_population_heatmap(df_temp, rec_id, bef, aft, 'ss1', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss1', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good')
fig.tight_layout()
# plt.savefig(out_dir+f'\heatmap_ss1.png', dpi=200) 
plt.show()
plt.close()

fig=plot_population_heatmap(df_temp, rec_id, bef, aft, 'ss2', prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting='ss2', activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good')
fig.tight_layout()
# plt.savefig(out_dir+f'\heatmap_ss1.png', dpi=200) 
plt.show()
plt.close()

# plot mean trace compare ss1 and ss2
fig, ax = plt.subplots(figsize=(3, 2), dpi=200)        
pf.plot_mean_trace(np.stack(df_temp.loc[df_temp['pyrUp_ss1'], 'run_cal_profile_raw_good_ss1']), 
                   ax, label='ss1')
pf.plot_mean_trace(np.stack(df_temp.loc[df_temp['pyrUp_ss2'], 'run_cal_profile_raw_good_ss2']),
                   ax, color='orange', label=f'ss2_{drug}')
ax.legend(frameon=False)
plt.show()
plt.close()