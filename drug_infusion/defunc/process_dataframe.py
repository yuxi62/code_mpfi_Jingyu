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
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
from drug_infusion.plot_functions import plot_population_heatmap
from drug_infusion.utils_infusion import load_profile, load_behaviour, calculate_ratio, sort_response
from drug_infusion.rec_lst_infusion import selected_rec_lst, df_session_info
from drug_infusion.trial_selection import select_good_trials

from common import plotting_functions_Jingyu as pf
from common.utils_basic import normalize
#%%
def plot_session(df_session, drug, out_dir=None):
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
        out_dir = out_dir_parent + r'\{}_{}'.format(drug, rec_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df_gcamp_profile = sort_response(df_gcamp_profile, thresh_up, thresh_down, active_thresh, ratio_type='run_ratio_raw')
        plot_session(df_gcamp_profile, drug, out_dir)
    
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
#%%
# selected subset of recordings
rec_lst = selected_rec_lst

drug = 'prazosin'
# drug = 'propranolol' 
# drug = 'SCH'                                                                                                                                                                                                                                                                                            
rec_drug = rec_lst[rec_lst['label'].apply(lambda x: f'{drug}' in x)]
rec_drug = rec_drug[rec_drug['conc'].apply(lambda x: np.nanmin(x)>=1)] # with concentration > 1 mg/ml
rec_ctrl = rec_lst[rec_lst['label'].apply(lambda x: 'ctrl' in x)]
# only keep ctrl sessions from animals that have drug sessions
rec_ctrl = rec_ctrl[rec_ctrl['anm'].isin(rec_drug['anm'])]
rec_all = pd.concat((rec_drug, rec_ctrl))

out_dir_parent = r"Z:\Jingyu\Code\2p_SCH23390_infusion\session_selected_trials_raw\{}".format(drug)
if not os.path.exists(out_dir_parent):
    os.makedirs(out_dir_parent)
    
df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
#%% data pooling
pre_window=(-1, -0.5)
post_window=(0.5, 1.5)
thresh_up = 1.3
thresh_down = 1/thresh_up
active_thresh = 0
bef, aft = 2, 4
plot_single_session = 0
# trace_type = 'raw' # using raw_dff, ulternative is 'rsd'

errors = []
# data pooling for drug sessions
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_lst), desc="Processing sessions"):
# for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC310')&(rec_all['date']=='20250819')].iterrows():
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
    
    # df_drug_pool = pd.concat((df_drug_pool, df_processed_profile))
    
# df_drug_pool.to_parquet(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\processed_dataframe\pooled_df_drug_pool_{}_{}_thresh={}.parquet"
#                     .format(pre_window, post_window, thresh_up))    
    
# data pooling for ctrl sessions
for rec_idx, rec in tqdm(rec_ctrl.iterrows(), total=len(rec_lst), desc="Processing sessions"):
# for rec_idx, rec in rec_all.loc[(rec_all['anm']=='AC986')&(rec_all['date']=='20250624')].iterrows():
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
    
    # df_ctrl_pool = pd.concat((df_ctrl_pool, df_processed_profile))


#%% testing code - look into one session
thresh_up = 1.5
thresh_down = 1/thresh_up

df_drug_pool_pyr = sort_response(df_drug_pool, thresh_up, thresh_down,
                                 active_thresh=0, ratio_type='run_ratio_raw_good')
# select recordings (rec_id) with n_keep_trials > 15 for both ss1 and ss2
valid_recs = df_drug_pool_pyr.groupby(['anm', 'date']).apply(
    lambda g: (g['n_keep_trials_ss1'].iloc[0] > 15) & (g['n_keep_trials_ss2'].iloc[0] > 15))
valid_recs = valid_recs[valid_recs].index
df_drug_pool_pyr = df_drug_pool_pyr.set_index(['anm', 'date']).loc[valid_recs].reset_index()

anm = 'AC986'
date  ='20250624'
df_temp = df_drug_pool_pyr.loc[(df_drug_pool_pyr['anm']== anm)&
                               (df_drug_pool_pyr['date']== date)]
out_dir_ss = r"\\mpfi.org\Public\Wang lab\Jingyu\Code\2p_SCH23390_infusion\session_selected_trials_raw\test_{}-{}_ss1_{}{}".format(anm, date, pre_window, post_window)
if not os.path.exists(out_dir_ss):
    os.makedirs(out_dir_ss)
cell_type = 'active_soma'
s = 'ss1'
prefix = s

fig=plot_population_heatmap(df_temp, rec_id, bef, aft, s, prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting=s, activity_profile='run_cal_profile_raw_good', ratio='run_ratio_raw_good')
plt.savefig(out_dir_ss+r'\heatmap_raw_dff.png')
plt.show()
plt.close()

fig=plot_population_heatmap(df_temp, rec_id, bef, aft, s, prefix=prefix, suffix=f'{cell_type}',
                            session_for_sorting=s, activity_profile='run_cal_profile_rsd_good', ratio='run_ratio_raw_good')
plt.savefig(out_dir_ss+r'\heatmap_rsd_dff.png')
plt.show()
plt.close()

for G in ['pyrUp', 'pyrDown']:
    temp_pyrDown = df_temp.loc[df_temp[f'{G}_{s}']]
    xaxis = np.arange(30*(bef+aft))/30-bef
    for idx, roi in temp_pyrDown.iterrows():
        roi_profile_raw = roi[f'run_cal_profile_raw_good_{s}']
        roi_profile_rsd = roi[f'run_cal_profile_rsd_good_{s}']
        run_ratio = roi[f'run_ratio_raw_good_{s}']
        roi_profile_raw = gaussian_filter1d(roi_profile_raw, sigma=1)
        roi_profile_rsd = gaussian_filter1d(roi_profile_rsd, sigma=1)
        fig, ax = plt.subplots(figsize=(2,2), dpi=200)
        ax.plot(xaxis, roi_profile_raw, lw=1, color='green', label='raw_dFF')
        ax.plot(xaxis, roi_profile_rsd, lw=1, color='orange', label='rsd_filtered')
        ax.set(title=f'{G}_ratio={run_ratio:.3f}', xlim = (-1.5, 4), xlabel='time (s)', ylabel='dF/F')
        ax.legend(frameon=False, prop={'size': 5})
        fig.tight_layout()
        plt.savefig(out_dir_ss + r'\{}_uid{}.png'.format(G, roi['unit_id']), dpi=200)
        # plt.show()
        plt.close()
    
#%% plot heatmap
