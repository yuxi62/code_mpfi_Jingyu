# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 17:26:51 2025

@author: Jingyu Cao

perpose: update session_info dataframe with behavior trial info
"""
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
if (r"Z:\Jingyu\code_mpfi_Jingyu\drug_infusion"in sys.path) == False:
    sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu\drug_infusion")
from rec_lst_infusion import rec_lst_infusion
#%% func
def update_session_info(df_session_info, rec, exp=r'GCaMP8s_infusion'):
    if not isinstance(df_session_info, pd.DataFrame) or df_session_info.empty:
        df_session_info = pd.DataFrame(columns=[
            'anm', 'date', 'session', 'label', 'conc', 'latency', 
            'valid_trials_ss1', 'valid_trials_ss2', 'valid_trials_ss3',
            'good_trials_ss1', 'good_trials_ss2', 'good_trials_ss3',
            'bad_trials_ss1', 'bad_trials_ss2', 'bad_trials_ss3',
            'n_valid_trials_ss1', 'n_valid_trials_ss2', 'n_valid_trials_ss3',
            'n_good_trials_ss1', 'n_good_trials_ss2', 'n_good_trials_ss3',
            'n_bad_trials_ss1', 'n_bad_trials_ss2', 'n_bad_trials_ss3'
        ])
    # ensure object dtype for columns that hold lists
    for col in ['label', 'session',
                'good_trials_ss1','good_trials_ss2','good_trials_ss3',
                'bad_trials_ss1','bad_trials_ss2','bad_trials_ss3']:
        if col in df_session_info.columns:
            df_session_info[col] = df_session_info[col].astype('object')

    anm_id = rec['anm']
    ss      = rec['session']                  # list like ['02','04',...]
    date    = rec['date']
    sessions = ['ss1','ss2','ss3'][:len(ss)]
    session_label = rec['label']              # list like ['baseline','SCH',...]
    row_idx = f"{anm_id}-{date}"

    # --- ensure row exists BEFORE any list assignment ---
    if row_idx not in df_session_info.index:
        # create an empty row with all columns present
        df_session_info.loc[row_idx] = {col: pd.NA for col in df_session_info.columns}

        # now single-cell assignments with lists are unambiguous
        df_session_info.at[row_idx, 'anm']     = anm_id
        df_session_info.at[row_idx, 'date']    = date
        df_session_info.at[row_idx, 'session'] = ss                 # list ok
        df_session_info.at[row_idx, 'label']   = session_label      # list ok
        df_session_info.at[row_idx, 'conc']    = rec.get('conc', [None]*len(ss))
        df_session_info.at[row_idx, 'latency'] = rec.get('latency', None)
    
        # Load behavior & assign trials (lists in single cells are fine)
        for i, s in enumerate(sessions):
            p_beh = rf"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\behaviour_profile\{anm_id}-{date}-{ss[i]}.pkl"
            beh = pd.read_pickle(p_beh)
            n_trials = len(beh['run_onsets'])
            valid_trials = np.where((~np.isnan(beh['reward_times']))&
                                    (np.array(beh['non_stop_trials'])==0)&
                                    (np.array(beh['non_fullstop_trials'])==0),
                                    True, False)
            df_session_info.at[row_idx, f'valid_trials_{s}']   = valid_trials
            df_session_info.at[row_idx, f'n_valid_trials_{s}']  = np.sum(valid_trials)
            df_session_info.at[row_idx, f'perc_valid_trials_{s}']  = np.sum(valid_trials)/n_trials
            
            # good_trials, bad_trials = utl.seperate_bad_trial(beh)
            # df_session_info.at[row_idx, f'good_trials_{s}']   = good_trials  # list ok
            # df_session_info.at[row_idx, f'bad_trials_{s}']    = bad_trials   # list ok
            # df_session_info.at[row_idx, f'n_good_trials_{s}'] = len(good_trials)
            # df_session_info.at[row_idx, f'n_bad_trials_{s}']  = len(bad_trials)
            # df_session_info.at[row_idx, f'perc_good_trials_{s}'] = len(good_trials)/n_trials
            # df_session_info.at[row_idx, f'perc_bad_trials_{s}']  = len(bad_trials)/n_trials

                 
            
    return df_session_info

#%%
out_dir = r"Z:\Jingyu\code_mpfi_Jingyu\drug_infusion"
if __name__ == "__main__":
    p_session_info = out_dir + r'\infusion_session_info.parquet'
    if not os.path.exists(p_session_info):
        df_session_info = []
    else:
        # df_session_info = pd.read_pickle(p_session_info)
        df_session_info = pd.read_parquet(p_session_info)
    
    for rec in tqdm(rec_lst_infusion):
        df_session_info = update_session_info(df_session_info, rec)

    # df_session_info.to_pickle(r'Z:\Jingyu\Code\Python\2p_SCH23390_infusion\session_info_new.pkl')
    df_session_info.to_parquet(p_session_info)
