# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:56:19 2026

@author: Jingyu Cao
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# Add parent directory to path
CODE_ROOT = r"\\mpfi.org\Public\Wang lab\Jingyu\code_mpfi_Jingyu"
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from common.plot_single_trial_function import align_frame_behaviour, plot_single_trial_html_roi
from common.utils_imaging import percentile_dff
from drug_infusion.rec_lst_infusion import selected_rec_lst


#%%
rec_id = 'AC316-20251024'
# for idx, rec in selected_rec_lst.iloc[].iterrows():
# anm, date = rec['anm'], rec['date']
# rec_id = anm + '-' + date
concat_path = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{rec_id}"
p_beh_ss1 = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile\{rec_id}-02.pkl" 
# p_beh_ss2 = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile\AC310-20250829-04.pkl" 

p_suite2p_ss1 = r"Z:\Jingyu\2P_Recording\AC310\AC310-20250829\02\suite2p_func_detec\plane0"
# p_suite2p_ss2 = r"Z:\Jingyu\2P_Recording\AC310\AC310-20250829\04\suite2p_func_detec\plane0"

#
beh_ss1 = pd.read_pickle(p_beh_ss1)
# beh_ss2 = pd.read_pickle(p_beh_ss2)

#
sig_master = xr.open_dataarray(os.path.join(concat_path, "sig_master_raw.nc"))
F_all =  sig_master.values.squeeze()

sig_master_neu = xr.open_dataarray(os.path.join(concat_path, "sig_master_neu_raw.nc"))
Fneu_all =  sig_master_neu.values.squeeze()

F_corr = F_all-0.7*Fneu_all

suite2p_ss1_ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
suite2p_ss1_stat = np.load(p_suite2p_ss1+r'\stat.npy', allow_pickle=True)


F_corr_ss1 = F_corr[:, :suite2p_ss1_ops['nframes']]
    
dff_ss1, baseline_ss1 = percentile_dff(F_corr_ss1, return_baseline=True)

figure_path = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS"

baseline_mean_ss1 = np.nanmean(baseline_ss1, axis=-1)
plt.hist(baseline_mean_ss1, bins=100)
plt.savefig(figure_path + rf"\{rec_id}_roi_dff_20th_no_filter_baseline_mean_hist.png", dpi=200)
#%%
# for roi in range(20):
#     beh_ss1_aligend = align_frame_behaviour(beh_ss1)
#     roi_f = F_corr_ss1[roi]
#     roi_dff, roi_baseline = percentile_dff(roi_f, return_baseline=True)
    
#     fig = plot_single_trial_html_roi(beh_ss1_aligend,
#                                      roi_f, roi_dff, roi_baseline + 0.1,
#                                      labels={'ch1': 'gcamp_rawF-neu',
#                                              'ch2': 'gcamp_dff',
#                                              'ch3': 'baseline'},
#                                      colors={'ch1': 'grey',
#                                              'ch2': 'green',
#                                              'ch3': 'blue'},
#                                      shared_yaxis=['ch1', 'ch3']
#                                      )
#     fig.write_html(figure_path + rf"\{rec}_roi_uid{roi}_dff_20th_no_filter_1e-8_test.html")
    # fig.show()








