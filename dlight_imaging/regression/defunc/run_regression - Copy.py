# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:46:49 2026

@author: Jingyu Cao

dLight regression script
"""
import sys
import os
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")

from dlight_imaging.regression import Regression_Red_From_Green_Single_Trial
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils as utl
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = ''


DILATION_STEPS = (0, 2, 4, 6, 8, 10)
# DILATION_STEPS = (0, )

correction_params = {
    'grid_size': 16,
    'pre_frames': 90,
    'post_frames': 120,
    'n_jobs': -1
}
#%%
# rec = 'AC967-20250225-04'
# rec = 'AC969-20250319-04'

regression_name ='single_trial_regression'
original_stdout = sys.stdout
error_lst = []
     
for rec in tqdm(rec_lst):
    try:
        start_time = time.time()
        print(f'\n{rec}_running regression pipeline...')
        out_path_regression = OUR_DIR_REGRESS / f"{rec}"
        if not os.path.exists(out_path_regression):
            os.makedirs(out_path_regression)
            
        log_file = open(out_path_regression/f'{rec}_regression_processing_log.txt', 'w')
        log_file.write(f"\n{'='*50}\nRecording: {rec}\n{'='*50}\n")
        log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        sys.stdout = log_file
        
        anm, date, ss = rec.split('-')
        p_rec = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly\suite2p\plane0"
        # beh = pd.read_pickle(OUT_DIR_RAW_DATA / 'behaviour_profile' /  f'{rec}.pkl')
        
        # load suite2p ops
        suite2p_ops = np.load(p_rec+r'\ops.npy', allow_pickle=True).item()
        # load masks path or generate mask file    
        p_masks = utl.load_masks_axon_dlight(rec, OUR_DIR_REGRESS)
        # load or generate behaviour file
        beh_data = utl.load_behaviour(rec, OUR_DIR_REGRESS, suite2p_ops['nframes'])
        
        ## run regression
        nframes = suite2p_ops['nframes']
        # nframes = 10000
        
        # load ch1 and ch2 movie
        # dlight
        mov = utl.load_bin_file(p_rec, r'\data.bin', n_frames=nframes, height=512, width=512)
        movr = utl.load_bin_file(p_rec, r"\data_chan2.bin", n_frames=nframes, height=512, width=512)
        
        global_axon_mask = np.load(p_masks / 'dilated_global_axon_k=0.npy')
        global_dlight_mask = np.load(p_masks / 'global_dlight_mask_enhanced.npy')
        
        for k_size in DILATION_STEPS:
            print(f'regression for dilation step = {k_size}.....')
            global_axon_mask_dilated = np.load(p_masks / f'dilated_global_axon_k={k_size}.npy')
            dilated_global_axon_and_dlight_mask = (global_axon_mask_dilated)&(global_dlight_mask)
            global_neuropil = np.load(p_masks / f'dilated_global_axon_neuropil_k={k_size}.npy')
            
            p_dilation_results = OUR_DIR_REGRESS / rec / regression_name / r'dilation_k={}'.format(k_size)
            p_res = p_dilation_results / f'{regression_name}_res_traces.npz'
            
            # if not p_res.exists():
            # Run the regression pipeline
            print("Starting regression pipeline ...")
            Regression_Red_From_Green_Single_Trial.run_regression_pipeline(
                mov=mov,
                movr=movr,
                global_mask_green=dilated_global_axon_and_dlight_mask, # green mask
                global_mask_red=global_axon_mask, # red axon mask
                global_neuropil=global_neuropil, # axon_or_dlight mask, for calculating local background
                run_onset_frames = beh_data['run_onset_frames'],
                mean_img=global_axon_mask,
                regression_name = regression_name,
                output_dir=p_dilation_results,
                correction_params = correction_params
            )        
        del mov, movr
        
        print(f"Completed: {rec}")
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        print(f"Total time: {int(mins)}m {secs:.1f}s")
        log_file.close()
        sys.stdout = original_stdout
        print(f"Completed: {rec} - Time: {int(mins)}m {secs:.1f}s")
    
    except Exception as e:
        sys.stdout = original_stdout
        error_lst.append(rec)
        print(f"Error processing {rec}: {str(e)}")
        continue
        
