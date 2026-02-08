# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 18:00:44 2026
Modified on 3 Feb 2025

process traces for identified somata in drug infusion experiments 

@author: Jingyu Cao
@modifier: Dinghao Luo
"""

#%% imports 
from pathlib import Path
import sys 

import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
import xarray as xr
from scipy.ndimage import shift as ndi_shift
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

# add parent directories to path
directories = [
    'Z:/Jingyu/code_mpfi_Jingyu/common', 
    'Z:/Jingyu/code_mpfi_Jingyu/drug_infusion',
    'Z:/Dinghao/code_mpfi_dinghao/utils'
    ]
sys.path.extend(directories)

from common.utils_basic import trace_filter
from common.mask import generate_masks
from common.robust_sd_filter import robust_filter_along_axis
from common.trial_selection import select_good_trials, seperate_valid_trial
from common.event_response_quantification import quantify_event_response


#%%
def warp_rois_rigid(roi_map, sh, fill=0):
    """
    roi_map: (T, H, W) or (H, W)
    sh: (2,) rigid shift [dy, dx] (most common) OR [dx, dy] if you swap
    """
    sh = np.asarray(sh).astype(float).ravel()
    dy, dx = sh[0], sh[1]

    if roi_map.ndim == 2:
        shift_vec = (dy, dx)
    elif roi_map.ndim == 3:
        shift_vec = (0.0, dy, dx)  # no shift along first axis
    else:
        raise ValueError(f"roi_map must be 2D or 3D, got shape {roi_map.shape}")

    return ndi_shift(roi_map, shift=shift_vec, order=0, mode="constant", cval=fill)


def roi_map_to_list(roi_map):
    """
    Convert ROI map of shape (n_roi, H, W) into a list of dicts
    [{'npix': int, 'ypix': array, 'xpix': array}, ...]
    """
    roi_list = []
    n_roi = roi_map.shape[0]

    for i in range(n_roi):
        ypix, xpix = np.where(roi_map[i] > 0)  # coordinates where roi is active
        roi_list.append({
            'npix': len(ypix),
            'ypix': ypix,
            'xpix': xpix
        })
    return roi_list

#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion")
OUTPUT_RES = OUT_DIR_RAW_DATA / "processed_dataframe"

pre_window=(-1, -0.5)
post_window=(0.5, 1.5)
thresh_pyrUp = 1.3
thresh_pyrDown = 1/thresh_pyrUp
active_thresh = 0
bef, aft = 2, 4
plot_single_session = 0

df_drug_pool = pd.DataFrame()
df_ctrl_pool = pd.DataFrame()
#%%
# import recording list
import drug_infusion.rec_lst_infusion as rec_info
# rec_drug, rec_drug_ctrl = rec_info.rec_SCH, rec_info.rec_SCH_ctrl
# for (rec_drug, rec_drug_ctrl) in [(rec_info.rec_SCH, rec_SCH_ctrl)
rec_drug = rec_info.rec_lst
error_lst = []
# Process each recording
for rec_idx, rec in tqdm(rec_drug.iterrows(), total=len(rec_drug), desc="Processing sessions"):
    anm = rec['anm']
    date = rec['date']
    print(f'\n{anm}-{date}')
    data_path = OUT_DIR_RAW_DATA/'raw_signals'/f'{anm}-{date}'
    
    if not(data_path/r'soma_class.npz').exists():
        if not(data_path/r'F_corr.npy').exists():
            sig_master = xr.open_dataarray(data_path/"sig_master_raw.nc")
            F_all =  sig_master.values.squeeze()
            sig_master_neu = xr.open_dataarray(data_path/"sig_master_neu_raw.nc")
            Fneu_all =  sig_master_neu.values.squeeze()
            F_corr = F_all-0.7*Fneu_all
            np.save(data_path/r'F_corr.npy', F_corr)
        else:
            F_corr = np.load(data_path/r'F_corr.npy')
        
        if not (data_path/r'gcamp_stats.npy').exists():
            # always use the first session for reference mean image
            p_suite2p_ss1 = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\02\suite2p_func_detec\plane0"
            suite2p_ss1_ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
            mean_img_ch1 = suite2p_ss1_ops['meanImg']
            
            A_master = xr.open_dataarray(data_path/"A_master.nc")
            roi_map = A_master.values.squeeze()
            shift_ds = xr.open_dataset(data_path/"shift_ds.nc")
            sh = shift_ds["shifts"].sel(animal=anm, session=f'{date}_02')
            roi_map_shifted = warp_rois_rigid(roi_map, (-sh).values)
            gcamp_stats = roi_map_to_list(roi_map_shifted)
            gcamp_stats = roi_map_to_list(roi_map)
            np.save(data_path/r'gcamp_stats.npy', np.asarray(gcamp_stats, dtype='object'))
        else:
            gcamp_stats = np.load(data_path/r'gcamp_stats.npy', allow_pickle=True)
    
        
        is_soma, is_active, is_active_soma = generate_masks.select_gcamp_rois(mean_img_ch1, F_corr,
                                         gcamp_stats, 
                                         path_result=r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS")
        np.savez_compressed(
            data_path/r'soma_class.npz',
            is_soma=is_soma,
            is_active=is_active,
            is_active_soma=is_active_soma,
        )
    else:
        is_active_soma = np.load(data_path/r'soma_class.npz')['is_soma']
    
    #%% loading behaviour file
    # try:
    p_beh_ss1 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-02.pkl'
    p_beh_ss2 = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{anm}-{date}-04.pkl'
    beh_ss1   = pd.read_pickle(p_beh_ss1)
    beh_ss2   = pd.read_pickle(p_beh_ss2)

    run_ss1       = np.array(beh_ss1['run_onset_frames'])
    run_valid_ss1 = run_ss1[(seperate_valid_trial(beh_ss1))&(run_ss1!=-1)]
    run_good_ss1  = run_ss1[(select_good_trials(beh_ss1))&(run_ss1!=-1)]
    run_good_ss1[:10] = 0 # exclude first 10 trials due to imaging intensity drifting
    run_ss2       = np.array(beh_ss2['run_onset_frames'])
    run_valid_ss2 = run_ss2[(seperate_valid_trial(beh_ss2))&(run_ss2!=-1)]
    run_good_ss2  = run_ss2[(select_good_trials(beh_ss2))&(run_ss2!=-1)]
    run_good_ss2[:10] = 0 # exclude first 10 trials due to imaging intensity drifting
    
    # only include active soma rois
    dff_ss1 = np.load(data_path/f'{anm}-{date}-02_dFF.npy')[is_active_soma]   
    dff_ss2 = np.load(data_path/f'{anm}-{date}-04_dFF.npy')[is_active_soma]
    
    dff_ss1_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss1, n_sd=5)
    dff_ss2_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss2, n_sd=5)

    dff_ss1_sm = cp_gaussian_filter1d(cp.array(dff_ss1_safe), 
                                                sigma=1).get()
    dff_ss2_sm = cp_gaussian_filter1d(cp.array(dff_ss2_safe), 
                                                sigma=1).get()
    
    shuffle_params={'times': 1000,
                    'pre_event_window':  2, # seconds
                    'post_event_window': 4 
                    }
    
    # Define configurations for each stats calculation
    configs = {
        'valid_ss1': (dff_ss1_sm, run_valid_ss1),
        'valid_ss2': (dff_ss2_sm, run_valid_ss2),
        'good_ss1':  (dff_ss1_sm, run_good_ss1),
        'good_ss2':  (dff_ss2_sm, run_good_ss2),
    }

    stats = {
        name: quantify_event_response(
                corrected_traces=traces,
                event_frames=events,
                baseline_window=(-1, 0),
                response_window=(0, 1.5),
                imaging_rate=30.0,
                shuffle_test=False,
                shuffle_params=shuffle_params
            ).drop(columns=['roi_id', 'dilation_k',
                            'shuff_response_amplitude', 'shuff_effect_size', 'shuff_response_ratio'],
                            errors='ignore'
                            )
            for name, (traces, events) in configs.items()
    }

    # Combine all with suffixes and add unit_id
    stats_combined = pd.concat(
        [df.add_suffix(f'_{name}') for name, df in stats.items()],
        axis=1
    )
    stats_combined.insert(0, 'unit_id', np.where(is_active_soma)[0])
    stats_combined.to_parquet(OUTPUT_RES/f'{anm}-{date}_raw_dff_profile.parquet')
    
    dff_ss1_rsd = robust_filter_along_axis(dff_ss1, gpu=1).get() # smoothed already, sigma=1
    dff_ss2_rsd = robust_filter_along_axis(dff_ss2, gpu=1).get() # smoothed already, sigma=1
    
    dff_ss1_rsd_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss1_rsd, n_sd=5)
    dff_ss2_rsd_safe = np.apply_along_axis(trace_filter, axis=-1, arr=dff_ss2_rsd, n_sd=5)
    
    shuffle_params={'times': 1000,
                    'pre_event_window':  2, # seconds
                    'post_event_window': 4 
                    }
    
    # Define configurations for each stats calculation
    configs = {
        'valid_ss1': (dff_ss1_rsd_safe, run_valid_ss1),
        'valid_ss2': (dff_ss2_rsd_safe, run_valid_ss2),
        'good_ss1':  (dff_ss1_rsd_safe, run_good_ss1),
        'good_ss2':  (dff_ss2_rsd_safe, run_good_ss2),
    }

    stats_rsd = {
        name: quantify_event_response(
                corrected_traces=traces,
                event_frames=events,
                baseline_window=(-1, 0),
                response_window=(0, 1.5),
                imaging_rate=30.0,
                shuffle_test=False,
                shuffle_params=shuffle_params
            ).drop(columns=['roi_id', 'dilation_k',
                            'shuff_response_amplitude', 'shuff_effect_size', 'shuff_response_ratio'
                            ],
                            errors='ignore'
                            )
            for name, (traces, events) in configs.items()
    }

    # Combine all with suffixes and add unit_id
    stats_rsd_combined = pd.concat(
        [df.add_suffix(f'_{name}') for name, df in stats_rsd.items()],
        axis=1
    )
    stats_rsd_combined.insert(0, 'unit_id', np.where(is_active_soma)[0])
    stats_rsd_combined.to_parquet(OUTPUT_RES/f'{anm}-{date}_rsd_dff_profile.parquet')
    # except:
    #     print('!!!ERROR')
    #     error_lst.append(f'{anm}-{date}')




    
    
    
    