# -*- coding: utf-8 -*-
"""
Process Calcium Traces for Drug Infusion Experiments

This module extracts and processes GCaMP calcium imaging data from concatenated
multi-session recordings. For each recording, it:
1. Loads concatenated fluorescence signals from NetCDF files
2. Segments traces by session (baseline, drug, post-drug)
3. Calculates dF/F using 10th percentile baseline
4. Applies robust SD filtering to identify active periods
5. Classifies ROIs as active somas vs background
6. Saves processed data as parquet files

Output:
    {anm}_{date}_df_gcamp_profile.parquet - DataFrame with columns:
        - anm_id, date, unit_id: Identifiers
        - map, npix: ROI spatial information
        - mean_chan2: Mean tdTomato channel intensity (for D1R identification)
        - active_soma: Boolean classification
        - baseline_ss1/ss2/ss3: Baseline fluorescence traces
        - full_dff_trace_raw_ss1/ss2/ss3: Raw dF/F traces
        - full_dff_trace_rsd_ss1/ss2/ss3: Robust SD filtered traces

Usage:
    Run as script to process all sessions meeting quality criteria:
        python process_calcium_traces.py

    Or import and use the main function:
        from process_calcium_traces import extract_session_traces
        df = extract_session_traces(rec, concat_path)

Author: Jingyu Cao
"""
#%% imports
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path Configuration
# Add paths for local imports (order matters - repo path first to avoid conflicts)
# if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
#     sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")

# Local imports (from this repo)
from common.utils_basic import nd_to_list
from common.utils_imaging import percentile_dff
# from common.utils_imaging import nanpercentile_dff as percentile_dff
from drug_infusion.robust_sd_filter import robust_filter_along_axis
from drug_infusion.rec_lst_infusion import selected_rec_lst
from drug_infusion import Select_GCaMP_ROIs
# External imports
import Generate_masks

#%% funcs
# helper function
def roi_map_to_stat_list(roi_map):
    """
    Convert ROI spatial map to list of stat dictionaries.

    Parameters
    ----------
    roi_map : ndarray
        ROI masks of shape (n_roi, height, width)

    Returns
    -------
    list of dict
        Each dict contains: npix, ypix, xpix for one ROI
    """
    roi_list = []
    for i in range(roi_map.shape[0]):
        ypix, xpix = np.where(roi_map[i] > 0)
        roi_list.append({
            'npix': len(ypix),
            'ypix': ypix,
            'xpix': xpix
        })
    return roi_list


def compute_roi_channel_means(roi_map, reference_image):
    """
    Compute mean intensity of reference image within each ROI.

    Parameters
    ----------
    roi_map : ndarray
        ROI masks of shape (n_roi, height, width)
    reference_image : ndarray
        Reference image (e.g., tdTomato channel) of shape (height, width)

    Returns
    -------
    ndarray
        Mean intensity for each ROI, shape (n_roi,)
    """
    ref_flat = reference_image.ravel()
    roi_binary = (roi_map > 0).astype(float)
    roi_flat = roi_binary.reshape(roi_map.shape[0], -1)

    roi_pixel_counts = roi_flat.sum(axis=1)
    roi_sums = roi_flat @ ref_flat

    return roi_sums / roi_pixel_counts


def get_session_frame_counts(anm_id, date, sessions):
    """
    Get frame counts for each session from Suite2P ops files.

    Parameters
    ----------
    anm_id : str
        Animal ID
    date : str
        Recording date (YYYYMMDD)
    sessions : list of str
        Session numbers (e.g., ['02', '04', '06'])

    Returns
    -------
    list of int
        Frame count for each session
    ndarray
        Reference channel 2 image from last session
    """
    nframes = []
    ref_chan2 = None

    for ss in sessions:
        ops_path = os.path.join(
            RECORDING_DIR, anm_id, f"{anm_id}-{date}", ss,
            "suite2p", "plane0", "ops.npy"
        )
        ops = np.load(ops_path, allow_pickle=True).item()
        nframes.append(ops['nframes'])
        ref_chan2 = ops['meanImg_chan2_corrected']

    return nframes, ref_chan2


# main funciton
def extract_session_traces(rec, concat_path, active_soma_only=True):
    """
    Extract and process calcium traces for a single recording.

    Parameters
    ----------
    rec : dict or Series
        Recording metadata with keys: anm, date, session, label
    concat_path : str
        Path to concatenated data directory
    active_soma_only : bool
        If True, return only ROIs classified as active somas

    Returns
    -------
    pd.DataFrame
        Processed calcium data for all ROIs
    """
    anm_id = rec['anm']
    date = rec['date']
    sessions = rec['session']

    # -------------------------------------------------------------------------
    # Load concatenated data
    # -------------------------------------------------------------------------
    sig_master = xr.open_dataarray(os.path.join(concat_path, "sig_master.nc"))
    sig_master = sig_master.rename({"master_uid": "unit_id"})

    A_master = xr.open_dataarray(os.path.join(concat_path, "A_master.nc"))
    A_master = A_master.rename({"master_uid": "unit_id"})

    F_all = np.array(sig_master).squeeze()
    roi_map = np.array(A_master).squeeze()
    n_rois = F_all.shape[0]

    # -------------------------------------------------------------------------
    # Classify ROIs as active somas
    # -------------------------------------------------------------------------
    gcamp_stats = roi_map_to_stat_list(roi_map)
    np.save(os.path.join(concat_path, 'unit_id_stat.npy'),
            np.asarray(gcamp_stats, dtype='object'))

    # Create output directory for masks
    mask_output_dir = Path(RESULTS_DIR) / f"{anm_id}-{date}"
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    # Load mean image for soma classification
    ops_path = os.path.join(
        RECORDING_DIR, anm_id, f"{anm_id}-{date}", "02",
        "suite2p", "plane0", "ops.npy"
    )
    ops = np.load(ops_path, allow_pickle=True).item()
    mean_img = ops['meanImg']

    # Generate global GCaMP mask
    global_gcamp_mask = Generate_masks.generate_and_save_dlight_mask(
        mean_img=mean_img,
        output_dir=mask_output_dir,
        output_filename_base="global_gcamp_mask",
        gaussian_sigma=1.5,
        peak_min_distance=5,
        adaptive_block_size=5
    )

    # Classify somas
    active_soma = Select_GCaMP_ROIs_Jingyu.classify_and_save_somas(
        stat_array=gcamp_stats,
        f=F_all,
        mean_img=mean_img,
        soma_mask=global_gcamp_mask,
        output_dir=mask_output_dir,
        thresholds=SOMA_THRESHOLDS,
        figure_on=False
    )

    # -------------------------------------------------------------------------
    # Get session frame counts and reference image
    # -------------------------------------------------------------------------
    nframes, ref_chan2 = get_session_frame_counts(anm_id, date, sessions)
    ch2_roi_means = compute_roi_channel_means(roi_map, ref_chan2)

    # -------------------------------------------------------------------------
    # Segment and process traces by session
    # -------------------------------------------------------------------------
    # Session 1
    F_ss1 = F_all[:, :nframes[0]]
    dff_ss1, baseline_ss1 = percentile_dff(
        F_ss1, window_size=DFF_WINDOW_SIZE, q=DFF_PERCENTILE, return_baseline=True
    )
    # dff_ss1 = trace_filter(dff_ss1)
    dff_rsd_ss1 = robust_filter_along_axis(dff_ss1, factor=2, gpu=1).get()
    
    # test for percentile dFF
    # fig, ax = plt.subplots(figsize=(10, 2), dpi=200)
    # ax.plot(F_ss1[10, 1000:2800], lw=1, color='blue', label='F raw', alpha=.5)
    # ax.plot(baseline_ss1[10, 1000:2800], lw=1, color='orange', label=f'{DFF_PERCENTILE}% baseline', 
    #         alpha=.5)
    # ax.set(ylabel='F a.u.')
    # ax.set_ylim(bottom=0)
    # ax.legend(frameon=False)
    # tax=ax.twinx()
    # tax.plot(dff_ss1[10, 1000:2800], lw=1, color='green', label='dF/F', alpha=.5)
    # tax.set(ylabel='dF/F')
    # tax.plot(dff_rsd_ss1[10, 1000:2800],lw=1, color='tab:red', label='rsd dF/F', alpha=.5)
    # tax.legend(frameon=False)
    # plt.show()
    
    # Session 2
    F_ss2 = F_all[:, nframes[0]:nframes[0]+nframes[1]]
    dff_ss2, baseline_ss2 = percentile_dff(
        F_ss2, window_size=DFF_WINDOW_SIZE, q=DFF_PERCENTILE, return_baseline=True
    )
    # dff_ss2 = trace_filter(dff_ss2)
    dff_rsd_ss2 = robust_filter_along_axis(dff_ss2, factor=2, gpu=1).get()

    # Session 3 (if exists)
    has_ss3 = len(sessions) >= 3
    if has_ss3:
        F_ss3 = F_all[:, nframes[0]+nframes[1]:]
        dff_ss3, baseline_ss3 = percentile_dff(
            F_ss3, window_size=DFF_WINDOW_SIZE, q=DFF_PERCENTILE, return_baseline=True
        )
        # dff_ss3 = trace_filter(dff_ss3)
        dff_rsd_ss3 = robust_filter_along_axis(dff_ss3, factor=2, gpu=1).get()

    # -------------------------------------------------------------------------
    # Build output DataFrame
    # -------------------------------------------------------------------------
    rows = []
    for roi in tqdm(range(n_rois), desc=f'Processing ROIs for {anm_id}-{date}'):
        row = {
            "anm_id": anm_id,
            "date": date,
            "unit_id": roi,
            "map": nd_to_list(roi_map[roi]),
            "npix": int(np.sum(roi_map[roi] > 0)),
            "mean_chan2": float(ch2_roi_means[roi]),
            "active_soma": bool(active_soma[roi]),

            "full_f_raw_trace_ss1": nd_to_list(F_ss1[roi]),
            "full_f_raw_trace_ss2": nd_to_list(F_ss2[roi]),
            "full_f_raw_trace_ss3": nd_to_list(F_ss3[roi]) if has_ss3 else [],
            
            "baseline_ss1": nd_to_list(baseline_ss1[roi]),
            "baseline_ss2": nd_to_list(baseline_ss2[roi]),
            "baseline_ss3": nd_to_list(baseline_ss3[roi]) if has_ss3 else [],
            
            "full_dff_trace_raw_ss1": nd_to_list(dff_ss1[roi]),
            "full_dff_trace_raw_ss2": nd_to_list(dff_ss2[roi]),
            "full_dff_trace_raw_ss3": nd_to_list(dff_ss3[roi]) if has_ss3 else [],

            "full_dff_trace_rsd_ss1": nd_to_list(dff_rsd_ss1[roi]),
            "full_dff_trace_rsd_ss2": nd_to_list(dff_rsd_ss2[roi]),
            "full_dff_trace_rsd_ss3": nd_to_list(dff_rsd_ss3[roi]) if has_ss3 else [],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if active_soma_only:
        df = df[df['active_soma']].reset_index(drop=True)

    return df
#%% Main
if __name__ == "__main__":
    # Constants
    RECORDING_DIR = r"Z:\Jingyu\2P_Recording"
    RESULTS_DIR = r"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\results_masks"
    OUTPUT_DIR = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\gcamp_profile"
    
    # selected subset of recordings
    rec_lst = selected_rec_lst
    print(f"Found {len(rec_lst)} sessions to process")

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    errors = []
    # Process each recording
    for rec_idx, rec in tqdm(rec_lst.iterrows(), total=len(rec_lst), desc="Processing sessions"):
    # test recording
    # for rec_idx, rec in rec_lst.loc[(rec_lst['anm']=='AC986')&(rec_lst['date']=='20250624')].iterrows():
        anm_id = rec['anm']
        date = rec['date']

        concat_path = os.path.join(RECORDING_DIR, anm_id, f"{anm_id}-{date}", "concat")
        output_path = os.path.join(OUTPUT_DIR, f"{anm_id}_{date}_df_gcamp_profile_20perc_dff.parquet")

        if os.path.exists(output_path):
            print(f"{anm_id}-{date}: Already processed, skipping")
            continue

        tqdm.write(f"\nProcessing {anm_id}-{date}")
    
        try:
            df = extract_session_traces(rec, concat_path, active_soma_only=True)
    
            df.to_parquet(
                output_path,
                engine="pyarrow",
                compression="zstd",
                compression_level=3,
                index=False
            )
    
            print(f"Saved {len(df)} ROIs to {output_path}")
    
        except Exception as e:
            print(f"ERROR processing {anm_id}-{date}: {e}")
            errors.append(f'{anm_id}-{date}')
            continue
