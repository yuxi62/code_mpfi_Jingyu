# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:38:52 2026

@author: Jingyu Cao
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.utils_imaging import percentile_dff

# sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")
# sys.path.append(r"Z:\Jingyu\code_mpfi_Jingyu")
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst    
import dlight_imaging.regression.utils as utl
#%%
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = ''
regression_name ='single_trial_regression'
# DILATION_STEPS = (0, 2, 4, 6, 8, 10)
DILATION_STEPS = (0, ) # for testing

figure_path = OUT_DIR_RAW_DATA / 'TEST_PLOTS'
#%%
rec_lst = ['AC969-20250319-04',  ] # for testing
for rec in rec_lst:
    # load run-onset event frames
    p_beh_file = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
    beh = pd.read_pickle(p_beh_file)
    run_onset_frames = np.array(beh['run_onset_frames'])
    # valid_trials = (utl.seperate_valid_trial(beh))&(run_onset_frames!=-1)
    # run_onset_frames_filtered = run_onset_frames[valid_trials]
    
    for k_size in DILATION_STEPS:
        # lode regression result traces
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / r'dilation_k={}'.format(k_size))
        regress_traces = np.load(p_regression / 'single_trial_regression_res_traces.npz')
        corrected_dlight = regress_traces['corrected_dlight']
        original_dlight = regress_traces['original_dlight']
        red_trace = regress_traces['red_trace']
        neuropil_dlight = regress_traces['neuropil_dlight']
        # list of grids with signals extracted
        valid_grids = utl.get_valid_grids(corrected_dlight)
        
        #%% test dFF calculation
        # ridx =100
        for ridx in range(10):
            roi_trace_dlight = corrected_dlight[valid_grids[ridx]]
            roi_trace_red = red_trace[valid_grids[ridx]]
            plt.hist(roi_trace_dlight, bins=50, color='green', alpha=.5)
            plt.show()
            plt.hist(roi_trace_red, bins=50, color='tab:red', alpha=.5)
            plt.show()
            
            from common.plot_single_trial_function import align_frame_behaviour, plot_single_trial_html_roi
            roi_dff_dlight, roi_baseline_dlight = percentile_dff(roi_trace_dlight,
                                                                 q=10,
                                                                 return_baseline=True)
            roi_dff_red = percentile_dff(roi_trace_red)
            beh_aligned = align_frame_behaviour(beh)
            fig = plot_single_trial_html_roi(beh_aligned,
                                             roi_trace_dlight, roi_dff_dlight, roi_baseline_dlight + 0.1,
                                             labels={'ch1': 'dlight_rawF',
                                                     'ch2': 'dlight_dff',
                                                     'ch3': 'dlight_baseline'},
                                             colors={'ch1': 'grey',
                                                     'ch2': 'green',
                                                     'ch3': 'blue'},
                                             shared_yaxis=['ch1', 'ch3']
                                             )
            fig.write_html(figure_path / rf"{rec}_grid{valid_grids[ridx]}_dff_10th_no_filter_1e-8.html")
            # fig.show()
        
        #%% test plot: preliminary grid response map
        from skimage.transform import resize
        
        def _extract_trials(trace, event_frames, pre_frames, post_frames):
            T = len(trace)
            all_trials = []
            for onset in event_frames:
                start, end = onset - pre_frames, onset + post_frames
                if start >= 0 and end < T:
                    all_trials.append(trace[start:end])
            return np.array(all_trials) if all_trials else None

        def plot_trial_average_maps_corrected(
            traces_grid: np.ndarray, 
            event_frames: np.ndarray,
            pre_frames: int, 
            post_frames: int, 
            mean_img: np.ndarray, 
            title: str,
            imaging_rate: float = 30.0,
            save_path=None
        ):
            """
            Generates a single, clean heatmap of the trial-averaged response amplitude,
            calculating amplitude as (Post - Pre) / Pre.
        
            Args:
                traces_grid (np.ndarray): The 3D (Y, X, T) array of traces to analyze.
                event_frames (np.ndarray): Array of frame onsets for trials.
                pre_frames (int): Number of pre-onset frames.
                post_frames (int): Number of post-onset frames.
                mean_img (np.ndarray): The anatomical mean image for the background.
                title (str): The title for the plot.
                imaging_rate (float): Imaging rate in Hz.
                vmax (float): The absolute limit for the symmetric color scale.
            """
            if traces_grid is None:
                print(f"Cannot plot '{title}', no valid trial data.")
                return
        
            n_blocks_y, n_blocks_x, T = traces_grid.shape
            
            # --- Step 1: Pre-calculate all trial averages ---
            window_len = pre_frames + post_frames
            mean_trials = np.full((n_blocks_y, n_blocks_x, window_len), np.nan)
        
            for i in range(n_blocks_y):
                for j in range(n_blocks_x):
                    trials = _extract_trials(traces_grid[i, j, :], event_frames, pre_frames, post_frames)
                    if trials is not None:
                        if trials.size > 0:
                            mean_trials[i, j, :] = np.mean(trials, axis=0)
            
            # --- Step 2: Calculate the Amplitude Map using the specified method ---
            zero_idx = pre_frames
            response_window_start = zero_idx + int(0.3 * imaging_rate)
            response_window_end = zero_idx + int(1.0 * imaging_rate)
            baseline_window_start = zero_idx - int(1.0 * imaging_rate)
            baseline_window_end = zero_idx - int(0.3 * imaging_rate)
            
            # Calculate baseline and response periods
            baseline = np.nanmean(mean_trials[:, :, baseline_window_start:baseline_window_end], axis=-1)
            response = np.nanmean(mean_trials[:, :, response_window_start:response_window_end], axis=-1)
            
            # Calculate amplitude as (response - baseline) / baseline
            # Add a small epsilon to the denominator to avoid division by zero errors
            epsilon = 1e-9
            amplitude_map = (response - baseline) # / (baseline + epsilon)
        
            # --- Step 3: Visualize the Result ---
            H, W = mean_img.shape
            amp_map_resized = resize(amplitude_map, (H, W), order=0) # order=0 for sharp blocks
            
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(mean_img, cmap='gray',  alpha=0.6)
        
            v_abs_max = np.nanmax(np.abs(amplitude_map))
        
            # Create a symmetric color range centered at zero.
            vmin = -v_abs_max
            vmax = v_abs_max
            
            # Use the provided vmax for a symmetric color scale
            im = ax.imshow(amp_map_resized, cmap='coolwarm', alpha=0.7, vmin=vmin, vmax=vmax)
            
            ax.set_title(title)
            ax.axis('off')
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('dF/F change (Post - Pre) / Pre')
            
            if save_path:
                print(f"  > Saving figure to {save_path}")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
            plt.close()
            plt.close(fig)
            
        correction_params = {
            'grid_size': 16,
            'pre_frames': 90,
            'post_frames': 120,
            'n_jobs': -1
        }
        global_axon_mask = np.load(OUR_DIR_REGRESS / rec / 'masks' / 'dilated_global_axon_k=0.npy')
        plot_trial_average_maps_corrected(
                                        traces_grid=corrected_dlight, # Simplified to one map
                                        event_frames=run_onset_frames,
                                        pre_frames=correction_params['pre_frames'], 
                                        post_frames=correction_params['post_frames'],
                                        mean_img=global_axon_mask,         
                                        title='Run Onset',
                                        save_path=figure_path / f'{rec}_corrected_response_map_run_onset.png'
                                        )
        
        
        #%%
        from dlight_imaging.regression import Regression_Red_From_Green_BehAvg_Template
        print("\nSteps 4-6: Generating and saving detailed neighborhood plots...")
        
        blocks_of_interest = {
            'Block_A': (8,8),
            'Block_B': (23,8),
            'Block_C': (8,23),
            'Block_D': (23,23)
        }
        neighborhood_size = 15
        pre_frames=correction_params['pre_frames']
        post_frames=correction_params['post_frames']
        imaging_rate = 30
        
        for name, center_coords in blocks_of_interest.items():
            print(f" > Plotting neighborhood for '{name}' block at {center_coords}...")
            
            # Original vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                original_dlight, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'{rec}_neighborhood_orig_vs_corrected_{name}.png'
            )
            
            # Neuropil vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                neuropil_dlight, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'{rec}_neighborhood_neuropil_vs_corrected_{name}.png'
            )
            
            # Red vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                red_trace, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'{rec}_neighborhood_red_vs_corrected_{name}.png'
            )
        
       

        
