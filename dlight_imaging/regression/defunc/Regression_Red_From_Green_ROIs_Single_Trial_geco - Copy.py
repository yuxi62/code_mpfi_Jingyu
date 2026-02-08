import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d # Import the Gaussian filter
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path 
from skimage.transform import resize
import Extract_dlight_masked_GECO_ROI_traces
import Regression_Red_From_Green_ROIs_geco

# Assume all previous helper functions are defined, especially the one below
def _extract_trials(trace, event_frames, pre_frames, post_frames):
    T = len(trace)
    all_trials = []
    for onset in event_frames:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T:
            all_trials.append(trace[start:end])
    return np.array(all_trials) if all_trials else None


def correct_green_trace_with_red_single_trials(
    orig_green_trace: np.ndarray, 
    green_bg_trace: np.ndarray, 
    red_trace: np.ndarray, 
    roi_idx: int, 
    event_frames: np.ndarray,
    pre_frames: int = 90, 
    post_frames: int = 120
):
    """
    Corrects a green trace using a robust trial-by-trial regression that handles
    variable inter-trial intervals to produce a continuous corrected trace.

    Args:
    orig_green_trace (np.ndarray): The full-length raw trace for the ROI.
    green_bg_trace (np.ndarray): The full-length green channel background trace.
    red_trace (np.ndarray): The full-length red trace.
    event_frames (np.ndarray): Array of frame onsets for trials.
    roi_idx (int): The suite2p ROI being analyzed.
    """
    T = orig_green_trace.shape[0]
    
    # --- Step 1: Initialize Accumulator Arrays ---
    # We will accumulate sums and then divide by counts for a proper average.
    sum_corrected_trace = np.zeros(T)
    sum_artifact_trace = np.zeros(T)
    overlap_counts = np.zeros(T)

    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        print(f"Skipping ROI {roi_idx}: No valid trials.")
        return None, None

    # --- Step 2: Loop through trials and perform "Correct-and-Paste" ---
    print(f"\n--- Performing Variable-Window Correction for ROI {roi_idx} ---")
    
    prev_trial_end = 0
    for i, onset in enumerate(valid_event_frames):
        # Define the start of the current analysis window
        # For the first trial, it starts at frame 0.
        # For subsequent trials, it starts where the last trial's post-window ended.
        start_frame = prev_trial_end if i > 0 else 0
        
        # The end of the analysis window is the end of the current trial's post-event period
        end_frame = onset + post_frames
        
        # Ensure we don't try to create a window of zero or negative length
        if start_frame >= end_frame:
            prev_trial_end = end_frame
            continue

        # --- Extract Slices for this specific window ---
        y_target = orig_green_trace[start_frame:end_frame]
        neuropil_regressor = green_bg_trace[start_frame:end_frame]
        red_regressor = red_trace[start_frame:end_frame]

        # --- Build and Fit a Model for this window ONLY ---
        X_regressors = np.vstack([neuropil_regressor, red_regressor]).T
        
        try:
            model = LinearRegression().fit(X_regressors, y_target)
            estimated_artifact_slice = model.predict(X_regressors)
            corrected_slice = y_target - estimated_artifact_slice

            # --- "Add" the results to the full-length sum arrays ---
            sum_corrected_trace[start_frame:end_frame] += corrected_slice
            sum_artifact_trace[start_frame:end_frame] += estimated_artifact_slice
            overlap_counts[start_frame:end_frame] += 1

        except ValueError as e:
            print(f"Warning: Could not fit model for trial {i} (frames {start_frame}-{end_frame}). Skipping. Error: {e}")

        # Update the end point for the next trial's start
        prev_trial_end = end_frame

    # --- Step 3: Final Averaging and Gap Filling ---
    final_corrected_trace = np.copy(orig_green_trace) # Start with original trace
    estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps

    # Find the indices where at least one trial contributed
    valid_indices = overlap_counts > 0
    
    # Calculate the average for these valid indices
    final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
    estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]
        
    # --- Step 3: Visualization ---
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    frames = np.arange(T)
    ax_1 = ax.twinx()
    ax.plot(frames, orig_green_trace, color='gray', label='Original Green Trace', alpha=0.6)
    ax.plot(frames, estimated_full_artifact, color='purple', label='Full Estimated Artifact', alpha=0.7)
    ax.plot(frames, final_corrected_trace, color='green', label='Final Corrected Green Trace', linewidth=1.5, alpha=0.7)
    ax_1.plot(frames, red_trace, color='red', label='Final Corrected Green Trace', linewidth=1.5, alpha=0.7)
    ax.legend()
    ax.set_title(f'Variable-Window Correction for Suite2p ROI {roi_idx}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, T-1)
    plt.show()

    return final_corrected_trace, estimated_full_artifact



def plot_correction_validation_suite_single_trial(
    original_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    final_corrected_trace: np.ndarray,
    red_trace: np.ndarray,
    event_frames: np.ndarray,
    roi_idx: tuple,
    pre_frames: int = 90,
    post_frames: int = 120,
    imaging_rate: float = 30.0,
    str_trial_type='Run Onset'
):
    """
    Generates a comprehensive 4-panel "before and after" visualization suite to
    validate the artifact correction for a single block.

    Args:
        original_trace (np.ndarray): The full-length raw trace for the ROI.
        neuropil_trace (np.ndarray): The full-length neuropil trace.
        final_corrected_trace (np.ndarray): The full-length final corrected trace.
        red_trace (np.ndarray): The full-length red trace.
        event_frames (np.ndarray): Array of frame onsets for trials.
        roi_idx (tuple): The suite2p ROI being analyzed.
        pre_frames (int): Number of frames before onset.
        post_frames (int): Number of frames after onset.
        imaging_rate (float): Imaging rate in Hz.
        str_trial_type (str): Trial type.
    """
    
    # Helper function to extract trials from a full-length trace
    def _extract_trials(trace, onsets, pre, post):
        T = len(trace)
        all_trials = []
        for onset in onsets:
            start, end = onset - pre, onset + post
            if start >= 0 and end < T:
                all_trials.append(trace[start:end])
        return np.array(all_trials) if all_trials else np.array([[]])

    # --- Part 1: Extract Trials for All Traces ---
    original_trials = _extract_trials(original_trace, event_frames, pre_frames, post_frames)
    neuropil_trials = _extract_trials(neuropil_trace, event_frames, pre_frames, post_frames)
    corrected_trials = _extract_trials(final_corrected_trace, event_frames, pre_frames, post_frames)
    red_trials = _extract_trials(red_trace, event_frames, pre_frames, post_frames)

    if original_trials.size == 0 or corrected_trials.size == 0:
        print(f"No valid trials to plot for ROI {roi_idx}.")
        return

    # --- Part 2: Create the Visualization Suite ---
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Full Correction & Validation for ROI {roi_idx}', fontsize=16)

    # --- Panel 1:  Averaged Traces (Original vs Red) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_right = ax1.twinx()
    time_axis = np.linspace(-pre_frames/imaging_rate, post_frames/imaging_rate, pre_frames+post_frames)
    ax1.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax1_right.plot(time_axis, np.mean(red_trials, axis=0), color='red', label='Red Avg')
    ax1.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax1.axvline(0, color='red', linestyle='--'); ax1.axhline(0, color='gray', linestyle=':')
    ax1.set_title('1. Trial-Averaged Traces (green vs red)'); ax1.set_xlabel(f'Time from {str_trial_type} (s)'); ax1.set_ylabel('F'); ax1.legend()
    

    # Find a common, symmetric color scale for the heatmaps
    vmax = np.nanpercentile(np.abs(original_trials), 99) # Use 99th percentile to be robust to outliers
    vmin = -vmax

    # --- Panel 2: "Before" Heatmap ---
    ax2 = fig.add_subplot(gs[:, 1])
    ax2_trace = ax2.twinx()
    im2 = ax2.imshow(original_trials, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax,
                     extent=[time_axis[0], time_axis[-1], len(original_trials), 0])
    ax2_trace.plot(time_axis, np.mean(original_trials, axis=0), color='black', lw=2, alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--')
    ax2.set_title('2. Before Correction (Original Data)'); ax2.set_xlabel(f'Time from {str_trial_type} (s)'); ax2.set_ylabel('Trial #')
    ax2_trace.set_ylabel('F (Avg)')

    # --- Panel 3: "After" Heatmap ---
    ax3 = fig.add_subplot(gs[:, 2], sharey=ax2)
    ax3_trace = ax3.twinx()
    im3 = ax3.imshow(corrected_trials, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax,
                     extent=[time_axis[0], time_axis[-1], len(corrected_trials), 0])
    ax3_trace.plot(time_axis, np.mean(corrected_trials, axis=0), color='black', lw=2, alpha=0.5)
    ax3.axvline(0, color='black', linestyle='--')
    ax3.set_title('3. After Correction (Final Trace)'); ax3.set_xlabel(f'Time from {str_trial_type} (s)')
    ax3_trace.set_ylabel('F (Avg)')
    plt.setp(ax3.get_yticklabels(), visible=False) # Hide y-labels as they are shared

    # --- Panel 4: Averaged Traces (Original vs Neuropil) ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax4.plot(time_axis, np.mean(neuropil_trials, axis=0), color='red', label='Neuropil Avg')
    ax4.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax4.axvline(0, color='red', linestyle='--'); ax4.axhline(0, color='gray', linestyle=':')
    ax4.set_title('4. Trial-Averaged Traces'); ax4.set_xlabel(f'Time from {str_trial_type} (s)'); ax4.set_ylabel('F'); ax4.legend()

    plt.show()



def _correct_single_roi_worker(
    orig_green_trace: np.ndarray, 
    green_bg_trace: np.ndarray, 
    red_trace: np.ndarray,  
    event_frames,
    pre_frames=90, 
    post_frames=120,
    red_trace_mask=None,
):
    """
    Worker function for parallel execution. Corrects a SINGLE green trace using
    its corresponding red trace and green neuropil trace as regressors.
    This function contains the core scientific logic.

    Args:
        original_green_trace (np.ndarray): The raw 1D fluorescence trace for the green channel ROI.
        green_neuropil_trace (np.ndarray): The 1D neuropil trace for the green channel ROI.
        red_trace (np.ndarray): The raw 1D fluorescence trace for the corresponding red channel ROI.
        smoothing_sigma (float): The sigma for the Gaussian filter applied to the red trace.

    Returns:
        dict: A dictionary containing the original, corrected, smoothed red, and neuropil traces.
    """
    
    T = orig_green_trace.shape[0]
    
    # --- Step 1: Initialize Accumulator Arrays ---
    # We will accumulate sums and then divide by counts for a proper average.
    sum_corrected_trace = np.zeros(T)
    sum_artifact_trace = np.zeros(T)
    overlap_counts = np.zeros(T)

    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        # Return dictionaries with NaN arrays of the correct shape on failure
        return {'corrected': np.full(T, np.nan), 'artifact': np.full(T, np.nan)}

    # --- Step 2: Loop through trials and perform "Correct-and-Paste" ---    
    prev_trial_end = 0
    for i, onset in enumerate(valid_event_frames):
        # Define the start of the current analysis window
        # For the first trial, it starts at frame 0.
        # For subsequent trials, it starts where the last trial's post-window ended.
        start_frame = prev_trial_end if i > 0 else 0
        
        # The end of the analysis window is the end of the current trial's post-event period
        end_frame = onset + post_frames
        
        # Ensure we don't try to create a window of zero or negative length
        if start_frame >= end_frame:
            prev_trial_end = end_frame
            continue

        # --- Extract Slices for this specific window ---
        y_target = orig_green_trace[start_frame:end_frame]
        neuropil_regressor = green_bg_trace[start_frame:end_frame]
        red_regressor = red_trace[start_frame:end_frame]

        # --- Handle red_trace_mask if provided ---
        # When mask is True, those frames should be excluded from fitting
        if red_trace_mask is not None:
            mask_slice = red_trace_mask[start_frame:end_frame]
            valid_mask = ~mask_slice  # False in original mask = valid for fitting
            n_valid_frames = np.sum(valid_mask)
        else:
            valid_mask = None
            n_valid_frames = len(y_target)

        # --- Build and Fit a Model for this window ONLY ---
        try:
            if valid_mask is not None and n_valid_frames < 100:
                # Too few valid frames in current segment: borrow frames from neighboring segments
                # Strategy: first borrow from next segment, then from previous if still needed

                extended_start = start_frame
                extended_end = end_frame
                total_valid = n_valid_frames

                # --- Step 1: Borrow from NEXT segment ---
                if i + 1 < len(valid_event_frames):
                    next_segment_end = min(valid_event_frames[i + 1] + post_frames, T)
                else:
                    next_segment_end = T  # Last segment, extend to end of trace

                # Borrow all valid frames from next segment (up to reaching 100)
                for frame_idx in range(end_frame, next_segment_end):
                    if not red_trace_mask[frame_idx]:  # This frame is valid
                        total_valid += 1
                    extended_end = frame_idx + 1
                    if total_valid >= 100:
                        break

                # --- Step 2: If still < 100, borrow from PREVIOUS segment ---
                if total_valid < 100 and i > 0:
                    # Previous segment starts from its start_frame (could be 0 for first trial)
                    prev_segment_start = 0 if i == 1 else valid_event_frames[i - 2] + post_frames
                    prev_segment_start = max(prev_segment_start, 0)

                    # Borrow backwards from start_frame - 1 down to prev_segment_start
                    for frame_idx in range(start_frame - 1, prev_segment_start - 1, -1):
                        if not red_trace_mask[frame_idx]:  # This frame is valid
                            total_valid += 1
                        extended_start = frame_idx
                        if total_valid >= 100:
                            break

                # --- Step 3: Use whatever valid frames we have (even if < 100) ---
                # Extract extended slices for fitting only
                y_fit_extended = orig_green_trace[extended_start:extended_end]
                neuropil_extended = green_bg_trace[extended_start:extended_end]
                red_extended = red_trace[extended_start:extended_end]
                mask_extended = ~red_trace_mask[extended_start:extended_end]  # valid = True

                # Fit using valid frames from extended range
                X_fit = np.vstack([neuropil_extended[mask_extended], red_extended[mask_extended]]).T
                y_fit = y_fit_extended[mask_extended]
                model = LinearRegression().fit(X_fit, y_fit)

                # Beta 1: neuropil regressor constant; Beta 2: red channel regressor constant
                b1, b2 = model.coef_[0], model.coef_[1]

            elif valid_mask is not None:
                # Use valid frames only for fitting, but predict on all frames
                X_fit = np.vstack([neuropil_regressor[valid_mask], red_regressor[valid_mask]]).T
                y_fit = y_target[valid_mask]
                model = LinearRegression().fit(X_fit, y_fit)
                
                ## Predict on ALL frames in the segment
                # X_predict = np.vstack([neuropil_regressor, red_regressor]).T
                # estimated_artifact_slice = model.predict(X_predict)
   
                #1/31/26
                # Beta 1: neuropil regressor constant; Beta 2: red channel regressor constant
                b1, b2 = model.coef_[0], model.coef_[1]
            else:
                # No mask provided: use all frames as before
                X_regressors = np.vstack([neuropil_regressor, red_regressor]).T
                model = LinearRegression().fit(X_regressors, y_target)
                # estimated_artifact_slice = model.predict(X_regressors)
                
                #1/31/26
                # Beta 1: neuropil regressor constant; Beta 2: red channel regressor constant
                b1, b2 = model.coef_[0], model.coef_[1]


            # corrected_slice = y_target - estimated_artifact_slice
            # estimated time-varying artifact components # 1/31/26
            # Gcleaned =Graw −[β1* (R(t)−mean(R(t))) + β2 * (N(t)-mean(N(t)))] 
            est_art = b1*(neuropil_regressor - np.nanmean(neuropil_regressor)) \
                        + b2*(red_regressor - np.nanmean(red_regressor))
            corrected_slice = y_target - est_art

            # --- "Add" the results to the full-length sum arrays ---
            sum_corrected_trace[start_frame:end_frame] += corrected_slice
            sum_artifact_trace[start_frame:end_frame] += est_art
            overlap_counts[start_frame:end_frame] += 1

        except ValueError as e:
            print(f"Warning: Could not fit model for trial {i} (frames {start_frame}-{end_frame}). Skipping. Error: {e}")

        # Update the end point for the next trial's start
        prev_trial_end = end_frame

    # --- Step 3: Final Averaging and Gap Filling ---
    final_corrected_trace = np.copy(orig_green_trace) # Start with original trace
    estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps

    # Find the indices where at least one trial contributed
    valid_indices = overlap_counts > 0

    # Calculate the average for these valid indices
    final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
    estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]

    # Return all the important results for this single ROI.
    return {
        'corrected': final_corrected_trace,
        'artifact':  estimated_full_artifact
    }


def run_red_channel_correction_single_trials_parallel(
    orig_green_traces: np.ndarray,
    green_bg_traces: np.ndarray,
    red_traces: np.ndarray,
    event_frames,
    pre_frames=90,
    post_frames=120,
    red_trace_mask=None,
    n_jobs: int = -1
):
    """
    Main orchestrator that runs red-channel correction in parallel for all ROIs.
    This function prepares the data and manages the parallel execution.

    Args:
        orig_green_traces (np.ndarray): 2D array (n_rois, n_frames) of raw green traces.
        green_bg_traces (np.ndarray): 2D array (n_rois, n_frames) of green neuropil traces.
        red_traces (np.ndarray): 2D array (n_rois, n_frames) of red channel traces.
        smoothing_sigma (float): Sigma for smoothing the red trace regressor.
        red_trace_mask (np.ndarray, optional): Boolean mask (n_frames,) or (n_rois, n_frames).
            When True, those frames are excluded from regression fitting.
        n_jobs (int): Number of parallel jobs for joblib.

    Returns:
        tuple: Four 2D arrays (n_rois, n_frames) for the original green, corrected green,
               final red regressor, and final neuropil regressor traces.
    """
    # --- Step 1: Validate Inputs ---
    num_rois, T = orig_green_traces.shape

    # Ensure all input arrays have the same dimensions
    assert orig_green_traces.shape == green_bg_traces.shape == red_traces.shape, \
        "All input trace arrays must have the same shape (n_rois, n_frames)."

    # Handle mask dimensions: can be 1D (shared) or 2D (per-ROI)
    if red_trace_mask is not None:
        print('!!!! red trace mask used for regression !!!')
        if red_trace_mask.ndim == 1:
            # 1D mask: same mask for all ROIs
            mask_is_shared = True
        else:
            # 2D mask: per-ROI mask
            mask_is_shared = False
            assert red_trace_mask.shape == (num_rois, T), \
                "2D red_trace_mask must have shape (n_rois, n_frames)."
    else:
        mask_is_shared = True  # No mask, doesn't matter

    print(f"Starting parallel correction for {num_rois} ROIs...")

    # --- Step 2: Run Parallel Correction ---
    # We pass only the data for a single ROI to each worker. This is much more efficient.
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_correct_single_roi_worker)(
            orig_green_traces[i, :], # Pass the 1D trace for ROI i
            green_bg_traces[i, :],
            red_traces[i, :],
            event_frames,
            pre_frames,
            post_frames,
            red_trace_mask if mask_is_shared else red_trace_mask[i, :]
        ) for i in tqdm(range(num_rois), desc="Correcting all ROIs")
    )
    
    # --- Step 3: Unpack and Organize Results ---
    # Prepare empty arrays to store the output data.
    corrected_traces_out = np.full((num_rois, T), np.nan)
    estimated_artifact = np.full((num_rois, T), np.nan)
    
    # Loop through the list of results and place each trace in the correct row.
    for i, res_dict in enumerate(results):
        if res_dict is not None:
            corrected_traces_out[i, :] = res_dict['corrected']
            estimated_artifact[i, :] = res_dict['artifact']
                
    return corrected_traces_out, estimated_artifact



def run_single_roi_analysis_suite(
    relative_soma_index: int,
    soma_indices: np.ndarray,
    stat_array: np.ndarray,
    mean_dlight: np.ndarray,
    dlight_traces: np.ndarray,
    dlight_bg_traces: np.ndarray,
    geco_traces: np.ndarray,
    run_onset_frames: np.ndarray,
    reward_frames: np.ndarray,
    pre_frames: int,
    post_frames: int,
    global_dlight_mask: np.ndarray,
    imaging_rate: float = 30.0
):
    """
    Orchestrates a full "deep dive" analysis and visualization suite for a
    single, specified ROI.

    Args:
        relative_soma_index (int): The index of the soma in your filtered list (e.g., 0 to n-1).
        soma_indices (np.ndarray): The array that maps the relative index to the absolute Suite2p index.
        (Other arguments are the full data arrays and parameters for the analysis)
    """
    
    # --- Step 0: Setup and Index Translation ---
    # Get the absolute Suite2p index for titles and stat array lookups
    absolute_roi_idx = soma_indices[relative_soma_index]
    
    print(f"\n{'='*20} Starting Full Analysis for Soma #{relative_soma_index} (Absolute ROI #{absolute_roi_idx}) {'='*20}")

    # --- STEP 1: Plot the trial-averaged dLight response map ---
    print("\n--- Step 1: Visualizing Trial-Averaged Response Map (Run Onset) ---")
    
    # Get the 2D trial-averaged trace for this one ROI
    mean_trace_for_roi = mean_dlight[relative_soma_index : relative_soma_index + 1, :]
    
    Extract_dlight_masked_GECO_ROI_traces.plot_roi_response_map(
        mean_trace_trials=mean_trace_for_roi,
        stat_array=stat_array,
        roi_indices_to_extract=np.array([absolute_roi_idx]),
        pre_frames=pre_frames,
        mean_img=global_dlight_mask,
        imaging_rate=imaging_rate, 
        str_align='Run Onset, dLight'
    )

    # --- STEP 2: Perform the regression correction ---
    print("\n--- Step 2: Performing Red-Channel Correction ---")
    # Get the 1D full-length traces for this one ROI
    green_trace_1d = dlight_traces[relative_soma_index, :]
    green_bg_trace_1d = dlight_bg_traces[relative_soma_index, :]
    red_trace_1d = geco_traces[relative_soma_index, :]
    
    # This call assumes a function that takes 1D traces and the absolute index
    final_corrected_trace, estimated_artifact = correct_green_trace_with_red_single_trials(
        green_trace_1d, 
        green_bg_trace_1d, 
        red_trace_1d, 
        absolute_roi_idx, 
        event_frames=run_onset_frames,
        pre_frames=pre_frames, 
        post_frames=post_frames
    )

    # --- STEP 3 & 4: Plot validation dashboards for both events ---
    print("\n--- Step 3: Visualizing Correction Aligned to Run Onset ---")
    plot_correction_validation_suite_single_trial(
        original_trace=green_trace_1d,
        neuropil_trace=green_bg_trace_1d,
        final_corrected_trace=final_corrected_trace,
        red_trace=red_trace_1d, 
        event_frames=run_onset_frames,
        roi_idx=absolute_roi_idx,
        pre_frames=pre_frames,
        post_frames=post_frames,
        imaging_rate=imaging_rate,
        str_trial_type='Run Onset'
    )
    
    print("\n--- Step 4: Visualizing Correction Aligned to Reward ---")
    plot_correction_validation_suite_single_trial(
        original_trace=green_trace_1d,
        neuropil_trace=green_bg_trace_1d,
        final_corrected_trace=final_corrected_trace,
        red_trace=red_trace_1d,
        event_frames=reward_frames, # Use reward frames here
        roi_idx=absolute_roi_idx,
        pre_frames=pre_frames,
        post_frames=post_frames,
        imaging_rate=imaging_rate,
        str_trial_type='Reward'
    )
    
    print(f"\n{'='*20} Analysis for ROI #{absolute_roi_idx} Complete {'='*20}")
    
    # Return the key results
    return final_corrected_trace, estimated_artifact


def run_and_save_motion_correction_results(
    dlight_traces: np.ndarray,
    dlight_bg_traces: np.ndarray,
    red_traces: np.ndarray,
    soma_indices: np.ndarray,
    event_frames: np.ndarray,
    output_dir: str,
    regression_name: str,
    pre_frames: int,
    post_frames: int,
    smoothing_sigma: float = 0.0,
    red_trace_mask = None,
    num_rois_to_plot = 10,
    n_jobs: int = -1
):
    """
    Orchestrates the full pipeline for correcting a green channel (dLight) using a
    red channel as a regressor for all specified ROIs.

    This function performs:
    1. Parallel regression correction for each ROI.
    2. Parallel generation and saving of a detailed diagnostic dashboard for each ROI.
    3. Saving of all key input and output traces to a single .npz file.

    Args:
        dlight_traces (np.ndarray): 2D array (n_rois, T) of original green traces.
        dlight_bg_traces (np.ndarray): 2D array of green neuropil traces.
        red_traces (np.ndarray): 2D array of red channel traces.
        soma_indices (np.ndarray): 1D array mapping relative indices to absolute Suite2p indices.
        event_frames (np.ndarray): Array of event onsets for dashboard plotting.
        output_dir (str): Main directory to save all results and figures.
        regression_name (str): 
        pre_frames, post_frames (int): Trial window parameters.
        smoothing_sigma (float): Sigma for smoothing the red trace regressor.
        n_jobs (int): Number of parallel jobs to use.
    """
    print(f"\n--- Starting Full Red-Channel Correction and Reporting Pipeline for '{regression_name}' ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    figure_path = output_path / regression_name
    figure_path.mkdir(parents=True, exist_ok=True)
    
    # --- STEP 1: Perform regression for each ROI in parallel ---
    print("\nStep 1: Performing parallel regression for all ROIs...")
    corrected_dlight, red_regressor = run_red_channel_correction_single_trials_parallel(
        orig_green_traces=dlight_traces,
        green_bg_traces=dlight_bg_traces,
        red_traces=red_traces, 
        event_frames=event_frames,
        pre_frames=pre_frames, 
        post_frames=post_frames,
        red_trace_mask=red_trace_mask
    )

    # --- STEP 2: Generate and save dashboard figures for each ROI in parallel ---
    print("\nStep 2: Generating and saving a dashboard for each ROI...")
    Regression_Red_From_Green_ROIs_geco.run_dashboard_creation_parallel(
        original_dlight=dlight_traces,
        corrected_dlight=corrected_dlight,
        # red_trace=red_regressor, # Use the smoothed red trace that was the actual regressor
        red_trace=red_traces,
        soma_indices=soma_indices,
        event_frames=event_frames,
        pre_frames=pre_frames,
        post_frames=post_frames,
        output_dir=figure_path, # Save figures in the dedicated subdirectory
        n_jobs=n_jobs,
        num_rois_to_plot=num_rois_to_plot # plot first 50 ROIs
    )

    # --- STEP 3: Save all regression results to a single file ---
    data_save_path = figure_path / f'{regression_name}_res_traces.npz'
    print(f"\nStep 3: Saving regression results to: {data_save_path}")
    
    np.savez_compressed(
        data_save_path,
        original_dlight=dlight_traces,
        corrected_dlight=corrected_dlight,
        red_trace_regressor=red_regressor, # Save the regressor used
        neuropil_dlight=dlight_bg_traces,
        soma_indices=soma_indices,
        event_frames=event_frames,
        pre_frames=pre_frames,
        post_frames=post_frames,
        smoothing_sigma=smoothing_sigma
    )
    print("Save complete.")
    
    print(f"\n--- Pipeline Complete. Results saved in {output_path} ---")
    
    # Return the key output data for immediate use
    # return corrected_dlight, red_regressor
