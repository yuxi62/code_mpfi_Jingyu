import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive plotting to prevent figure windows from showing
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d # Import the Gaussian filter
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path 
from skimage.transform import resize
from . import Regression_Red_From_Green_BehAvg_Template

# Assume all previous helper functions are defined, especially the one below
def _extract_trials(trace, event_frames, pre_frames, post_frames):
    T = len(trace)
    all_trials = []
    for onset in event_frames:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T:
            all_trials.append(trace[start:end])
    return np.array(all_trials) if all_trials else None


def correct_single_trial_with_red(mov, movr, global_mask_green, global_mask_red, global_neuropil_rev, event_frames, block_coords, grid_size=16,
                                 pre_frames=90, post_frames=120):
    """
    Corrects a green color trace by the red channel and using it in a multi-regressor model.

    Args:
        (Familiar arguments...)
        smoothing_sigma (float): The standard deviation of the Gaussian kernel
                                        used to smooth the estimated artifact kernel.
                                        Value is in units of frames.
    """
    print(f"\n--- Data-Driven Kernel Correction for Block {block_coords} ---")
    T = mov.shape[0]
    row, col = block_coords

    # --- 1. Extract the Target Axon Trace (green channel) ---
    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])
    local_axon_mask = global_mask_green[block_slice]
    if np.sum(local_axon_mask) < 10: 
        print(f"local mask pixel number {np.sum(local_axon_mask)}.")
        return None, None, None, None
    axon_mask_full = np.zeros_like(global_mask_green); axon_mask_full[block_slice] = local_axon_mask
    original_axon_trace = mov[:, axon_mask_full].mean(axis=1)

    local_neuropil_mask = ~global_neuropil_rev[block_slice]
    neuropil_mask_full = np.zeros_like(global_mask_green); neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_trace = mov[:, neuropil_mask_full].mean(axis=1) if np.sum(neuropil_mask_full) >= 10 else np.zeros(T)

    # --- 2. Extract the Target Axon Trace (red channel) ---
    local_red_mask = global_mask_red[block_slice]
    if np.sum(local_red_mask) < 10: 
        print(f"local red mask pixel number {np.sum(local_red_mask)}.")
        return None, None, None, None
    red_mask_full = np.zeros_like(global_mask_red); red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)

    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        print(f"Skipping Block {block_coords}: No valid trials.")
        return None, None, None, None

    # --- 3. Initialize Accumulator Arrays ---
    # We will accumulate sums and then divide by counts for a proper average.
    sum_corrected_trace = np.zeros(T)
    sum_artifact_trace = np.zeros(T)
    overlap_counts = np.zeros(T) 

    # --- 4: Loop through trials and perform "Correct-and-Paste" ---
    print(f"\n--- Performing Variable-Window Correction for Block {block_coords} ---")
    
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
        y_target = original_axon_trace[start_frame:end_frame]
        neuropil_regressor = neuropil_trace[start_frame:end_frame]
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

    # --- 5: Final Averaging and Gap Filling ---
    final_corrected_trace = np.copy(original_axon_trace) # Start with original trace
    estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps

    # Find the indices where at least one trial contributed
    valid_indices = overlap_counts > 0
    
    # Calculate the average for these valid indices
    final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
    estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]
    
    # --- 6. Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), constrained_layout=True)

    # --- Top Subplot (Full Traces) ---
    frames = np.arange(T)
    ax.plot(frames, original_axon_trace, color='gray', label='Original Green Trace', alpha=0.6)
    ax.plot(frames, neuropil_trace, color='yellow', label='Neuropil', alpha=0.6)
    ax.plot(frames, red_trace, color='red', label='Red trace', linewidth=1.5, alpha=0.6)
    ax.plot(frames, final_corrected_trace, color='green', label='Final Corrected Green Trace', linewidth=1.5, alpha=0.5)
    
    ax.legend()
    ax.set_title(f'Correction for Block {block_coords}')
    ax.set_xlabel('Frame Number') # Add x-label for clarity
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, T-1) # Set limits to the full range

    plt.close()

    return final_corrected_trace, original_axon_trace, neuropil_trace, red_trace


def plot_correction_validation_suite_single_trial(
    original_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    final_corrected_trace: np.ndarray,
    red_trace: np.ndarray,
    event_frames: np.ndarray,
    block_coords: tuple,
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
        block_coords (tuple): The (row, col) of the block being analyzed.
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
        print(f"No valid trials to plot for block {block_coords}.")
        return

    # --- Part 2: Create the Visualization Suite ---
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Full Correction & Validation for Block {block_coords}', fontsize=16)

    # --- Panel 1:  Averaged Traces (Original vs Red) ---
    ax1 = fig.add_subplot(gs[0, 0])
    time_axis = np.linspace(-pre_frames/imaging_rate, post_frames/imaging_rate, pre_frames+post_frames)
    ax1.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax1.plot(time_axis, np.mean(red_trials, axis=0), color='red', label='Red Avg')
    ax1.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax1.axvline(0, color='red', linestyle='--'); ax1.axhline(0, color='gray', linestyle=':')
    ax1.set_title('1. Trial-Averaged Traces (green vs red)'); ax1.set_xlabel(f'Time from {str_trial_type} (s)'); ax1.set_ylabel('dF/F'); ax1.legend()
    

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
    ax2_trace.set_ylabel('dF/F (Avg)')

    # --- Panel 3: "After" Heatmap ---
    ax3 = fig.add_subplot(gs[:, 2], sharey=ax2)
    ax3_trace = ax3.twinx()
    im3 = ax3.imshow(corrected_trials, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax,
                     extent=[time_axis[0], time_axis[-1], len(corrected_trials), 0])
    ax3_trace.plot(time_axis, np.mean(corrected_trials, axis=0), color='black', lw=2, alpha=0.5)
    ax3.axvline(0, color='black', linestyle='--')
    ax3.set_title('3. After Correction (Final Trace)'); ax3.set_xlabel(f'Time from {str_trial_type} (s)')
    ax3_trace.set_ylabel('dF/F (Avg)')
    plt.setp(ax3.get_yticklabels(), visible=False) # Hide y-labels as they are shared

    # --- Panel 4: Averaged Traces (Original vs Neuropil) ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax4.plot(time_axis, np.mean(neuropil_trials, axis=0), color='red', label='Neuropil Avg')
    ax4.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax4.axvline(0, color='red', linestyle='--'); ax4.axhline(0, color='gray', linestyle=':')
    ax4.set_title('4. Trial-Averaged Traces'); ax4.set_xlabel(f'Time from {str_trial_type} (s)'); ax4.set_ylabel('dF/F'); ax4.legend()

    plt.close()


def _correct_single_trial_worker(block_idx, mov, movr, global_mask_green, global_mask_red, global_neuropil,
                                 event_frames, grid_size, pre_frames, post_frames):
    """
    Worker function for parallel execution. Corrects the green trace for a single block
    using trial-by-trial regression against the red channel and neuropil.

    Returns:
        tuple: (traces_dict, qc_dict) where qc_dict contains grid coordinates, mask counts,
               and lists of regression metrics for all events.
    """
    T = mov.shape[0]
    H, W = mov.shape[1], mov.shape[2]
    n_blocks_x = W // grid_size
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x

    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])

    # --- 1. Extract Full-Length Traces ---
    local_green_mask = global_mask_green[block_slice]
    n_green_mask = int(np.sum(local_green_mask))
    if n_green_mask < 10: return None
    green_mask_full = np.zeros_like(global_mask_green); green_mask_full[block_slice] = local_green_mask
    original_green_trace = mov[:, green_mask_full].mean(axis=1)

    local_neuropil_mask = global_neuropil[block_slice]
    n_neuropil_mask = int(np.sum(local_neuropil_mask))
    if n_neuropil_mask < 10: return None
    neuropil_mask_full = np.zeros_like(global_mask_green); neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_trace = mov[:, neuropil_mask_full].mean(axis=1)

    local_red_mask = global_mask_red[block_slice]
    n_red_mask = int(np.sum(local_red_mask))
    if n_red_mask < 10: return None
    red_mask_full = np.zeros_like(global_mask_red); red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)

    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        return None

    # --- 3. Initialize Accumulator Arrays ---
    # We will accumulate sums and then divide by counts for a proper average.
    sum_corrected_trace = np.zeros(T)
    sum_artifact_trace = np.zeros(T)
    overlap_counts = np.zeros(T)

    # Lists to collect regression metrics across all events
    r2_list = []
    b1_list = []
    b2_list = []
    event_frame_list = []
    start_frame_list = []
    end_frame_list = []

    # --- 4: Loop through trials and perform "Correct-and-Paste" ---    
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
        y_target = original_green_trace[start_frame:end_frame]
        neuropil_regressor = neuropil_trace[start_frame:end_frame]
        red_regressor = red_trace[start_frame:end_frame]

        # --- Build and Fit a Model for this window ONLY ---
        X_regressors = np.vstack([neuropil_regressor, red_regressor]).T
        
        try:
            # Non-negative least squares
            model = LinearRegression().fit(X_regressors, y_target) 
            # estimated_artifact_slice = model.predict(X_regressors)
            # corrected_slice = y_target - estimated_artifact_slice
            
            # The Fix: Do not subtract the intercept. 
            # Subtract only the time-varying components: 
            # Gcleaned =Graw −[β1* (R(t)−mean(R(t))) + β2 * (N(t)-mean(N(t)))] 
            # Beta 1: neuropil regressor constant; Beta 2: red channel regressor constant
            b1, b2 = model.coef_[0], model.coef_[1]

            # Collect regression metrics for this event
            r2_list.append(model.score(X_regressors, y_target))
            b1_list.append(b1)
            b2_list.append(b2)
            event_frame_list.append(onset)
            start_frame_list.append(start_frame)
            end_frame_list.append(end_frame)
            
            # estimated time-varying artifact components
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

    # --- 5: Final Averaging and Gap Filling ---
    final_corrected_trace = np.copy(original_green_trace) # Start with original trace
    estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps

    # Find the indices where at least one trial contributed
    valid_indices = overlap_counts > 0
    
    # Calculate the average for these valid indices
    final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
    estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]

    # Build QC dictionary for this block
    qc_dict = {
        'grid_y': row,
        'grid_x': col,
        'n_green_mask': n_green_mask,
        'n_neuropil_mask': n_neuropil_mask,
        'n_red_mask': n_red_mask,
        'r2': r2_list,
        'b1': b1_list,
        'b2': b2_list,
        'event_frame': event_frame_list,
        'start_frame': start_frame_list,
        'end_frame': end_frame_list
    }

    return {'original': original_green_trace, 'corrected': final_corrected_trace, 'red': red_trace, 'neuropil': neuropil_trace}, qc_dict

def _correct_single_trial_BG_worker(block_idx, mov, movr, global_mask_green, global_mask_red, global_neuropil, 
                                    fov_trace, # including whole FOV dlight (BG) as an regressor ## potentially can use red_fov_trace
                                    event_frames, grid_size, pre_frames, post_frames): 
    """
    Worker function for parallel execution. Corrects the green trace for a single block
    using trial-by-trial regression against the red channel and neuropil.
    """
    T = mov.shape[0]
    H, W = mov.shape[1], mov.shape[2]
    n_blocks_x = W // grid_size
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x
    
    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])
    
    # --- 1. Extract Full-Length Traces ---
    local_green_mask = global_mask_green[block_slice]
    if np.sum(local_green_mask) < 10: return None
    green_mask_full = np.zeros_like(global_mask_green); green_mask_full[block_slice] = local_green_mask
    original_green_trace = mov[:, green_mask_full].mean(axis=1)

    local_neuropil_mask = global_neuropil[block_slice]
    if np.sum(local_neuropil_mask) < 10: return None
    neuropil_mask_full = np.zeros_like(global_mask_green); neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_trace = mov[:, neuropil_mask_full].mean(axis=1)

    local_red_mask = global_mask_red[block_slice]
    if np.sum(local_red_mask) < 10: return None
    red_mask_full = np.zeros_like(global_mask_red); red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)

    # Get a list of onsets that are valid within the movie bounds
    valid_event_frames = sorted([e for e in event_frames if (e - pre_frames >= 0 and e + post_frames < T)])
    if not valid_event_frames:
        return None, None, None, None

    # --- 3. Initialize Accumulator Arrays ---
    # We will accumulate sums and then divide by counts for a proper average.
    sum_corrected_trace = np.zeros(T)
    sum_artifact_trace = np.zeros(T)
    overlap_counts = np.zeros(T) 

    # --- 4: Loop through trials and perform "Correct-and-Paste" ---    
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
        y_target = original_green_trace[start_frame:end_frame]
        neuropil_regressor = neuropil_trace[start_frame:end_frame]
        red_regressor = red_trace[start_frame:end_frame]
        fov_regressor = fov_trace[start_frame:end_frame]
        # --- Build and Fit a Model for this window ONLY ---
        X_regressors = np.vstack([neuropil_regressor, fov_regressor, red_regressor]).T
        
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

    # --- 5: Final Averaging and Gap Filling ---
    final_corrected_trace = np.copy(original_green_trace) # Start with original trace
    estimated_full_artifact = np.zeros(T) # Artifact is zero in gaps

    # Find the indices where at least one trial contributed
    valid_indices = overlap_counts > 0
    
    # Calculate the average for these valid indices
    final_corrected_trace[valid_indices] = sum_corrected_trace[valid_indices] / overlap_counts[valid_indices]
    estimated_full_artifact[valid_indices] = sum_artifact_trace[valid_indices] / overlap_counts[valid_indices]
    
    return {'original': original_green_trace, 'corrected': final_corrected_trace, 'red': red_trace, 'neuropil': neuropil_trace}

# --- Main Orchestrator Function (Modified) ---
def run_single_trial_correction_parallel(mov, movr, global_mask_green, global_mask_red, global_neuropil,
                                         event_frames, BG_regressor, **kwargs):
    """
    Main orchestrator that runs single-trial red-channel correction in parallel.
    """
    T, H, W = mov.shape
    grid_size = kwargs.get('grid_size', 16)
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    num_blocks = n_blocks_y * n_blocks_x
    
    n_jobs = kwargs.pop('n_jobs', -1)
    
    # Extract only the parameters needed by the worker function
    worker_params = {
        'grid_size': kwargs.get('grid_size', 16),
        'pre_frames': kwargs.get('pre_frames', 90),
        'post_frames': kwargs.get('post_frames', 120)
    }
    
    if BG_regressor is None:
        all_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_correct_single_trial_worker)(
                i, mov, movr, global_mask_green, global_mask_red, global_neuropil, event_frames, **worker_params
            ) for i in tqdm(range(num_blocks), desc="Correcting all blocks (single-trial)")
        )
        # Unpack: each worker returns (res_dict, qc_dict)
        results = [r[0] if r is not None else None for r in all_results]
        all_qc_dicts = [r[1] if r is not None else None for r in all_results]
    else:
        print('*** using FOV trace for single trial regression ***')
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_correct_single_trial_BG_worker)(
                i, mov, movr, global_mask_green, global_mask_red, global_neuropil, BG_regressor, # BG_regressor: dlight FOV trace or red FOV trace
                event_frames, **worker_params
            ) for i in tqdm(range(num_blocks), desc="Correcting all blocks (single-trial)")
        )
        all_qc_dicts = None  # Not available for BG_worker

    # Unpack results into grid arrays
    original_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    corrected_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)

    for i, res_dict in enumerate(results):
        row, col = i // n_blocks_x, i % n_blocks_x
        if res_dict is not None and type(res_dict) is not tuple: # handle some weird exception
            original_traces_grid[row, col, :] = res_dict['original']
            corrected_traces_grid[row, col, :] = res_dict['corrected']
            red_traces_grid[row, col, :] = res_dict['red']
            neuropil_traces_grid[row, col, :] = res_dict['neuropil']

    # Build regression QC DataFrame (only available when BG_regressor is None)
    if all_qc_dicts is not None:
        # Filter out None entries and build DataFrame
        valid_qc_dicts = [qc for qc in all_qc_dicts if qc is not None]
        regress_qc_df = pd.DataFrame(valid_qc_dicts)
    else:
        regress_qc_df = None

    return original_traces_grid, corrected_traces_grid, red_traces_grid, neuropil_traces_grid, regress_qc_df



def run_regression_pipeline(
    mov: np.ndarray,
    movr: np.ndarray,
    global_mask_green: np.ndarray,
    global_mask_red: np.ndarray,
    global_neuropil: np.ndarray,
    run_onset_frames: np.ndarray,
    mean_img: np.ndarray,
    regression_name: str,
    output_dir: str,
    blocks_of_interest: dict={},
    correction_params: dict={},
    neighborhood_size: int=15,
    imaging_rate: int=30,
    BG_regressor=None, # BG_regressor: dlight FOV trace or red FOV trace
    plot_neighborhoods: bool=False,
):
    """
    Orchestrates the full pipeline: runs the red-channel correction, saves all
    resulting data, and generates a comprehensive suite of validation figures.

    Args:
        mov, movr (np.ndarray): Green and red channel movies.
        global_..._mask (np.ndarray): Various required masks.
        mean_img (np.ndarray): Anatomical mean image for plot backgrounds.
        regression_name (str): A descriptive string of the regression method used
        output_dir (str): Main directory to save all results and figures.
        blocks_of_interest (dict): A dictionary specifying coordinates for detailed
                                   plots, e.g., {'red_center': (2,13), 'blue_center': (14,1)}.
        correction_params (dict): A dictionary of parameters for the correction.
        neighborhood_size (int): number of neighboring grids to plot
    """
    print(f"\n--- Starting Full Correction & Validation Pipeline ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    # figure_path = output_path / regression_name
    figure_path = output_path
    figure_path.mkdir(parents=True, exist_ok=True)
    
    # --- STEP 1: Run the Correction ---
    print("\nStep 1: Running parallel red-channel correction...")

    original_dlight, corrected_dlight, red_trace, neuropil_dlight, regress_qc_df = run_single_trial_correction_parallel(
        mov=mov,
        movr=movr,
        global_mask_green=global_mask_green,
        global_mask_red=global_mask_red,
        global_neuropil=global_neuropil,
        event_frames=run_onset_frames,
        BG_regressor=BG_regressor,
        **correction_params
    )

    # --- STEP 2: Save All Processed Trace Data ---
    data_save_path = figure_path / f'{regression_name}_res_traces.npz'
    print(f"\nStep 2: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        original_dlight=original_dlight,
        corrected_dlight=corrected_dlight,
        red_trace=red_trace,
        neuropil_dlight=neuropil_dlight,
        global_mask_green=global_mask_green,
        global_mask_red=global_mask_red,
        global_neuropil=global_neuropil,
        **correction_params # Save parameters for reproducibility
    )

    # Save regression QC data as CSV (DataFrame format)
    if regress_qc_df is not None:
        qc_save_path = figure_path / f'{regression_name}_qc.pkl'
        regress_qc_df.to_pickle(qc_save_path)
        print(f"  Saved regression QC data to {qc_save_path}")

    # --- STEP 3: Visualize Global Response Map ---
    print("\nStep 3: Generating and saving summary response heatmap...")

    pre_frames = correction_params['pre_frames']
    post_frames = correction_params['post_frames']
    
    Regression_Red_From_Green_BehAvg_Template.plot_trial_average_maps_corrected(
        traces_grid=corrected_dlight, # Simplified to one map
        event_frames=run_onset_frames,
        pre_frames=pre_frames, 
        post_frames=post_frames,
        mean_img=mean_img,         
        title='Run Onset',
        save_path=figure_path / 'corrected_response_map_run_onset.png'
    )
    
    # --- STEPS 4, 5, 6: Generate and Save Detailed Neighborhood Plots ---
    if plot_neighborhoods:
        print("\nSteps 4-6: Generating and saving detailed neighborhood plots...")

        for name, center_coords in blocks_of_interest.items():
            print(f" > Plotting neighborhood for '{name}' block at {center_coords}...")
            
            # Original vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                original_dlight, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_orig_vs_corrected_{name}.png'
            )
            
            # Neuropil vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                neuropil_dlight, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_neuropil_vs_corrected_{name}.png'
            )
            
            # Red vs. Corrected
            Regression_Red_From_Green_BehAvg_Template.plot_neighborhood_correction_grid(
                red_trace, corrected_dlight, run_onset_frames, 
                pre_frames, post_frames, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_red_vs_corrected_{name}.png'
            )
    else:
        print("\nSteps 4-6: Skipping neighborhood plots (plot_neighborhoods=False)")

    print(f"\n--- Pipeline Complete. Results and figures saved in {output_path} ---")

    