import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive plotting to prevent figure windows from showing
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d # Import the Gaussian filter
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path 
from skimage.transform import resize

# Assume all previous helper functions are defined, especially the one below
def _extract_trials(trace, event_frames, pre_frames, post_frames):
    T = len(trace)
    all_trials = []
    for onset in event_frames:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T:
            all_trials.append(trace[start:end])
    return np.array(all_trials) if all_trials else None


def correct_green_trace_with_estimated_red_kernel(mov, movr, global_mask_green, global_mask_red, global_neuropil_rev, block_coords, 
                                        event_frames1, event_frames2, grid_size=32, pre_frames=90, post_frames=120,
                                        imaging_rate=30.0, baseline_period_s=(-3.0, -1.0),
                                        kernel_smoothing_sigma=1.5):
    """
    Corrects a green color trace by estimating the artifact kernel directly from the trial-averaged
    data in the red channel and using it in a multi-regressor model.

    Args:
        (Familiar arguments...)
        baseline_period_s (tuple): The time window (in seconds, relative to event onset)
                                   to use for baseline subtraction of the kernel.
        kernel_smoothing_sigma (float): The standard deviation of the Gaussian kernel
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
        return None, None, None, None, None
    axon_mask_full = np.zeros_like(global_mask_green); axon_mask_full[block_slice] = local_axon_mask
    original_axon_trace = mov[:, axon_mask_full].mean(axis=1)

    local_neuropil_mask = ~global_neuropil_rev[block_slice]
    neuropil_mask_full = np.zeros_like(global_mask_green); neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_trace = mov[:, neuropil_mask_full].mean(axis=1) if np.sum(neuropil_mask_full) >= 10 else np.zeros(T)

    # --- 2. Extract the Target Axon Trace (red channel) ---
    local_red_mask = global_mask_red[block_slice]
    if np.sum(local_red_mask) < 10: 
        print("local red mask pixel number {np.sum(local_red_mask)}.")
        return None, None, None, None, None
    red_mask_full = np.zeros_like(global_mask_red); red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)

    # --- 3. Estimate the Artifact Kernel from Red Channel ---
    # Extract all trials from the red channel trace
    all_trials_red = _extract_trials(red_trace, event_frames1, pre_frames, post_frames)

    if all_trials_red is None:
        print("No valid trials for this block.")
        return None, None, None, None, None
                
    # The trial-averaged trace IS our estimated kernel
    raw_kernel = np.mean(all_trials_red, axis=0)

    # Smooth the raw kernel to get our final, clean template
    print(f"Smoothing estimated kernel with Gaussian filter (sigma={kernel_smoothing_sigma} frames)...")
    # smooth_kernel = gaussian_filter1d(raw_kernel, sigma=kernel_smoothing_sigma)
    smooth_kernel = raw_kernel
    
    # Baseline-correct the kernel using the specified time window
    baseline_start_idx = pre_frames + int(baseline_period_s[0] * imaging_rate)
    baseline_end_idx = pre_frames + int(baseline_period_s[1] * imaging_rate)
    baseline_mean = np.mean(smooth_kernel[baseline_start_idx:baseline_end_idx])
    estimated_kernel = smooth_kernel - baseline_mean

    # --- 4. Extract the Target Axon Trace (red channel), if event_frames2 exists ---
    if event_frames2.size != 0:    
        # --- Estimate the Artifact Kernel from Red Channel ---
        # Extract all trials from the red channel trace
        all_trials_red2 = _extract_trials(red_trace, event_frames2, pre_frames, post_frames)

        if all_trials_red2 is None:
            print("No valid trials for this block.")
            return None, None, None, None, None
                    
        # The trial-averaged trace IS our estimated kernel
        raw_kernel2 = np.mean(all_trials_red2, axis=0)
    
        # Smooth the raw kernel to get our final, clean template
        print(f"Smoothing estimated kernel 2 with Gaussian filter (sigma={kernel_smoothing_sigma} frames)...")
        # smooth_kernel2 = gaussian_filter1d(raw_kernel2, sigma=kernel_smoothing_sigma)
        smooth_kernel2 = raw_kernel2
        
        # Baseline-correct the kernel using the specified time window
        baseline_mean2 = np.mean(smooth_kernel2[baseline_start_idx:baseline_end_idx])
        estimated_kernel2 = smooth_kernel2 - baseline_mean2
    
    # --- 3. Create the Regressors ---
    # Regressor 1: Slow Z-Motion (Neuropil Trace)
    slow_z_regressor = neuropil_trace
    
    # Regressor 2: The Event-Locked Artifact    
    # Convolve the events with our data-driven kernel, replace convolution, which will somehow introducing unwanted delays
    event_locked_regressor = np.zeros(T)
    for onset in event_frames1:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T: event_locked_regressor[start:end] += estimated_kernel

    # Regressor 3: The Event2-Locked Artifact    
    # Convolve the events with our data-driven kernel, replace convolution, which will somehow introducing unwanted delays
    event_locked_regressor2 = np.zeros(T)
    for onset in event_frames2:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T: event_locked_regressor2[start:end] += estimated_kernel2

    # --- 4. Build and Fit the Final Model ---
    X_regressors = np.vstack([slow_z_regressor, event_locked_regressor, event_locked_regressor2]).T
    y_target = original_axon_trace
    
    model = LinearRegression().fit(X_regressors, y_target)
    
    estimated_full_artifact = model.predict(X_regressors)
    final_corrected_trace = y_target - estimated_full_artifact

    # --- 5. Visualization ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [3, 1, 1]},
                         constrained_layout=True)

    # --- Top Subplot (Full Traces) ---
    frames = np.arange(T)
    ax_top = axes[0]
    ax_top.plot(frames, original_axon_trace, color='gray', label='Original Green Trace', alpha=0.6)
    ax_top.plot(frames, estimated_full_artifact, color='red', label='Full Estimated Artifact', alpha=0.7)
    ax_top.plot(frames, final_corrected_trace, color='green', label='Final Corrected Green Trace', linewidth=1.5, alpha=0.7)
    ax_top.plot(frames, event_locked_regressor, color='purple', label='Convolved Event with Red Channel Kernel', linewidth=1.5, alpha=0.6)
    ax_top.plot(frames, event_locked_regressor2, color='cyan', label='Convolved Event2 with Red Channel Kernel', linewidth=1.5, alpha=0.6)
    ax_top.legend()
    ax_top.set_title(f'Kernel-Based Correction for Block {block_coords}')
    ax_top.set_xlabel('Frame Number') # Add x-label for clarity
    ax_top.set_ylabel('Fluorescence (a.u.)')
    ax_top.grid(True, linestyle='--')
    ax_top.set_xlim(0, T-1) # Set limits to the full range

    # --- Middle Subplot (Artifact Kernel) ---
    ax_middle = axes[1]
    window_len = len(estimated_kernel)
    time_axis = np.linspace(-pre_frames / imaging_rate, 
                            (window_len - pre_frames) / imaging_rate, 
                            window_len)
    
    ax_middle.plot(time_axis, estimated_kernel, color='black')
    ax_middle.axvline(0, color='red', linestyle='--') # Made the t=0 line red for visibility
    ax_middle.axhline(0, color='gray', linestyle=':')
    ax_middle.axvspan(baseline_period_s[0], baseline_period_s[1], color='blue', alpha=0.1, label='Baseline Window')
    ax_middle.set_title('Red-Channel-Driven Artifact Kernel (Estimated from Trial Average)')
    ax_middle.set_xlabel('Time from Event (s)')
    ax_middle.set_ylabel('dF/F')
    ax_middle.legend()
    ax_middle.grid(True, linestyle='--')
    ax_middle.set_xlim(time_axis[0], time_axis[-1]) # Set limits to the event window

    # --- Bottom Subplot (Artifact Kernel) ---
    ax_bottom = axes[2]
    window_len = len(estimated_kernel2)
    time_axis = np.linspace(-pre_frames / imaging_rate, 
                            (window_len - pre_frames) / imaging_rate, 
                            window_len)
    
    ax_bottom.plot(time_axis, estimated_kernel2, color='black')
    ax_bottom.axvline(0, color='red', linestyle='--') # Made the t=0 line red for visibility
    ax_bottom.axhline(0, color='gray', linestyle=':')
    ax_bottom.axvspan(baseline_period_s[0], baseline_period_s[1], color='blue', alpha=0.1, label='Baseline Window')
    ax_bottom.set_title('Red-Channel-Driven Artifact Kernel2 (Estimated from Trial Average)')
    ax_bottom.set_xlabel('Time from Event2 (s)')
    ax_bottom.set_ylabel('dF/F')
    ax_bottom.legend()
    ax_bottom.grid(True, linestyle='--')
    ax_bottom.set_xlim(time_axis[0], time_axis[-1]) # Set limits to the event window

    plt.close()

    return final_corrected_trace, original_axon_trace, estimated_full_artifact, estimated_kernel, estimated_kernel2



def plot_correction_validation_suite_two_kernels(
    original_trace: np.ndarray,
    estimated_artifact: np.ndarray,
    final_corrected_trace: np.ndarray,
    artifact_template: np.ndarray,
    artifact_template2: np.ndarray,
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
        estimated_artifact (np.ndarray): The full-length estimated artifact trace.
        final_corrected_trace (np.ndarray): The full-length final corrected trace.
        artifact_template (np.ndarray): The short, trial-averaged artifact kernel/template.
        artifact_template2 (np.ndarray): The short, trial-averaged artifact kernel/template, for the second event frames.
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
    artifact_trials = _extract_trials(estimated_artifact, event_frames, pre_frames, post_frames)
    corrected_trials = _extract_trials(final_corrected_trace, event_frames, pre_frames, post_frames)

    if original_trials.size == 0 or corrected_trials.size == 0:
        print(f"No valid trials to plot for block {block_coords}.")
        return

    # --- Part 2: Create the Visualization Suite ---
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Full Correction & Validation for Block {block_coords}', fontsize=16)

    # --- Panel 1: Artifact Template ---
    ax1 = fig.add_subplot(gs[0, 0])
    time_axis = np.linspace(-pre_frames/imaging_rate, post_frames/imaging_rate, len(artifact_template))
    ax1.plot(time_axis, artifact_template, color='black')
    ax1.plot(time_axis, artifact_template2, color='red')
    ax1.axvline(0, color='red', linestyle='--'); ax1.axhline(0, color='gray', linestyle=':')
    ax1.set_title('1. Derived Artifact Template'); ax1.set_xlabel(f'Time from {str_trial_type} (s)'); ax1.set_ylabel('dF/F')

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

    # --- Panel 4: Averaged Traces ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax4.plot(time_axis, np.mean(artifact_trials, axis=0), color='red', label='Artifact Avg')
    ax4.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax4.axvline(0, color='red', linestyle='--'); ax4.axhline(0, color='gray', linestyle=':')
    ax4.set_title('4. Trial-Averaged Traces'); ax4.set_xlabel(f'Time from {str_trial_type} (s)'); ax4.set_ylabel('dF/F'); ax4.legend()

    plt.close()


def _correct_green_with_red_worker(block_idx, mov, movr, global_mask_green, global_mask_red, global_neuropil_rev,
                                   event_frames1, event_frames2, grid_size, pre_frames, post_frames, 
                                   imaging_rate, baseline_period_s, kernel_smoothing_sigma):
    """
    Worker function for parallel execution. Corrects the green trace for a single block
    using a kernel derived from the red trace.
    """
    T = mov.shape[0]
    H, W = mov.shape[1], mov.shape[2]
    n_blocks_x = W // grid_size
    row, col = block_idx // n_blocks_x, block_idx % n_blocks_x
    
    block_slice = tuple(slice(r * grid_size, (r + 1) * grid_size) for r in [row, col])
    
    # Extract green channel traces
    local_green_mask = global_mask_green[block_slice]
    if np.sum(local_green_mask) < 10: return None
    green_mask_full = np.zeros_like(global_mask_green); green_mask_full[block_slice] = local_green_mask
    original_green_trace = mov[:, green_mask_full].mean(axis=1)

    local_neuropil_mask = ~global_neuropil_rev[block_slice]
    if np.sum(local_neuropil_mask) < 10: return None
    neuropil_mask_full = np.zeros_like(global_mask_green); neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_trace = mov[:, neuropil_mask_full].mean(axis=1)

    # Extract red channel trace for kernel estimation
    local_red_mask = global_mask_red[block_slice]
    if np.sum(local_red_mask) < 10: return None
    red_mask_full = np.zeros_like(global_mask_red); red_mask_full[block_slice] = local_red_mask
    red_trace = movr[:, red_mask_full].mean(axis=1)

    # Estimate artifact kernel from red channel, for event_frames1
    all_trials_red = _extract_trials(red_trace, event_frames1, pre_frames, post_frames)
    if all_trials_red is None: return None
        
    raw_kernel = np.mean(all_trials_red, axis=0)
    # smooth_kernel = gaussian_filter1d(raw_kernel, sigma=kernel_smoothing_sigma)
    smooth_kernel = raw_kernel
    
    baseline_start_idx = pre_frames + int(baseline_period_s[0] * imaging_rate)
    baseline_end_idx = pre_frames + int(baseline_period_s[1] * imaging_rate)
    baseline_mean = np.mean(smooth_kernel[baseline_start_idx:baseline_end_idx])
    estimated_kernel = smooth_kernel - baseline_mean

    # Estimate artifact kernel from red channel, for event_frames2
    if event_frames2.size != 0:    
        # --- Estimate the Artifact Kernel from Red Channel ---
        # Extract all trials from the red channel trace
        all_trials_red2 = _extract_trials(red_trace, event_frames2, pre_frames, post_frames)
        if all_trials_red is None: return None
            
        # The trial-averaged trace IS our estimated kernel
        raw_kernel2 = np.mean(all_trials_red2, axis=0)
    
        # Smooth the raw kernel to get our final, clean template
        # smooth_kernel2 = gaussian_filter1d(raw_kernel2, sigma=kernel_smoothing_sigma)
        smooth_kernel2 = raw_kernel2
        
        # Baseline-correct the kernel using the specified time window
        baseline_mean2 = np.mean(smooth_kernel2[baseline_start_idx:baseline_end_idx])
        estimated_kernel2 = smooth_kernel2 - baseline_mean2

    # Create regressors
    event_locked_regressor = np.zeros(T)
    for onset in event_frames1:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T: event_locked_regressor[start:end] += estimated_kernel

    event_locked_regressor2 = np.zeros(T)
    for onset in event_frames2:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T: event_locked_regressor2[start:end] += estimated_kernel2

    X_regressors = np.vstack([neuropil_trace, event_locked_regressor, event_locked_regressor2]).T
    
    # Fit model and correct the GREEN trace
    model = LinearRegression().fit(X_regressors, original_green_trace)
    estimated_full_artifact = model.predict(X_regressors)
    final_corrected_trace = original_green_trace - estimated_full_artifact

    return {'original': original_green_trace, 'corrected': final_corrected_trace, 'red': red_trace, 'neuropil': neuropil_trace}


# --- Main Orchestrator Function ---
def run_red_channel_correction_parallel(mov, movr, global_mask_green, global_mask_red, global_neuropil_rev,
                                        event_frames1, event_frames2, **kwargs):
    """
    Main orchestrator that runs red-channel correction in parallel for all blocks
    and generates the final summary visualization.
    """
    T, H, W = mov.shape
    grid_size = kwargs.get('grid_size', 32)
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    num_blocks = n_blocks_y * n_blocks_x
    
    n_jobs = kwargs.pop('n_jobs', -1)
    
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_correct_green_with_red_worker)(
            i, mov, movr, global_mask_green, global_mask_red, global_neuropil_rev, event_frames1, event_frames2, **kwargs
        ) for i in tqdm(range(num_blocks), desc="Correcting all blocks")
    )
    
    original_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    corrected_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    red_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    neuropil_traces_grid = np.full((n_blocks_y, n_blocks_x, T), np.nan)
    
    for i, res_dict in enumerate(results):
        if res_dict is not None:
            row, col = i // n_blocks_x, i % n_blocks_x
            original_traces_grid[row, col, :] = res_dict['original']
            corrected_traces_grid[row, col, :] = res_dict['corrected']
            red_traces_grid[row, col, :] = res_dict['red']
            neuropil_traces_grid[row, col, :] = res_dict['neuropil']
                
    return original_traces_grid, corrected_traces_grid, red_traces_grid, neuropil_traces_grid


def plot_neighborhood_correction_grid(original_traces, corrected_traces, event_frames,
                                      pre_frames, post_frames, imaging_rate,
                                      center_coords, neighborhood_size=15,
                                      save_path=None):
    """
    Creates a focused grid of dual-axis trial-averaged traces centered around a
    specific block of interest.

    Args:
        original_traces, corrected_traces: 3D data arrays (Y, X, T).
        event_frames, pre_frames, post_frames, imaging_rate: Trial parameters.
        center_coords (tuple): The (row, col) of the central block for the view.
        neighborhood_size (int): The size of the grid to plot (e.g., 5 for a 5x5 grid).
    """
    center_row, center_col = center_coords
    
    # --- 1. Create the subplot grid for the neighborhood ---
    fig, axes = plt.subplots(neighborhood_size, neighborhood_size, 
                             figsize=(int(neighborhood_size * 2.5), neighborhood_size * 2),
                             sharex=True, constrained_layout=True)
    fig.suptitle(f'Neighborhood Correction View Centered on Block {center_coords}', fontsize=16)

    # --- 2. Extract trial-averaged data ONLY for the blocks in the neighborhood ---
    half_size = neighborhood_size // 2
    
    # Calculate the data slice for the neighborhood
    row_start = max(0, center_row - half_size)
    row_end = min(original_traces.shape[0], center_row + half_size + 1)
    col_start = max(0, center_col - half_size)
    col_end = min(original_traces.shape[1], center_col + half_size + 1)
    
    # Extract the relevant subset of traces
    orig_subset = original_traces[row_start:row_end, col_start:col_end, :]
    corr_subset = corrected_traces[row_start:row_end, col_start:col_end, :]

    # Pre-calculate all trial averages for this subset
    n_y, n_x, T = orig_subset.shape
    window_len = pre_frames + post_frames
    
    subset_orig_avg = np.full((n_y, n_x, window_len), np.nan)
    subset_corr_avg = np.full((n_y, n_x, window_len), np.nan)
    
    for i in range(n_y):
        for j in range(n_x):
            original_trials = _extract_trials(orig_subset[i, j, :], event_frames, pre_frames, post_frames)
            corrected_trials = _extract_trials(corr_subset[i, j, :], event_frames, pre_frames, post_frames)
            if original_trials.size > 0: subset_orig_avg[i, j, :] = np.mean(original_trials, axis=0)
            if corrected_trials.size > 0: subset_corr_avg[i, j, :] = np.mean(corrected_trials, axis=0)

    # --- 3. Determine y-axis ranges from the SUBSET of data ---
    final_ylim_orig = (np.nanmin(subset_orig_avg) * 1.05, np.nanmax(subset_orig_avg) * 1.05)
    final_ylim_corr = (np.nanmin(subset_corr_avg) * 1.05, np.nanmax(subset_corr_avg) * 1.05)
    
    # --- 4. Create the plot elements ---
    time_axis = np.linspace(-pre_frames/imaging_rate, (window_len - pre_frames)/imaging_rate, window_len)

    for i in range(neighborhood_size):
        for j in range(neighborhood_size):
            ax_orig = axes[i, j]
            ax_corr = ax_orig.twinx()
            
            # Calculate the corresponding data block coordinates
            data_row = center_row - half_size + i
            data_col = center_col - half_size + j
            
            # Check if this data coordinate is valid
            if (0 <= data_row < original_traces.shape[0]) and (0 <= data_col < original_traces.shape[1]):
                
                # Calculate the index within our pre-calculated SUBSET
                subset_row = data_row - row_start
                subset_col = data_col - col_start
                
                original_trials_mean = subset_orig_avg[subset_row, subset_col, :]
                corrected_trials_mean = subset_corr_avg[subset_row, subset_col, :]

                if not np.all(np.isnan(original_trials_mean)):
                    ax_orig.plot(time_axis, original_trials_mean, color='gray')
                    ax_corr.plot(time_axis, corrected_trials_mean, color='green')
                    
                    ax_orig.set_ylim(final_ylim_orig)
                    ax_corr.set_ylim(final_ylim_corr)
                    ax_orig.set_title(f'({data_row},{data_col})', fontsize=8)
                else:
                    ax_orig.axis('off'); ax_corr.axis('off')
            else:
                ax_orig.axis('off'); ax_corr.axis('off')

            if j != 0: ax_orig.set_yticklabels([])
            if j != neighborhood_size - 1: ax_corr.set_yticklabels([]) 
            if i != 0: ax_orig.set_xticklabels([])

            ax_orig.tick_params(axis='y', colors='gray', labelsize=7)
            ax_corr.tick_params(axis='y', colors='green', labelsize=7)
    
    # --- 5. Highlight Center ---
    center_ax = axes[half_size, half_size]
    center_ax.set_facecolor('#fffde7')
    plt.setp(center_ax.spines.values(), color='red', linewidth=2)
   
    # Add shared axis labels
    fig.supxlabel('Time from Event Onset (s)', fontsize=12)
    fig.text(0, 0.5, 'Original dF/F (Avg)', color='gray', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(1, 0.5, 'Corrected dF/F (Avg)', color='green', ha='center', va='center', rotation='vertical', fontsize=12)
    
    if save_path:
        print(f"  > Saving figure to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
    plt.close(fig)


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


def run_regression_pipeline(
    mov: np.ndarray,
    movr: np.ndarray,
    global_mask_green: np.ndarray,
    global_mask_red: np.ndarray,
    global_neuropil_rev: np.ndarray,
    run_onset_frames: np.ndarray,
    reward_frames: np.ndarray,
    mean_img: np.ndarray,
    mask_str: str,
    output_dir: str,
    blocks_of_interest: dict,
    correction_params: dict,
    neighborhood_size: int=15,
    plot_neighborhoods: bool=False
):
    """
    Orchestrates the full pipeline: runs the red-channel correction, saves all
    resulting data, and generates a comprehensive suite of validation figures.

    Args:
        mov, movr (np.ndarray): Green and red channel movies.
        global_..._mask (np.ndarray): Various required masks.
        run_onset_frames, reward_frames (np.ndarray): Event frame arrays.
        mean_img (np.ndarray): Anatomical mean image for plot backgrounds.
        mask_str (str): A descriptive string of the mask used
        output_dir (str): Main directory to save all results and figures.
        blocks_of_interest (dict): A dictionary specifying coordinates for detailed
                                   plots, e.g., {'red_center': (2,13), 'blue_center': (14,1)}.
        correction_params (dict): A dictionary of parameters for the correction.
        neighborhood_size (int): number of neighboring grids to plot
    """
    print(f"\n--- Starting Full Correction & Validation Pipeline ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    figure_path = output_path / 'regression_with_template' / mask_str
    figure_path.mkdir(parents=True, exist_ok=True)
    
    # --- STEP 1: Run the Correction ---
    print("\nStep 1: Running parallel red-channel correction...")
    original_dlight, corrected_dlight, red_trace, neuropil_dlight = run_red_channel_correction_parallel(
        mov=mov,
        movr=movr,
        global_mask_green=global_mask_green,
        global_mask_red=global_mask_red,
        global_neuropil_rev=global_neuropil_rev,
        event_frames1=reward_frames,
        event_frames2=run_onset_frames,
        **correction_params
    )

    # --- STEP 2: Save All Processed Trace Data ---
    data_save_path = figure_path / f'all_corrected_traces_regress_with_template.npz'
    print(f"\nStep 2: Saving processed data to {data_save_path}")
    np.savez_compressed(
        data_save_path,
        original_dlight=original_dlight,
        corrected_dlight=corrected_dlight,
        red_trace=red_trace,
        neuropil_dlight=neuropil_dlight,
        global_mask_green=global_mask_green,
        global_mask_red=global_mask_red,
        global_neuropil_rev=global_neuropil_rev,
        **correction_params # Save parameters for reproducibility
    )

    # --- STEP 3: Visualize Global Response Map ---
    print("\nStep 3: Generating and saving summary response heatmap...")
    
    plot_trial_average_maps_corrected(
        traces_grid=corrected_dlight, # Simplified to one map
        event_frames=run_onset_frames,
        pre_frames=correction_params['pre_frames'], 
        post_frames=correction_params['post_frames'],
        mean_img=mean_img,         
        title='Run Onset',
        save_path=figure_path / 'corrected_response_map_run_onset.png'
    )
    
    # --- STEPS 4, 5, 6: Generate and Save Detailed Neighborhood Plots ---
    if plot_neighborhoods:
        print("\nSteps 4-6: Generating and saving detailed neighborhood plots...")
        
        pre = correction_params['pre_frames']
        post = correction_params['post_frames']
        imaging_rate = correction_params['imaging_rate']

        for name, center_coords in blocks_of_interest.items():
            print(f" > Plotting neighborhood for '{name}' block at {center_coords}...")
            
            # Original vs. Corrected
            plot_neighborhood_correction_grid(
                original_dlight, corrected_dlight, run_onset_frames, 
                pre, post, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_orig_vs_corrected_{name}.png'
            )
            
            # Neuropil vs. Corrected
            plot_neighborhood_correction_grid(
                neuropil_dlight, corrected_dlight, run_onset_frames, 
                pre, post, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_neuropil_vs_corrected_{name}.png'
            )
            
            # Red vs. Corrected
            plot_neighborhood_correction_grid(
                red_trace, corrected_dlight, run_onset_frames, 
                pre, post, imaging_rate, center_coords, neighborhood_size,
                save_path=figure_path / f'neighborhood_red_vs_corrected_{name}.png'
            )
    else:
        print("\nSteps 4-6: Skipping neighborhood plots (plot_neighborhoods=False)")

    print(f"\n--- Pipeline Complete. Results and figures saved in {output_path} ---")

    