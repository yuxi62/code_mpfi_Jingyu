import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d # Import the Gaussian filter
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path 
from skimage.transform import resize
from . import Extract_dlight_masked_GECO_ROI_traces

# Assume all previous helper functions are defined, especially the one below
def _extract_trials(trace, event_frames, pre_frames, post_frames):
    T = len(trace)
    all_trials = []
    for onset in event_frames:
        start, end = onset - pre_frames, onset + post_frames
        if start >= 0 and end < T:
            all_trials.append(trace[start:end])
    return np.array(all_trials) if all_trials else None


def correct_green_trace_with_red(orig_green_trace, green_bg_trace, red_trace, roi_idx,
                                        smoothing_sigma=1.5):
    """
    Corrects a green color trace by the red channel and using it in a multi-regressor model.

    Args:
        orig_green_trace (np.ndarray): The full-length raw trace for the ROI.
        green_bg_trace (np.ndarray): The full-length green channel background trace.
        red_trace (np.ndarray): The full-length red trace.
        roi_idx (int): The suite2p ROI being analyzed.
        smoothing_sigma (float): The standard deviation of the Gaussian kernel
                                        used to smooth the estimated artifact kernel.
                                        Value is in units of frames.
    """
    T = orig_green_trace.shape[0]
        
    # --- 1. Smooth the raw kernel to get our final, clean template
    print(f"Smoothing estimated kernel 2 with Gaussian filter (sigma={smoothing_sigma} frames)...")
    if smoothing_sigma != 0:
        smooth_red_trace = gaussian_filter1d(red_trace, sigma=smoothing_sigma)
    else:
        smooth_red_trace = red_trace
    
    # --- 2. Create the Regressors ---
    # Regressor 1: Slow Z-Motion (Neuropil Trace)
    slow_z_regressor = green_bg_trace
    
    # --- 3. Build and Fit the Final Model ---
    X_regressors = np.vstack([slow_z_regressor, smooth_red_trace]).T
    y_target = orig_green_trace
    
    model = LinearRegression().fit(X_regressors, y_target)
    
    estimated_full_artifact = model.predict(X_regressors)
    final_corrected_trace = y_target - estimated_full_artifact

    # --- 4. Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), constrained_layout=True)

    # --- 5. Top Subplot (Full Traces) ---
    frames = np.arange(T)
    ax.plot(frames, orig_green_trace, color='gray', label='Original Green Trace', alpha=0.6)
    ax.plot(frames, estimated_full_artifact, color='red', label='Full Estimated Artifact', alpha=0.7)
    ax.plot(frames, final_corrected_trace, color='green', label='Final Corrected Green Trace', linewidth=1.5, alpha=0.7)
    ax.plot(frames, red_trace, color='purple', label='Red trace', linewidth=1.5, alpha=0.6)
    ax.legend()
    ax.set_title(f'Correction for Suite2p ROI {roi_idx}')
    ax.set_xlabel('Frame Number') # Add x-label for clarity
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, T-1) # Set limits to the full range

    plt.show()

    return final_corrected_trace, estimated_full_artifact, smooth_red_trace



def plot_correction_validation_suite(
    original_trace: np.ndarray,
    estimated_artifact: np.ndarray,
    final_corrected_trace: np.ndarray,
    red_trace: np.ndarray,
    event_frames: np.ndarray,
    roi_to_analyze: int,
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
        red_trace (np.ndarray): The full-length red trace.
        event_frames (np.ndarray): Array of frame onsets for trials.
        roi_to_analyze (int): The suite2p ROI being analyzed.
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
    red_trials = _extract_trials(red_trace, event_frames, pre_frames, post_frames)

    if original_trials.size == 0 or corrected_trials.size == 0:
        print(f"No valid trials to plot for ROI {roi_to_analyze}.")
        return

    # --- Part 2: Create the Visualization Suite ---
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5])
    fig.suptitle(f'Full Correction & Validation for ROI {roi_to_analyze}', fontsize=16)

    # --- Panel 1: Averaged Traces (Original vs Red) ---
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
    ax4.plot(time_axis, np.mean(artifact_trials, axis=0), color='red', label='Artifact Avg')
    ax4.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', lw=2, label='Corrected Avg')
    ax4.axvline(0, color='red', linestyle='--'); ax4.axhline(0, color='gray', linestyle=':')
    ax4.set_title('4. Trial-Averaged Traces'); ax4.set_xlabel(f'Time from {str_trial_type} (s)'); ax4.set_ylabel('F'); ax4.legend()

    plt.show()


def _correct_single_roi_worker(
    original_green_trace: np.ndarray, 
    green_neuropil_trace: np.ndarray, 
    red_trace: np.ndarray, 
    smoothing_sigma: float
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
    # --- Step 1: Prepare the Regressors ---
    
    # Smooth the red channel trace to create a cleaner regressor.
    # This assumes the red trace primarily captures the artifact.
    if smoothing_sigma > 0:
        smooth_red_regressor = gaussian_filter1d(red_trace, sigma=smoothing_sigma)
    else:
        smooth_red_regressor = red_trace

    # The green neuropil trace serves as the second regressor for slow drifts.
    neuropil_regressor = green_neuropil_trace

    # --- Step 2: Build and Fit the Linear Model ---
    
    # Create the model matrix X with the two regressors as columns.
    # The shape will be (n_frames, 2).
    X_regressors = np.vstack([neuropil_regressor, smooth_red_regressor]).T
    
    # The target variable y is the original green trace.
    y_target = original_green_trace
    
    # Fit the linear model: y_target ≈ a*neuropil + b*red_artifact
    model = LinearRegression().fit(X_regressors, y_target)
    
    # The full estimated artifact is the model's prediction based on both regressors.
    estimated_full_artifact = model.predict(X_regressors)
    
    # The final corrected trace is the residual of the model.
    final_corrected_trace = y_target - estimated_full_artifact

    # Return all the important results for this single ROI.
    return {
        'corrected': final_corrected_trace,
        'red_regressor': smooth_red_regressor
    }


def run_red_channel_correction_parallel(
    orig_green_traces: np.ndarray, 
    green_bg_traces: np.ndarray, 
    red_traces: np.ndarray,
    smoothing_sigma: float = 1.5,
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
    
    print(f"Starting parallel correction for {num_rois} ROIs...")
    
    # --- Step 2: Run Parallel Correction ---
    # We pass only the data for a single ROI to each worker. This is much more efficient.
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_correct_single_roi_worker)(
            orig_green_traces[i, :], # Pass the 1D trace for ROI i
            green_bg_traces[i, :],
            red_traces[i, :],
            smoothing_sigma
        ) for i in tqdm(range(num_rois), desc="Correcting all ROIs")
    )
    
    # --- Step 3: Unpack and Organize Results ---
    # Prepare empty arrays to store the output data.
    corrected_traces_out = np.full((num_rois, T), np.nan)
    red_regressor_traces_out = np.full((num_rois, T), np.nan)
    
    # Loop through the list of results and place each trace in the correct row.
    for i, res_dict in enumerate(results):
        if res_dict is not None:
            corrected_traces_out[i, :] = res_dict['corrected']
            red_regressor_traces_out[i, :] = res_dict['red_regressor']
                
    return corrected_traces_out, red_regressor_traces_out



def create_and_save_roi_dashboard(
    roi_idx: int,
    original_trace: np.ndarray,
    corrected_trace: np.ndarray,
    red_trace: np.ndarray,
    estimated_artifact: np.ndarray,
    event_frames: np.ndarray,
    pre_frames: int,
    post_frames: int,
    output_dir: str,
    imaging_rate: float = 30.0
):
    """
    Creates and saves a comprehensive, multi-panel diagnostic and results
    dashboard for a single ROI.

    Args:
        roi_idx (int): The index number of the ROI.
        original_trace (np.ndarray): The full-length original green trace.
        corrected_trace (np.ndarray): The full-length final corrected trace.
        red_trace (np.ndarray): The full-length red channel trace.
        estimated_artifact (np.ndarray): The full-length estimated artifact trace.
        event_frames, pre_frames, post_frames: Trial parameters.
        output_dir (str): The directory to save the output PNG file.
        imaging_rate (float): Imaging rate in Hz.
    """
    # --- Part 1: Prepare data for trial-based plots ---
    original_trials = _extract_trials(original_trace, event_frames, pre_frames, post_frames)
    corrected_trials = _extract_trials(corrected_trace, event_frames, pre_frames, post_frames)
    red_trials = _extract_trials(red_trace, event_frames, pre_frames, post_frames)
    artifact_trials = _extract_trials(estimated_artifact, event_frames, pre_frames, post_frames)
    
    if original_trials.size == 0:
        print(f"Skipping ROI {roi_idx}: No valid trials found.")
        return

    # --- Part 2: Create the Figure and GridSpec Layout ---
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1])
    fig.suptitle(f'Full Correction & Validation for ROI {roi_idx}', fontsize=16)

    # --- Panel 1: Full Time-Series Correction (top row) ---
    ax_full = fig.add_subplot(gs[0, :])
    T = len(original_trace)
    frames = np.arange(T)
    ax_full.plot(frames, original_trace, color='gray', label='Original Green Trace', alpha=0.7)
    ax_full.plot(frames, estimated_artifact, color='red', label='Full Estimated Artifact', alpha=0.7)
    ax_full.plot(frames, corrected_trace, color='green', label='Final Corrected Green Trace')
    ax_full.plot(frames, red_trace, color='purple', label='Red Trace', alpha=0.6, linewidth=0.8)
    ax_full.set_title(f'Correction for ROI {roi_idx}'); ax_full.set_xlabel('Frame Number'); ax_full.set_ylabel('Fluorescence (a.u.)')
    ax_full.legend(); ax_full.grid(True, linestyle=':')
    ax_full.set_xlim(0, T - 1)

    # --- Create common elements for trial-aligned plots ---
    time_axis = np.linspace(-pre_frames/imaging_rate, post_frames/imaging_rate, original_trials.shape[1])
    
    # --- Panels 2 & 3: Heatmaps (middle row) ---
    # Before
    ax_hm_before = fig.add_subplot(gs[1:, 1])
    ax_hm_before_1 = ax_hm_before.twinx()
    vmax_before = np.nanpercentile(np.abs(original_trials), 98)
    ax_hm_before.imshow(original_trials, cmap='coolwarm', vmin=-vmax_before, vmax=vmax_before, aspect='auto', extent=[time_axis[0], time_axis[-1], len(original_trials), 0])
    ax_hm_before_1.plot(time_axis, np.mean(original_trials, axis=0), color='black', lw=1.5, alpha=0.7)
    ax_hm_before.axvline(0, color='black', linestyle='--'); ax_hm_before.set_title('2. Before Correction');
    ax_hm_before.set_xlabel('Time from Event (s)');
    ax_hm_before.set_ylabel('Trial number')
    ax_hm_before_1.set_ylabel('F')
    
    # After
    ax_hm_after = fig.add_subplot(gs[1:, 2], sharey=ax_hm_before)
    ax_hm_after_1 = ax_hm_after.twinx()
    vmax_after = np.nanpercentile(np.abs(corrected_trials), 98)
    ax_hm_after.imshow(corrected_trials, cmap='coolwarm', vmin=-vmax_after, vmax=vmax_after, aspect='auto', extent=[time_axis[0], time_axis[-1], len(corrected_trials), 0])
    ax_hm_after_1.plot(time_axis, np.mean(corrected_trials, axis=0), color='black', lw=1.5, alpha=0.7)
    ax_hm_after.axvline(0, color='black', linestyle='--'); ax_hm_after.set_title('3. After Correction');
    ax_hm_after.set_xlabel('Time from Event (s)');
    ax_hm_after_1.set_ylabel('F')
    
    # --- Panels 4 & 5: Trial-Averaged Traces (bottom row) ---
    # Green vs Red
    ax_avg_gvr = fig.add_subplot(gs[1, 0])
    ax_avg_gvr_right = ax_avg_gvr.twinx()
    ax_avg_gvr.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax_avg_gvr_right.plot(time_axis, np.mean(red_trials, axis=0), color='red', label='Red Avg')
    ax_avg_gvr.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', label='Corrected Avg')
    ax_avg_gvr.set_title('1. Trial-Averaged Traces (green vs red)'); ax_avg_gvr.legend()

    # Original vs Corrected
    ax_avg_ovc = fig.add_subplot(gs[2, 0])
    ax_avg_ovc.plot(time_axis, np.mean(original_trials, axis=0), color='gray', label='Original Avg')
    ax_avg_ovc.plot(time_axis, np.mean(artifact_trials, axis=0), color='red', label='Artifact Avg')
    ax_avg_ovc.plot(time_axis, np.mean(corrected_trials, axis=0), color='green', label='Corrected Avg')
    ax_avg_ovc.set_title('4. Trial-Averaged Traces'); ax_avg_ovc.legend()

    for ax in [ax_avg_gvr, ax_avg_ovc]:
        ax.axvline(0, color='red', linestyle='--'); ax.axhline(0, color='gray', linestyle=':')
        ax.set_xlabel('Time from Event (s)'); ax.set_ylabel('F')

    # --- Part 3: Save the Figure ---
    output_path = Path(output_dir) / f'ROI_{roi_idx}_correction.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dashboard for ROI {roi_idx} to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close figure to free memory


def _create_dashboard_worker(
    i: int,
    original_dlight_traces: np.ndarray,
    corrected_dlight_traces: np.ndarray,
    red_traces: np.ndarray,
    soma_indices_map: np.ndarray,
    event_frames: np.ndarray,
    pre_frames: int,
    post_frames: int,
    output_dir: str
):
    """
    Worker function for parallel execution. Creates and saves the full
    diagnostic dashboard for a single ROI.

    Args:
        i (int): The relative index of the ROI in the data arrays (0 to n-1).
        soma_indices_map (np.ndarray): The array that maps the relative index `i`
                                       to the absolute Suite2p ROI index.
        (Other arguments are the data arrays and parameters)
    """
    # Extract the data for this single ROI using the relative index `i`
    original_trace = original_dlight_traces[i, :]
    corrected_trace = corrected_dlight_traces[i, :]
    red_trace0 = red_traces[i, :]
    
    # Calculate the artifact trace
    artifact_trace = original_trace - corrected_trace
    
    # Get the absolute ROI index for file naming and titles
    absolute_roi_idx = soma_indices_map[i]
    
    # Check if the data is valid before plotting
    if not np.all(np.isnan(original_trace)):
        # Call the original plotting and saving function
        # (Make sure this function is imported or defined)
        create_and_save_roi_dashboard(
            roi_idx=absolute_roi_idx,
            original_trace=original_trace,
            corrected_trace=corrected_trace,
            red_trace=red_trace0,
            estimated_artifact=artifact_trace,
            event_frames=event_frames,
            pre_frames=pre_frames,
            post_frames=post_frames,
            output_dir=output_dir
        )
    return i # Return the index to confirm completion


# --- 2. The Main Orchestrator Function ---
def run_dashboard_creation_parallel(
    original_dlight: np.ndarray,
    corrected_dlight: np.ndarray,
    red_trace: np.ndarray,
    soma_indices: np.ndarray,
    event_frames: np.ndarray,
    pre_frames: int,
    post_frames: int,
    output_dir: str,
    num_rois_to_plot=None,
    n_jobs: int = -1
):
    """
    Main orchestrator that generates and saves the diagnostic dashboard
    for all specified ROIs in parallel.

    Args:
        original_dlight (np.ndarray): 2D array (n_rois, T) of original traces.
        corrected_dlight (np.ndarray): 2D array of corrected traces.
        red_trace (np.ndarray): 2D array of red channel traces.
        soma_indices (np.ndarray): 1D array mapping relative indices to absolute Suite2p indices.
        event_frames (np.ndarray): Array of event onsets (e.g., run_onset_frames).
        pre_frames (int): Number of pre-event frames.
        post_frames (int): Number of post-event frames.
        output_dir (str): Path to the folder to save the figures.
        n_jobs (int): Number of parallel jobs for joblib.
    """
    if not num_rois_to_plot:
        num_rois_to_plot = original_dlight.shape[0]
    else:
        num_rois_to_plot = min(num_rois_to_plot, original_dlight.shape[0])
    print(f"Starting parallel dashboard generation for {num_rois_to_plot} ROIs...")
    
    # joblib.Parallel distributes the calls to the worker function across cores
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_create_dashboard_worker)(
            i,
            original_dlight,
            corrected_dlight,
            red_trace,
            soma_indices,
            event_frames,
            pre_frames,
            post_frames,
            output_dir
        ) for i in tqdm(range(num_rois_to_plot), desc="Generating dashboards")
    )
    
    print("All dashboard figures have been saved.")


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
    final_corrected_trace, estimated_artifact, red_regressor = correct_green_trace_with_red(
        green_trace_1d, 
        green_bg_trace_1d, 
        red_trace_1d, 
        absolute_roi_idx, 
        smoothing_sigma=0
    )

    # --- STEP 3 & 4: Plot validation dashboards for both events ---
    print("\n--- Step 3: Visualizing Correction Aligned to Run Onset ---")
    plot_correction_validation_suite(
        original_trace=green_trace_1d,
        estimated_artifact=estimated_artifact,
        final_corrected_trace=final_corrected_trace,
        red_trace=red_regressor, 
        event_frames=run_onset_frames,
        roi_to_analyze=absolute_roi_idx,
        pre_frames=pre_frames,
        post_frames=post_frames,
        str_trial_type='Run Onset'
    )

    
    print("\n--- Step 4: Visualizing Correction Aligned to Reward ---")
    plot_correction_validation_suite(
        original_trace=green_trace_1d,
        estimated_artifact=estimated_artifact,
        final_corrected_trace=final_corrected_trace,
        red_trace=red_regressor,
        event_frames=reward_frames, # Use reward frames here
        roi_to_analyze=absolute_roi_idx,
        pre_frames=pre_frames,
        post_frames=post_frames,
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
    event_name: str,
    pre_frames: int,
    post_frames: int,
    smoothing_sigma: float = 0.0,
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
        event_name (str): A descriptive name for the event (e.g., 'RunOnset') used in filenames.
        pre_frames, post_frames (int): Trial window parameters.
        smoothing_sigma (float): Sigma for smoothing the red trace regressor.
        n_jobs (int): Number of parallel jobs to use.
    """
    print(f"\n--- Starting Full Red-Channel Correction and Reporting Pipeline for '{event_name}' ---")
    
    # --- Setup Paths ---
    output_path = Path(output_dir)
    figure_path = output_path / 'regression_whole_red_trace'
    figure_path.mkdir(parents=True, exist_ok=True)
    
    # --- STEP 1: Perform regression for each ROI in parallel ---
    print("\nStep 1: Performing parallel regression for all ROIs...")
    corrected_dlight, red_regressor = run_red_channel_correction_parallel(
        orig_green_traces=dlight_traces,
        green_bg_traces=dlight_bg_traces,
        red_traces=red_traces, 
        smoothing_sigma=smoothing_sigma,
        n_jobs=n_jobs
    )

    # --- STEP 3: Save all regression results to a single file ---
    data_save_path = figure_path / f'dlight_regression_whole_trace_results_{event_name}.npz'
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
    
    # --- STEP 2: Generate and save dashboard figures for each ROI in parallel ---
    print("\nStep 2: Generating and saving a dashboard for each ROI...")
    run_dashboard_creation_parallel(
        original_dlight=dlight_traces,
        corrected_dlight=corrected_dlight,
        red_trace=red_regressor, # Use the smoothed red trace that was the actual regressor
        soma_indices=soma_indices,
        event_frames=event_frames,
        pre_frames=pre_frames,
        post_frames=post_frames,
        output_dir=figure_path, # Save figures in the dedicated subdirectory
        n_jobs=n_jobs
    )
    
    print(f"\n--- Pipeline Complete. Results saved in {output_path} ---")
    
    # Return the key output data for immediate use
    return corrected_dlight, red_regressor
