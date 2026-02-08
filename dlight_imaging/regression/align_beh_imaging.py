import pickle
import numpy as np
import os
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_resample_speed(file_path: str, imaging_rate: float = 30.0, verbose: bool = True):
    """
    Loads behavioral data from a pickle file, resamples the speed trace to a
    fixed imaging rate, and aligns it to the imaging frame times.

    The function assumes the pickle file contains a dictionary with at least two keys:
    - 'speed_times_aligned': A list where each element is a NumPy array of shape (N, 2),
                             with column 0 being timestamps (ms) and column 1 being speed (cm/s).
    - 'frame_times': A list or array of timestamps (ms) for each imaging frame.

    Args:
        file_path (str): The full path to the .pkl behavioral file.
        imaging_rate (float): The imaging frame rate in Hz (e.g., 30.0).
        verbose (bool): If True, prints diagnostic information during processing.

    Returns:
        np.ndarray: A 1D NumPy array containing the speed trace, resampled and aligned
                    to the imaging frames. Returns None if an error occurs.
    """
    # --- Step 0: Load the Data and Validate ---
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check for required keys in the loaded data
    if 'speed_times_aligned' not in data or 'frame_times' not in data:
        print("Error: Pickle file must contain 'speed_times_aligned' and 'frame_times' keys.")
        return None
        
    if not data['speed_times_aligned'] or len(data['frame_times']) == 0:
        print("Error: 'speed_times_aligned' or 'frame_times' data is empty.")
        return None

    # --- Step 1: Concatenate Speed Data from All Trials ---
    all_speed_pairs = []
    for trial in data['speed_times_aligned']:
        trial = np.array(trial)
        # Ensure the trial data is a valid 2-column array
        if trial.ndim == 2 and trial.shape[1] == 2:
            all_speed_pairs.append(trial)
            
    if not all_speed_pairs:
        print("Error: No valid speed data found after filtering trials.")
        return None

    concatenated = np.vstack(all_speed_pairs)
    timestamps = concatenated[:, 0]  # in milliseconds
    speeds = concatenated[:, 1]      # in cm/s
    
    # Ensure timestamps are monotonically increasing for interpolation
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    speeds = speeds[sort_indices]

    # --- Step 2: Create New Timestamps at the Imaging Rate ---
    start_time_ms = timestamps[0]
    end_time_ms = timestamps[-1]
    interval_ms = 1000.0 / imaging_rate
    
    # Create an evenly spaced time vector for the new, resampled data
    downsampled_timestamps_ms = np.arange(start_time_ms, end_time_ms, interval_ms)

    # --- Step 3: Interpolate Speed Values at the New Timestamps ---
    # Create an interpolation function from the original data
    interp_fn = interp1d(timestamps, speeds, kind='linear', bounds_error=False, fill_value="extrapolate")
    # Calculate the speed at each new timestamp
    speed_resampled = interp_fn(downsampled_timestamps_ms)

    # --- Step 4: Align Resampled Speed Trace with Imaging Frames ---
    frame_times_ms = np.array(data['frame_times'])
    first_frame_time_ms = frame_times_ms[0]

    # Find the index in our new 30Hz timestamps that is closest to the start of imaging
    frame_start_index = np.argmin(np.abs(downsampled_timestamps_ms - first_frame_time_ms))

    # Slice the resampled speed trace so its "frame 0" aligns with the imaging "frame 0"
    speed_aligned = speed_resampled[frame_start_index:]
    
    if verbose:
        print("-" * 30)
        print(f"Behavioral data loaded from: {os.path.basename(file_path)}")
        print(f"Resampling speed to {imaging_rate} Hz.")
        print(f"First imaging frame time: {first_frame_time_ms:.2f} ms")
        print(f"Closest resampled timestamp: {downsampled_timestamps_ms[frame_start_index]:.2f} ms")
        print(f"Alignment start index: {frame_start_index}")
        print(f"Final aligned speed trace length: {len(speed_aligned)} frames")
        print("-" * 30)

    return speed_aligned



def load_and_resample_lick(file_path: str, imaging_rate: float = 30.0, verbose: bool = True):
    """
    Loads behavioral data from a pickle file, resamples the lick times to a
    fixed imaging rate, and aligns it to the imaging frame times.

    The function assumes the pickle file contains a dictionary with at least two keys:
    - 'lick_times': A list where each element is a NumPy array of shape (N, 2),
                             with column 0 being timestamps (ms) and column 1 being speed (cm/s).
    - 'frame_times': A list or array of timestamps (ms) for each imaging frame.

    Args:
        file_path (str): The full path to the .pkl behavioral file.
        imaging_rate (float): The imaging frame rate in Hz (e.g., 30.0).
        verbose (bool): If True, prints diagnostic information during processing.

    Returns:
        np.ndarray: A 1D NumPy array containing the speed trace, resampled and aligned
                    to the imaging frames. Returns None if an error occurs.
    """
    # --- Step 0: Load the Data and Validate ---
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check for required keys in the loaded data
    if 'lick_times' not in data or 'frame_times' not in data:
        print("Error: Pickle file must contain 'speed_times_aligned' and 'frame_times' keys.")
        return None
        
    if not data['lick_times'] or len(data['frame_times']) == 0:
        print("Error: 'lick_times' or 'frame_times' data is empty.")
        return None

    # --- Step 1: Concatenate Lick Data from All Trials ---
    all_lick_pairs = []
    for trial in data['lick_times']:
        trial = np.array(trial)
        # Ensure the trial data is a valid 2-column array
        if trial.ndim == 2 and trial.shape[1] == 2:
            all_lick_pairs.append(trial)
            
    if not all_lick_pairs:
        print("Error: No valid lick data found after filtering trials.")
        return None

    concatenated = np.vstack(all_lick_pairs)
    all_licks_ms = concatenated[:, 0]  # in milliseconds
    
    # Ensure timestamps are monotonically increasing for interpolation
    all_licks_ms = np.sort(all_licks_ms)

    # --- Step 2: Define Time Bins Based on Imaging Frames ---
    frame_times_ms = np.array(data['frame_times'])
    
    # The bins are defined by the start times of consecutive frames.
    # The last bin extends to a hypothetical next frame time.
    interval_ms = 1000.0 / imaging_rate
    bin_edges = np.append(frame_times_ms, frame_times_ms[-1] + interval_ms)

    # --- Step 3: Count Licks per Bin (Binned Statistic) ---
    # np.histogram gives us the count of lick timestamps that fall into each bin.
    lick_counts_per_frame, _ = np.histogram(all_licks_ms, bins=bin_edges)
    
    # The result `lick_counts_per_frame` is now a 1D array where each element `i`
    # is the number of licks between frame_times[i] and frame_times[i+1].
    # Its length is already aligned to the number of frames.

    if verbose:
        print("-" * 30)
        print(f"Lick data loaded from: {os.path.basename(file_path)}")
        print(f"Binned {len(all_licks_ms)} lick events into {len(lick_counts_per_frame)} imaging frames.")
        print(f"Total licks counted in final trace: {np.sum(lick_counts_per_frame)}")
        print(f"Final aligned lick trace length: {len(lick_counts_per_frame)} frames")
        print("-" * 30)

    return lick_counts_per_frame
    


def clean_event_frames(file_path: str, event_str: str) -> np.ndarray:
    """
    Cleans a raw array of event timestamps from behavioral data.

    This function performs two main cleaning steps:
    1. Removes any initial invalid entries (e.g., zeros or placeholders) by
       finding the first frame greater than 0.
    2. Removes any trailing invalid entries (e.g., -1 placeholders) by finding
       the last valid frame.

    Args:
        file_path (str): The full path to the .pkl behavioral file.
        event_str (str): The parameter to analyze.

    Returns:
        np.ndarray: A cleaned 1D array containing only the valid event frames.
                    Returns an empty array if no valid frames are found.
    """
    # --- Step 0: Load the Data and Validate ---
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Ensure input is a NumPy array
    event_frames = np.array(data[event_str])

    # Handle empty input array
    if event_frames.size == 0:
        return np.array([])

    # --- Step 1: Remove initial invalid entries ---
    # Create a mask to find all frames > 0
    mask_positive = event_frames > 0
    
    # If there are any positive frames, find the index of the first one
    if np.any(mask_positive):
        first_valid_idx = np.argmax(mask_positive)
        # Slice the array to start from the first valid frame
        event_frames = event_frames[first_valid_idx:]
    else:
        # If no positive frames exist at all, return an empty array
        return np.array([])

    # --- Step 2: Remove trailing invalid entries (often marked as -1) ---
    # Create a mask to find all frames that are not -1
    mask_valid = event_frames != -1
    
    # If there are any valid frames left, find the index of the last one
    if np.any(mask_valid):
        # np.where returns a tuple of arrays; we want the first one
        # np.max gets the last index where the condition is true
        last_valid_idx = np.max(np.where(mask_valid))
        # Slice the array to include up to the last valid frame
        event_frames = event_frames[:last_valid_idx + 1]
    else:
        # If no non--1 frames exist, return an empty array
        return np.array([])

    # --- Step 3: Filter out any remaining -1 values ---
    # Create a final boolean mask to select all elements that are not -1
    final_mask = event_frames != -1
    cleaned_frames = event_frames[final_mask]

    return cleaned_frames



def find_first_lick_frames(file_path: str, time_threshold: float = 800, verbose: bool = True):
    """
    For each trial, finds the first lick event that occurs after the animal has
    exceeded a distance threshold and returns the index of the closest imaging frame.

    Assumes the pickle file contains a dictionary with keys:
    - 'lick_times': A list of trial data. Each element is a NumPy array of shape (N_licks, 2)
                    with column 0 being timestamps (ms) and column 1 being some other value.
    - 'run_onsets': A list of run onset timestamps (ms) for individual trials
    - 'frame_times': A list or array of timestamps (ms) for each imaging frame.

    Args:
        file_path (str): The full path to the .pkl behavioral file.
        time_threshold (float): The time in ms the animal must exceed for a lick
                                    to be considered.
        verbose (bool): If True, prints diagnostic information.

    Returns:
        np.ndarray: A 1D array of frame indices corresponding to the found events.
                    Each element corresponds to a trial where the event was found.
                    Returns an empty array if no events are found or an error occurs.
    """
    # --- Step 0: Load Data and Validate ---
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return np.array([])
        
    required_keys = ['lick_times', 'run_onsets', 'frame_times']
    if not all(key in data for key in required_keys):
        print(f"Error: Pickle file must contain all keys: {required_keys}")
        return np.array([])

    if len(data['lick_times']) != len(data['run_onsets']):
        print("Error: Mismatch in the number of trials between 'lick_times' and 'run_onsets'.")
        trial_no = min(len(data['lick_times']),len(data['run_onsets']))
        print(f'Trial no.: {trial_no}')
        lick_times = data['lick_times'][:trial_no]
        run_onsets = data['run_onsets'][:trial_no]
        
    
    frame_times_ms = np.array(data['frame_times'])
    if frame_times_ms.size == 0:
        print("Error: 'frame_times' data is empty.")
        return np.array([])
        
    # --- Step 1: Iterate Through Trials to Find Events ---
    event_frame_indices = []
    
    # Use zip to iterate through corresponding trials of lick times and distances
    for trial_idx, (trial_licks,trial_run_onset) in enumerate(zip(lick_times, run_onsets)):
        trial_licks = np.array(trial_licks)
        trial_run_onset = np.array(trial_run_onset)
        # print(f"Run onset at {trial_run_onset: .2f} ms. ")
        
        # Ensure data for this trial is valid
        if trial_licks.ndim != 2 or trial_licks.shape[1] != 2:
            if verbose: print(f"Skipping trial {trial_idx}: lick_times is not 2 dimensional.")
            continue
        
        lick_timestamps = trial_licks[:, 0]

        # Find all licks that occur *after* the distance threshold is crossed
        mask = lick_timestamps - trial_run_onset > time_threshold
        
        
        # Check if the condition was ever met in this trial
        if np.any(mask):
            # Find the index of the *first* lick that meets the condition
            first_event_index = np.argmax(mask)
            
            # Get the timestamp of that specific lick event
            event_time_ms = lick_timestamps[first_event_index]
            
            # --- Step 2: Find the Closest Imaging Frame ---
            # Calculate the absolute difference between the event time and all frame times
            time_diffs = np.abs(frame_times_ms - event_time_ms)
            
            # The index of the minimum difference is the closest frame
            closest_frame_index = np.argmin(time_diffs)

            if(event_time_ms >= frame_times_ms[-1]):
                if verbose:
                    print(f"Trial {trial_idx}: Outside the frame range.")
                continue
            
            event_frame_indices.append(closest_frame_index)

            if verbose:
                print(f"Trial {trial_idx}: Event found at {event_time_ms:.2f} ms. Run onset at {trial_run_onset: .2f} ms. "
                      f"Closest frame is #{closest_frame_index} at {frame_times_ms[closest_frame_index]:.2f} ms.")
        else:
            if verbose:
                print(f"Trial {trial_idx}: No lick event found with time > {time_threshold} ms.")

    event_frame_indices = np.array(event_frame_indices)
    event_frame_indices = event_frame_indices[event_frame_indices > 1]

    return event_frame_indices


def process_and_save_behavioral_data(
    input_pkl_path: str,
    output_dir: str,
    num_imaging_frames: int,
    imaging_rate: float = 30.0,
    do_plots: bool = True
) -> dict:
    """
    Loads a behavioral pickle file, processes multiple event types (speed, licks,
    onsets), aligns them to imaging frames, and saves the results to an .npz file.

    Args:
        input_pkl_path (str): The full path to the source .pkl behavioral file.
        output_dir (str): The path to the directory where the output .npz file will be saved.
        num_imaging_frames (int): The total number of frames in the corresponding imaging movie.
                                  This is crucial for correctly truncating the traces.
        imaging_rate (float): The imaging rate in Hz.
        do_plots (bool): If True, generates and displays verification plots.

    Returns:
        dict: A dictionary containing all the processed and aligned behavioral arrays.
              Returns an empty dictionary if processing fails.
    """
    print(f"\n--- Processing Behavioral File: {os.path.basename(input_pkl_path)} ---")
    
    input_path = Path(input_pkl_path)
    output_path = Path(output_dir)
    
    # --- Ensure output directory exists ---
    # output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Dictionary to store all final results ---
    processed_data = {}

    # --- 1. Process Speed ---
    print("\n1. Processing speed trace...")
    speed_trace = load_and_resample_speed(input_path, imaging_rate=imaging_rate, verbose=False)
    if speed_trace is not None:
        # Truncate or pad the trace to match the exact number of imaging frames
        if len(speed_trace) > num_imaging_frames:
            speed_trace = speed_trace[:num_imaging_frames]
        elif len(speed_trace) < num_imaging_frames:
            padding = np.zeros(num_imaging_frames - len(speed_trace))
            speed_trace = np.concatenate([speed_trace, padding])
        
        processed_data['speed'] = speed_trace
        print(f" > Final speed trace shape: {speed_trace.shape}")
        
        if do_plots:
            plt.figure(figsize=(10, 3)); plt.plot(speed_trace, color='darkgray', linewidth=0.5); plt.title('Aligned Speed Trace')
            plt.xlabel('Frame Number'); plt.ylabel('Speed (cm/s)'); plt.grid(True)
            plt.savefig(str(output_dir) + '/speed_trace_aligned.png', dpi=150, bbox_inches='tight')
            plt.close()

    # --- 2. Process Licks ---
    print("\n2. Processing lick trace...")
    lick_trace = load_and_resample_lick(input_path, imaging_rate=imaging_rate, verbose=False)
    if lick_trace is not None:
        if len(lick_trace) > num_imaging_frames:
            lick_trace = lick_trace[:num_imaging_frames]
        elif len(lick_trace) < num_imaging_frames:
            padding = np.zeros(num_imaging_frames - len(lick_trace))
            lick_trace = np.concatenate([lick_trace, padding])

        processed_data['licks'] = lick_trace
        print(f" > Final lick trace shape: {lick_trace.shape}")
        if do_plots:
            plt.figure(figsize=(10, 3)); plt.plot(lick_trace, drawstyle='steps-mid', color='darkgray', linewidth=0.5); plt.title('Aligned Lick Trace')
            plt.xlabel('Frame Number'); plt.ylabel('Lick Count'); plt.grid(True)
            plt.savefig(str(output_dir) + '/lick_trace_aligned.png', dpi=150, bbox_inches='tight')
            plt.close()

    # --- 3. Process Event Frames ---
    event_keys = ['run_onset_frames', 'reward_frames', 'start_cue_frames', 'first_lick_frames']
    print("\n3. Processing event frame timestamps...")
    
    with open(input_path, 'rb') as f:
        raw_data = pickle.load(f)

    for key in event_keys:
        print(f" > Cleaning '{key}'...")
        if key == 'first_lick_frames':
            # Handle the special case of first_lick_frames
            event_frames = find_first_lick_frames(input_path, time_threshold=600.0, verbose=False)
        elif key in raw_data:
            event_frames = clean_event_frames(input_pkl_path, key)
        else:
            print(f"   - Key '{key}' not found in file. Skipping.")
            continue
        
        # Final validation: remove any frames that are outside the movie bounds
        valid_mask = (event_frames >= 0) & (event_frames < num_imaging_frames)
        processed_data[key] = event_frames[valid_mask]
        print(f"   - Found {len(processed_data[key])} valid events.")

    # --- 4. Save All Processed Data to a Single .npz File ---
    output_file_path = output_path
    print(f"\n4. Saving all processed data to: {output_file_path}")
    np.savez_compressed(output_file_path, **processed_data)
    print("Save complete.")
    
    return processed_data