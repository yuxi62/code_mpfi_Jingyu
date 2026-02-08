import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from scipy.ndimage import binary_dilation
from skimage.transform import resize
# Import the new morphological tools
from skimage.morphology import white_tophat, disk, remove_small_objects, binary_closing
from skimage.filters import median, threshold_otsu, gaussian, threshold_local
from skimage import img_as_ubyte, img_as_uint
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi




def create_global_axon_mask(mean_img, tophat_radius=5, intensity_quantile=0.97, min_size=10, clahe_clip_limit=0.008,
                            output_filename_base=None, output_dir=None):
    """
    Creates a single, clean, global mask of axons and terminals from a mean image
    using a robust morphological and intensity-based hybrid approach.

    Args:
        mean_img (np.ndarray): The 2D mean image of the entire field of view.
        tophat_radius (int): Radius for the top-hat filter to find axons.
        intensity_quantile (float): High quantile to find the brightest dots (terminals).
        min_size (int): The minimum number of pixels for a connected object to be kept.
                        Effectively removes salt-and-pepper noise.

    Returns:
        np.ndarray: A 2D boolean array representing the final clean mask.
    """
    print("--- Creating Global Axon Mask ---")
    
     # --- Step 1: Normalize and Denoise (in 16-bit) ---
    median_radius = 2
    print(f"1. Normalizing and Denoising with Median Filter (radius=median_radius)...")
    # First, ensure the input is treated as a 16-bit image.
    # We can normalize it to the full 16-bit range [0, 65535] for consistency.
    if mean_img.dtype != np.uint16:
        print(" > Converting image to uint16 for 16-bit processing.")
        # Normalize to [0, 1] float, then scale to 16-bit integer range
        img_norm_float = (mean_img - np.min(mean_img)) / (np.max(mean_img) - np.min(mean_img))
        mean_img_16bit = (img_norm_float * 65535).astype(np.uint16)
    else:
        mean_img_16bit = mean_img

    denoised_img = median(mean_img_16bit, disk(median_radius))

    # --- Step 2: Enhance Local Contrast with CLAHE ---
    # clahe_clip_limit = 0.008
    print(f"2. Enhancing contrast with CLAHE (clip limit={clahe_clip_limit})...")
    
    # CLAHE works directly and very effectively on uint16 images.
    enhanced_img = equalize_adapthist(denoised_img, clip_limit=clahe_clip_limit)
    # The output of equalize_adapthist is float64 in range [0,1], let's scale it back for consistency
    enhanced_img_scaled = (enhanced_img * 65535).astype(np.uint16)

    # --- Step 3: Segment using Hybrid Strategy ---
    print(f"3. Segmenting axons (r={tophat_radius}) and terminals (q={intensity_quantile})...")
    
    # Part A: Find axons with Top-Hat
    struct_el = disk(tophat_radius)
    tophat_img = white_tophat(enhanced_img_scaled, struct_el)
    tophat_thresh = threshold_otsu(tophat_img)
    mask_axons = tophat_img > tophat_thresh

    # Part B: Find terminals with Intensity
    dot_thresh = np.quantile(enhanced_img_scaled[enhanced_img_scaled > 0], intensity_quantile)
    mask_dots = denoised_img > dot_thresh

    # Part C: Combine into a raw mask
    raw_mask = np.logical_or(mask_axons, mask_dots)

    # --- Step 4: Post-Processing / Cleaning ---
    # Part A: Remove small, noisy objects
    cleaned_mask_1 = remove_small_objects(raw_mask, min_size=min_size, connectivity=1)
    
    # Part B: Fill small holes within axons
    # A small structuring element for closing is usually best
    final_mask = binary_closing(cleaned_mask_1, disk(1))

    # --- Visualization of the process ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(mean_img, cmap='gray')
    axes[0].set_title('Original Mean Image')
    axes[1].imshow(enhanced_img, cmap='gray')
    axes[1].set_title('Enhanced Mean Image')
    axes[2].imshow(raw_mask, cmap='gray')
    axes[2].set_title('Raw Combined Mask')
    axes[3].imshow(cleaned_mask_1, cmap='gray')
    axes[3].set_title(f'After Removing Objects < {min_size}px')
    axes[4].imshow(final_mask, cmap='gray')
    axes[4].set_title('Final Cleaned Mask')
    for ax in axes: ax.axis('off')
    plt.suptitle('Global Axon Mask Generation', fontsize=16)
    fig.tight_layout()
    if output_dir:
        plt.savefig(output_dir / f'{output_filename_base}_preview.png')
        plt.close()
    else:
        plt.show()

    return final_mask


def generate_membrane_mask_from_dLight(mean_img, 
                                      gaussian_sigma=1.5, 
                                      peak_min_distance=5,
                                      adaptive_block_size=5,
                                      output_dir=None,
                                      ): # New parameter
    """
    Generates a mask of bright membrane regions using a watershed algorithm,
    constrained by a robust ADAPTIVE global mask.
    """
    print("--- Starting Membrane Mask Generation ---")
    
    # --- Step 1: Pre-processing ---
    image_smoothed = gaussian(mean_img, sigma=gaussian_sigma)

    # --- Step 2: Create a Robust Containment Mask with Adaptive Thresholding ---
    print("2. Creating containment mask with adaptive thresholding...")
    # This method calculates a threshold for each pixel based on its local neighborhood
    # `block_size` must be an odd number. It defines the size of the neighborhood.
    local_thresh = threshold_local(image_smoothed, block_size=adaptive_block_size, method='gaussian')
    watershed_containment_mask = image_smoothed > local_thresh

    # --- Step 3: Image Inversion (same as before) ---
    print("3. Inverting image to highlight dark somas...")
    image_inverted = np.max(image_smoothed) - image_smoothed

    # --- Step 4: Find Seeds (same as before) ---
    print(f"4. Finding seeds (min_distance={peak_min_distance})...")
    local_maxi_coords = peak_local_max(image_inverted, 
                                       min_distance=peak_min_distance,
                                       labels=ndi.label(watershed_containment_mask)[0])
    markers_mask = np.zeros(image_inverted.shape, dtype=bool)
    markers_mask[tuple(local_maxi_coords.T)] = True
    markers = ndi.label(markers_mask)[0]

    # --- Step 5: Watershed Segmentation with the Adaptive Mask ---
    print("5. Running Watershed algorithm with adaptive containment mask...")
    labels = watershed(-image_inverted, markers, mask=watershed_containment_mask)

    # --- Step 6: Final Mask Generation and Inversion ---
    print("6. Creating and inverting soma mask...")
    membrane_mask = labels > 0
    final_membrane_mask = membrane_mask & watershed_containment_mask
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    stages = {
        'Original Image': mean_img,
        'Adaptive Containment Mask': watershed_containment_mask,
        'Inverted Image': image_inverted,
        'Segmented Somas': labels,
        'Final Membrane Mask': final_membrane_mask
    }
    axes[2].plot(local_maxi_coords[:, 1], local_maxi_coords[:, 0], 'r.')
    for ax, (title, img) in zip(axes, stages.items()):
        ax.imshow(img, cmap='gray' if title not in ['Seeds', 'Segmented Somas'] else 'nipy_spectral')
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle('Global dLight Mask Generation', fontsize=16)
    if output_dir:
        plt.savefig(output_dir/r'Global dLight Mask Generation.png')
        plt.close()
    else:
        plt.show()

    return final_membrane_mask


def generate_feature_mask_enhanced_tophat_dLight(
    mean_img, 
    median_radius=2, 
    clahe_clip_limit=0.015,
    tophat_radius=10,
    min_object_size=20
):
    """
    Generates a mask of bright curvilinear structures using a refined pipeline:
    Denoise -> Enhance (CLAHE) -> Feature Extract (Top-Hat) -> Threshold -> Clean.

    This method is robust to uneven illumination and enhances faint features.

    Args:
        mean_img (np.ndarray): The 16-bit 2D mean image.
        median_radius (int): Radius for the median filter.
        clahe_clip_limit (float): Contrast limit for CLAHE.
        tophat_radius (int): Radius for the top-hat filter.
        min_object_size (int): Minimum pixel size for objects to be kept.

    Returns:
        np.ndarray: The final 2D boolean mask.
    """
    print("--- Starting Enhanced Top-Hat Feature Extraction Pipeline ---")
    
    # --- Step 1: Denoise (in 16-bit) ---
    print(f"1. Denoising with Median Filter (radius={median_radius})...")
    # Ensure the input is uint16 for the median filter
    if mean_img.dtype != np.uint16:
        img_norm = (mean_img - np.min(mean_img)) / (np.max(mean_img) - np.min(mean_img))
        mean_img_16bit = (img_norm * 65535).astype(np.uint16)
    else:
        mean_img_16bit = mean_img
    image_denoised = median(mean_img_16bit, disk(median_radius))

    # --- Step 2: Enhance Local Contrast with CLAHE ---
    print(f"2. Enhancing contrast with CLAHE (clip limit={clahe_clip_limit})...")
    # CLAHE works directly on uint16 images and returns a uint16 image
    enhanced_img = equalize_adapthist(image_denoised, clip_limit=clahe_clip_limit)
    # The output of equalize_adapthist on an integer image is an integer image of the same type.
    # No need to scale back manually if the input is uint16.

    # --- Step 3: Feature Enhancement with White Top-Hat ---
    print(f"3. Enhancing features with Top-Hat filter (radius={tophat_radius})...")
    # The Top-Hat filter is now applied to the contrast-enhanced image.
    struct_el = disk(tophat_radius)
    tophat_img = white_tophat(enhanced_img, struct_el)

    # --- Step 4: Global Thresholding of the Enhanced Image ---
    print("4. Applying global Otsu threshold to Top-Hat image...")
    if np.all(tophat_img == 0):
        print("Warning: Top-hat filter produced a black image. Try adjusting parameters.")
        return np.zeros_like(mean_img, dtype=bool)
        
    thresh = threshold_otsu(tophat_img)
    raw_mask = tophat_img > thresh

    # --- Step 5: Morphological Cleaning ---
    print(f"5. Cleaning mask (removing objects < {min_object_size}px)...")
    cleaned_mask_1 = remove_small_objects(raw_mask, min_size=min_object_size)
    cleaned_mask_2 = binary_closing(cleaned_mask_1, disk(2))

    # --- Visualization ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    stages = {
        'Original Image': mean_img,
        'CLAHE Enhanced': enhanced_img,
        'Top-Hat Output': tophat_img,
        'Raw Thresholded Mask': raw_mask,
        'Final Cleaned Mask': cleaned_mask_2
    }
    
    for ax, (title, img) in zip(axes, stages.items()):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle('Global dLight Mask Generation (tophat)', fontsize=16)
    plt.show()

    return cleaned_mask_2



def create_artifact_map_from_global_mask(mov, global_axon_mask, grid_size=32, imaging_rate=30.0):
    """
    Recalculates the artifact map using a pre-computed, high-quality global mask.
    This provides a much more robust and artifact-free analysis.

    Args:
        mov (np.ndarray): The registered movie (T, H, W).
        global_axon_mask (np.ndarray): The 2D boolean global mask of all axons/terminals.
        grid_size (int): The size of the grid blocks in pixels.
        imaging_rate (float): The imaging rate in Hz for the noise filter.
    """
    T, H, W = mov.shape
    mean_img_for_display = mov.mean(axis=0)

    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    print(f"Recalculating artifact maps using global mask on a {n_blocks_y}x{n_blocks_x} grid...")

    z_motion_map = np.full((n_blocks_y, n_blocks_x), np.nan) # Use NaN for empty blocks
    noise_map = np.full((n_blocks_y, n_blocks_x), np.nan)

    b, a = butter(3, 5.0 / (imaging_rate / 2.0), btype='high') # 5Hz high-pass filter

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block_slice = (slice(i * grid_size, (i + 1) * grid_size),
                           slice(j * grid_size, (j + 1) * grid_size))
            
            # The new, robust way to define the local ROI
            local_roi_mask = global_axon_mask[block_slice]

            # If this block contains no signal from our global mask, skip it
            if np.sum(local_roi_mask) < 5:
                continue

            # Create a full-size mask for this local ROI to index the movie
            roi_mask_full = np.zeros_like(global_axon_mask)
            roi_mask_full[block_slice] = local_roi_mask

            # Perform the same diagnostics as before
            dilated_mask = binary_dilation(roi_mask_full, iterations=5)
            neuropil_mask = dilated_mask & ~roi_mask_full
            
            if np.sum(neuropil_mask) < 5: continue

            roi_trace = mov[:, roi_mask_full].mean(axis=1)
            neuropil_trace = mov[:, neuropil_mask].mean(axis=1)
            
            correlation = scipy.stats.pearsonr(roi_trace, neuropil_trace)[0]
            z_motion_map[i, j] = -correlation

            filtered_trace = filtfilt(b, a, roi_trace)
            noise_map[i, j] = np.std(filtered_trace)

    # --- Visualize the Results ---
    print("--- Visualizing Final Artifact Maps ---")
    
    z_motion_resized = resize(z_motion_map, (H, W), order=0, preserve_range=True, anti_aliasing=False)
    noise_map_resized = resize(noise_map, (H, W), order=0, preserve_range=True, anti_aliasing=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Recalculated Artifact Maps from Global Mask', fontsize=16)

    ax1.imshow(mean_img_for_display, cmap='gray')
    im1 = ax1.imshow(z_motion_resized, cmap='RdBu_r', alpha=0.6, vmin=-0.5, vmax=0.5)
    ax1.set_title('Z-Motion Signature Map')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='ROI-Neuropil Correlation\n(Negative = Z-motion)')

    ax2.imshow(mean_img_for_display, cmap='gray')
    im2 = ax2.imshow(noise_map_resized, cmap='viridis', alpha=0.6)
    ax2.set_title('High-Frequency Noise Map')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Std. Dev. of High-Pass Filtered Trace')

    plt.show()
    return z_motion_map, noise_map



def plot_block_diagnostics(mov, mov_mean_trace, ops_file_path, global_axon_mask, block_coords, grid_size=32):
    """
    Generates a detailed, 3-panel diagnostic time-series plot for a single
    manually selected block, using a global mask to define the ROI.

    Args:
        mov (np.ndarray): The registered movie (T, H, W).
        ops_file_path (str): Path to the Suite2p ops.npy file.
        global_axon_mask (np.ndarray): The 2D boolean global mask of all structures.
        block_coords (tuple): The (row, col) index of the block to analyze.
        grid_size (int): The size of the grid blocks in pixels.
    """
    T, H, W = mov.shape
    row, col = block_coords

    print(f"\n--- Generating Diagnostic Dashboard for Block (row={row}, col={col}) ---")
    
    # --- 1. Define the Local ROI from the Global Mask ---
    y_start, y_end = row * grid_size, (row + 1) * grid_size
    x_start, x_end = col * grid_size, (col + 1) * grid_size
    block_slice = (slice(y_start, y_end), slice(x_start, x_end))
    
    # The ROI is the part of the global mask inside this block
    local_roi_mask = global_axon_mask[block_slice]
    
    # Create a full-size mask for this local ROI to index the movie
    roi_mask_full = np.zeros_like(global_axon_mask)
    roi_mask_full[block_slice] = local_roi_mask

    if np.sum(roi_mask_full) < 5:
        print(f"Block ({row},{col}) contains fewer than 5 pixels from the global mask. Skipping.")
        return

    # --- 2. Perform Diagnostic Calculations ---
    ops = np.load(ops_file_path, allow_pickle=True).item()
    
    # ROI and Neuropil Traces
    dilated_mask = binary_dilation(roi_mask_full, iterations=5)
    neuropil_mask = dilated_mask & ~roi_mask_full
    
    roi_trace = mov[:, roi_mask_full].mean(axis=1)
    neuropil_trace = mov[:, neuropil_mask].mean(axis=1) if np.sum(neuropil_mask) > 0 else np.zeros(T)
    
    # Sharpness Trace (std dev of pixels within the ROI over time)
    sharpness_trace = np.array([np.std(frame[roi_mask_full]) for frame in mov])
    
    # Global Motion Trace
    rigid_disp = np.sqrt(ops['xoff']**2 + ops['yoff']**2)[:T]

    # Helper to Z-score for plotting on the same scale
    def zscore(x):
        std = np.std(x)
        return (x - np.mean(x)) / (std if std > 0 else 1.0)

    # --- 3. Create the 3-Panel Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Diagnostic Traces for Block (row={row}, col={col})', fontsize=16)
    frames = np.arange(T)

    # Panel 1: Z-Motion Diagnostic (ROI vs. Neuropil)
    axes[0].plot(frames, zscore(roi_trace), label='ROI Intensity', color='green')
    axes[0].plot(frames, zscore(neuropil_trace), label='Neuropil Intensity', color='purple', alpha=0.8)
    axes[0].set_title('ROI vs. Neuropil Intensity (Anti-correlation suggests Z-motion)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].legend()

    # Panel 2: Focus Diagnostic (ROI Sharpness)
    axes[1].plot(frames, zscore(sharpness_trace), label='ROI Sharpness', color='orange')
    axes[1].plot(frames, zscore(roi_trace), label='ROI Intensity', color='green', alpha=0.5)
    axes[1].set_title('ROI Sharpness (Dips suggest out-of-focus Z-motion)')
    axes[1].set_ylabel('Normalized Metric')
    axes[1].legend()

    # Panel 3: Movie Mean Trace
    axes[2].plot(frames, mov_mean_trace, label='Movie Mean Trace', color='blue')
    axes[2].set_title('Movie Mean Trace')
    axes[2].set_xlabel('Frame Number')
    axes[2].set_ylabel('Movie Mean Trace')
    axes[2].legend()
    
    for ax in axes: ax.grid(True, linestyle='--')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def analyze_large_region_fluctuations(mov, ops_file_path, global_axon_mask, region_coords):
    """
    Analyzes fluctuations within a large, user-defined rectangular region to
    overcome single-ROI shot noise and reveal underlying slow signals.

    Args:
        mov (np.ndarray): The registered movie (T, H, W).
        ops_file_path (str): Path to the Suite2p ops.npy file.
        global_axon_mask (np.ndarray): The 2D boolean global mask of all structures.
        region_coords (tuple): A tuple (y_start, y_end, x_start, x_end) defining the
                               large rectangular region to analyze.
    """
    y_start, y_end, x_start, x_end = region_coords
    
    # --- 1. Define the Large Local ROI ---
    # The ROI is the part of the global mask inside the specified region
    region_slice = (slice(y_start, y_end), slice(x_start, x_end))
    local_roi_mask = global_axon_mask[region_slice]

    # Create the full-size mask for indexing
    roi_mask_full = np.zeros_like(global_axon_mask)
    roi_mask_full[region_slice] = local_roi_mask
    
    num_roi_pixels = np.sum(roi_mask_full)
    if num_roi_pixels < 50: # Require a decent number of pixels for a stable trace
        print(f"Region contains only {num_roi_pixels} pixels from the global mask. Signal may be noisy.")
        if num_roi_pixels == 0:
            return

    print(f"\n--- Analyzing Large Region ({y_start}:{y_end}, {x_start}:{x_end}) with {num_roi_pixels} pixels ---")

    # --- 2. Perform Diagnostic Calculations ---
    # This part is identical to our single-block diagnostic function
    T = mov.shape[0]
    ops = np.load(ops_file_path, allow_pickle=True).item()
    
    dilated_mask = binary_dilation(roi_mask_full, iterations=5)
    neuropil_mask = dilated_mask & ~roi_mask_full
    
    roi_trace = mov[:, roi_mask_full].mean(axis=1)
    neuropil_trace = mov[:, neuropil_mask].mean(axis=1) if np.sum(neuropil_mask) > 0 else np.zeros(T)
    sharpness_trace = np.array([np.std(frame[roi_mask_full]) for frame in mov if np.sum(roi_mask_full) > 0])
    
    # Let's also bring back the Movie Mean Trace for direct comparison
    movie_mean_trace = mov.mean(axis=(1, 2))
    
    def zscore(x):
        std = np.std(x)
        return (x - np.mean(x)) / (std if std > 0 else 1.0)

    # --- 3. Create the Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Diagnostic Traces for Large Region ({y_start}:{y_end}, {x_start}:{x_end})', fontsize=16)
    frames = np.arange(T)

    # Panel 1: The key comparison
    axes[0].plot(frames, zscore(roi_trace), label='Large Region ROI Intensity', color='green', alpha=0.6)
    axes[0].plot(frames, zscore(movie_mean_trace), label='Global Movie Mean Intensity', color='blue', alpha=0.6)
    axes[0].set_title('Local Region vs. Global Mean (Hypothesis: they should now correlate)')
    axes[0].legend()

    # Panel 2: The Z-motion test
    axes[1].plot(frames, zscore(roi_trace), label='Large Region ROI Intensity', color='green', alpha=0.6)
    axes[1].plot(frames, zscore(neuropil_trace), label='Neuropil Intensity', color='purple', alpha=0.6)
    axes[1].set_title('ROI vs. Neuropil (Test for anti-correlation)')
    axes[1].legend()
    
    # Panel 3: Sharpness
    axes[2].plot(frames, zscore(sharpness_trace), label='Large Region ROI Sharpness', color='orange', alpha=0.6)
    axes[2].plot(frames, zscore(roi_trace), label='Large Region ROI Intensity', color='green', alpha=0.5)
    axes[2].set_title('Focus/Sharpness Diagnostic')
    axes[2].set_xlabel('Frame Number')
    axes[2].legend()
    
    for ax in axes: ax.grid(True, linestyle='--')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def map_axon_vs_background_z_motion(mov, global_axon_mask, grid_size=32):
    """
    Creates a side-by-side comparison of how "in-focus" axons vs. the
    "out-of-focus" background correlate with the global Z-motion signal.

    Args:
        mov (np.ndarray): The registered movie (T, H, W).
        global_axon_mask (np.ndarray): The 2D boolean global mask of all structures.
        grid_size (int): The size of the analysis grid blocks in pixels.
    """
    T, H, W = mov.shape
    mean_img_for_display = mov.mean(axis=0)

    # 1. Define the Global Z-Motion Signal Proxy
    global_z_trace = mov.mean(axis=(1, 2))

    # 2. Loop through each block to calculate correlations for both masks
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    print(f"Comparing axon vs. background Z-motion on a {n_blocks_y}x{n_blocks_x} grid...")

    axon_corr_map = np.full((n_blocks_y, n_blocks_x), np.nan)
    background_corr_map = np.full((n_blocks_y, n_blocks_x), np.nan)

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block_slice = (slice(i * grid_size, (i + 1) * grid_size),
                           slice(j * grid_size, (j + 1) * grid_size))
            
            # --- Define Axon ROI for the block ---
            local_axon_mask = global_axon_mask[block_slice]
            
            # --- Define Background ROI for the block ---
            local_background_mask = ~local_axon_mask

            # --- Analyze Axon Trace ---
            if np.sum(local_axon_mask) >= 10:
                axon_mask_full = np.zeros_like(global_axon_mask)
                axon_mask_full[block_slice] = local_axon_mask
                local_axon_trace = mov[:, axon_mask_full].mean(axis=1)
                correlation_axon, _ = scipy.stats.pearsonr(local_axon_trace, global_z_trace)
                axon_corr_map[i, j] = correlation_axon

            # --- Analyze Background Trace ---
            if np.sum(local_background_mask) >= 20:
                background_mask_full = np.zeros_like(global_axon_mask)
                background_mask_full[block_slice] = local_background_mask
                local_background_trace = mov[:, background_mask_full].mean(axis=1)
                correlation_bg, _ = scipy.stats.pearsonr(local_background_trace, global_z_trace)
                background_corr_map[i, j] = correlation_bg

    # 3. Visualize the side-by-side maps
    print("--- Visualizing Comparative Z-Motion Maps ---")
    axon_corr_resized = resize(axon_corr_map, (H, W), order=0, preserve_range=True, anti_aliasing=False)
    background_corr_resized = resize(background_corr_map, (H, W), order=0, preserve_range=True, anti_aliasing=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Z-Motion Source Comparison', fontsize=16)
    
    # Axon Map (In-Focus)
    ax1.imshow(mean_img_for_display, cmap='gray')
    im1 = ax1.imshow(axon_corr_resized, cmap='coolwarm', alpha=0.6, vmin=-0.5, vmax=0.5)
    ax1.set_title('Correlation (Axons vs. Global Z-Trace)')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Pearson Correlation (r)')

    # Background Map (Out-of-Focus/Neuropil)
    ax2.imshow(mean_img_for_display, cmap='gray')
    im2 = ax2.imshow(background_corr_resized, cmap='coolwarm', alpha=0.6, vmin=-0.5, vmax=0.5)
    ax2.set_title('Correlation (Background vs. Global Z-Trace)')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Pearson Correlation (r)')
    
    plt.show()
    return axon_corr_map, background_corr_map



def plot_archetype_block_traces(mov, global_axon_mask, axon_corr_map, background_corr_map, 
                              blocks_to_plot, grid_size=32):
    """
    For a list of selected block coordinates, plots a comparison of the global Z-trace,
    the local axon trace, and the local background trace.

    Args:
        mov (np.ndarray): The registered movie (T, H, W).
        global_axon_mask (np.ndarray): The 2D boolean global mask.
        axon_corr_map (np.ndarray): The 2D map of axon-global correlations.
        background_corr_map (np.ndarray): The 2D map of background-global correlations.
        blocks_to_plot (list of tuples): A list of (row, col) coordinates to plot.
        grid_size (int): The size of the grid blocks.
    """
    T = mov.shape[0]
    global_z_trace = mov.mean(axis=(1, 2))

    def zscore(x):
        std = np.std(x)
        return (x - np.mean(x)) / (std if std > 0 else 1.0)

    for row, col in blocks_to_plot:
        # --- Extract local traces for this block ---
        block_slice = (slice(row * grid_size, (row + 1) * grid_size),
                       slice(col * grid_size, (col + 1) * grid_size))
        
        local_axon_mask = global_axon_mask[block_slice]
        local_bg_mask = ~local_axon_mask
        
        axon_mask_full = np.zeros_like(global_axon_mask)
        axon_mask_full[block_slice] = local_axon_mask
        
        bg_mask_full = np.zeros_like(global_axon_mask)
        bg_mask_full[block_slice] = local_bg_mask

        local_axon_trace = mov[:, axon_mask_full].mean(axis=1) if np.sum(local_axon_mask) > 0 else np.zeros(T)
        local_bg_trace = mov[:, bg_mask_full].mean(axis=1) if np.sum(local_bg_mask) > 0 else np.zeros(T)

        # --- Create the Plot ---
        fig, ax = plt.subplots(figsize=(15, 5))
        frames = np.arange(T)

        ax.plot(frames, zscore(global_z_trace), color='blue', label='Global Z-Trace (Movie Mean)', alpha=0.5)
        ax.plot(frames, zscore(local_axon_trace), color='green', label='Local Axon Trace', linewidth=1.5, alpha=0.5)
        ax.plot(frames, zscore(local_bg_trace), color='red', label='Local Background Trace', alpha=0.5)

        # Get correlation values for the title
        axon_corr = axon_corr_map[row, col]
        bg_corr = background_corr_map[row, col]
        
        ax.set_title(f"Trace Comparison for Block ({row}, {col})\n"
                     f"Axon Corr: {axon_corr:.2f}, Background Corr: {bg_corr:.2f}", fontsize=14)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Z-Scored Intensity')
        ax.legend()
        ax.grid(True, linestyle='--')
        plt.show()


def correct_and_plot_roi_trace_neuropil(mov, global_axon_mask, block_coords, grid_size=32):
    """
    Performs local neuropil regression to correct the trace for a single block
    and visualizes the result.

    Args:
        mov (np.ndarray): The registered movie.
        global_axon_mask (np.ndarray): The global mask.
        block_coords (tuple): The (row, col) of the block to correct.
        grid_size (int): The size of the grid blocks.
    """
    print(f"\n--- Performing Artifact Correction for Block {block_coords} ---")
    T, H, W = mov.shape
    row, col = block_coords

    # --- 1. Extract Traces (same as before) ---
    block_slice = (slice(row * grid_size, (row + 1) * grid_size),
                   slice(col * grid_size, (col + 1) * grid_size))
    
    local_axon_mask = global_axon_mask[block_slice]
    if np.sum(local_axon_mask) < 5: return
    
    axon_mask_full = np.zeros_like(global_axon_mask)
    axon_mask_full[block_slice] = local_axon_mask
    local_axon_trace = mov[:, axon_mask_full].mean(axis=1)

    # Use a slightly larger dilation for a more robust neuropil signal
    dilated_mask = binary_dilation(axon_mask_full, iterations=2)
    neuropil_mask = dilated_mask & ~axon_mask_full
    if np.sum(neuropil_mask) < 20: return
    local_bg_trace = mov[:, neuropil_mask].mean(axis=1)

    # --- 2. Perform Linear Regression ---
    # We want to model: axon_trace = a * bg_trace + intercept
    X = local_bg_trace.reshape(-1, 1) # Reshape for sklearn
    y = local_axon_trace
    
    reg = LinearRegression().fit(X, y)
    scaling_factor = reg.coef_[0]
    
    # The estimated artifact is the scaled background trace
    estimated_artifact = reg.predict(X)

    # --- 3. Calculate the Corrected Trace ---
    corrected_trace = y - estimated_artifact
    
    print(f"Optimal scaling factor for background subtraction: {scaling_factor:.3f}")

    # --- 4. Visualize the Correction ---
    fig, ax = plt.subplots(figsize=(15, 6))
    frames = np.arange(T)

    ax.plot(frames, y, color='gray', label='Original Axon Trace', alpha=0.5)
    ax.plot(frames, estimated_artifact, color='red', label='Estimated Z-Motion Artifact', alpha=0.5)
    ax.plot(frames, corrected_trace, color='green', label='Corrected Axon Trace', linewidth=1.5, alpha=0.5)

    ax.set_title(f'Artifact Correction for Block {block_coords}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.show()

    return corrected_trace, local_axon_trace, local_bg_trace


def correct_and_plot_roi_trace_bk(mov, global_axon_mask, block_coords, grid_size=32):
    """
    Performs local neuropil regression to correct the trace for a single block
    and visualizes the result.
    
    *** CORRECTED VERSION: Uses the "Block Inversion" method for defining neuropil,
    consistent with our diagnostic framework. ***

    Args:
        mov (np.ndarray): The registered movie.
        global_axon_mask (np.ndarray): The global mask.
        block_coords (tuple): The (row, col) of the block to correct.
        grid_size (int): The size of the grid blocks.
    """
    print(f"\n--- Performing Artifact Correction for Block {block_coords} ---")
    T, H, W = mov.shape
    row, col = block_coords

    # --- 1. Define Local Axon and Neuropil ROIs ---
    block_slice = (slice(row * grid_size, (row + 1) * grid_size),
                   slice(col * grid_size, (col + 1) * grid_size))
    
    local_axon_mask = global_axon_mask[block_slice]
    if np.sum(local_axon_mask) < 5: 
        print(f"Skipping Block {block_coords}: Not enough axon pixels.")
        return

    # CORRECTED NEUROPIL DEFINITION
    local_neuropil_mask = ~local_axon_mask
    if np.sum(local_neuropil_mask) < 20: 
        print(f"Skipping Block {block_coords}: Not enough neuropil pixels.")
        return

    # Create full-size masks for indexing
    axon_mask_full = np.zeros_like(global_axon_mask)
    axon_mask_full[block_slice] = local_axon_mask
    local_axon_trace = mov[:, axon_mask_full].mean(axis=1)

    neuropil_mask_full = np.zeros_like(global_axon_mask)
    neuropil_mask_full[block_slice] = local_neuropil_mask
    local_bg_trace = mov[:, neuropil_mask_full].mean(axis=1)

    # --- 2. Perform Linear Regression ---
    # Model: axon_trace = a * bg_trace + intercept + neural_signal
    X = local_bg_trace.reshape(-1, 1)
    y = local_axon_trace
    
    reg = LinearRegression().fit(X, y)
    scaling_factor = reg.coef_[0]
    
    # The estimated artifact is the scaled background trace
    estimated_artifact = reg.predict(X)

    # --- 3. Calculate the Corrected Trace ---
    # Corrected trace is the residual of the model, which is our best
    # estimate of the true neural activity.
    corrected_trace = y - estimated_artifact
    
    print(f"Optimal scaling factor for background subtraction: {scaling_factor:.3f}")

    # --- 4. Visualize the Correction ---
    fig, ax = plt.subplots(figsize=(15, 6))
    frames = np.arange(T)

    ax.plot(frames, y, color='gray', label='Original Axon Trace', alpha=0.5)
    ax.plot(frames, estimated_artifact, color='red', label='Estimated Z-Motion Artifact', alpha=0.5)
    ax.plot(frames, corrected_trace, color='green', label='Corrected Axon Trace', linewidth=1.5, alpha = 0.5)

    ax.set_title(f'Artifact Correction for Block {block_coords}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.show()

    return corrected_trace, local_axon_trace, local_bg_trace




def correct_trace_with_generalized_model(mov, ops_file_path, global_axon_mask, 
                                         block_coords, grid_size=32):
    """
    Performs a comprehensive artifact correction using a generalized multi-regressor
    model that accounts for slow Z-motion and fast global motion jerks.

    Args:
        mov, global_axon_mask, block_coords, grid_size: Same as before.
        ops_file_path (str): Path to the Suite2p ops.npy file.
    """
    print(f"\n--- Performing Generalized Artifact Correction for Block {block_coords} ---")
    T, H, W = mov.shape
    row, col = block_coords

    # --- 1. Extract Target Trace and Neuropil Regressor ---
    block_slice = (slice(row * grid_size, (row + 1) * grid_size),
                   slice(col * grid_size, (col + 1) * grid_size))
    
    local_axon_mask = global_axon_mask[block_slice]
    if np.sum(local_axon_mask) < 5: 
        print(f"Skipping Block {block_coords}: Not enough axon pixels.")
        return None, None, None

    axon_mask_full = np.zeros_like(global_axon_mask)
    axon_mask_full[block_slice] = local_axon_mask
    original_axon_trace = mov[:, axon_mask_full].mean(axis=1)

    local_neuropil_mask = ~local_axon_mask
    if np.sum(local_neuropil_mask) < 20: 
        print(f"Skipping Block {block_coords}: Not enough neuropil pixels.")
        return None, None, None

    neuropil_mask_full = np.zeros_like(global_axon_mask)
    neuropil_mask_full[block_slice] = local_neuropil_mask
    neuropil_regressor = mov[:, neuropil_mask_full].mean(axis=1)

    # --- 2. Prepare Additional Global Jerk Regressors ---
    ops = np.load(ops_file_path, allow_pickle=True).item()
    
    # Regressor for Fast X/Y Jerk (derivative of rigid shifts)
    # np.diff reduces length by 1, so we pad with a 0 at the start to maintain length T
    rigid_jerk_regressor = np.sqrt(np.diff(ops['xoff'][:T], prepend=ops['xoff'][0])**2 + 
                                   np.diff(ops['yoff'][:T], prepend=ops['yoff'][0])**2)
    
    # Regressor for Fast Z-Jerk / Quality Dip (derivative of registration quality)
    z_jerk_regressor = np.abs(np.diff(ops['corrXY'][:T], prepend=ops['corrXY'][0]))

    # --- 3. Build and Fit the Generalized Linear Model ---
    # Create the model matrix X with all three regressors as columns
    X_regressors = np.vstack([
        neuropil_regressor, 
        rigid_jerk_regressor, 
        z_jerk_regressor
    ]).T
    
    # Define the target variable y
    y_target = original_axon_trace
    
    model = LinearRegression().fit(X_regressors, y_target)
    
    # The full estimated artifact is the model's prediction based on all regressors
    estimated_full_artifact = model.predict(X_regressors)

    # --- 4. Calculate the Final Corrected Trace ---
    final_corrected_trace = y_target - estimated_full_artifact
    
    print("Optimal scaling factors (Neuropil, Rigid Jerk, Z-Jerk): "
          f"{model.coef_[0]:.3f}, {model.coef_[1]:.3f}, {model.coef_[2]:.3f}")

    # --- 5. Visualize the Correction ---
    fig, ax = plt.subplots(figsize=(15, 6))
    frames = np.arange(T)

    ax.plot(frames, y_target, color='gray', label='Original Axon Trace', alpha=0.5)
    ax.plot(frames, estimated_full_artifact, color='red', label='Full Estimated Artifact', alpha=0.5)
    ax.plot(frames, final_corrected_trace, color='green', label='Final Corrected Trace', linewidth=1.5, alpha=0.8)

    ax.set_title(f'Generalized Artifact Correction for Block {block_coords}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Fluorescence (a.u.)')
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.show()

    return final_corrected_trace, original_axon_trace, neuropil_regressor
    



def analyze_trace_for_noise(corrected_trace, trace_name="Corrected Axon Trace", sigma=2.0):
    """
    Performs a 3-part diagnostic analysis on a trace to determine if it
    is primarily noise or contains structured signal.

    Args:
        corrected_trace (np.ndarray): The 1D time-series trace to analyze.
        trace_name (str): The name of the trace for plot titles.
        sigma (float): The standard deviation of the Gaussian kernel for low-pass filtering.
                       A larger sigma means more smoothing.
    """
    print(f"\n--- Analyzing '{trace_name}' for Noise vs. Signal ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Noise vs. Signal Analysis for: {trace_name}', fontsize=16)

    # --- Test 1: Low-Pass Filtering ---
    # Apply a Gaussian filter to smooth out high-frequency noise
    filtered_trace = gaussian_filter1d(corrected_trace, sigma=sigma)
    
    axes[0].plot(corrected_trace, color='gray', alpha=0.6, label='Original Corrected Trace')
    axes[0].plot(filtered_trace, color='red', linewidth=1, label=f'Low-Pass Filtered (sigma={sigma})')
    axes[0].set_title('Test 1: Low-Pass Filter')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Fluorescence (a.u.)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--')

    # --- Test 2: Amplitude Distribution ---
    axes[1].hist(corrected_trace, bins=100, density=True, color='green', alpha=0.7)
    
    # Fit a Gaussian to the data to see how well it matches
    mu, std = scipy.stats.norm.fit(corrected_trace)
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu, std)
    axes[1].plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
    axes[1].set_title('Test 2: Amplitude Histogram')
    axes[1].set_xlabel('Fluorescence (a.u.)')
    axes[1].set_ylabel('Probability Density')
    axes[1].legend()

    # --- Test 3: Temporal Structure (Autocorrelation) ---
    # We use plt.acorr which computes and plots the autocorrelation
    axes[2].acorr(filtered_trace, maxlags=200, usevlines=True, normed=True, lw=2)
    axes[2].set_title('Test 3: Autocorrelation')
    axes[2].set_xlabel('Time Lag (frames)')
    axes[2].set_ylabel('Correlation of Filtered Trace')
    axes[2].grid(True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_pixel_count_heatmap(global_axon_mask: np.ndarray, 
                             grid_size: int, 
                             mean_img: np.ndarray = None):
    """
    Creates a heatmap showing the number of active (True) pixels from a mask
    within each block of a grid.

    This is useful for visualizing the density of segmented structures across the FOV.

    Args:
        global_axon_mask (np.ndarray): The 2D boolean global mask to analyze.
        grid_size (int): The size of the grid blocks in pixels.
        mean_img (np.ndarray, optional): The 2D mean image to use as a background
                                         for anatomical context. Defaults to None.
    """
    H, W = global_axon_mask.shape
    
    # Calculate the number of blocks in the grid
    n_blocks_y = H // grid_size
    n_blocks_x = W // grid_size
    
    print(f"Calculating pixel counts for a {n_blocks_y}x{n_blocks_x} grid...")

    # Create an empty array to store the pixel counts for each block
    pixel_count_map = np.zeros((n_blocks_y, n_blocks_x))

    # --- Loop through each block and count the True pixels ---
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            # Define the slice for the current block
            block_slice = (slice(i * grid_size, (i + 1) * grid_size),
                           slice(j * grid_size, (j + 1) * grid_size))
            
            # Extract the part of the mask corresponding to this block
            local_mask = global_axon_mask[block_slice]
            
            # Count the number of True pixels by summing the boolean array
            pixel_count = np.sum(local_mask)
            
            # Store the count in our map
            pixel_count_map[i, j] = pixel_count

    # --- Visualize the Results ---
    print("Visualizing pixel count heatmap...")
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # If a mean image is provided, display it as a gray background
    if mean_img is not None:
        ax.imshow(mean_img, cmap='gray', alpha=0.6)
        # Upscale the count map to overlay it
        # order=0 (nearest neighbor) keeps the sharp blocky look
        count_map_resized = resize(pixel_count_map, mean_img.shape, order=0, 
                                   preserve_range=True, anti_aliasing=False)
        im = ax.imshow(count_map_resized, cmap='viridis', alpha=0.7)
    else:
        # If no mean image, just show the pixel count map directly
        im = ax.imshow(pixel_count_map, cmap='viridis', alpha=0.7)

    ax.set_title(f'Pixel Count per {grid_size}x{grid_size} Block')
    ax.axis('off')
    
    # Add a colorbar to show what the colors mean
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Mask Pixels per Block')
    
    plt.show()

    return pixel_count_map
