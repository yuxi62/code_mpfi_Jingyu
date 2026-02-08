import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.measure import regionprops
from scipy.stats import skew, pearsonr
import matplotlib.cm as cm # Import the colormap module
from pathlib import Path
import tifffile

def calculate_rois_hollowness(stat_array, soma_mask):
    """
    Calculate how "hollow" each ROI is by comparing it
    to a ground-truth mask of soma locations.

    Args:
        stat_array (np.ndarray): The array of dictionaries from 'stat.npy'.
        soma_mask_ground_truth (np.ndarray): A 2D boolean mask where True indicates
                                             the location of a dark soma center.

    """
    num_rois = len(stat_array)
    hollowness_scores = np.zeros(num_rois)  
    for i, roi in enumerate(stat_array):
        # Get the pixel coordinates for the current Suite2p ROI
        ypix = roi['ypix']
        xpix = roi['xpix']
        total_pixels_in_roi = len(ypix)

        if total_pixels_in_roi == 0:
            continue

        # --- The Hollowness Test ---
        # Count how many of the ROI's pixels fall on the bright membrane mask
        # We can do this by checking the value of the membrane mask at the ROI's coordinates
        pixels_on_membrane = np.sum(soma_mask[ypix, xpix])
        
        # Calculate the hollowness ratio
        hollowness_ratio = pixels_on_membrane / total_pixels_in_roi
        hollowness_scores[i] = hollowness_ratio
                    
    return hollowness_scores
    

def calculate_rois_robust(stat_array, f_path, fneu_path):      
    """
    Calculate signal skewness and ROI-neuropil correlation of each ROI.
    
    Args:
        stat_array: stat loaded from stat.npy
    """
    # --- Load all data (same as before) ---
    F_raw = np.load(f_path)
    Fneu_raw = np.load(fneu_path)

    num_rois = len(stat_array)
    skewness_all = np.zeros(num_rois)
    correlation_all = np.zeros(num_rois)
    
    # Pre-calculate a standard corrected trace for skewness measurement
    F_corr_std = F_raw - 0.7 * Fneu_raw
    
    print(f"Classifying {num_rois} ROIs with robust metrics...")
    
    for i, roi in enumerate(stat_array):
        # --- Signal Skewness ---
        f_corrected = F_corr_std[i, :]
        skewness = skew(f_corrected)
        skewness_all[i] = skewness
        
        # --- ROI-to-Neuropil Correlation ---
        f_roi = F_raw[i, :]
        f_neu = Fneu_raw[i, :]
        # Check for sufficient variance before calculating correlation
        if np.std(f_roi) > 0 and np.std(f_neu) > 0:
            correlation = pearsonr(f_roi, f_neu)[0]
            correlation_all[i] = correlation
        else: # If one trace is flat, it can't be correlated
            continue
        
    return skewness_all, correlation_all


def calculate_rois_aspect_ratio(stat_array):
    """
    Calculate the aspect_ratio for each ROI.

    Args:
        stat_array: stat loaded from stat.npy
    """
    
    num_rois = len(stat_array)
    aspect_ratio_all = np.zeros(num_rois)
    
    for i, roi in enumerate(stat_array):
        # --- Calculate compactness ---
        ymin, xmin = np.min(roi['ypix']), np.min(roi['xpix'])
        ymax, xmax = np.max(roi['ypix']), np.max(roi['xpix'])
        roi_img = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        roi_img[roi['ypix'] - ymin, roi['xpix'] - xmin] = 1
        props = regionprops(roi_img)[0]
    
        aspect_ratio =  props.major_axis_length/(props.minor_axis_length+1e-9)
        aspect_ratio_all[i] = aspect_ratio
            
    return aspect_ratio_all
    
def calculate_rois_compactness(stat_array):
    """
    Calculate the compactness for each ROI.

    Args:
        stat_array: stat loaded from stat.npy
    """
    
    num_rois = len(stat_array)
    
    compactness_all = np.zeros(num_rois)
    area_all = np.zeros(num_rois)
    perimeter_all = np.zeros(num_rois)
    aspect_ratio_all = np.zeros(num_rois)
    
    for i, roi in enumerate(stat_array):
        # --- Calculate compactness ---
        try:
            ymin, xmin = np.min(roi['ypix']), np.min(roi['xpix'])
            ymax, xmax = np.max(roi['ypix']), np.max(roi['xpix'])
            roi_img = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            roi_img[roi['ypix'] - ymin, roi['xpix'] - xmin] = 1
            area_all[i] = (ymax - ymin)*(xmax - xmin) # max area around the ROI
            props = regionprops(roi_img)[0]
            
            perimeter = props.perimeter
            compactness = 0 if perimeter == 0 else 4 * np.pi * props.area / (perimeter**2)
            ar =  props.major_axis_length/(props.minor_axis_length+1e-9)
            compactness_all[i] = compactness
            perimeter_all[i] = perimeter
            aspect_ratio_all[i] = ar
        except:
            compactness_all[i] = np.nan
            perimeter_all[i] = np.nan
            aspect_ratio_all[i] = np.nan
            
    return compactness_all, area_all, aspect_ratio_all, perimeter_all


def calculate_rois_npix(stat_array):
    """
    Calculate the number of pixels for each ROI.

    Args:
        stat_array: stat loaded from stat.npy
    """
    
    num_rois = len(stat_array)
    
    roi_npix_all = np.zeros(num_rois)
    
    print(f"Classifying {num_rois} ROIs based on npix...")
    
    for i, roi in enumerate(stat_array):
        # --- Calculate number of pixels ---
        roi_npix_all[i] = roi['npix']
        
    return roi_npix_all


def visualize_soma_classification_masks(
    mean_img, 
    stat_array, 
    is_soma,
    output_dir: str = None,
    file_name: str = "soma_classification.png"
):
    """
    Creates a two-panel plot to visualize classification results.
    - Left panel: Shows only ROIs classified as somas.
    - Right panel: Shows only ROIs classified as non-somas.
    Each individual ROI is assigned a unique random color for clarity.

    Args:
        mean_img (np.ndarray): The 2D anatomical mean image.
        stat_array (np.ndarray): The array of dictionaries from 'stat.npy'.
        is_soma (np.ndarray): A 1D boolean array from the classification function.
    """
    H, W = mean_img.shape
    num_rois = len(stat_array)

    # --- Step 1: Generate a unique color for every ROI ---
    # We use a perceptually uniform colormap to get a nice range of distinct colors.
    colors = cm.nipy_spectral(np.linspace(0, 1, num_rois))
    
    # --- Step 2: Create two separate overlay masks ---
    soma_color_mask = np.zeros((H, W, 4), dtype=np.float32)
    non_soma_color_mask = np.zeros((H, W, 4), dtype=np.float32)
    
    soma_count = 0
    non_soma_count = 0

    print("Generating colored masks for visualization...")
    for i, roi in enumerate(stat_array):
        ypix, xpix = roi['ypix'], roi['xpix']
        
        # Get the unique color for this ROI
        roi_color = colors[i]
        
        # Set the alpha (transparency) for the overlay
        roi_color[3] = 0.7 
        
        if is_soma[i]:
            soma_color_mask[ypix, xpix] = roi_color
            soma_count += 1
        else:
            non_soma_color_mask[ypix, xpix] = roi_color
            non_soma_count += 1

    # --- Step 3: Create the final two-panel plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Soma Classification Results", fontsize=16)

    # Panel 1: Somas
    ax1.imshow(mean_img, cmap='gray', alpha=0.7)
    ax1.imshow(soma_color_mask, alpha=0.6)
    ax1.set_title(f"Somas (n={soma_count})")
    
    # Panel 2: Non-Somas
    ax2.imshow(mean_img, cmap='gray', alpha=0.7)
    ax2.imshow(non_soma_color_mask, alpha=0.6)
    ax2.set_title(f"Non-Somas (n={non_soma_count})")
    
    for ax in [ax1, ax2]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Step 4: Save the Figure ---
    if output_dir:
        # Create a Path object for robust path handling
        output_path = Path(output_dir) / file_name
        
        print(f"Saving figure to: {output_path}")
        fig.savefig(output_path, dpi=400, bbox_inches='tight')
        print("Save complete.")
    
    # --- Step 5: Display the Figure ---
    # plt.show()
    
    # It's good practice to close the figure object after saving/showing
    # to free up memory, especially in a loop.
    plt.close(fig)


def visualize_and_save_soma_masks(
    mean_img: np.ndarray, 
    stat_array: np.ndarray, 
    is_soma: np.ndarray,
    output_dir: str = None,
    file_basename: str = "soma_roi_mask",
    save_mask: bool = True
) -> tuple:
    """
    Visualizes soma classification, creates separate binary masks for somas and
    non-somas, and saves the masks and the figure to disk.

    Args:
        mean_img (np.ndarray): The 2D anatomical mean image.
        stat_array (np.ndarray): The array of dictionaries from 'stat.npy'.
        is_soma (np.ndarray): A 1D boolean array from the classification function.
        output_dir (str, optional): The directory to save the plot and mask. If None,
                                    nothing is saved. Defaults to None.
        file_basename (str, optional): The base name for the output files.
        save_mask (bool, optional): If True, saves the binary soma mask. Defaults to True.

    Returns:
        tuple: A tuple containing two 2D boolean masks: (somas_mask, non_somas_mask).
               - somas_mask: True where pixels belong to a classified soma.
               - non_somas_mask: True where pixels belong to a classified non-soma.
    """
    H, W = mean_img.shape
    num_rois = len(stat_array)

    # --- Create empty binary masks for both populations ---
    somas_mask = np.zeros((H, W), dtype=bool) 
    # FIX: Removed extra parenthesis
    non_somas_mask = np.zeros((H, W), dtype=bool)
    
    # Calculate counts for titles
    soma_count = np.sum(is_soma)
    # FIX: Defined non_soma_count
    non_soma_count = num_rois - soma_count

    print("Generating binary masks for somas and non-somas...")
    for i, roi in enumerate(stat_array):
        ypix, xpix = roi['ypix'], roi['xpix']
        
        if is_soma[i]:
            # Set pixels to True (or 1) in the soma mask
            somas_mask[ypix, xpix] = True
        else:
            # Set pixels to True (or 1) in the non-soma mask
            non_somas_mask[ypix, xpix] = True

    # --- Create the three-panel visualization plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("Soma ROI masks", fontsize=16)

    ax1.imshow(mean_img, cmap='gray')
    ax1.set_title("Mean Image")

    ax2.imshow(somas_mask, cmap='gray') # Use cmap='gray' for binary masks
    ax2.set_title(f"Somas (n={soma_count})")
    
    ax3.imshow(non_somas_mask, cmap='gray')
    ax3.set_title(f"Non-Somas (n={non_soma_count})")
    
    for ax in [ax1, ax2, ax3]: ax.axis('off')

    # --- Save the Figure and the Masks ---
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_mask:
            # Save the soma mask
            mask_path_npy = output_path / f'{file_basename}.npy'
            mask_path_tiff = output_path / f'{file_basename}.tiff'
            print(f"Saving binary soma mask to:\n  NPY: {mask_path_npy}\n  TIFF: {mask_path_tiff}")
            # FIX: Used the correct variable name `somas_mask`
            np.save(mask_path_npy, somas_mask)
            tifffile.imwrite(mask_path_tiff, somas_mask.astype('uint8') * 255)

            # Save the non-soma mask
            non_mask_path_npy = output_path / f'non_{file_basename}.npy'
            non_mask_path_tiff = output_path / f'non_{file_basename}.tiff'
            # FIX: Corrected the print statement
            print(f"Saving binary non-soma mask to:\n  NPY: {non_mask_path_npy}\n  TIFF: {non_mask_path_tiff}")
            np.save(non_mask_path_npy, non_somas_mask)
            tifffile.imwrite(non_mask_path_tiff, non_somas_mask.astype('uint8') * 255)
    
    # plt.show()
    plt.close(fig)
    
    # Return both binary masks
    return somas_mask, non_somas_mask


def classify_and_save_somas(
    stat_path: str,
    f_path: str,
    fneu_path: str,
    mean_img: np.ndarray,
    soma_mask: np.ndarray,
    output_dir: str,
    thresholds: dict,
    figure_on: bool = True
) -> np.ndarray:
    """
    Performs a full ROI classification pipeline to identify somas based on a
    suite of morphological and temporal features.

    This function:
    1. Calculates features: hollowness, compactness, size, skewness, and correlation.
    2. Applies user-defined thresholds to classify ROIs.
    3. Optionally visualizes the feature distributions and final classification.
    4. Saves all calculated features, parameters, and the final classification to a .npz file.

    Args:
        stat_path, f_path, fneu_path (str): Paths to the required Suite2p .npy files.
        mean_img (np.ndarray): The mean image for visualization backgrounds.
        soma_mask (np.ndarray): The ground-truth soma mask for the hollowness calculation.
        output_dir (str): The directory to save results and figures.
        thresholds (dict): A dictionary containing all classification thresholds.
        figure_on (bool): If True, all intermediate and final plots will be displayed.

    Returns:
        np.ndarray: A 1D boolean array where True indicates a classified soma.
    """
    print("\n--- Starting Full Soma Classification Pipeline ---")

    # --- Setup Paths and Load Data ---
    output_path = Path(output_dir)
    stat_array = np.load(stat_path, allow_pickle=True)
    
    # --- STEP 1: Calculate All ROI Features ---
    print("\nStep 1: Calculating features for all ROIs...")
    
    hollowness_scores = calculate_rois_hollowness(stat_array=stat_array, soma_mask=soma_mask)
    compactness, area, aspect_ratio_all, _ = calculate_rois_compactness(stat_array=stat_array)
    roi_npix = calculate_rois_npix(stat_array=stat_array)
    skewness, correlation = calculate_rois_robust(stat_array, f_path, fneu_path)
    
    # --- (Optional) STEP 2: Visualize Feature Distributions ---
    if figure_on:
        print("\nStep 2: Visualizing feature distributions...")
        
        # Hollowness Plot
        plt.figure(figsize=(6, 4)); plt.hist(hollowness_scores, bins=100, edgecolor='black')
        plt.title("Distribution of ROI Hollowness Scores"); plt.xlabel("Hollowness Ratio"); plt.ylabel("# ROIs"); plt.show()
        
        # Morphological Plots
        fig_morph, axes_morph = plt.subplots(1, 3, figsize=(18, 4))
        axes_morph[0].hist(compactness, bins=100, edgecolor='black'); axes_morph[0].set_title("Compactness")
        axes_morph[1].hist(area, bins=100, edgecolor='black'); axes_morph[1].set_title("Bounding Box Area")
        axes_morph[2].hist(roi_npix, bins=100, edgecolor='black'); axes_morph[2].set_title("Pixel Count (npix)")
        plt.suptitle("Distribution of Morphological Features"); plt.show()
        
        # Temporal Plots
        fig_temp, axes_temp = plt.subplots(1, 2, figsize=(12, 4))
        axes_temp[0].hist(skewness, bins=100, edgecolor='black'); axes_temp[0].set_title("Skewness")
        axes_temp[1].hist(correlation, bins=100, edgecolor='black'); axes_temp[1].set_title("F-Fneu Correlation")
        plt.suptitle("Distribution of Temporal Features"); plt.show()

    # --- STEP 3: Classify ROIs using Thresholds ---
    print("\nStep 3: Classifying ROIs based on thresholds...")
    is_soma = (
        (roi_npix > thresholds['min_soma_roi_npix']) &
        (roi_npix < thresholds['max_soma_roi_npix']) &
        (area < thresholds['area_threshold']) &
        (compactness > thresholds['compactness_threshold_min']) & # Corrected logic for compactness
        (aspect_ratio_all < thresholds['aspect_ratio_max'])&
        (hollowness_scores > thresholds['hollowness_threshold_min']) &
        (skewness > thresholds['skewness_threshold'])
    )
    print(f" > Found {np.sum(is_soma)} ROIs passing all criteria.")

    # --- STEP 4: Visualize Final Classification ---
    print("\nStep 4: Visualizing final classification...")
    visualize_soma_classification_masks(
        mean_img, stat_array, is_soma, 
        output_dir=output_path, 
        file_name='Soma_classification_results.png' # Pass file name
    )
    
    # --- STEP 5: Save soma roi masks ---
    print("\nStep 5: Save soma roi masks...")
    visualize_and_save_soma_masks(
        mean_img, 
        stat_array, 
        is_soma,
        output_dir=output_path,
        file_basename="soma_roi_mask",
        save_mask=True
    )

    # --- STEP 6: Save All Results and Parameters ---
    data_save_path = output_path / 'soma_classification_results.npz'
    print(f"\nStep 6: Saving all data and parameters to {data_save_path}")
    
    np.savez_compressed(
        data_save_path,
        is_soma=is_soma,
        roi_npix=roi_npix,
        area=area,
        compactness=compactness,
        aspect_ratio_all=aspect_ratio_all, 
        hollowness_scores=hollowness_scores,
        skewness=skewness,
        correlation=correlation,
        **thresholds # Save the entire thresholds dictionary
    )
    print("Save complete.")
    
    return is_soma