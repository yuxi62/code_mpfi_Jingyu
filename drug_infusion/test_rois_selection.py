# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:31:55 2026

@author: Jingyu Cao
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from common.mask import generate_masks

from scipy.ndimage import shift as ndi_shift

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

def post_processing_suite2p_gui(img_orig):
    """
    apply percentile-based contrast normalisation and rescale to 8-bit for GUI display.
    
    parameters:
    - img_orig: np.ndarray
        input 2D image.
    
    returns:
    - img_proc: np.ndarray
        normalised image (uint8, range 0–255).
    """
    # normalize to 1st and 99th percentile
    perc_low, perc_high = np.percentile(img_orig, [1, 99])
    img_proc = (img_orig - perc_low) / (perc_high - perc_low)
    img_proc = np.maximum(0, np.minimum(1, img_proc))

    # convert to uint8
    img_proc *= 255
    img_proc = img_proc.astype(np.uint8)

    return img_proc


#%%
# rec_id = 'AC316-20251024'
rec_id = 'AC317-20251126'
# for idx, rec in selected_rec_lst.iloc[].iterrows():
# anm, date = rec['anm'], rec['date']
# rec_id = anm + '-' + date
anm, date = rec_id.split('-')
concat_path = Path(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{rec_id}")
p_dff_ss1 = concat_path / f'{rec_id}-02_dFF.npy'
dff_all_ss1 = np.load(p_dff_ss1)
p_beh_ss1 = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile\{rec_id}-02.pkl" 
# p_beh_ss2 = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\behaviour_profile\AC310-20250829-04.pkl" 

p_suite2p_ss1 = rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\02\suite2p_func_detec\plane0"
# p_suite2p_ss2 = r"Z:\Jingyu\2P_Recording\AC310\AC310-20250829\04\suite2p_func_detec\plane0"

#
beh_ss1 = pd.read_pickle(p_beh_ss1)
# beh_ss2 = pd.read_pickle(p_beh_ss2)

#
sig_master = xr.open_dataarray(os.path.join(concat_path, "sig_master_raw.nc"))
F_all =  sig_master.values.squeeze()

sig_master_neu = xr.open_dataarray(os.path.join(concat_path, "sig_master_neu_raw.nc"))
Fneu_all =  sig_master_neu.values.squeeze()

A_master = xr.open_dataarray(os.path.join(concat_path, "A_master.nc"))
roi_map = A_master.values.squeeze()

shift_ds = xr.open_dataset(os.path.join(concat_path, "shift_ds.nc"))
sh = shift_ds["shifts"].sel(animal=anm, session=f'{date}_02') # use the first session's reference mean image
roi_map_shifted = warp_rois_rigid(roi_map, (-sh).values)
gcamp_stats = roi_map_to_list(roi_map_shifted)

F_corr = F_all-0.7*Fneu_all

# from scipy.stats import median_abs_deviation as rsd
# all_F_mean = np.nanmean(F_corr, axis=-1) # (n_rois, )
# all_F_std  = np.nanstd(F_corr, axis=-1)  # (n_rois, )
# all_F_median = np.nanmedian(F_corr, axis=-1)
# all_F_rsd = 1.4826 *rsd(F_corr, axis=-1, nan_policy="omit")
# act_thresh = (all_F_mean+2*all_F_std)[:, None]
# r = 10
# fig, ax = plt.subplots()
# ax.plot(F_corr[r, 10000:15000])
# ax.axhline(all_F_mean[r] + 2*all_F_std[r], lw=1, c='green')
# # ax.axhline(all_F_median[r] + 2*all_F_rsd[r], lw=1, c='tab:red')
# plt.show()

suite2p_ss1_ops = np.load(p_suite2p_ss1+r'\ops.npy', allow_pickle=True).item()
mean_img_ch1 = suite2p_ss1_ops['meanImg']

F_corr_ss1 = F_corr[:, :suite2p_ss1_ops['nframes']]

is_soma, is_active, is_active_soma = generate_masks.select_gcamp_rois(mean_img_ch1, F_corr,
                                 gcamp_stats, 
                                 path_result=r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS")


#%%
selected_roi_map = np.any(roi_map_shifted[is_soma], axis=0)
non_roi_map = np.any(roi_map_shifted[~is_soma], axis=0)
plt.imshow(selected_roi_map, cmap='Greys', alpha=0.5)
plt.show()
plt.imshow(non_roi_map, cmap='Greys', alpha=0.5)
plt.show()

from scipy.ndimage import gaussian_filter1d
from matplotlib.transforms import blended_transform_factory
def plot_random_dff_traces_with_scalebar(
    F_soma, n_show=30, seed=None, q_baseline=10, eps=1e-6,
    fs=None,                       # Hz (optional). If None, x is frames
    tbar_s=5.0,                     # time scalebar length in seconds (or frames if fs=None)
    ybar_dff=0.2,                   # dF/F scalebar height
    robust_prc=(1, 99),             # robust limits for shared scaling
    figsize=(12, 14), dpi=200
):
    """
    Plots n_show random ROI dF/F traces as stacked panels with:
      - shared y-limits across panels
      - no per-ROI y-axis labels/ticks
      - one uniform time scalebar + one uniform dF/F scalebar (bottom-left)
    """
    F = np.asarray(F_soma)
    n_rois, n_frames = F.shape
    n_show = min(n_show, n_rois)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rois, size=n_show, replace=False)

    # dF/F with per-ROI percentile baseline
    F0 = np.nanpercentile(F[idx], q_baseline, axis=1, keepdims=True)
    F0 = np.maximum(F0, eps)
    dff = (F[idx] - F0) / F0  # (n_show, n_frames)

    # shared y-limits (robust)
    # lo, hi = np.nanpercentile(dff.reshape(-1), robust_prc)
    lo, hi = -1, 6
    # lo, hi = 0, 10
    pad = 0.10 * (hi - lo + 1e-12)
    ylo, yhi = lo - pad, hi + pad

    # x-axis in frames or seconds
    if fs is None:
        x = np.arange(n_frames)
        xlabel = "Frame"
        tbar = tbar_s  # interpret as frames
        tbar_label = f"{int(tbar_s)} frames" if float(tbar_s).is_integer() else f"{tbar_s:g} frames"
    else:
        x = np.arange(n_frames) / float(fs)
        xlabel = "Time (s)"
        tbar = float(tbar_s)
        tbar_label = f"{tbar_s:g} s"

    fig, axes = plt.subplots(
        n_show, 1, sharex=True, figsize=figsize, dpi=dpi,
        gridspec_kw={"hspace": 0.05}
    )
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, dff[i], lw=0.8)
        ax.set_ylim(ylo, yhi)

        # clean look: no y ticks/labels per ROI
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel(xlabel)
    axes[0].set_title(f"Random {n_show} soma ROIs: ΔF/F (baseline={q_baseline}th percentile)")

    # ---- Add ONE scalebar on the bottom axis (data-coord anchored) ----
    ax0 = axes[-1]

    # Position scalebar near bottom-left in axes-fraction for x, data for y
    trans = blended_transform_factory(ax0.transAxes, ax0.transData)

    # x start at 5% into the axis; x end determined by tbar in data units
    x_start = 0.05
    # Convert an axes-fraction x_start to data, then add tbar in data units
    x0_data = ax0.get_xlim()[0] + x_start * (ax0.get_xlim()[1] - ax0.get_xlim()[0])
    x1_data = x0_data + tbar

    # y start slightly above bottom y-limit
    y0 = ylo + 0.08 * (yhi - ylo)
    y1 = y0 + ybar_dff

    # time bar (horizontal)
    ax0.plot([x0_data, x1_data], [y0, y0], lw=2)

    # dF/F bar (vertical)
    ax0.plot([x0_data, x0_data], [y0, y1], lw=2)

    # labels
    ax0.text(x0_data, y0 - 0.03 * (yhi - ylo), tbar_label, ha="left", va="top", fontsize=9)
    ax0.text(x0_data - 0.01 * (ax0.get_xlim()[1] - ax0.get_xlim()[0]),
             y0 + 0.5 * (ybar_dff),
             f"{ybar_dff:g} ΔF/F", ha="right", va="center", rotation=90, fontsize=9)

    plt.tight_layout()
    return fig, axes, idx


def plot_random_dff_traces(F_soma, n_show=30, seed=None, q_baseline=10, eps=1e-6,
                           use_common_ylim=False, figsize=(12, 14), dpi=200):
    """
    F_soma: (n_rois, n_frames) fluorescence array (numpy).
    n_show: number of ROIs to plot
    q_baseline: percentile baseline per ROI for dF/F
    use_common_ylim: if True, use a common y-lim across panels (robust)
    """
    F = np.asarray(F_soma)
    n_rois, n_frames = F.shape
    n_show = min(n_show, n_rois)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rois, size=n_show, replace=False)
    
    dff=F_soma


    # robust scaling helpers
    def robust_ylim(y):
        lo, hi = np.nanpercentile(y, [1, 99])
        pad = 0.10 * (hi - lo + 1e-12)
        return lo - pad, hi + pad

    if use_common_ylim:
        ylo, yhi = robust_ylim(dff.reshape(-1))
    else:
        ylo = yhi = None

    fig, axes = plt.subplots(
        n_show, 1, sharex=True,
        figsize=figsize, dpi=dpi,
        gridspec_kw={"hspace": 0.05}
    )
    if n_show == 1:
        axes = [axes]

    x = np.arange(n_frames)

    for i, ax in enumerate(axes):
        y = dff[idx[i]]
        ax.plot(x, y, lw=0.8)
        # ax.set_ylabel(f"ROI {idx[i]}", rotation=0, labelpad=25, va="center")

        if use_common_ylim:
            ax.set_ylim(ylo, yhi)
        else:
            lo, hi = robust_ylim(y)
            ax.set_ylim(lo, hi)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # keep y ticks minimal to reduce clutter
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"Random {n_show} soma ROIs: ΔF/F (baseline={q_baseline}th percentile)")

    plt.tight_layout()
    return fig, axes, idx

# Example usage:
# F_soma = F_corr[is_soma]   # (n_rois, n_frames)
# fig, axes, roi_idx = plot_random_dff_traces(F_soma, n_show=30, seed=0, q_baseline=10)
# plt.show()


# Example:
# F_soma = F_corr[is_soma]
# fig, axes, roi_idx = plot_random_dff_traces_with_scalebar(F_soma, n_show=30, seed=0, fs=30, tbar_s=5, ybar_dff=0.2)
# plt.show()

# Example usage:
trace_soma = dff_all_ss1[is_active_soma]   # (n_rois, n_tracerames)
trace_soma_sm = gaussian_filter1d(trace_soma, sigma=1)
trace_non_soma = dff_all_ss1[(is_soma)&(~(is_active))] 
trace_non_soma_sm = gaussian_filter1d(trace_non_soma, sigma=1)

traceig, axes, roi_idx = plot_random_dff_traces(trace_soma_sm, n_show=30)
plt.show()

traceig, axes, roi_idx = plot_random_dff_traces(trace_non_soma_sm, n_show=30)
plt.show()




#%% plot test masks 

# from tqdm import tqdm

# save_stem = Path('Z:/Jingyu/LC_HPC_manuscript/raw_data/drug_infusion/TEST_PLOTS/soma_mask_inspection')
# save_stem.mkdir(exist_ok=True)
# save_path = save_stem / f'{rec_id}-02'
# save_path.mkdir(exist_ok=True)

# mean_img_ch1_enhanced = np.flipud(post_processing_suite2p_gui(mean_img_ch1))

# for i, soma in tqdm(enumerate(is_soma), total=len(is_soma)):
#     if soma:
#         curr_f    = F_corr[i,:1000]
#         curr_mask = roi_map_shifted[i,:,:]
        
#         fig, axs = plt.subplots(1,3,figsize=(10,3))
        
#         axs[0].imshow(mean_img_ch1_enhanced, aspect='auto')
        
#         axs[1].imshow(curr_mask, aspect='auto')
        
#         axs[2].plot(curr_f)
        
#         fig.savefig(save_path / f'ROI {i}.png',
#                     dpi=300)
        
        






    
# dff_ss1, baseline_ss1 = percentile_dff(F_corr_ss1, return_baseline=True)

# figure_path = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\TEST_PLOTS"

# baseline_mean_ss1 = np.nanmean(baseline_ss1, axis=-1)
# plt.hist(baseline_mean_ss1, bins=100)
# plt.savefig(figure_path + rf"\{rec_id}_roi_dff_20th_no_filter_baseline_mean_hist.png", dpi=200)