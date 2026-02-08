# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:03:53 2026

@author: CaoJ
"""
import numpy as np
import pandas as pd
from pathlin import Path
import matplotlib.pyplot as plt
from common.utils_basic import normalize
from common.utils_imaging import align_trials
import common.plotting_functions_Jingyu as pf
from scipy.ndimage import gaussian_filter1d



def get_valid_grids(res_traces):
    """
    Get list of all valid grids (those without NaN traces).

    Args:
        res_traces: Loaded npz file with trace arrays

    Returns:
        list: List of tuples (grid_y, grid_x) for valid grids
    """
    corrected_dlight = res_traces['corrected_dlight']
    n_blocks_y, n_blocks_x, _ = corrected_dlight.shape

    valid_grids = []
    for gy in range(n_blocks_y):
        for gx in range(n_blocks_x):
            if not np.all(np.isnan(corrected_dlight[gy, gx, :])):
                valid_grids.append((gy, gx))

    print(f"Found {len(valid_grids)} valid grids out of {n_blocks_y * n_blocks_x} total grids")
    return valid_grids



def plot_regression_qc_check(res_traces, qc_info, beh, grid_y, grid_x, save_fig=False,
                             max_events_per_fig=64):
    """
    Plot regression QC visualization for a specific grid block.

    For each event in the block, plots the original, corrected, red, and neuropil
    traces within the regression window (start_frame:end_frame), with regression
    coefficients (b1, b2) and R² displayed in the title.

    If number of events exceeds max_events_per_fig (default 64 = 8x8), saves as
    a series of figures.

    Args:
        rec: Recording name
        regression_name: Name of the regression method
        k_size: Dilation kernel size
        grid_y: Y coordinate of the grid block
        grid_x: X coordinate of the grid block
        save_fig: Whether to save the figure
        max_events_per_fig: Maximum events per figure (default 64 = 8x8 grid)
    """
    # Extract trace grids
    original_dlight = res_traces['original_dlight']
    corrected_dlight = res_traces['corrected_dlight']
    red_trace = res_traces['red_trace']
    neuropil_dlight = res_traces['neuropil_dlight']

    # Check if the specified grid has valid data
    if np.all(np.isnan(original_dlight[grid_y, grid_x, :])):
        print(f"Grid ({grid_y}, {grid_x}) has no valid traces (all NaN).")
        return

    # Get QC info for this specific grid
    grid_qc = qc_info[(qc_info['grid_y'] == grid_y) & (qc_info['grid_x'] == grid_x)]

    if len(grid_qc) == 0:
        print(f"No QC info found for grid ({grid_y}, {grid_x}).")
        return

    # Extract the QC data for this grid (should be one row)
    qc_row = grid_qc.iloc[0]
    r2_list = qc_row['r2']
    b1_list = qc_row['b1']
    b2_list = qc_row['b2']
    start_frame_list = qc_row['start_frame']
    end_frame_list = qc_row['end_frame']
    event_frame_list = qc_row['event_frame']

    n_events = len(r2_list)

    # Extract traces for this grid
    orig_trace = original_dlight[grid_y, grid_x, :]
    corr_trace = corrected_dlight[grid_y, grid_x, :]
    red_tr = red_trace[grid_y, grid_x, :]
    neuropil_tr = neuropil_dlight[grid_y, grid_x, :]
    
    # Create save directory if needed
    if save_fig:
        save_dir = p_dilation_results / 'regression_qc_check'
        save_dir.mkdir(exist_ok=True)
        
    # plot trial averaging and heatmap
    bef, aft = 1, 4
    xaxis = np.arange(30*(bef+aft))/30- bef
    orig_trace_aligned = align_trials(orig_trace, 'run', beh, bef, aft)
    corr_trace_aligned = align_trials(corr_trace, 'run', beh, bef, aft)
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    ax = axs[0]
    pf.plot_mean_trace(orig_trace_aligned, ax, xaxis, color='grey', label='raw_dlight')
    pf.plot_mean_trace(corr_trace_aligned, ax, xaxis, color='green', label='corrected_dlight')
    ax.legend(frameon=False, loc='upper right', fontsize=6)
    ax = axs[1]
    ax.imshow(gaussian_filter1d(normalize(orig_trace_aligned), sigma=1), 
              cmap='Greys',
              aspect='auto')
    ax.set(title='raw_dlight')
    ax = axs[2]
    ax.imshow(gaussian_filter1d(normalize(corr_trace_aligned), sigma=1), 
              cmap='Greys',
              aspect='auto')
    ax.set(title='corrected_dlight')
    fig.tight_layout()
    if save_fig:
        fig_name = f'qc_grid_{grid_y}_{grid_x}_overview.png'
        plt.savefig(save_dir / fig_name, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {save_dir / fig_name}")
    else:
        plt.show()
            
    # plot single event res
    # Calculate number of figures needed
    n_figs = (n_events + max_events_per_fig - 1) // max_events_per_fig

    # Create figures
    for fig_idx in range(n_figs):
        # Calculate event range for this figure
        start_event = fig_idx * max_events_per_fig
        end_event = min((fig_idx + 1) * max_events_per_fig, n_events)
        events_in_fig = end_event - start_event

        # Determine grid layout (up to 8x8)
        n_cols = min(8, events_in_fig)
        n_rows = (events_in_fig + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

        # Title with page info if multiple figures
        title_suffix = f' (Page {fig_idx+1}/{n_figs})' if n_figs > 1 else ''
        fig.suptitle(f'Regression QC - Grid ({grid_y}, {grid_x}){title_suffix}\n'
                     f'npix_green={qc_row["n_green_mask"]}, npix_neuropil={qc_row["n_neuropil_mask"]}, npix_red={qc_row["n_red_mask"]}',
                     fontsize=14)

        for i_local in range(events_in_fig):
            i_global = start_event + i_local
            row_idx = i_local // n_cols
            col_idx = i_local % n_cols
            ax = axes[row_idx, col_idx]

            start_f = start_frame_list[i_global]
            end_f = end_frame_list[i_global]
            event_f = event_frame_list[i_global]
            r2 = r2_list[i_global]
            b1 = b1_list[i_global]
            b2 = b2_list[i_global]

            # Time axis relative to event onset
            frames = np.arange(start_f, end_f)
            time_rel = frames - event_f  # Relative to event frame

            # Plot traces
            ax.plot(time_rel, orig_trace[start_f:end_f], 'gray', label='Original', alpha=0.7)
            ax.plot(time_rel, corr_trace[start_f:end_f], 'g', label='Corrected', linewidth=1.5)
            ax.plot(time_rel, red_tr[start_f:end_f], 'r', label='Red', alpha=0.6)
            ax.plot(time_rel, neuropil_tr[start_f:end_f], 'orange', label='Neuropil', alpha=0.6)

            # Mark event onset
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)

            ax.set_title(f'Event {i_global+1}: b1={b1:.3f}, b2={b2:.3f}, R²={r2:.3f}', fontsize=9)
            ax.set_xlabel('Frames from event')
            ax.set_ylabel('Fluorescence')

            # Only show legend on first subplot of first figure
            if i_global == 0:
                ax.legend(loc='upper right', fontsize=7)

        # Hide empty subplots
        for i_local in range(events_in_fig, n_rows * n_cols):
            row_idx = i_local // n_cols
            col_idx = i_local % n_cols
            axes[row_idx, col_idx].set_visible(False)

        plt.tight_layout()

        if save_fig:
            if n_figs > 1:
                fig_name = f'qc_grid_{grid_y}_{grid_x}_page{fig_idx+1}.png'
            else:
                fig_name = f'qc_grid_{grid_y}_{grid_x}.png'
            plt.savefig(save_dir / fig_name, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved figure to {save_dir / fig_name}")
        else:
            plt.show()


def plot_regression_qc_summary(rec, regression_name, k_size, save_fig=False):
    """
    Plot a summary heatmap of regression quality metrics (mean R², b1, b2) across all grids.

    Args:
        rec: Recording name
        regression_name: Name of the regression method
        k_size: Dilation kernel size
        save_fig: Whether to save the figure
    """
    import matplotlib.pyplot as plt

    p_dilation_results = OUR_DIR_REGRESS / rec / regression_name / r'dilation_k={}'.format(k_size)
    res_traces = np.load(p_dilation_results / f'{regression_name}_res_traces.npz')
    qc_info = pd.read_pickle(p_dilation_results / f'{regression_name}_qc.pkl')

    # Get grid dimensions from traces
    n_blocks_y, n_blocks_x, _ = res_traces['corrected_dlight'].shape

    # Initialize arrays for mean metrics
    mean_r2 = np.full((n_blocks_y, n_blocks_x), np.nan)
    mean_b1 = np.full((n_blocks_y, n_blocks_x), np.nan)
    mean_b2 = np.full((n_blocks_y, n_blocks_x), np.nan)

    # Fill in values from QC DataFrame
    for _, row in qc_info.iterrows():
        gy, gx = int(row['grid_y']), int(row['grid_x'])
        mean_r2[gy, gx] = np.mean(row['r2'])
        mean_b1[gy, gx] = np.mean(row['b1'])
        mean_b2[gy, gx] = np.mean(row['b2'])

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Regression QC Summary - {rec} (k={k_size})', fontsize=14)

    # R² heatmap
    im0 = axes[0].imshow(mean_r2, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Mean R²')
    plt.colorbar(im0, ax=axes[0])

    # b1 heatmap (neuropil coefficient)
    vmax_b1 = np.nanpercentile(np.abs(mean_b1), 95)
    im1 = axes[1].imshow(mean_b1, cmap='coolwarm', vmin=-vmax_b1, vmax=vmax_b1)
    axes[1].set_title('Mean β1 (Neuropil)')
    plt.colorbar(im1, ax=axes[1])

    # b2 heatmap (red coefficient)
    vmax_b2 = np.nanpercentile(np.abs(mean_b2), 95)
    im2 = axes[2].imshow(mean_b2, cmap='coolwarm', vmin=-vmax_b2, vmax=vmax_b2)
    axes[2].set_title('Mean β2 (Red)')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')

    plt.tight_layout()

    if save_fig:
        save_path = p_dilation_results / 'qc_summary_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()
    # return fig


rec = ''
k_size=0
regression_name ='single_trial_regression'
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = ''

p_dilation_results = OUR_DIR_REGRESS / rec / regression_name / r'dilation_k={}'.format(k_size)
res_traces = np.load(p_dilation_results / f'{regression_name}_res_traces.npz')
qc_info = pd.read_pickle(p_dilation_results / f'{regression_name}_qc.pkl')
beh = pd.read_pickle(OUT_DIR_RAW_DATA / 'behaviour_profile' /  f'{rec}.pkl')

# Get and print valid grids
valid_grids = get_valid_grids(res_traces)
print(f"Valid grids: {valid_grids}")

for grid_y, grid_x in valid_grids[:10]:
    plot_regression_qc_check(res_traces, qc_info, beh,
                             grid_y=grid_y, 
                             grid_x=grid_x,
                             save_fig=True)