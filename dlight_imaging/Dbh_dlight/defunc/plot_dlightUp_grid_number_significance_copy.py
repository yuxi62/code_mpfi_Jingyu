# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:06:40 2025

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
from common import plotting_functions_Jingyu as pf
save_fig = pf.save_fig
pf.mpl_formatting()
#%% PATHS AND PARAMS
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
OUR_DIR_REGRESS = OUT_DIR_RAW_DATA / 'regression_res'
OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")
    
baseline_window=(-1, 0)
response_window=(0, 1.5)
effect_size_thresh = 0.05
amp_shuff_thresh_up = 95
amp_shuff_thresh_down = 5
regression_name ='single_trial_regression'

p_pooled_df = OUT_DIR_RAW_DATA / rf"df_population_pooled_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.pkl"
df_pool = pd.read_pickle(p_pooled_df)
#%%
all_shuffle_ups = []
all_real_ups = []
all_shuffle_ups_perc = []
all_real_ups_perc = []

df_pool = pd.read_pickle(p_pooled_df)

for rec_id, roi_stats in df_pool.groupby('rec_id', sort=False):
    roi_stats = roi_stats.reset_index(drop=True)

    n_roi = len(roi_stats)
    if n_roi == 0:
        continue

    # real ups
    up_mask = roi_stats['Up'].to_numpy(dtype=bool)
    n_ups = up_mask.sum()
    n_ups_perc = n_ups / n_roi

    # --- stack shuffles into (n_roi, n_shuff) ---
    # each entry is same-length 1D array/list
    amps = np.vstack(roi_stats['shuff_response_amplitude'].to_numpy())
    effects = np.vstack(roi_stats['shuff_effect_size'].to_numpy())

    # per-ROI scalars -> (n_roi, 1) for broadcasting
    edge0 = (roi_stats['edge'].to_numpy() == 0)[:, None]
    amp_thresh = roi_stats['shuffle_amps_thresh_up'].to_numpy()[:, None]

    # boolean matrix: (n_roi, n_shuff)
    shuff_ups = edge0 & (amps > amp_thresh) & (effects > effect_size_thresh)

    # counts per shuffle: (n_shuff,)
    shuff_ups_count = shuff_ups.sum(axis=0)
    shuff_ups_count_perc = shuff_ups_count / n_roi

    all_shuffle_ups.append(shuff_ups_count)
    all_shuffle_ups_perc.append(shuff_ups_count_perc)
    all_real_ups.append(n_ups)
    all_real_ups_perc.append(n_ups_perc)

# stack across recs
all_shuffle_ups = np.stack(all_shuffle_ups)                 # (n_rec, n_shuff)
all_shuffle_ups_perc = np.stack(all_shuffle_ups_perc)       # (n_rec, n_shuff)
all_real_ups = np.asarray(all_real_ups)                     # (n_rec,)
all_real_ups_perc = np.asarray(all_real_ups_perc)           # (n_rec,)

# 95th percentile across shuffles for each rec
all_shuffle_ups_95 = np.percentile(all_shuffle_ups, 95, axis=1)
all_shuffle_ups_perc_95 = np.percentile(all_shuffle_ups_perc, 95, axis=1)

all_shuffle_ups_99 = np.percentile(all_shuffle_ups, 99, axis=1)
all_shuffle_ups_perc_99 = np.percentile(all_shuffle_ups_perc, 99, axis=1)
#%%
out_dir = r"Z:\Jingyu\LC_HPC_manuscript\figure5_Dbh_dLight"


fig, ax, stats = pf.plot_overlay_1bar(100*np.array(all_shuffle_ups_perc_95), 
                 100*np.array(all_real_ups_perc),
                 colname=['shuffle_95', 'data'],
                 ylabel='% roi',
                 bar_width=0.15,
                 point_size=5,
                 jitter=0.03,
                 # ylim=(0, 45),
                 annotation=True)
ax.annotate(f'shuff_99_mean: {100*np.nanmean(all_shuffle_ups_perc_99):.3f}', 
            xy=(0.4, 50), size=8)
save_fig(fig, OUT_DIR_FIG, r'dlightUp_grid_number_signicance_ES={}_amp={}.pdf'
            .format(effect_size_thresh, amp_shuff_thresh_up), save=0)

