# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 12:24:42 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
from common import plotting_functions_Jingyu as pf
pf.mpl_formatting()
from common.utils_basic import normalize
# Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst
#%%
def classify_roi(df_stat):
    df_stat = df_stat.copy()
    df_stat['dlight_valid'] = df_stat['mean_profile'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    df_stat['red_valid'] = df_stat['mean_profile_red'].apply(lambda x: np.all(np.abs(x)<1, axis=-1))
    df_stat = df_stat.loc[(df_stat['dlight_valid'])&(df_stat['red_valid'])]

    edges = [0, 31]
    df_stat['edge'] = df_stat['roi_id'].apply(lambda rc: any(v in edges for v in rc))
    df_stat['shuffle_amps_thresh_up'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_stat['shuffle_amps_thresh_down'] = df_stat['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))
    
    df_stat['Up'] = np.where(
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']>df_stat['shuffle_amps_thresh_up'])&
                                (df_stat['effect_size']>0.05),
                                True, False)

    df_stat['Down'] = np.where(
                                ~(df_stat['edge'])&
                                (df_stat['response_amplitude']<df_stat['shuffle_amps_thresh_down'])&
                                (df_stat['effect_size']< -0.05),
                                True, False)
    return df_stat
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
DILATION_STEPS = (0, 2, 4, 6, 8, 10)

df_stats_all = pd.DataFrame()
for rec in rec_lst:
    print(f'loading: {rec}-------------------------------------------')
    anm, date, ss = rec.split('-')
    p_data = r"Z:\Jingyu\2P_Recording\{}\{}\{}\RegOnly".format(anm, f'{anm}-{date}', ss)
    for k in DILATION_STEPS:
        p_regression = (OUR_DIR_REGRESS / rec / regression_name 
                        / f'dilation_k={k}')
        p_stats = p_regression / f'{rec}_profile_stat.parquet'
        p_stats_red = p_regression / f'{rec}_profile_stat_red.parquet'
        roi_stats = pd.read_parquet(p_stats)
        mean_profile_red = pd.read_parquet(p_stats_red)['mean_profile']
        roi_stats['mean_profile_red'] = mean_profile_red
        roi_stats['rec_id'] = rec
        roi_stats = classify_roi(roi_stats)
        df_stats_all = pd.concat((df_stats_all, roi_stats))

#%% plot session mean +/- sem for different dialtion steps
df_dlightUp_all  = pd.DataFrame()
df_dilation_stat = pd.DataFrame()
for g_idx, df_rec in df_stats_all.groupby('rec_id'):
    # start with rois that are dLightUp without dilation
    dlightUp_mask = (df_rec['dilation_k'] == 0) & (df_rec['Up'])
    dlightUp_rois = df_rec.loc[dlightUp_mask, 'roi_id'].apply(tuple).values
    df_dlightUp_dilations = df_rec[
        df_rec['roi_id'].apply(tuple).isin(dlightUp_rois)
    ]
    df_dlightUp_all = pd.concat((df_dlightUp_all, df_dlightUp_dilations))
    
df_dilation_stat = df_dlightUp_all.groupby(['rec_id', 'dilation_k']
                                           )['response_amplitude'].agg(list).reset_index()

df_dilation_stat['response_amplitude_mean'] = df_dilation_stat['response_amplitude'].apply(lambda x: np.nanmean(x))
# df_dilation_stat['response_amplitude_sem'] = df_dilation_stat['response_amplitude'].apply(lambda x: sem(x, nan_policy='omit'))


#%% organise data 
rec_dict = {}
recname  = df_dilation_stat['rec_id'][0]
curr_dils = []; curr_amps = []
for _, dilation in df_dilation_stat.iterrows():
    if dilation['rec_id'] == recname:
        curr_dil = dilation['dilation_k']
        curr_amp = dilation['response_amplitude_mean']
        curr_dils.append(curr_dil)
        curr_amps.append(curr_amp)
    else:
        rec_dict[recname] = [curr_dils, curr_amps]
        curr_dils = []; curr_amps = []
        
        # new 
        recname  = dilation['rec_id']
        curr_dil = dilation['dilation_k']
        curr_amp = dilation['response_amplitude_mean']
        curr_dils.append(curr_dil)
        curr_amps.append(curr_amp)
        
from scipy.optimize import curve_fit
def _exp_decay(d, A, tau):
    return A * np.exp(-d / tau)

amp_axis = [0, 2, 4, 6, 8, 10]
amps     = {'0':  [],
            '2':  [],
            '4':  [],
            '6':  [],
            '8':  [],
            '10': []}

tau_results_all = []
for recname in rec_dict:
    if len(rec_dict[recname][0]) <= 3:
        print(f'{recname} has <= 3 dilation steps; skipped')
    else:
        try:
            amps['0'].append(rec_dict[recname][1][0])
            amps['2'].append(rec_dict[recname][1][1])
            amps['4'].append(rec_dict[recname][1][2])
            amps['6'].append(rec_dict[recname][1][3])
            amps['8'].append(rec_dict[recname][1][4])
            amps['10'].append(rec_dict[recname][1][5])
        except:
            pass
        
        R = np.array(rec_dict[recname][1])
        
        # define C as min over R 
        C = np.min(R)
        R0 = R - C
        
        # fit params 
        d_fit = rec_dict[recname][0]
        y_fit = R0
        
        try:
            popt, _ = curve_fit(
                _exp_decay,
                d_fit,
                y_fit,
                bounds=([0, 0.5], [np.inf, 20])
                )
            tau_hat = float(popt[1])
        except Exception:
            tau_hat = np.nan
    tau_results_all.append(tau_hat)


#%% histogram of spatial tau per time window
# hist_stem = save_stem / 'tau_histograms'
# hist_stem.mkdir(parents=True, exist_ok=True)

TAU_MAX = 20
BIN_WIDTH = 0.5
bins = np.arange(0, TAU_MAX + BIN_WIDTH, BIN_WIDTH)

vals = tau_results_all

fig, ax = plt.subplots(figsize=(2.4, 2.1))

ax.hist(
    vals,
    bins=bins,
    color='lightblue',
    edgecolor='none',
    linewidth=0.4
)

med = np.median(vals)
ax.axvline(med, color='teal', linestyle='--')
ax.text(
    0.9, 0.98,
    f'Median = {med:.2f}',
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=7,
    color='teal'
)

ax.spines[['top', 'right']].set_visible(False)
ax.set(
    xlabel=r'Spatial $\tau$ (px)',
    xlim=(-1,21),
    ylabel='Session count',
    )

# for ext in ['.pdf', '.png']:
#     fig.savefig(
#         hist_stem / f'tau_hist_{wname.replace(" ", "").replace("-", "")}{ext}',
#         dpi=300,
#         bbox_inches='tight'
#         )


#%% median radial decay curves with IQR
# lineplot_stem = save_stem / 'mean_radial_curves'
# lineplot_stem.mkdir(parents=True, exist_ok=True)

mean_curve = [np.nanmean(list(amps.values())[i]) for i in range(6)]
sem_curve  = [sem(list(amps.values())[i]) for i in range(6)]

fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=300)

ax.errorbar(
    amp_axis,
    mean_curve,
    yerr=sem_curve,
    fmt='_',              # short horizontal line for the mean
    color='teal',
    ecolor='lightblue',
    elinewidth=1.2,
    capsize=3,
    capthick=1.2,
    markersize=6
)

ax.spines[['top', 'right']].set_visible(False)
ax.set(
    xlabel='Dilation (px)',
    xlim=(-1,11),
    ylabel='Mean dLight RI',
)

# for ext in ['.pdf', '.png']:
#     fig.savefig(
#         lineplot_stem / f'mean_radial_curve_{wname.replace(" ", "").replace("-", "")}{ext}',
#         dpi=300,
#         bbox_inches='tight'
#     )