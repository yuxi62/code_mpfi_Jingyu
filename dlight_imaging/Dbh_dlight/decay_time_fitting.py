# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:04:05 2026

@author: Jingyu Cao
"""
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np

def exp_decay(t, A, tau, C):
    return A * np.exp(-(t - t[0]) / tau) + C

def compute_tau_with_qc(t, y, peak_index=None, fit_win=None, min_points=5):
    """
    Fit single-exponential decay after the peak and return tau + fit quality.

    Returns a dict:
        {
            'tau': ...,
            'A': ...,
            'C': ...,
            'tau_se': ...,
            'r2': ...,
            'ss_res': ...,
            'n_points': ...
        }
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if peak_index is None:
        peak_index = int(np.nanargmax(y))

    t0 = t[peak_index]
    if fit_win is not None:
        mask = (t >= t0 + fit_win[0]) & (t <= t0 + fit_win[1])
    else:
        mask = (t >= t0)

    if mask.sum() < min_points:
        return {
            'peak': np.nan,
            'tau': np.nan,
            'A': np.nan,
            'C': np.nan,
            'tau_se': np.nan,
            'r2': np.nan,
            'ss_res': np.nan,
            'n_points': mask.sum(),
        }

    t_fit = t[mask]-t0
    y_fit = y[mask]

    C = y_fit.min()
    def fitfun(t, A, tau):
        return exp_decay(t, A, tau, C)

    try:
        popt, pcov = curve_fit(fitfun, t_fit, y_fit)
        A, tau = popt

        # predicted fit
        y_pred = exp_decay(t_fit, A, tau, C)

        # residuals and R²
        residuals = y_fit - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - y_fit.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # standard error of parameters from covariance matrix
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*3
        # A_se, tau_se, C_se = perr
        A_se, tau_se = perr

        return {
            'peak': float(t0),
            'tau': float(tau),
            'A': float(A),
            'C': float(C),
            'tau_se': float(tau_se),
            'r2': float(r2),
            'ss_res': float(ss_res),
            'n_points': int(len(y_fit)),
        }

    except Exception:
        return {
            'peak': np.nan,
            'tau': np.nan,
            'A': np.nan,
            'C': np.nan,
            'tau_se': np.nan,
            'r2': np.nan,
            'ss_res': np.nan,
            'n_points': int(len(y_fit)),
        }
def plot_tau_fit(t, y, fit_info, peak_index=None, fit_win=None, ax=None):
    """
    Plot original trace together with the fitted exponential decay.

    Parameters
    ----------
    t, y : arrays 
        Full trace data.
    fit_info : dict 
        Output from compute_tau_with_qc
    peak_index : int or None
        Peak index; if None, peak is found automatically.
    fit_win : tuple
        Fit window (same units as t).
    ax : matplotlib axis
        Optional axis to draw into.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,3), dpi=150)

    # Handle bad fits
    if fit_info['tau'] is np.nan or np.isnan(fit_info['tau']):
        ax.plot(t, y, color='gray', lw=1.2)
        ax.set_title("Fit failed / insufficient points")
        return fig, ax

    # Peak
    if peak_index is None:
        peak_index = np.argmax(y)

    t0 = t[peak_index]

    # Fit window
    if fit_win is not None:
        mask = (t >= t0 + fit_win[0]) & (t <= t0 + fit_win[1])
    else:
        mask = (t >= t0)
    # mask = (t >= t0 + fit_win[0]) & (t <= t0 + fit_win[1])
    t_fit = t[mask]
    y_fit = y[mask]

    # Reconstruct predicted fit
    A = fit_info['A']
    tau = fit_info['tau']
    C = fit_info['C']
    y_pred = A * np.exp(-(t_fit - t_fit[0]) / tau) + C
    # y_pred = A * np.exp(-(t_fit - t_fit[0]) / tau) + y_fit.min()

    # --- Plots ---
    ax.plot(t, y, label="Original trace", color="black", lw=1)
    ax.plot(t_fit, y_fit, "o", label="Data used for fit", ms=4, color="tab:green")
    ax.plot(t_fit, y_pred, "-", label=f"Fit (tau={tau:.2f})", color="tab:red", lw=2)

    # Peak marker
    ax.axvline(t0, color="blue", ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.legend(frameon=False)
    ax.set_title(f"Tau fit (R²={fit_info['r2']:.3f})")

    return fig, ax

#%% Main
if __name__ == '__main__':
    bef, aft = 2, 4
    baseline_window=(-1, 0)
    response_window=(0, 1.5)
    effect_size_thresh = 0.05
    amp_shuff_thresh_up = 95
    amp_shuff_thresh_down = 5
    
    OUT_DIR_FIG = Path(r"Z:\Jingyu\LC_HPC_manuscript\fig_Dbh_dlight")
    OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight")
    p_pooled_df = OUT_DIR_RAW_DATA / rf"df_population_pooled_pre{baseline_window}_post{response_window}_ES={effect_size_thresh}_shuff{amp_shuff_thresh_up}.pkl"
    df_roi_pool = pd.read_pickle(p_pooled_df)

    dlightUp_traces_dlight = 100*np.stack(df_roi_pool.loc[df_roi_pool['Up'], 'mean_profile'])

    # fit peak value and calculate tau
    taus = []
    peaks = []
    n_rois = dlightUp_traces_dlight.shape[0]
    for r in range(n_rois):
        roi_trace = dlightUp_traces_dlight[r, :]
        roi_trace = gaussian_filter1d(roi_trace, sigma=5)

        roi_tau_dic = compute_tau_with_qc(np.arange(30*(bef+aft)), roi_trace)
        taus.append(roi_tau_dic)
        # fig, ax = plot_tau_fit(np.arange(150), roi_trace, roi_tau_dic, fit_win=None)
        # fig.tight_layout()
        # plt.savefig(out_dir+r'\{}_tau_fitting_roi{}.png'.format(rec, r), dpi=200)
        # plt.close()
        
    df_taus = pd.DataFrame(taus)
    up_tau = df_taus.loc[(df_taus['r2']>0.7)&(df_taus['peak']<(bef+2)*30), 'tau']/30
    up_peak = df_taus.loc[(df_taus['r2']>0.7)&(df_taus['peak']<(bef+2)*30), 'peak']/30-bef
    # up_tau  = df_taus.loc[(df_taus['r2']>0.7), 'tau']/30
    # up_peak = df_taus.loc[(df_taus['r2']>0.7), 'peak']/30-bef
    print('tau = {:.3f}+/-{:.3f}'.format(up_tau.mean(), up_tau.sem()))
    print('peak = {:.3f}+/-{:.3f}'.format(up_peak.mean(), up_peak.sem()))

    # plot histogram
    all_up_roi_mean = np.nanmean(dlightUp_traces_dlight, axis=0)
    fit_info = compute_tau_with_qc(np.arange(30*(bef+aft)), all_up_roi_mean)

    xaxis = np.arange(30*(bef+aft))/30-bef   
    
    # plot decaying time histogram
    Q1 = np.nanpercentile(up_tau, 25)
    Q3 = np.nanpercentile(up_tau, 75)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.hist(up_tau, bins=30, range=(0, 4), facecolor='darkgreen', edgecolor='none', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(np.median(up_tau), color="darkgreen", ls="--", lw=1, alpha=0.8)
    ax.annotate(f"tau median = {up_tau.median():.3f}",
                xy=(0.5, 0.88), xycoords="axes fraction",
                ha="left", va="top")
    ax.annotate('tau mean = {:.3f}\n+/-{:.3f}'.format(up_tau.mean(), up_tau.sem()),
                xy=(0.5, 0.78), xycoords="axes fraction",
                ha="left", va="top")
    ax.annotate('IQR = [{:.3f}, {:.3f}] s'.format(Q1, Q3),
                xy=(0.5, 0.60), xycoords="axes fraction",
                ha="left", va="top")
    ax.set(ylabel='roi count', xlabel='tau (s)')
    plt.tight_layout()
    plt.savefig(OUT_DIR_FIG/r'pupulation_tau_hist_ES={}_amp={}.pdf'
                .format(effect_size_thresh, amp_shuff_thresh_up), dpi=300)
    plt.show()
    
    # plot peak time histogram
    Q1 = np.nanpercentile(up_peak, 25)
    Q3 = np.nanpercentile(up_peak, 75)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.hist(up_peak, bins=30, facecolor='steelblue', edgecolor='none', alpha=.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(np.median(up_peak), color="blue", ls="--", lw=1, alpha=0.6)
    ax.annotate(f"peak median = {up_peak.median():.3f}",
                xy=(0.5, 0.88), xycoords="axes fraction",
                ha="left", va="top")
    ax.annotate('peak mean = {:.3f}\n+/-{:.3f}'.format(up_peak.mean(), up_peak.sem()),
                xy=(0.5, 0.78), xycoords="axes fraction",
                ha="left", va="top")
    ax.annotate('IQR = [{:.3f}, {:.3f}] s'.format(Q1, Q3),
                xy=(0.5, 0.60), xycoords="axes fraction",
                ha="left", va="top")

    ax.set(ylabel='roi count', xlabel='peak time (s)')
    plt.tight_layout()
    plt.savefig(OUT_DIR_FIG/r'pupulation_peak_hist_ES={}_amp={}.pdf'
                .format(effect_size_thresh, amp_shuff_thresh_up), dpi=300)
    plt.show()