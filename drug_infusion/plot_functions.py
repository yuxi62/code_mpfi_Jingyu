# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:41:47 2025

@author: Jingyu Cao
"""
import os
import sys
import numpy as np
from scipy.stats import sem
from scipy.ndimage import median_filter, gaussian_filter1d
import matplotlib.pyplot as plt
if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
import utils_Jingyu as utl

def plot_ratio_distribution(df_rec, colors=['steelblue', 'orange', 'purple'],
                            suffix=''):
    
    sessions = ['ss1', 'ss2']

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    rec = df_rec.loc[0]
    session_label = rec['label']
    fig.suptitle('{}_{}_{}_conc.={}{}'.format(rec['anm'], 
                                            rec['date'], 
                                            rec['label'][1], 
                                            rec['conc'][1],
                                            suffix),
                 size=12)
    ax = axs[0]
    for i, ss in enumerate(sessions):
        ss_ratios = df_rec[f'run_ratio_rsd_good_{ss}']
        ax.hist(ss_ratios, bins=50, range=(0,10), facecolor=colors[i], 
                edgecolor='none', alpha=.3, 
                label = session_label[i])
    ax.set(xlabel='run_ratio', ylabel='hist.')
    ax.legend(frameon=False, prop={'size':6}, loc=(.7, .8))
    
    ax = axs[1]
    for i, ss in enumerate(sessions):
        ss_ratios = df_rec[f'run_ratio_rsd_good_{ss}']
        sorted_ratios = np.sort(ss_ratios)
        cdf = np.linspace(0, 1, len(sorted_ratios))
    
        ax.plot(sorted_ratios, cdf, color=colors[i], alpha=0.8, lw=1.5,
                label=session_label[i])
    
    ax.set(xlabel='run_ratio', ylabel='CDF', 
           xlim = (0,10))
    ax.legend(frameon=False, prop={'size':6}, loc=(.7, .1))
    fig.tight_layout()
    
def plot_population_overview(rec, df_rec, session_info, bef, aft,
                             prefix=None, suffix='', 
                             f_out='', save_plot=1, sub_folder=1):
    
    anm_id = rec['anm']
    ss = rec['session']
    n_sessions = len(ss)
    date = rec['date']
    sessions = ['ss1', 'ss2', 'ss3'][:len(rec['session'])]
    session_label = rec['label']
    
    # if not os.path.exists(f_out):
    os.makedirs(f_out, exist_ok=True)
    if sub_folder:
        f_out = f_out+r'\{}-{}_{}_conc={}'.format(anm_id, 
                                                  date,
                                                  session_label[1],
                                                  rec['conc'][1])
        os.makedirs(f_out, exist_ok=True)
    # population heatmap
    fig, axs = plt.subplots(n_sessions, n_sessions+2, 
                            figsize=(2+(n_sessions+2)*2, 2+n_sessions*2), 
                            dpi=200)
    
    fig.suptitle('{}_{}_{}_conc.={}{}'.format(anm_id, 
                                            date, 
                                            session_label[1], 
                                            rec['conc'][1],
                                            suffix),
                 x=0.55, y=1, size=12
                 )
    
    colors = ['steelblue', 'orange', 'purple']
    
    # robust_sd_factor = 2
    for i, s in enumerate(sessions): # sorted by, each row
        for ii, ss in enumerate(sessions): # potted session, each column
            activity_profile = f'run_cal_profile_rsd_good_{ss}'
            ax = axs[i,ii]
            run_aligned_sorted = np.stack(df_rec.sort_values(by=f'run_ratio_rsd_good_{s}', 
                                                                 ascending=False)[activity_profile]
                                          ).squeeze()
            run_aligned_sorted_nr = utl.normalize(run_aligned_sorted)
            # run_aligned_sorted_sm = gaussian_filter1d(run_aligned_sorted_nr, sigma=1, axis=-1).get()
            # run_aligned_sorted_sm = robust_std_filter(run_aligned_sorted_sm, robust_sd_factor)
            tot_roi = run_aligned_sorted.shape[0]
            ax.imshow(run_aligned_sorted_nr, 
                      cmap='Greys', 
                      aspect='auto', 
                      extent=[-bef, aft, 0, tot_roi],
                      interpolation=None)
            ax.axvline(0, 0, lw=1, ls='--', color='darkred')
            ax.set(xlabel='time (s)', ylabel='roi sorted by {}'.format(s), 
                   title='{}_{}_n_good={}'.format(ss, session_label[ii], len(session_info[r'good_trials_'+ss])))
            
            xaxis=np.arange(30*(bef+aft))/30-bef
            ax1 = axs[i, -2]
            up_profile = np.stack(df_rec.loc[df_rec[f'pyrUp_{s}'], 
                                             activity_profile])
            # up_profile = utl.normalize(up_profile)
            up_profile = gaussian_filter1d(up_profile, sigma=1, axis=-1)
            # up_profile = robust_std_filter(up_profile, robust_sd_factor)
            up_mean = np.nanmean(up_profile , axis=0)
            up_sem = sem(up_profile , axis=0, nan_policy='omit')
            ax1.plot(xaxis,up_mean, c=colors[ii], label=f'{ss}')
            ax1.fill_between(xaxis, up_mean+up_sem, up_mean-up_sem,
                             facecolor=colors[ii], edgecolor='none', alpha=.3)
            ax1.legend(frameon=False, prop={'size':6}, loc=(.8, 1))
            
            ax1 = axs[i, -1]
            down_profile = np.stack(df_rec.loc[df_rec[f'pyrDown_{s}'], 
                                             activity_profile])
            down_profile = gaussian_filter1d(down_profile, sigma=1, axis=-1)
            # down_profile = robust_std_filter(down_profile, robust_sd_factor)
            down_mean = np.nanmean(down_profile , axis=0)
            down_sem = sem(down_profile , axis=0, nan_policy='omit')
            ax1.plot(xaxis, down_mean, c=colors[ii], label=f'{ss}')
            ax1.fill_between(xaxis, down_mean+down_sem, down_mean-down_sem,
                             facecolor=colors[ii], edgecolor='none', alpha=.3)
            ax1.legend(frameon=False, prop={'size':6}, loc=(.8, 1))
            
            n_ups = len(up_profile)
            n_downs = len(down_profile)
            ax.axhspan(n_downs, n_downs, color='tab:red')
            ax.axhspan(tot_roi-n_ups, tot_roi-n_ups, color='tab:red')
    fig.tight_layout()   
    
    if save_plot:
        plt.savefig(f_out+ r'\{}_{}-run_aligned_{}_conc.={}{}.png'.format(anm_id, 
                                                                          date, 
                                                                          session_label[1],
                                                                          rec['conc'][1],
                                                                          suffix), 
                      bbox_inches='tight')
        plt.close()        
    
    # df_pyUp = df_session.loc[df_session[f'pyrUp_{ss}']].sort_values(by=f'run_ratio_{ss}', ascending=False)
    # df_pyDown = df_session.loc[df_session[f'pyrDown_{ss}']].sort_values(by=f'run_ratio_{ss}', ascending=False)
    
    # tot_pyUp = len(df_pyUp)
    # tot_pyDown = len(df_pyDown)
    
    
    # fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    # ax.imshow(utl.normalize(np.stack(df_pyUp[f'run_cal_profile_{ss}'])), 
    #           cmap='Greys', 
    #           aspect='auto', 
    #           extent=[-bef, aft, 0, tot_pyUp],
    #           interpolation=None)
    # ax.set(title='{}_pyUP_dFF_GCaMP_{}'.format(session, ss))
    
    
    # fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
    # ax.imshow(utl.normalize(np.stack(df_pyDown[f'run_cal_profile_{ss}'])), 
    #           cmap='Greys', 
    #           aspect='auto', 
    #           extent=[-bef, aft, 0, tot_pyDown],
    #           interpolation=None)
    # ax.set(title='{}_pyDown_dFF_GCaMP_{}'.format(session, ss))
    
    # trial average
    # tmp_lst = []
    perc_pyup = []
    perc_pydown = []
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    colors = ['steelblue', 'orange', 'purple']
    xaxis = np.arange(30*(bef+aft))/30-bef
    
    # n_act = len(df_rec)
    for i, ss in enumerate(sessions):
        n_act = len(df_rec.loc[(df_rec[f'pyrAct_{sessions[i]}'])])
        
        activity_profile = f'run_cal_profile_rsd_good_{ss}'
        df_rec_up = df_rec.loc[(df_rec[f'pyrUp_{sessions[i]}'])]
        df_rec_up_profile = np.vstack(df_rec_up[activity_profile])
        # df_rec_up_profile = robust_std_filter(df_rec_up_profile, 3)
        ups_avg = np.nanmean(df_rec_up_profile, axis=0)
        ups_sem = sem(df_rec_up_profile, axis=0, nan_policy='omit')
        # tmp_lst.append(trial_avg)
        
        perc_pyup.append(len(df_rec_up)/n_act)
        
        df_rec_down = df_rec.loc[(df_rec[f'pyrDown_{sessions[i]}'])]
        df_rec_down_profile = np.vstack(df_rec_down[activity_profile])
        df_rec_down_profile
        # df_rec_down_profile = robust_std_filter(df_rec_down_profile, 3)
        downs_avg = np.nanmean(df_rec_down_profile, axis=0)
        downs_sem = sem(df_rec_down_profile, axis=0, nan_policy='omit')
        perc_pydown.append(len(df_rec_down)/n_act)
        
        axs[0].plot(xaxis, ups_avg, color=colors[i], label = session_label[i]+'_Ups')
        axs[0].fill_between(xaxis, ups_avg+ups_sem, ups_avg-ups_sem, facecolor=colors[i],
                            edgecolor='none', alpha=.3)
        axs[0].plot(xaxis, downs_avg, color=colors[i], label = session_label[i]+'_Downs', alpha=.6)
        axs[0].fill_between(xaxis, downs_avg+downs_sem, downs_avg-downs_sem, facecolor=colors[i],
                            edgecolor='none', alpha=.3)
        axs[0].set(ylabel='dF/F', xlabel='time from run (s)')
        axs[0].legend(frameon=False, prop={'size':6})
        
        
    # print(perc_pyup)
    axs[1].scatter(session_label, perc_pyup, color='darkred', label='pyUp')
    axs[1].set(ylabel='perc.')
    # tax = axs[1].twinx()
    axs[1].scatter(session_label, perc_pydown,  color='green', label='pyDown')
    # axs[1].set(ylabel='perc_pydown')
    axs[1].legend(frameon=False, prop={'size':6}, loc=(.8, 1))
    # tax.legend(frameon=False, prop={'size':6}, loc=(.45, 1))

    fig.suptitle('{}_{}_{}_conc.={}{}'.format(anm_id, 
                                            date, 
                                            session_label[1], 
                                            rec['conc'][1],
                                            suffix),
                 x=0.55, y=.9, size=12
                 )
    fig.tight_layout()
    
    if save_plot:
        plt.savefig(f_out+ r'\{}_{}-run_aligned_avg_{}_conc.={}{}.png'.format(anm_id, 
                                                                             date, 
                                                                             session_label[1],
                                                                             rec['conc'][1],
                                                                             suffix
                                                                             ), 
                    bbox_inches='tight')
                      
        plt.close()
    
    # ratio distribution
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    fig.suptitle('{}_{}_{}_conc.={}{}'.format(anm_id, 
                                            date, 
                                            session_label[1], 
                                            rec['conc'][1],
                                            suffix),
                 size=12)
    ax = axs[0]
    for i, ss in enumerate(sessions):
        ss_ratios = df_rec[f'run_ratio_rsd_good_{ss}']
        ax.hist(ss_ratios, bins=50, range=(0,10), facecolor=colors[i], 
                edgecolor='none', alpha=.3, 
                label = session_label[i])
    ax.set(xlabel='run_ratio', ylabel='hist.')
    ax.legend(frameon=False, prop={'size':6}, loc=(.7, .8))
    ax = axs[1]
    for i, ss in enumerate(sessions):
        ss_ratios = df_rec[f'run_ratio_rsd_good_{ss}']
        sorted_ratios = np.sort(ss_ratios)
        cdf = np.linspace(0, 1, len(sorted_ratios))
    
        ax.plot(sorted_ratios, cdf, color=colors[i], alpha=0.8, lw=1.5,
                label=session_label[i])
    
    ax.set(xlabel='run_ratio', ylabel='CDF', 
           xlim = (0,10))
    ax.legend(frameon=False, prop={'size':6}, loc=(.7, .1))
    fig.tight_layout()
    
    if save_plot:
        plt.savefig(f_out+r'\{}_{}-run_aligned_ratio_{}_conc.={}{}.png'.format(anm_id, date, 
                                                                      session_label[1],
                                                                      rec['conc'][1],
                                                                      suffix
                                                                      ),
                      bbox_inches='tight')                                                   
        plt.close()

def plot_population_heatmap(df_rec, rec_id, bef, aft, s, 
                            prefix='', suffix='', show_pyr_lines=True, session_for_sorting=None,
                            activity_profile = 'run_cal_profile_rsd',
                            ratio = 'run_ratio_rsd',
                            cross_sess_norm=False):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    activity_profile = f'{activity_profile}_{s}'
    
    # Sort data by run_ratio
    if session_for_sorting is None:
       session_for_sorting = s
       
    df_sorted = df_rec.sort_values(by=f'{ratio}_{session_for_sorting}', ascending=False)
    # df_sorted = df_rec.dropna(subset=[f'{ratio}_{session_for_sorting}']).sort_values(by=f'{ratio}_{session_for_sorting}')
    
    run_aligned_sorted = np.stack(df_sorted[activity_profile]).squeeze()
    run_aligned_sorted_nr = utl.normalize(run_aligned_sorted)
    run_aligned_sorted_sm = gaussian_filter1d(run_aligned_sorted_nr, sigma=1, axis=-1)
    # run_aligned_sorted_sm = robust_std_filter(run_aligned_sorted_sm, robust_sd_factor)
    tot_roi = run_aligned_sorted.shape[0]
    
    # Plot heatmap
    ax.imshow(run_aligned_sorted_sm, 
              cmap='Greys', 
              aspect='auto', 
              extent=[-bef, aft, 0, tot_roi],
              interpolation=None)
    
    # Add lines to indicate pyrUp and pyrDown boundaries
    if show_pyr_lines:
        # Get the sorted indices for pyrUp and pyrDown
        pyr_up_col = f'pyrUp_{s}'
        pyr_down_col = f'pyrDown_{s}'
        
        if pyr_up_col in df_sorted.columns and pyr_down_col in df_sorted.columns:
            # Count pyrUp cells (they should be at the top since sorted by ratio descending)
            n_pyr_up = df_sorted[pyr_up_col].sum()
            # Count pyrDown cells (they should be at the bottom)
            n_pyr_down = df_sorted[pyr_down_col].sum()
            
            # Draw horizontal line at the boundary of pyrUp cells
            if n_pyr_up > 0:
                ax.axhline(y=tot_roi-n_pyr_up, color='purple', linestyle='-', linewidth=1.5, 
                          alpha=0.7, label=f'pyrUp boundary (n={n_pyr_up})')
                print(n_pyr_up)
            # Draw horizontal line at the boundary of pyrDown cells
            if n_pyr_down > 0:
                ax.axhline(y=n_pyr_down, color='darkorange', linestyle='-', 
                          linewidth=1.5, alpha=0.7, label=f'pyrDown boundary (n={n_pyr_down})')
                print(n_pyr_down)
                
            # Add legend if lines were drawn
            # if n_pyr_up > 0 or n_pyr_down > 0:
            #     ax.legend(loc='upper right', fontsize=6, frameon=False)
    
    ax.axvline(0, 0, lw=1, ls='--', color='darkred')

    ax.set(xlabel='time (s)', 
           ylabel='roi sorted by {}'.format(session_for_sorting), 
           xlim=(-1, 4),
           title='{}_{}_{}'.format(prefix,
                                   rec_id,
                                   suffix)
           )
    fig.tight_layout()
    return fig

def plot_single_roi(rec, roi_idx, df_roi, session_info, ref_mean, bef, aft, f_out):
    
    anm_id = rec['anm']
    date = rec['date']
    session_label = rec['label']
    
    roi_map = np.stack(df_roi['map'])
    roi_mask = np.where(roi_map>0, 1, np.nan)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200); fig.tight_layout()
    plt.suptitle(r'{}_roi{}_pyr={}'
                .format(anm_id+'-'+date, roi_idx, 'NA'), size=8)
    ax.imshow(ref_mean,
              cmap='gray',
              vmin=np.percentile(ref_mean, 1),
              vmax=np.percentile(ref_mean, 99))
    ax.imshow(roi_mask,
              cmap='Set1', alpha=.3)
    ax.set_axis_off() 
    plt.savefig(f_out+r'\{}_{}_roi{}_FOV.png'.format(anm_id, date, roi_idx),
                  bbox_inches='tight')                                                   
    plt.close()
    
    raw_sess_array_ss1 = np.stack(df_roi['run_cal_profile_all_ss1'])
    raw_sess_array_ss2 = np.stack(df_roi['run_cal_profile_all_ss2'])
    raw_mean_ss1 = np.nanmean(raw_sess_array_ss1, axis=0)
    raw_mean_ss2 = np.nanmean(raw_sess_array_ss2, axis=0)        
    raw_sess_array_ss1_good = raw_sess_array_ss1.copy()
    raw_sess_array_ss1_good[session_info['bad_trials_ss1']] = 0
    raw_sess_array_ss2_good = raw_sess_array_ss2.copy()
    raw_sess_array_ss2_good[session_info['bad_trials_ss2']] = 0
    
    raw_sess_array_ss1_bad = raw_sess_array_ss1.copy()
    raw_sess_array_ss1_bad[session_info['good_trials_ss1']] = 0
    raw_sess_array_ss2_bad = raw_sess_array_ss2.copy()
    raw_sess_array_ss2_bad[session_info['good_trials_ss2']] = 0
    
    # raw_mean_ss1_good = np.nanmean(raw_sess_array_ss1[session_info['good_trials_ss1']], axis=0)
    # raw_mean_ss2_good = np.nanmean(raw_sess_array_ss2[session_info['good_trials_ss2']], axis=0)
    raw_mean_ss1_good = np.nanmean(raw_sess_array_ss1_good, axis=0)
    raw_mean_ss2_good = np.nanmean(raw_sess_array_ss2_good, axis=0)
    raw_mean_ss1_bad = np.nanmean(raw_sess_array_ss1[session_info['bad_trials_ss1']], axis=0)
    raw_mean_ss2_bad = np.nanmean(raw_sess_array_ss2[session_info['bad_trials_ss2']], axis=0)
    
    
    plt.rcParams['axes.titlesize'] = 10
    xaxis = np.arange(30*(bef+aft))/30-bef
    fig, axs = plt.subplots(3, 3, figsize=(6, 5), dpi=200); fig.tight_layout()
    plt.suptitle(r'{}_roi{}_pyr={}_active={}'
                .format(anm_id+'-'+date, roi_idx, 'NA', 'NA'), 
                y=1.05)
    ax = axs[0, 0]
    ax.imshow(raw_sess_array_ss1, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='ss1_n_all_trials={}'
           .format(raw_sess_array_ss1.shape[0]))
    ax = axs[0, 1]
    ax.imshow(raw_sess_array_ss1_good, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='n_good_trials_ss1={}'
           .format(len(session_info['good_trials_ss1'])))
    ax = axs[0, 2]
    ax.imshow(raw_sess_array_ss1_bad, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='n_bad_trials_ss1={}'
           .format(len(session_info['bad_trials_ss1'])))
    
    ax = axs[1, 0]
    ax.imshow(raw_sess_array_ss2, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='ss2_n_all_trials={}'
           .format(raw_sess_array_ss2.shape[0]))
    ax = axs[1, 1]
    ax.imshow(raw_sess_array_ss2_good, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='n_good_trials_ss2={}'
           .format(len(session_info['good_trials_ss2'])))
    ax = axs[1, 2]
    ax.imshow(raw_sess_array_ss2_bad, aspect='auto',
              extent=[-bef, aft,
                      raw_sess_array_ss1.shape[0], 0])
    ax.set(title='n_bad_trials_ss2={}'
           .format(len(session_info['bad_trials_ss2'])))
    
    ax = axs[0, 2]
    ax.set(title='n_bad_trials_ss1={}'
           .format(len(session_info['bad_trials_ss1'])))
    ax = axs[1, 2]
    ax.set(title='n_bad_trials_ss2={}'
           .format(len(session_info['bad_trials_ss2'])))
    
    ax = axs[2, 0]
    ax.plot(xaxis, raw_mean_ss1, color='green', lw=1)
    ax.plot(xaxis, raw_mean_ss2, color='orange', lw=1)
    ax = axs[2, 1]
    ax.plot(xaxis, raw_mean_ss1_good, color='green', lw=1)
    ax.plot(xaxis, raw_mean_ss2_good, color='orange', lw=1)
    ax = axs[2, 2]
    ax.plot(xaxis, raw_mean_ss1_bad, color='green', lw=1, alpha=.8)
    ax.plot(xaxis, raw_mean_ss2_bad, color='orange', lw=1, alpha=.8)
    plt.savefig(f_out+r'\{}_{}-roi{}_run_aligned_{}_conc.={}_raw.png'
                .format(anm_id, date, roi_idx, 
                        session_label[1],
                        rec['conc'][1],
                        ),
                  bbox_inches='tight')                                                   
    plt.close()
    
    rsd_sess_array_ss1 = np.stack(df_roi['run_cal_profile_all_rsd_ss1'])
    rsd_sess_array_ss2 = np.stack(df_roi['run_cal_profile_all_rsd_ss2'])
    rsd_mean_ss1 = np.nanmean(rsd_sess_array_ss1, axis=0)
    rsd_mean_ss2 = np.nanmean(rsd_sess_array_ss2, axis=0)
    rsd_sess_array_ss1_good = rsd_sess_array_ss1.copy()
    rsd_sess_array_ss1_good[session_info['bad_trials_ss1']] = 0
    rsd_sess_array_ss2_good = rsd_sess_array_ss2.copy()
    rsd_sess_array_ss2_good[session_info['bad_trials_ss2']] = 0
    
    rsd_sess_array_ss1_bad = rsd_sess_array_ss1.copy()
    rsd_sess_array_ss1_bad[session_info['good_trials_ss1']] = 0
    rsd_sess_array_ss2_bad = rsd_sess_array_ss2.copy()
    rsd_sess_array_ss2_bad[session_info['good_trials_ss2']] = 0
    
    # rsd_mean_ss1_good = np.nanmean(rsd_sess_array_ss1[session_info['good_trials_ss1']], axis=0)
    # rsd_mean_ss2_good = np.nanmean(rsd_sess_array_ss2[session_info['good_trials_ss2']], axis=0)
    rsd_mean_ss1_good = np.nanmean(rsd_sess_array_ss1_good, axis=0)
    rsd_mean_ss2_good = np.nanmean(rsd_sess_array_ss2_good, axis=0)
    rsd_mean_ss1_bad = np.nanmean(rsd_sess_array_ss1[session_info['bad_trials_ss1']], axis=0)
    rsd_mean_ss2_bad = np.nanmean(rsd_sess_array_ss2[session_info['bad_trials_ss2']], axis=0)

    xaxis = np.arange(30*(bef+aft))/30-bef
    fig, axs = plt.subplots(3, 3, figsize=(6, 5), dpi=200); fig.tight_layout()
    plt.suptitle(r'{}_roi{}_pyr={}_active={}'
                .format(anm_id+'-'+date, roi_idx, 'NA', 'NA'), 
                y=1.05)
    ax = axs[0, 0]
    ax.imshow(rsd_sess_array_ss1, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='ss1_n_all_trials={}'
           .format(rsd_sess_array_ss1.shape[0]))
    ax = axs[0, 1]
    ax.imshow(rsd_sess_array_ss1_good, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='n_good_trials_ss1={}'
           .format(len(session_info['good_trials_ss1'])))
    ax = axs[0, 2]
    ax.imshow(rsd_sess_array_ss1_bad, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='n_bad_trials_ss1={}'
           .format(len(session_info['bad_trials_ss1'])))
    
    ax = axs[1, 0]
    ax.imshow(rsd_sess_array_ss2, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='ss2_n_all_trials={}'
           .format(rsd_sess_array_ss2.shape[0]))
    ax = axs[1, 1]
    ax.imshow(rsd_sess_array_ss2_good, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='n_good_trials_ss2={}'
           .format(len(session_info['good_trials_ss2'])))
    ax = axs[1, 2]
    ax.imshow(rsd_sess_array_ss2_bad, aspect='auto',
              extent=[-bef, aft,
                      rsd_sess_array_ss1.shape[0], 0])
    ax.set(title='n_bad_trials_ss2={}'
           .format(len(session_info['bad_trials_ss2'])))
    
    ax = axs[0, 2]
    ax.set(title='n_bad_trials_ss1={}'
           .format(len(session_info['bad_trials_ss1'])))
    ax = axs[1, 2]
    ax.set(title='n_bad_trials_ss2={}'
           .format(len(session_info['bad_trials_ss2'])))
    
    ax = axs[2, 0]
    ax.plot(xaxis, rsd_mean_ss1, color='green', lw=1)
    ax.plot(xaxis, rsd_mean_ss2, color='orange', lw=1)
    ax = axs[2, 1]
    ax.plot(xaxis, rsd_mean_ss1_good, color='green', lw=1)
    ax.plot(xaxis, rsd_mean_ss2_good, color='orange', lw=1)
    ax = axs[2, 2]
    ax.plot(xaxis, rsd_mean_ss1_bad, color='green', lw=1, alpha=.8)
    ax.plot(xaxis, rsd_mean_ss2_bad, color='orange', lw=1, alpha=.8)
    plt.savefig(f_out+r'\{}_{}-roi{}_run_aligned_{}_conc.={}_rsd.png'
                .format(anm_id, date, roi_idx, 
                        session_label[1],
                        rec['conc'][1],
                        ),
                  bbox_inches='tight')                                                   
    plt.close()
    
