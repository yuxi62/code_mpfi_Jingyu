# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:55:42 2025

@author: Jingyu Cao
"""

DEFAULT_COLORS = {
    'ch1': 'green',
    'ch2': 'red',
    'ch3': 'blue',
    'run': 'orange',
    'reward': '#12d5e3',
    'speed': '#649dcc',
    'lick': '#493991',
    }
DEFAULT_LABELS = {
    'ch1': 'dLight',
    'ch2': 'Ai14',
    'ch3': 'pix_correlation'
    }
import numpy as np
from tqdm import tqdm
import plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

# bef, aft = 1, 3

def plot_single_trial_html_roi(beh, ch1=None, ch2=None, ch3=None, colors=None, labels=None, shared_yaxis=None):
        """
        Plot single trial with multiple channels.

        Parameters
        ----------
        ch1, ch2, ch3 : array-like, optional
            Data arrays for each channel.
        colors : dict, optional
            Override specific colors. Only pass keys you want to change.
            Example: {'ch1': 'blue'} changes ch1 color, keeps others as default.
        labels : dict, optional
            Override specific labels. Only pass keys you want to change.
            Example: {'ch1': 'GCaMP'} changes ch1 label, keeps others as default.
        shared_yaxis : list of str, optional
            List of channel names to plot on the same y-axis.
            Examples:
                - ['ch1', 'ch2'] - share ch1 and ch2 on primary y-axis
                - ['ch1', 'ch2', 'ch3'] - share all three on primary y-axis
                - ['ch2', 'ch3'] - share ch2 and ch3 (ch2 becomes primary if ch1 not provided)
            Default is None (each channel on separate y-axis).
        """
        # Merge user-provided colors/labels with defaults
        colors = {**DEFAULT_COLORS, **(colors or {})}
        labels = {**DEFAULT_LABELS, **(labels or {})}

        time_s = np.arange(len(beh['frame_times']))/30

        # Determine which channels share y-axis
        if shared_yaxis is None:
            shared_yaxis = []

        # Determine y-axis assignment for each channel
        # Channels in shared_yaxis will use the same y-axis ('y')
        ch1_yaxis = 'y'
        ch2_yaxis = 'y' if 'ch2' in shared_yaxis else 'y2'
        ch3_yaxis = 'y' if 'ch3' in shared_yaxis else 'y5'

        # Plot
        run_onsets = beh['run_onsets']
        rewards = beh['rewards']
        lick_times = beh['lick_times']
        upsampled_speeds = beh['upsampled_speeds']
        speed_t = beh['speed_times']
        fig = go.Figure()

        if ch1 is not None:
            # F trace
            fig.add_trace(go.Scatter(
                x=time_s,
                y=ch1,
                mode='lines',
                name=labels['ch1'],
                yaxis=ch1_yaxis,
                line=dict(color=colors['ch1']),# default y-axis
            ))

        if ch2 is not None:
            # Channel 2 trace (right y-axis or shared)
            fig.add_trace(go.Scatter(
                x=time_s,
                y=ch2,
                mode='lines',
                name=labels['ch2'],
                yaxis=ch2_yaxis,
                line=dict(color=colors['ch2']),# assign to secondary y-axis or shared
            ))
        
        # addtional signal
        if ch3 is not None:
            fig.add_trace(go.Scatter(
                x=time_s,
                y=ch3,
                # y=ch2_dff[roi, :],
                mode='lines',
                name=labels['ch3'],
                yaxis=ch3_yaxis,
                line=dict(color=colors['ch3']),# assign to secondary y-axis or shared
            ))
        
        
        # Add speed plot
        fig.add_trace(go.Scatter(
        x=speed_t,
        y=upsampled_speeds,
        mode='lines',
        name='Speed',
        yaxis='y3',  # Assign to a new y-axis
        line=dict(color=colors['speed'])
        ))
        
        
        # Add vertical lines for run-onset and reward
        shapes = []
        
        # Run-onset lines
        for t in run_onsets:
            shapes.append(dict(
                type="line",
                x0=t, x1=t,
                y0=0, y1=1,
                xref='x',
                yref='paper',  # full height regardless of y-axis scale
                line=dict(color=colors['run'], width=2),
            ))
        
        # Reward lines 
        for t in rewards:
            shapes.append(dict(
                type="line",
                x0=t, x1=t,
                y0=0, y1=1,
                xref='x',
                yref='paper',
                line=dict(color=colors['reward'], width=2),
            ))
        
        # plot licks
        lick_y = np.tile([0.0, 1.0, None], len(lick_times))  # full height in the lick axis
        lick_x = np.repeat(lick_times, 3)
        
        fig.add_trace(go.Scattergl(
            x=lick_x,
            y=lick_y,
            mode='lines',
            line=dict(color=colors['lick'], width=1),
            name='Licks',
            showlegend=False,
            hoverinfo='skip',
            yaxis='y4'
        ))
        fig.update_layout(
        yaxis4=dict(
            domain=[0.85, 0.90],    # place at the top 5% of the figure
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title=None,
            fixedrange=True,
            matches=None
        )
    )
        
        # # Add shapes to layout
        fig.update_layout(shapes=shapes)
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # invisible point
            mode='lines',
            line=dict(color=colors['run'], 
                      #dash='dot'
                      ),
            name='run-onset'
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # invisible point
            mode='lines',
            line=dict(color=colors['reward'], 
                      #dash='dot'
                      ),
            name='reward'
        ))
        
        # Determine y-axis title for primary axis based on shared channels
        shared_labels = []
        if ch1 is not None:
            shared_labels.append(labels['ch1'])
        if ch2 is not None and 'ch2' in shared_yaxis:
            shared_labels.append(labels['ch2'])
        if ch3 is not None and 'ch3' in shared_yaxis:
            shared_labels.append(labels['ch3'])
        yaxis_title = ' / '.join(shared_labels) if shared_labels else ''

        # add speed to layout
        fig.update_layout(
        xaxis=dict(domain=[0.0, 0.88]),  # shift plot area left to make space
        yaxis=dict(
            title=yaxis_title,
            side='left',
            showgrid=True,
            autorange=True
        ),

        yaxis3=dict(
            domain=[0.0, 0.5],
            title="Speed",
            overlaying='free',
            side='right',
            anchor='free',
            position=1,          # far to the right
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor=colors['speed'],
            tickfont=dict(color=colors['speed']),
            # titlefont=dict(color='blue'),
            fixedrange=False,
            layer='above traces'    # ← Important: ensures axis line appears on top
        ),
    )
        if ch2 is not None and 'ch2' not in shared_yaxis:
            fig.update_layout(
            yaxis2=dict(
                title=labels['ch2'],
                side='right',
                overlaying='y',
                position=0.93,
                showgrid=False,
                autorange=True
            ))
        if ch3 is not None and 'ch3' not in shared_yaxis:
            fig.update_layout(
            yaxis5=dict(
                title=labels['ch3'],
                side='right',
                overlaying='y',
                position=0.88,
                showgrid=False,
                autorange=True
            ),
            )
    
        
        # trial number annotation
        # trial_times = [(rewards[t] + run_onsets[t])/2 for t in range(len(rewards))]
        trial_times = [run_onsets[t] + 1 if run_onsets[t]>0 else rewards[t]+5 for t in range(len(rewards))]
        for t, t_times in enumerate(trial_times):
            fig.add_annotation(
                x=t_times,
                y=1.05,  # Position slightly above top (use `yref='paper'`!)
                yref='paper',
                text='tiral_{}'.format
                (t),
                showarrow=False,
                font=dict(size=10, color="black"),
                align="center",
                bgcolor="rgba(255,255,255,0.7)",  # semi-transparent background
                bordercolor="black",
                borderwidth=1
            )
        
        # add slider bar
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=10, label="10s", step="second", stepmode="backward"),
                        dict(count=60, label="1m", step="second", stepmode="backward"),
                        dict(step="all")
                    ]
                )
            )
        )
            
        fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
        
        # fig.show()
        return fig

def align_frame_behaviour(beh):
    lick_times = [np.vstack(trial)[:,0] if len(trial)>0 else np.nan
                 for trial in beh['lick_times']
                 ]
    lick_times = np.hstack(lick_times)
    lick_times = lick_times[(lick_times-beh['frame_times'][0]>0)&
                            (lick_times-beh['frame_times'][-1]<0)]
    lick_times = (lick_times-beh['frame_times'][0])/1000
    
    upsampled_speeds = beh['upsampled_speed_cm_s']
    upsampled_time = beh['upsampled_timestamps_ms']
    
    idx_first_frame = np.abs(upsampled_time - beh['frame_times'][0]).argmin()
    idx_last_frame = np.abs(upsampled_time - beh['frame_times'][-1]).argmin()
    
    # align upsampled speed to frames
    upsampled_speeds = upsampled_speeds[idx_first_frame:idx_last_frame]
    
    # Run-onset and reward times (in seconds)
    run_onsets = [t/30 for t in beh['run_onset_frames']]
    rewards = [t/30 for t in beh['reward_frames']]
    
    speed_t = np.linspace(0, upsampled_speeds.shape[0] / 1000, upsampled_speeds.shape[0])  # Speed time vector at 1000 Hz
    
    tmp_beh = {'run_onsets': run_onsets,
               'rewards': rewards,
               'upsampled_speeds': upsampled_speeds,
               'lick_times': lick_times,
               'speed_times': speed_t,
               'frame_times': beh['frame_times'],
               }
    
    return tmp_beh

#%%
if __name__ == "__main__":
    import pandas as pd
    rec = r'AC986-20250620-04'
    anm, date, ss = rec.split('-')
    p_beh = r"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\behaviour_profile\{}.pkl".format(rec)
    beh = pd.read_pickle(p_beh)
    beh = align_frame_behaviour(beh)

    fig = plot_single_trial_html_roi(beh)
    fig.write_html(r"Z:\Jingyu\Code\Python\2p_SCH23390_infusion\temp_plots\{}_single_trial_beh.html".format(rec))
    fig.show()
    
    