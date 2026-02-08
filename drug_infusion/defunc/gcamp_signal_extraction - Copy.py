# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 20:53:58 2025

@author: Phil Dong, Jingyu Cao
"""
# %% imports and definition
import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_centroids,
    calculate_mapping,
    fill_mapping,
    group_by_session,
    resolve_mapping,
)
from minian.motion_correction import apply_transform, estimate_motion
from tqdm.auto import tqdm

sys.path.append(r"Z:\Jingyu\Code\Python\concat_Phil\routine")
# from io import load_bin, load_footprint
from io_Phil import load_bin, load_footprint

PARAM_DIST = 5
PARAM_BASE_Q = 0.05

def set_window(wnd):
    return wnd == wnd.min()


def baseline_sub(sig):
    base = np.quantile(sig, PARAM_BASE_Q)
    return sig - base

# path and data structure initialisation 
from drug_infusion.rec_lst_infusion import selected_rec_lst
from drug_infusion.neuropil_mask import roi_map_to_neuropil_masks


#%% main 
OUT_DIR_RAW_DATA = r'Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals'
for ss_idx, date_ss in selected_rec_lst.iterrows() :
    temps = []
    
    t0 = time.time()
    
    anm = date_ss['anm']
    sessions = date_ss['session']
    date = date_ss['date']
    rec_id = anm+'-'+date
    
    print('{}_{}_processing...--------------------------'.format(anm, date))
    
    INT_PATH = OUT_DIR_RAW_DATA + rf'\{rec_id}'
    # FIG_PATH = r"Z:\Jingyu\2P_Recording\{}\{}\concat_anat_drd1\figs".format(anm, anm+'-'+date)
    
    if os.path.exists(os.path.join(INT_PATH, "sig_master_raw.nc")):
        print('{}_{}_results existed'.format(anm, date))
    else:
        os.makedirs(INT_PATH, exist_ok=True)
        # os.makedirs(FIG_PATH, exist_ok=True)
        
        for ss in sessions:
            session = date+'_'+ss
            p_suite2p_bin = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                anm+'-'+date, 
                                                                ss,
                                                                'suite2p',
                                                                 )
            p_suite2p_ops = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                anm+'-'+date, 
                                                                ss,
                                                                'suite2p_func_detec'
                                                                 )
            dat = load_bin(p_suite2p_bin,  p_suite2p_ops)
            temp = (
                dat.max("frame")
                .compute()
                .assign_coords(animal=anm, session=session)
            )
            temps.append(temp.rename("temps"))
        
    
        temps = xr.combine_nested([temps], ["animal", "session"]).chunk()
        shifts = estimate_motion(temps, dim="session").compute().rename("shifts")
        temps_sh = apply_transform(temps, shifts).compute().rename("temps_shifted")
        window = temps_sh.isnull().sum("session").rename("window")
        window, _ = xr.broadcast(window, temps_sh)
        window = xr.apply_ufunc(
            set_window,
            window,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
        )
        shift_ds = xr.merge([temps, shifts, temps_sh, window])
        # fig = px.imshow(shift_ds["temps_shifted"].squeeze(), facet_col="session")
        # fig.write_html(os.path.join(FIG_PATH, "temps_shifted.html"))
        
        try:
            shift_ds.to_netcdf(os.path.join(INT_PATH, "shift_ds.nc"))
        except PermissionError:
            print('Shift data already saved; skipped')
        
        print(r'compute templates and shifts: {:.2f} s'.format(time.time()-t0))
        # %% apply shifts
        # sscsv = pd.read_csv(IN_SS_CSV)
        shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
        A_shifted = []
        for ss in sessions:
            session = date+'_'+ss
            p_suite2p = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                anm+'-'+date, 
                                                                ss,
                                                                # 'suite2p',
                                                                'suite2p_func_detec'
                                                                 )
            temp = shift_ds["temps"].sel(animal=anm, session=session)
            # TODO: confirm whether the footprints correspond to cropped movie
            A = load_footprint(
                os.path.join(p_suite2p, "stat.feather"),
                temp.sizes["height"],
                temp.sizes["width"],
            )
            sh = shift_ds["shifts"].sel(animal=anm, session=session)
            A_sh = apply_transform(A, sh)
            A_shifted.append(A_sh)
            
        A_shifted = xr.combine_nested([A_shifted], ["animal", "session"])
        
        try:
            A_shifted.to_netcdf(os.path.join(INT_PATH, "A_shifted.nc"))
        except PermissionError:
            print('Shifted ROI masks already saved; skipped')
        
        print(r'apply shifts: {:.2f} s'.format(time.time()-t0))
        # %% compute mapping
        shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
        A_shifted = xr.open_dataarray(os.path.join(INT_PATH, "A_shifted.nc"))
        cents = calculate_centroids(A_shifted, shift_ds["window"])
        dist = calculate_centroid_distance(cents, index_dim=["animal"])
        dist_ft = dist[dist["variable", "distance"] < PARAM_DIST].copy()
        dist_ft = group_by_session(dist_ft)
        mappings = calculate_mapping(dist_ft)
        
        mappings_meta = resolve_mapping(mappings)
        mappings_meta_fill = fill_mapping(mappings_meta, cents)
        mappings_meta_fill.to_pickle(os.path.join(INT_PATH, "mappings_meta_fill.pkl"))
        print(r'compute mapping: {:.2f} s'.format(time.time()-t0))
        # %% compute master spatial footprint
        A_shifted = xr.open_dataarray(os.path.join(INT_PATH, "A_shifted.nc"))
        mappings = pd.read_pickle(os.path.join(INT_PATH, "mappings_meta_fill.pkl"))
        A_master = []
        for anm, map_anm in mappings.groupby(("meta", "animal")):
            A_anm = []
            for uid, Arow in tqdm(map_anm.iterrows(), total=len(map_anm)):
                A_ls = []
                for ss, sid in Arow["session"].dropna().items():
                    A_ls.append(
                        A_shifted.sel(animal=anm, session=ss, unit_id=sid).drop_vars("unit_id")
                    )
                curA = xr.concat(A_ls, "session").sum("session")
                curA = (curA / curA.sum()).assign_coords({"master_uid": uid})
                A_anm.append(curA)
            A_master.append(A_anm)
        A_master = xr.combine_nested(A_master, ["animal", "master_uid"])
        
        try:
            A_master.to_netcdf(os.path.join(INT_PATH, "A_master.nc"))
        except PermissionError:
            print('ROI masks already saved; skipped')
            
        print(r'compute master spatial footprint: {:.2f} s'.format(time.time()-t0))

        # %% extract signals
        # sscsv = pd.read_csv(IN_SS_CSV)
        A_master = xr.open_dataarray(os.path.join(INT_PATH, "A_master.nc"))
        shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
        sig_master = []
        sig_master_raw = []  # new 29 Jan 2026
        sig_master_neu_raw = [] # new 29 Jan 2026
        
        # sigs = []
        # extract rois
        for anm, Am in A_master.groupby("animal"):
            sigs = []  # bug? why is this here??
            sigs_raw = []  # new 29 Jan 2026
            sigs_neuropil = [] # new 29 Jan 2026
            for ss in sessions:
                session = date+'_'+ss
                p_suite2p_bin = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                    anm+'-'+date, 
                                                                    ss,
                                                                    'suite2p',
                                                                     )
                p_suite2p_ops = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                    anm+'-'+date, 
                                                                    ss,
                                                                    'suite2p_func_detec'
                                                                     )
                dat = load_bin(p_suite2p_bin,  p_suite2p_ops)
                sh = shift_ds["shifts"].sel(animal=anm, session=session)
                curA = apply_transform(Am, -sh)
                cur_sig = curA.dot(dat).compute()
                
                # new 29 Jan 2026
                cur_sig_raw = cur_sig.copy()
                sigs_raw.append(cur_sig_raw)
                
                cur_sig = xr.apply_ufunc(
                    baseline_sub,
                    cur_sig,
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                )
                sigs.append(cur_sig)
        
        # extract neuropils
        roi_map = np.array(A_master).squeeze().astype(bool)
        
        neuropil_map = roi_map_to_neuropil_masks(roi_map,
                                                 inner_neuropil_radius=3,
                                                 min_neuropil_pixels=350)
        
        A_neuropil = A_master.copy()
        A_neuropil.values = neuropil_map.reshape(A_master.shape)
        # Save to netCDF file in the same directory as A_master
        A_neuropil.to_netcdf(os.path.join(INT_PATH, "A_master_neu.nc"))
        
        for anm, Am in A_neuropil.groupby("animal"):
            sigs_neu_raw = [] # new 29 Jan 2026
            for ss in sessions:
                session = date+'_'+ss
                p_suite2p_bin = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                    anm+'-'+date, 
                                                                    ss,
                                                                    'suite2p',
                                                                     )
                p_suite2p_ops = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}\plane0".format(anm, 
                                                                    anm+'-'+date, 
                                                                    ss,
                                                                    'suite2p_func_detec'
                                                                     )
                # dat = load_bin(p_suite2p_bin,  p_suite2p_ops)
                sh = shift_ds["shifts"].sel(animal=anm, session=session)
                curA_neu = apply_transform(Am, -sh)
                cur_sig_neu = curA_neu.dot(dat).compute()
                
                # new 29 Jan 2026
                sigs_neu_raw.append(cur_sig_neu)

        sigs = xr.concat(sigs, "frame")
        sigs = sigs.assign_coords(frame=np.arange(sigs.sizes["frame"]))
        sig_master.append(sigs)
        
        sig_master = xr.concat(sig_master, "animal").compute()
        sig_master.to_netcdf(os.path.join(INT_PATH, "sig_master.nc"))
        
        # raw sigs without subtracting 5th, new 29 Jan 2026
        sigs_raw = xr.concat(sigs_raw, "frame")
        sigs_raw = sigs_raw.assign_coords(frame=np.arange(sigs_raw.sizes["frame"]))
        sig_master_raw.append(sigs_raw)
        
        sig_master_raw = xr.concat(sig_master_raw, "animal").compute()
        sig_master_raw.to_netcdf(os.path.join(INT_PATH, "sig_master_raw.nc"))
        
        # neuropil sigs, new 29 Jan 2026
        sigs_neu_raw = xr.concat(sigs_neu_raw, "frame")
        sigs_neu_raw = sigs_neu_raw.assign_coords(frame=np.arange(sigs_raw.sizes["frame"]))
        sig_master_neu_raw.append(sigs_neu_raw)
        
        sig_master_neu_raw = xr.concat(sig_master_neu_raw, "animal").compute()
        sig_master_neu_raw.to_netcdf(os.path.join(INT_PATH, "sig_master_neu_raw.nc"))
        
        print(r'extract signals: {:.2f} s'.format(time.time()-t0))
        


        

