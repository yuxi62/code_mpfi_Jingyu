# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:30:51 2024

@author: Jingyu Cao

run dopamine dataset wwtih suite2p-wang-lab, plot iterations of ROI detection in plot_roi_iterations.py

"""
import sys
if "Z:\Jingyu\Code\Python" not in sys.path:
    sys.path.append("Z:\Jingyu\Code\Python")
import anm_list_running as anm
from utils_Jingyu import send_notification
import traceback

import suite2p
from pathlib import Path
import os
import numpy as np
from contextlib import redirect_stdout
#%% params and sessions
# ops = np.load(r"Z:\Jingyu\2P_Recording\AC918\AC918-20231028\04\ROI_detection_test_2.0\suite2p\plane0\ops.npy", allow_pickle=True).item()
ops = np.load(r"Z:\Jingyu\2P_Recording\suite2p_ops\RegOnly.npy", allow_pickle=True).item()
# ops["input_format"]=="binary"
# ops['sparse_mode'] = True
ops['roidetect'] = 0
ops['reg_tif']=1
ops['reg_tif_chan2']=1
ops['align_by_chan']= 1 # align by dLight channel, Jingyu, 5/30/2025
ops['nonrigid']=0 # Jingyu, 5/30/2025
# tmp_lst = ['AC977-20250501-02',]
# tmp_lst = anm.AC964 + anm.AC967 + anm.AC969 + anm.AC963 + anm.AC970 + anm.AC966
tmp_lst = anm.AC992 + anm.AC991 +anm.AC304+anm.AC305
# exp='axon-GCaMP_RdLight'
exp = r'dlight_GECO_Drd1'
#%% Main
reg_lst = []
reg_f_lst = []
for s in tmp_lst:
    try:
        if 'AC971' in s:
            ops['functional_chan']=2
        else:
            ops['functional_chan']=1
        
        if 'AC977' in s:
            ops['align_by_chan']=1
        
        if exp=='axon-GCaMP_RdLight':
            ops['align_by_chan']=2
        
        if exp == r'dlight_GECO_Drd1':
            ops['functional_chan']=2
            ops['roidetect'] = 1
        
        anm = s[:5]
        date = s[6:14]
        ss = s[15:]
        p_data = os.path.join(r'Z:\Jingyu\2P_Recording',anm, anm+'-'+date, ss)

    # ops['do_extraction'] = False
                
        p_out = p_data + r'\rigid_reg'
        
        if os.path.exists(p_out+r'\suite2p\plane0\data.bin') == False:
            os.makedirs(p_out, exist_ok=True)
            # ops['path_roi_iterations'] = p_outops['save_path0']
            ops['save_path0'] = p_out
            
            
            print('INFO: Running suite2p-wang-lab for {}'.format(p_data))
            
            # save output to 'run.log' file
            os.makedirs(p_out, exist_ok = True)
            p_log = p_out+r'/run_suite2p-wang-lab.log'    
            print(f'INFO: Saving text output to {p_log}')
            
            # run suite2p
            db = {
                'data_path': [ str(p_data) ],
                'save_path0': p_out,
                # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
            }
            
            with open(p_log, 'w') as f:
                with redirect_stdout(f):
                    print(f'Running suite2p v{suite2p.version} from Spyder')
                    suite2p.run_s2p(ops=ops, db=db)
                
                print('Finished')
                
            subject = "Registrations for\n{}\nFinished Successfully".format(s)
            message = "Your Python script has finished running successfully."
            
            reg_lst.append(s)
            # try:
            #     send_notification(subject, message)
            #     print("Notification sent.")
            # except Exception as notify_err:
            #     print("Failed to send notification email:")
            #     print(notify_err)
        else:
            print(f'--------{s}_RegOnly finished already-------------')
            
    except Exception as e:
        print(f'Error Raised-----------{s}-----------')
        subject = "suite2p registration Error Occurred for {}".format(s)
        message = f"Your Python script encountered an error:\n\n{traceback.format_exc()}"
        reg_f_lst.append(s)
        try:
            send_notification(subject, message)
            print("Notification sent.")
        except Exception as notify_err:
            print("Failed to send notification email:")
            print(notify_err)
        
subject = "All Sessions - Registrations finished"
message = "finished session list:\n{}\nerror sessions:\n{}".format(reg_lst, reg_f_lst)
send_notification(subject, message)