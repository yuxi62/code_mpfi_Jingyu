# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:30:51 2024

@author: Jingyu Cao

MAKE SURE TO RUN THIS SCRIPT IN DEFAULT SUITE2P ENVIRONMENT!!

"""
#%%
import sys
sys.path.append("Z:\Jingyu\Code\Python")
import anm_list_running as anm

import suite2p
from pathlib import Path
import os
import numpy as np
from contextlib import redirect_stdout
from shutil import copytree, ignore_patterns

#%%

sessions = [
# 'AC986-20250614-02',
# 'AC986-20250614-04',
# 'AC986-20250614-06',

# 'AC988-20250614-02',
# 'AC988-20250614-04',
# 'AC988-20250614-06',

# 'AC985-20250612-02',
# 'AC985-20250612-04',
# 'AC985-20250612-06',


# 'AC986-20250703-02',
# 'AC986-20250703-04',
# 'AC986-20250624-06',

# 'AC988-20250702-02',
# 'AC988-20250702-04',
# 'AC988-20250629-06',

# 'AC989-20250717-02',
# 'AC989-20250717-04',
# 'AC989-20250628-06',

# 'AC314-20251103-02',
# 'AC314-20251103-04',
# 'AC314-20251104-02',
# 'AC314-20251104-04',

# 'AC316-20251104-02',
# 'AC316-20251104-04',

# 'AC314-20251107-02',
# 'AC314-20251107-04',
# 'AC314-20251113-02',
# 'AC314-20251113-04',
# 'AC314-20251118-02',
# 'AC314-20251118-04',
# 'AC314-20251119-02',
# 'AC314-20251119-04',
# 'AC314-20251122-02',
# 'AC314-20251122-04',

# 'AC317-20251110-02',
# 'AC317-20251110-04',
# 'AC317-20251113-02',
# 'AC317-20251113-04',
# 'AC317-20251118-02',
# 'AC317-20251118-04',
# 'AC317-20251119-02',
# 'AC317-20251119-04',
# 'AC317-20251121-02',
# 'AC317-20251121-04',
'AC317-20251125-02',
'AC317-20251125-04',
'AC317-20251126-02',
'AC317-20251126-04',
# 'AC302-20250728-02',
    ]
for s in sessions:
    
    p_data = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:])
    ops = np.load(r"Z:\Jingyu\2P_Recording\suite2p_ops\suite2p.npy", allow_pickle=True).item()
    p_out = os.path.join(p_data, 'suite2p')
    
    print(f'INFO: Running suite2p for {p_data}')
    
    # save output to 'run.log' file
    os.makedirs(p_out, exist_ok = True)
    p_log = p_out+r'/run_suite2p.log'    
    print(f'INFO: Saving text output to {p_log}')
    
    # run suite2p
    db = {
        'data_path': [ str(p_data) ],
        # 'save_path0': p_out,
        # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
    }
    
    with open(p_log, 'w') as f:
        with redirect_stdout(f):
            print(f'Running suite2p v{suite2p.version} from Spyder')
            suite2p.run_s2p(ops=ops, db=db)

            suite2p_ori = p_data+r'\suite2p'
            suite2p_new = p_data+r'\suite2p_func_detec'
            if not os.path.exists(suite2p_new+r'\plane0\stat.npy'):
                copytree(
                suite2p_ori,
                suite2p_new,
                # Any entry matching these names/patterns will be skipped
                ignore=ignore_patterns(
                    # directories to skip by name
                    "reg_tif", "reg_tif_chan2",
                    # files to skip by name or glob
                    "*.bin",
                ),
                dirs_exist_ok=True
                )
                    
    print(f'{s}------Finished------')

