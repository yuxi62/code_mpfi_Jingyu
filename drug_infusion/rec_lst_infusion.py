# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 11:42:29 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd

exp = r'GCaMP8s_infusion'

rec_lst_infusion = [
{'anm': 'AC986', 'date': '20250606', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH_2']},
{'anm': 'AC985', 'date': '20250606', 'session': ['02', '04'], 'label': ['baseline', 'SCH']},
{'anm': 'AC986', 'date': '20250609', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH_2']},
{'anm': 'AC985', 'date': '20250609', 'session': ['02', '04'], 'label': ['baseline', 'ctrl'], 'conc': [np.nan, np.nan]},
{'anm': 'AC986', 'date': '20250610', 'session': ['02', '04'], 'label': ['baseline', 'SCH'],  'conc': [np.nan, 0.027]},
{'anm': 'AC986', 'date': '20250611', 'session': ['02', '04'], 'label': ['baseline', 'ctrl'], 'conc': [np.nan, np.nan]},
{'anm': 'AC986', 'date': '20250612', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.027, 0.027]},
{'anm': 'AC986', 'date': '20250613', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5]}, ## Brainphys + SCH, solution issue

{'anm': 'AC986', 'date': '20250616', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5]},
{'anm': 'AC988', 'date': '20250616', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5]},
{'anm': 'AC986', 'date': '20250617', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan]},
{'anm': 'AC988', 'date': '20250617', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan]},
# from here increase waiting time to 20 min
{'anm': 'AC986', 'date': '20250618', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5], 'latency': 20,}, 
{'anm': 'AC988', 'date': '20250618', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 0.5, 0.5], 'latency': 20,}, 
{'anm': 'AC986', 'date': '20250619', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1], 'latency': 20,}, 
{'anm': 'AC988', 'date': '20250619', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 1, 1], 'latency': 20,}, 
{'anm': 'AC986', 'date': '20250620', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan], 'latency': 20,},
{'anm': 'AC988', 'date': '20250620', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour issue, running speed, licks
{'anm': 'AC986', 'date': '20250621', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1], 'latency': 20,}, # behaviour issue, second sessions stop licking
# {'anm': 'AC988', 'date': '20250623', 'session': ['02', '04', ], 'label': ['baseline', 'SCH', ], 'conc': [np.nan, 1, 1]}, # behaviour issue, running speed, imaging quality not good
{'anm': 'AC986', 'date': '20250624', 'session': ['02', '04', '06'], 'label': ['baseline', 'SCH', 'SCH2'], 'conc': [np.nan, 1, 1], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250625', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250626', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # some long stops
# {'anm': 'AC986', 'date': '20250626', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]}, # missed some trials, bad imaging quality
{'anm': 'AC986', 'date': '20250627', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250627', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250628', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC986', 'date': '20250628', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 0.5,]}, #????????? don't know why commented before
{'anm': 'AC988', 'date': '20250629', 'session': ['02', '04', '06'], 'label': ['baseline', 'ctrl', 'ctrl2'], 'conc': [np.nan, np.nan, np.nan], 'latency': 20,},
{'anm': 'AC989', 'date': '20250630', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250701', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC988', 'date': '20250702', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250702', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250703', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC986', 'date': '20250703', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250704', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250706', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250707', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250708', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]},
{'anm': 'AC989', 'date': '20250709', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250710', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250711', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,}, 
{'anm': 'AC989', 'date': '20250712', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]},
{'anm': 'AC301', 'date': '20250713', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,]}, # acclimation
{'anm': 'AC989', 'date': '20250715', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250716', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC989', 'date': '20250717', 'session': ['02', '04',], 'label': ['baseline', 'propranolol',], 'conc': [np.nan, 1,], 'latency': 20,},
{'anm': 'AC996', 'date': '20250719', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},# behaviour not perfect
{'anm': 'AC996', 'date': '20250721', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},# behaviour not perfect
{'anm': 'AC996', 'date': '20250725', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250725', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC996', 'date': '20250726', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 0.5,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250726', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  # behaviour not perfect
{'anm': 'AC302', 'date': '20250727', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,},
{'anm': 'AC302', 'date': '20250728', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC996', 'date': '20250728', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC302', 'date': '20250729', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC302', 'date': '20250730', 'session': ['02', '04',], 'label': ['baseline', 'muscimol',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC302', 'date': '20250731', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 30,},  
{'anm': 'AC302', 'date': '20250801', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 30,},  
{'anm': 'AC310', 'date': '20250818', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, # behaviour not perfect
{'anm': 'AC310', 'date': '20250819', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250820', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
{'anm': 'AC310', 'date': '20250821', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 1,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250822', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250823', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
{'anm': 'AC310', 'date': '20250826', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 2,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250827', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 3,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250828', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 20,},  
{'anm': 'AC310', 'date': '20250829', 'session': ['02', '04',], 'label': ['baseline', 'ctrl',], 'conc': [np.nan, np.nan,], 'latency': 20,}, 
# {'anm': 'AC310', 'date': '20250830', 'session': ['02', '04',], 'label': ['baseline', 'SCH',], 'conc': [np.nan, 5,], 'latency': 20,},  # bad imaging quality
]

muscimol_imaging = [
{'anm': 'AC986', 'date': '20250614', 'session': ['02', '04', '06'], 'label': ['baseline', 'muscimol', 'muscimol2'], 'conc': [np.nan, 0.25, 0.25]},    
{'anm': 'AC985', 'date': '20250614', 'session': ['02', '04', '06'], 'label': ['baseline', 'muscimol', 'muscimol2'], 'conc': [np.nan, 0.25, 0.25]},        
    ]

#%% processed sessions metadata
SESSION_INFO_PATH = r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\infusion_session_info.parquet"
df_session_info = pd.read_parquet(SESSION_INFO_PATH)

# Filter for high-quality sessions
selected_rec_lst = df_session_info[
                    (df_session_info['perc_valid_trials_ss1']>0.6)&
                    (df_session_info['perc_valid_trials_ss2']>0.6)&
                    (df_session_info['latency']==20)&
                    (df_session_info['anm']!='AC996') # different strain: Ai14:D1R
                    ]