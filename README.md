# Two-Photon Calcium Imaging Analysis Pipeline

Analysis pipeline for two-photon calcium imaging experiments, including drug infusion studies and dLight dopamine imaging.

## Overview

This codebase processes and analyzes two-photon calcium imaging data from mice performing reward-based behavioral tasks. It supports two main experimental paradigms:

### 1. Drug Infusion Experiments (GCaMP8s)
Pharmacological interventions to study neuromodulatory effects on striatal neural populations:
- **SCH23390**: D1 dopamine receptor antagonist (0.027-5 μM)
- **Propranolol**: Beta-adrenergic receptor antagonist
- **Muscimol**: GABA-A receptor agonist
- **Control**: Vehicle solution

### 2. dLight Dopamine Imaging
Direct measurement of dopamine dynamics using dLight sensors:
- **Axon-dLight**: Axonal dopamine release imaging
- **GECO-dLight**: Combined calcium and dopamine imaging with regression analysis

## Code structure

```
code_mpfi_Jingyu/
├── common/                          # Shared utility functions
│   ├── __init__.py
│   ├── plotting_functions_Jingyu.py # Visualization library (~2800 lines)
│   ├── utils_basic.py               # Basic utilities, trace filtering
│   ├── utils_imaging.py             # GPU-accelerated dF/F calculation
│   ├── utils_behaviour.py           # Behavioral data utilities
│   ├── trial_selection.py           # Trial classification functions
│   ├── robust_sd_filter.py          # Robust SD-based trace filtering
│   ├── event_response_quantification.py  # Event-aligned response analysis
│   ├── plot_single_trial_function.py     # Single trial visualization
│   ├── shuffle_funcs.py             # Statistical shuffling
│   └── mask/                        # ROI mask utilities
│       ├── generate_masks.py
│       ├── neuropil_mask.py
│       └── utils_mask.py
│
├── drug_infusion/                   # GCaMP drug infusion pipeline
│   ├── __init__.py
│   ├── rec_lst_infusion.py          # Recording session metadata (80+ sessions)
│   ├── session_metadata.py          # Session info management
│   ├── gcamp_signal_extraction.py   # ROI extraction from Suite2P
│   ├── process_calcium_traces.py    # dF/F calculation pipeline
│   ├── utils_infusion.py            # Drug response analysis utilities
│   ├── plot_functions.py            # Population heatmaps, distributions
│   ├── plot_pyrUp_Down_stats.py     # Pyramidal cell response statistics
│   ├── plot_pyrUp_Down_single_session.py  # Single session analysis
│   ├── plot_behaviour_trace.py      # Behavioral trace visualization
│   ├── run-suite2p_GCaMP.py         # Suite2P batch processing
│   └── defunc/                      # Deprecated scripts
│
├── dlight_imaging/                  # dLight dopamine imaging
│   ├── __init__.py
│   ├── session_selection.py         # Session filtering utilities
│   ├── session_selection_geco.py    # GECO-specific session selection
│   │
│   ├── Dbh_dlight/                  # DBH-Cre axon dLight imaging
│   │   ├── recording_list.py        # Session metadata
│   │   ├── plot_grid_single_session_profile.py
│   │   ├── plot_grid_population_profile.py
│   │   ├── plot_dilated_grid_stat.py
│   │   ├── plot_dlightUp_grid_number_significance.py
│   │   ├── decay_time_fitting.py    # Dopamine decay kinetics
│   │   └── whole_FOV_correlation.py
│   │
│   ├── geco_dlight/                 # GECO + dLight dual imaging
│   │   ├── recording_list.py
│   │   └── plot_roi_population_profile_geco.py
│   │
│   ├── regression/                  # Regression-based signal unmixing
│   │   ├── __init__.py
│   │   ├── align_beh_imaging.py     # Behavior-imaging alignment
│   │   ├── utils_regression.py      # Regression utilities
│   │   ├── regression_axon_dlight.py
│   │   ├── regression_geco_dlight.py
│   │   ├── run_response_stats_axon_dlight.py
│   │   ├── run_response_stats_geco_dlight.py
│   │   └── utils_regression_geco/   # GECO-specific regression
│   │       ├── Extract_dlight_masked_GECO_ROI_traces.py
│   │       ├── Regression_Red_From_Green_ROIs_geco.py
│   │       └── Regression_Red_From_Green_ROIs_Single_Trial_geco.py
│   │
│   └── run-suite2p-*.py             # Suite2P configuration scripts
│
└── requirements.txt
```

## Installation

### Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for large datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/yuxi62/code_mpfi_Jingyu.git
cd code_mpfi_Jingyu

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (adjust CUDA version as needed)
pip install cupy-cuda11x
```

### For Spyder/Interactive Use

```python
import sys
if r"Z:\Jingyu\code_mpfi_Jingyu" not in sys.path:
    sys.path.insert(0, r"Z:\Jingyu\code_mpfi_Jingyu")

# Then import modules
from common.utils_imaging import percentile_dff
from drug_infusion.rec_lst_infusion import rec_lst_infusion
```

## Usage

### Drug Infusion Analysis

```python
# 1. Load session metadata
from drug_infusion.rec_lst_infusion import rec_lst_infusion, df_session_info

# 2. Process calcium traces
from drug_infusion.utils_infusion import load_profile, calculate_ratio, sort_response

# 3. Analyze responses
df_profile = load_profile(rec)
df_profile = calculate_ratio(df_profile, session='ss2', ...)
df_profile = sort_response(df_profile, thresh_up=1.5, thresh_down=0.67)
```

### dLight Imaging Analysis

```python
# 1. Load recording list
from dlight_imaging.Dbh_dlight.recording_list import rec_lst

# 2. Run regression to separate dLight from motion artifacts
from dlight_imaging.regression.regression_axon_dlight import run_regression

# 3. Quantify run-onset responses
from dlight_imaging.regression.run_response_stats_axon_dlight import analyze_responses
```

## Key Analysis Features

### Signal Processing
- **percentile_dff()**: GPU-accelerated dF/F calculation using rolling percentile baseline
- **trace_filter_sd()**: Outlier removal based on standard deviation threshold
- **robust_sd_filter()**: Robust median-based filtering

### Trial Selection
- Automatic trial validation (reward delivery, running behavior)
- Configurable exclusion criteria (first N trials, stopped trials)

### Response Classification
- **pyrUp/pyrDown**: Cells with increased/decreased run-onset response
- Threshold-based classification with configurable ratios

### Visualization
- Population heatmaps sorted by response magnitude
- Mean trace plots with SEM shading
- Paired/unpaired statistical comparisons

## Dependencies

| Package | Purpose |
|---------|---------|
| NumPy | Numerical computing |
| Pandas | Data manipulation & parquet I/O |
| SciPy | Scientific computing |
| CuPy | GPU-accelerated arrays |
| OpenCV | Image processing |
| Matplotlib | Visualization |
| scikit-image | Image analysis |
| xarray | NetCDF data loading |
| tqdm | Progress bars |

## Author

Jingyu Cao
Max Planck Florida Institute for Neuroscience

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```
[Citation information to be added upon publication]
```

## Acknowledgments

- Max Planck Florida Institute for Neuroscience
- Wang Lab