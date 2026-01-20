# Two-Photon Calcium Imaging Analysis for Drug Infusion Experiments

Analysis pipeline for investigating the effects of dopamine receptor antagonists on striatal neural populations using two-photon calcium imaging.

## Overview

This codebase processes and analyzes two-photon calcium imaging data from mice performing a reward-based running task, with pharmacological interventions including:
- **SCH23390**: D1 dopamine receptor antagonist (various concentrations: 0.027-5 μM)
- **Propranolol**: Beta-adrenergic receptor antagonist
- **Muscimol**: GABA-A receptor agonist
- **Control**: Vehicle solution

## Project Structure

```
code_mpfi_Jingyu/
├── common/                     # Shared utility functions
│   ├── plotting_functions_Jingyu.py  # Visualization library
│   ├── utils_basic.py          # GPU-accelerated signal processing
│   ├── utils_imaging.py        # Imaging utilities
│   ├── utils_behaviour.py      # Behavioral data utilities
│   └── shuffle_funcs.py        # Statistical shuffling functions
│
├── drug_infusion/              # Main analysis pipeline
│   ├── rec_lst_infusion.py     # Recording session metadata
│   ├── df_roi_info.py          # ROI data extraction & DFF calculation
│   ├── df_roi_info_drd1.py     # D1R neuron-specific analysis
│   ├── drd1_detection.py       # Automated cell matching (GCaMP-tdTomato)
│   ├── trial_selection.py      # Behavioral trial classification
│   ├── utils_infusion.py       # Drug infusion analysis utilities
│   ├── plot_functions.py       # Experiment-specific plotting
│   └── update_session_info.py  # Session metadata management
│
└── dLight_imaging/             # dLight imaging experiments (planned)
```

## Installation

### Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for large datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/code_mpfi_Jingyu.git
cd code_mpfi_Jingyu

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (adjust CUDA version as needed)
pip install cupy-cuda11x
```

## Data

**Note**: Data files are not included in this repository due to size constraints.

### Expected Data Structure
Behavioral data files should be placed in:
```
drug_infusion/behaviour_profile/
├── AC986-20250606-02.pkl
├── AC986-20250606-04.pkl
└── ...
```

### Data Format
Each `.pkl` file contains a dictionary with:
- `run_onsets`: Trial start times
- `run_offsets`: Trial end times
- `reward_times`: Reward delivery timestamps
- `lick_times`: Licking event timestamps
- `non_stop_trials`: Trial validity flags
- `non_fullstop_trials`: Trial validity flags

## Usage

### 1. Update Session Information
```python
from drug_infusion.update_session_info import update_session_info
from drug_infusion.rec_lst_infusion import rec_lst_infusion

# Process all recording sessions
for rec in rec_lst_infusion:
    df_session_info = update_session_info(df_session_info, rec)
```

### 2. Extract ROI Data
```python
from drug_infusion.df_roi_info import extract_roi_data
# Process calcium imaging data and calculate DFF traces
```

### 3. Analyze D1R Neurons
```python
from drug_infusion.drd1_detection import mser_detection
# Match GCaMP signals to tdTomato-labeled D1R neurons
```

## Key Analysis Steps

1. **Trial Validation**: Filter valid trials based on reward delivery and running behavior
2. **DFF Calculation**: Compute ΔF/F₀ from raw calcium fluorescence
3. **Cell Classification**: Identify D1R-expressing neurons via fluorescent labeling
4. **Response Quantification**: Calculate pre/post-stimulus response ratios
5. **Population Analysis**: Generate heatmaps and statistical summaries

## Dependencies

| Package | Purpose |
|---------|---------|
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| SciPy | Scientific computing |
| CuPy | GPU-accelerated arrays |
| OpenCV | Image processing |
| Matplotlib | Visualization |
| scikit-image | Image analysis |
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
- [Additional acknowledgments]
