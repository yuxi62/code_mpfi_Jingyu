import numpy as np
import cupy as cp
import sys

print("Starting test...")
sys.stdout.flush()

# Test array
n_rois, n_frames = 1000, 10000
F_array = np.random.rand(n_rois, n_frames).astype(np.float32) * 100 + 50
# Add some NaN values
F_array[0, 100:200] = np.nan
F_array[10, 500:600] = np.nan

print(f'Test array shape: {F_array.shape}')
print(f'NaN count: {np.isnan(F_array).sum()}')
sys.stdout.flush()

# Import and test
from common.utils_imaging import nanpercentile_filter, nanpercentile_dff

# Test with small window first
print('Testing nanpercentile_filter with window_size=100...')
sys.stdout.flush()

arr_gpu = cp.asarray(F_array)
try:
    result = nanpercentile_filter(arr_gpu, 10, size=(1, 100))
    print(f'Result shape: {result.shape}')
    print(f'Result NaN count: {cp.isnan(result).sum().get()}')
    print('SUCCESS with small window!')
except Exception as e:
    import traceback
    print(f'ERROR: {e}')
    traceback.print_exc()

sys.stdout.flush()
print("Test complete.")
