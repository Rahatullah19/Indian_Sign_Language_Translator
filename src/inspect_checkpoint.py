from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import h5py

# Load weights or inspect file
file_path = 'checkpoints/model_epoch_best.weights.h5'

try:
    # Try loading as a complete model
    model = load_model(file_path)
    model.summary()
except:
    print("Could not load as a complete model. Inspecting as an HDF5 file...")

    # Inspect the HDF5 structure
    with h5py.File(file_path, 'r') as f:
        print("Keys in HDF5 file:", list(f.keys()))
        for key in f.keys():
            print(f"Contents under '{key}': {list(f[key].keys())}")
