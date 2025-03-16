import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from models.cnn_1d import cnn_1d
from dataset import hyperspectral_dataset

#######################################################################################
#######################################################################################
#######################################################################################

"""
infer.py - Hyperspectral Image Classification Inference Script

This script performs pixel-wise classification on a raw hyperspectral image using a pre-trained 
1D CNN model. It loads the trained model, processes the input image, runs inference on each pixel, 
and saves the predicted classification as a NumPy array.

Main Steps:
1. **Setup and Model Loading**:
   - Loads the trained CNN model (`best_model.pth`) and prepares it for inference.
   - Uses CUDA (GPU) if available, otherwise defaults to CPU.
   - Fixes potential `_orig_mod.` prefix issues when loading model state.

2. **Data Processing**:
   - Reads a raw `.bip` hyperspectral image from the specified path.
   - Converts the image into a PyTorch tensor and reshapes it to match the model's input format.

3. **Inference (Pixel-Wise Classification)**:
   - Iterates through each pixel in the image and passes it through the model.
   - Predicts one of three classes: **Land, Sea, or Cloud**.
   - Stores predictions in a NumPy array.

4. **Saving Predictions**:
   - Reshapes the predictions into a 2D image of shape `(598, 1092)`.
   - Saves the output as a `.npy` file in the appropriate directory.

Key Features:
- Utilizes `torch.no_grad()` for efficient inference.
- Prints progress using `tqdm` for tracking prediction status.
- Automatically creates directories for saving predictions.
"""

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = "models/best_model.pth"
RAW_DATA_PATH = "raw_data/mariamadre/mariamadre_2025-02-10T18-06-07Z/mariamadre_2025-02-10T18-06-07Z.bip"

# Load the model
model = cnn_1d(input_dim=110, num_classes=3)
state_dict = torch.load(MODEL_PATH, map_location=device)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()} # Fix `_orig_mod.`-prefix
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Load the raw data
try:
    dataset = hyperspectral_dataset(RAW_DATA_PATH, label_path=None)
    image_data = dataset.image_data

    if not isinstance(image_data, torch.Tensor):
        image_data = torch.from_numpy(image_data)
    
    image_data = image_data.to(device, dtype=torch.float32)
    image_data = image_data.contiguous().reshape(1, 120, -1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Run prediction (inference)
with torch.no_grad():
    num_pixels = image_data.shape[-1]

    print(num_pixels)

    # Empty array for predictions
    prediction_classes = np.zeros(num_pixels, dtype=np.uint8)

    for i in tqdm(range(num_pixels), desc="Predicting pixels", unit="px"):
        pixel_data = image_data[:, :, i].unsqueeze(0).unsqueeze(-1).squeeze(1) # (1, 120, 1)
        prediction = model(pixel_data)  # Model output
        prediction_classes[i] = torch.argmax(prediction, dim=1).item()  # Save prediction

# Reshape to (598, 1092)
prediction_classes = prediction_classes.reshape((598, 1092))

# Save prediction
file_path_split = RAW_DATA_PATH.split('/')
save_dir = f"raw_data/{file_path_split[1]}/{file_path_split[2]}/"
print(save_dir)
os.makedirs(save_dir, exist_ok=True)

np.save(f"{save_dir}PREDICTION_{file_path_split[2]}.npy", prediction_classes)
print(f"Prediction saved to 'prediction_classes.npy'")
print(f"Prediction shape: {prediction_classes.shape}")  # (598, 1092)