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

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = "models/best_model.pth"
RAW_DATA_PATH = "raw_data/mariamadre/mariamadre_2025-02-10T18-06-07Z/mariamadre_2025-02-10T18-06-07Z.bip"

# Load the model
model = cnn_1d(input_dim=120, num_classes=3)
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
    num_pixels = image_data.shape[-1]  # Hvor mange piksler vi har

    print(num_pixels)

    # Empty array for predictions
    prediction_classes = np.zeros(num_pixels, dtype=np.uint8)

    for i in tqdm(range(num_pixels), desc="Predicting pixels", unit="px"):
        pixel_data = image_data[:, :, i].unsqueeze(0).unsqueeze(-1).squeeze(1) # (1, 120, 1)
        prediction = model(pixel_data)  # Model output
        prediction_classes[i] = torch.argmax(prediction, dim=1).item()  # Save prediction

# Reshape to (598, 1092)
prediction_classes = prediction_classes.reshape((598, 1092))

np.save("prediction_classes.npy", prediction_classes)
print(f"Prediction saved to 'prediction_classes.npy'")
print(f"Prediction shape: {prediction_classes.shape}")  # Skal være (598, 1092)