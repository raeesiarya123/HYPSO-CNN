import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

"""
visualization.py - Visualizing Hyperspectral Image Predictions

This script overlays the model's classification results on an original hyperspectral image, 
generating a color-coded visualization where each class (land, sea, cloud) is assigned a specific color.

Main Steps:
1. **Load Prediction Data**:
   - Loads the `.npy` file containing pixel-wise classification results.
   - Defines a color mapping for the three classes:
     - **Sea** → Purple
     - **Land** → Cyan
     - **Cloud** → Red

2. **Prepare and Overlay the Color Map**:
   - Creates an RGB color map based on the predicted classes.
   - Loads the original image from a `.png` file.
   - Ensures the dimensions of the original image and prediction map match.

3. **Apply Transparency (Alpha Blending)**:
   - Merges the original image with the color-coded classification map using alpha blending.
   - Adjusts transparency to make predictions clearly visible while preserving the original image details.

4. **Save and Display the Visualization**:
   - Saves the final overlay image as a `.png` file in the appropriate directory.
   - Displays the result using `matplotlib`.

Key Features:
- Dynamically constructs paths based on the input file structure.
- Uses alpha blending to enhance visualization clarity.
- Ensures proper alignment of the prediction and original image dimensions.
"""

RAW_DATA_PATH = "raw_data/mariamadre/mariamadre_2025-02-10T18-06-07Z/mariamadre_2025-02-10T18-06-07Z.bip"
PNG_PATH = "raw_data/mariamadre/mariamadre_2025-02-10T18-06-07Z/mariamadre_2025-02-10T18-06-07Z.png"
file_path_split = RAW_DATA_PATH.split('/')

prediction_classes = np.load(f"raw_data/{file_path_split[1]}/{file_path_split[2]}/PREDICTION_{file_path_split[2]}.npy")

colors = {
    0: [0.5, 0, 0.5],  # Purple (Sea)
    1: [0, 1, 1],      # Cyan (Land)
    2: [1, 0, 0]       # Red (Cloud)
}

height, width = prediction_classes.shape
color_map = np.zeros((height, width, 3))

original_image = mpimg.imread(PNG_PATH)

for class_id, color in colors.items():
    color_map[prediction_classes == class_id] = color

if original_image.shape[:2] != (height, width):
    original_image = np.transpose(original_image, (1, 0, 2))

alpha = 0.3
overlay = (1 - alpha) * original_image[:, :, :3] + alpha * color_map

save_dir = f"raw_data/{file_path_split[1]}/{file_path_split[2]}/"
print(save_dir)
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.imshow(overlay)
plt.axis("off")
plt.title(f"Prediction: Land/Sea/Cloud\n{height}x{width}")
plt.savefig(os.path.join(save_dir, f"PREDICTION_{file_path_split[2]}.png"), dpi=300)
plt.show()