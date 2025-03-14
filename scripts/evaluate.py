import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from models.cnn_1d import cnn_1d
from dataset import hyperspectral_dataset
from csv_management import read_csv_file

#######################################################################################
#######################################################################################
#######################################################################################

"""
evaluate.py - Model evaluation script for the 1D CNN on hyperspectral images.

This script loads a trained CNN model and evaluates its performance on a test dataset. 
It computes the model's accuracy, generates a confusion matrix, and prints a classification report.

Main steps:
- Configures the device (GPU or CPU).
- Loads the test dataset from CSV file paths.
- Initializes and loads the trained model.
- Performs inference on the test set.
- Computes accuracy, confusion matrix, and classification report.

Key Features:
- Uses CUDA if available for faster inference.
- Handles `_orig_mod.` prefix in model state_dict to avoid loading errors.
- Uses batch processing for efficient evaluation.
"""

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = "models/best_model.pth"

# Load dataset
TEST_DATA_PATHS, TEST_LABEL_PATHS = read_csv_file("evaluate_files.csv")
test_datasets = [hyperspectral_dataset(data_path, label_path) for data_path, label_path in zip(TEST_DATA_PATHS, TEST_LABEL_PATHS)]
test_dataset = torch.utils.data.ConcatDataset(test_datasets)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=3)

# Load model
model = cnn_1d(input_dim=598, num_classes=3).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Evaluate
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)
            
            output = model(inputs)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy().flatten())
        
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy*100:.2f}%")

    return all_preds, all_labels

preds, labels = evaluate(model, test_loader)

conf_matrix = confusion_matrix(labels, preds)
class_names = ["Sea", "Land", "Cloud"]

# Opprett mappe hvis den ikke finnes
os.makedirs("plots/validation_plots", exist_ok=True)

plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("plots/validation_plots/confusion_matrix.png")
plt.show()

print(classification_report(labels, preds, target_names=class_names))
