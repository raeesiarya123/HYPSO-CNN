import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from models.cnn_1d import cnn_1d
#from models.cnn_1d import HyperspectralCNN
from dataset import hyperspectral_dataset
from scripts.data_management import read_csv_file
from functions_train import normalize_spectrum

#######################################################################################
#######################################################################################
#######################################################################################

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = "models/best_model_EPOCH_1.pth"

# Load dataset
TEST_DATA_PATHS, TEST_LABEL_PATHS, TEST_PNG_PATHS = read_csv_file("evaluate_files.csv")
test_datasets = [hyperspectral_dataset(data_path, label_path, apply_augment=False) for data_path, label_path in zip(TEST_DATA_PATHS, TEST_LABEL_PATHS)]
test_dataset = torch.utils.data.ConcatDataset(test_datasets)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1092, shuffle=False, num_workers=8)

# Load model
model = cnn_1d(input_dim=110, num_classes=3).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
#model.train()

all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluering pågår", leave=True, colour="red"):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = normalize_spectrum(inputs)
        inputs = inputs.unsqueeze(1)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nModellens nøyaktighet: {accuracy * 100:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
classes = ["Cloud", "Land", "Sea"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predikerte etiketter")
plt.ylabel("Sanne etiketter")
plt.title("Konfusjonsmatrise")
plt.savefig("plots/validation_plots/confusion_matrix.png")
plt.show()

print(classification_report(all_labels, all_preds, target_names=classes))