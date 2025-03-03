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
model = cnn_1d(input_dim=120, num_classes=3).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}  # Fix `_orig_mod.`-prefix
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Evaluate
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)
            
            output = model(inputs)
            _, preds = torch.max(output, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy*100:.2f}%")

    return all_preds, all_labels

labels, preds = evaluate(model, test_loader)

conf_matrix = confusion_matrix(labels, preds)
class_names = ["Land", "Sea", "Cloud"]

plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(labels, preds, target_names=class_names))