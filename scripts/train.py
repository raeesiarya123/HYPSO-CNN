import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from dataset import hyperspectral_dataset
from models.cnn_1d import cnn_1d
#from models.cnn_1d import HyperspectralCNN
from scripts.data_management import read_csv_file
from scripts.functions_train import *

#######################################################################################
#######################################################################################
#######################################################################################

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Optimalization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Parameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
COUNTER_LR = 0
LABEL_SMOOTHING = 0.1

print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Label smoothing: {LABEL_SMOOTHING}")

# Dataset
TRAIN_DATA_PATHS, TRAIN_LABEL_PATHS, TRAIN_PNG_PATHS = read_csv_file("train_files.csv")

train_datasets = [hyperspectral_dataset(data_path, label_path, apply_augment=False) for data_path, label_path in zip(TRAIN_DATA_PATHS, TRAIN_LABEL_PATHS)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

print(f"DEBUG - Train dataset size: {len(train_dataset)}")

# Debug: Check what train_dataset contains

"""
for i in range(len(train_dataset)):
    data, label = train_dataset[i]
    print(f"Sample {i}: Data shape: {data.shape}, Label: {label}")
    if i >= 5:  # Print only the first 5 samples for brevity
        break
"""

train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True, 
                          pin_memory=True, 
                          num_workers=8,
                          persistent_workers=False
                          )

print(f"Amount of images in dataset: {len(train_dataset)/(598*1092)}")

# Model
model = cnn_1d(input_dim=110, num_classes=3).to(device)
# Tell antall parametere i modellen
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Get best accuracy from earlier trained model
best_accuracy = get_earlier_model(device, model)

class_weights = get_weights_training().to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=LABEL_SMOOTHING)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
#criterion = FocalLoss(alpha=0.25, gamma=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, verbose=True)

def train_loop(model, train_loader, criterion, optimizer, save_path="models/best_model.pth", counter_lr=COUNTER_LR):
    model.train()
    global best_accuracy
    epoch_num = 1

    print(f"DEBUG - Number of batches: {len(train_loader)}")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        labels_per_epoch = []
        predictions_per_epoch = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False, colour="red")

        for batch in loop:
            inputs, labels = batch
            spectrum = inputs
            inputs = inputs.unsqueeze(1)
            labels = labels.squeeze(1)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            spectrum = normalize_spectrum(spectrum)
            spectrum = spectrum.unsqueeze(1)
            # DEBUG: Skriv ut labels for første batch
            #unique_labels = torch.unique(labels).cpu().numpy()
            #print(f"### DEBUG: Labels in first batch: {unique_labels}")
            #print(f"### DEBUG: Labels: {labels}")
            #print(f"### DEBUG: Length labels: {len(labels)}")
            #print(f"DEBUG: spectrum: {spectrum}")
            #print(f"DEBUG: spectrum shape: {spectrum.shape}")

            optimizer.zero_grad()
            output = model(inputs)

            # Plotting info
            predictions_per_epoch.append(output)
            labels_per_epoch.append(labels)

            loss = criterion(output, labels)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        save_path_epoch = f"{save_path[:-4]}_EPOCH_{epoch+1}.pth"

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'model_state_dict': model.state_dict(),
                        'best_accuracy': best_accuracy},
                        save_path_epoch)
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

        scheduler.step(accuracy)
        
        predictions_per_epoch = torch.cat(predictions_per_epoch).detach().cpu().numpy()
        labels_per_epoch = torch.cat(labels_per_epoch).detach().cpu().numpy()
        print("\n")

print("Starting training...")
train_loop(model, train_loader, criterion, optimizer)
print("Training finished.")