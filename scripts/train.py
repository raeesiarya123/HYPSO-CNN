import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from dataset import hyperspectral_dataset
from models.cnn_1d import cnn_1d
from csv_management import read_csv_file

#######################################################################################
#######################################################################################
#######################################################################################

"""
train.py - Training script for the 1D CNN model on hyperspectral images.

This script sets up and executes the training process, including:
- Configuring the device (GPU or CPU) for training.
- Loading the training dataset from CSV file paths.
- Initializing and compiling the 1D CNN model.
- Defining the loss function (CrossEntropy with label smoothing).
- Using the AdamW optimizer with a StepLR scheduler for learning rate adjustment.
- Implementing the training loop with loss calculation, backpropagation, and model saving.

Key Features:
- Uses CUDA if available for faster training.
- Applies data augmentation during dataset loading.
- Utilizes PyTorch's `torch.compile` for performance optimization.
- Saves the best-performing model based on accuracy.
"""

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Optimalization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Parameters
EPOCHS = 36
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
COUNTER_LR = 0
LABEL_SMOOTHING = 0.1

print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Label smoothing: {LABEL_SMOOTHING}")

# Dataset
TRAIN_DATA_PATHS, TRAIN_LABEL_PATHS = read_csv_file("train_files.csv")

train_datasets = [hyperspectral_dataset(data_path, label_path, apply_augment=True) for data_path, label_path in zip(TRAIN_DATA_PATHS, TRAIN_LABEL_PATHS)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True, 
                          pin_memory=True, 
                          num_workers=12,
                          persistent_workers=True,
                          prefetch_factor=4
                          )

print(f"Amount of images in dataset: {len(train_dataset)/(598*1092)}")

# Model
model = cnn_1d(input_dim=120, num_classes=3).to(device)
model = torch.compile(model)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5 if COUNTER_LR < 3 else 1)

def train_loop(model, train_loader, criterion, optimizer, save_path="models/best_model.pth", counter_lr=COUNTER_LR):
    model.train()
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for batch in loop:
            inputs, labels = batch
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if inputs.dim() == 2:  
                inputs = inputs.unsqueeze(-1)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

        print("\n")

print("Starting training...")
train_loop(model, train_loader, criterion, optimizer)
print("Training finished.")