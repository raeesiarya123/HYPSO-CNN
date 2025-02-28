import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from dataset import hyperspectral_dataset
from models.cnn_1d import cnn_1d

#######################################################################################
#######################################################################################
#######################################################################################

# GPU (CUDA) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")
print(torch.cuda.is_available())  # Skal være True
print(torch.cuda.device_count())  # Skal være minst 1
print(torch.version.cuda)  # Skal matche CUDA 12.7
print(torch.backends.cudnn.version())  # Skal gi et tall hvis CUDNN er aktiv
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Hyperparametere
EPOCHS = 75
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# Last inn datasettet
TRAIN_DATA_PATHS = ["raw_data/bluenile/bluenile_2025-01-25T08-23-16Z/bluenile_2025-01-25T08-23-16Z.bip@",
                    "raw_data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z/aeronetgalata_2025-01-02T08-52-34Z.bip@",
                    "raw_data/gulfofcalifornia/gulfofcalifornia_2025-01-14T18-19-49Z/gulfofcalifornia_2025-01-14T18-19-49Z.bip@"]

TRAIN_LABEL_PATHS = ["training_data/bluenile/bluenile_2025-01-25T08-23-16Z-l1a_products_dn_class.dat",
                   "training_data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z-l1a_products_dn_class.dat",
                   "training_data/gulfofcalifornia/gulfofcalifornia_2025-01-14T18-19-49Z-l1a_products_dn_class.dat"]

train_datasets = [hyperspectral_dataset(data_path, label_path) for data_path, label_path in zip(TRAIN_DATA_PATHS, TRAIN_LABEL_PATHS)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=3)

# Modell
model = cnn_1d(input_dim=120, num_classes=3).to(device)
model = torch.compile(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_loop(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for batch in loop:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1).squeeze(2)

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

print("Starter trening...")
train_loop(model, train_loader, criterion, optimizer)
print("Trening ferdig!")