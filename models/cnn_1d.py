import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class cnn_1d(nn.Module):
    def __init__(self, input_dim=110, num_classes=3,
                conv1_filters=8,
                conv2_filters=16,
                conv3_filters=32,
                conv4_filters=64,
                ):
        super(cnn_1d, self).__init__()
        torch.backends.cudnn.benchmark = True

        # Layer 1
        # 1) Convolution
        # 2) Batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_filters, kernel_size=7, stride=1, padding="same")
        #self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2
        # 1) Convolution
        # 2) Batch normalization
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=5, stride=1, padding="same")
        #self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3
        # 1) Convolution
        # 2) Batch normalization
        self.conv3 = nn.Conv1d(in_channels=conv2_filters, out_channels=conv3_filters, kernel_size=3, stride=1, padding="same")
        #self.bn3 = nn.BatchNorm1d(conv3_filters)
        self.mp3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4
        # 1) Convolution
        # 2) Batch normalization
        self.conv4 = nn.Conv1d(in_channels=conv3_filters, out_channels=conv4_filters, kernel_size=3, stride=1, padding="same")
        #self.bn4 = nn.BatchNorm1d(conv4_filters)
        self.mp4 = nn.MaxPool1d(kernel_size=2)

        # Global Average Pooling (compresses data from (batch, 128, 598) → (batch, 128, 1))
        # Prevents overfitting and reduces the number of parameters --> less overtraining, faster training
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout1 = nn.Dropout(p=0.1) # Dropout to prevent overfitting
        self.dropout2 = nn.Dropout(p=0.3) # Dropout to prevent overfitting

        # Fully connected layer
        self.fc = nn.Linear(conv4_filters, num_classes)

    def forward(self, x):

        #print(f"### DEBUG: Model Input Shape: {x.shape}")

        # Layer 1
        x = self.conv1(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.mp1(x)

        #print(f"DEBUG - After Conv1: {x.shape}")  

        # Layer 2
        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.mp2(x)
        #print(f"DEBUG - After Conv2: {x.shape}")  

        # Layer 3
        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.mp3(x)
        #print(f"DEBUG - After Conv3: {x.shape}") 

        # Layer 4
        x = self.conv4(x)
        #x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.mp4(x)
        #print(f"DEBUG - After Conv4: {x.shape}") 

        x = self.dropout1(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        #print(f"DEBUG - After Global Avg Pooling: {x.shape}")  

        # Flatten
        x = x.view(x.shape[0], -1)
        #print(f"DEBUG - After Flatten: {x.shape}")  

        x = self.dropout2(x)

        # Fully connected layer
        # Classifies the pixel as land, sea or cloud
        x = self.fc(x)
        #print(f"DEBUG - After Fully Connected Layer: {x.shape}")
        #print(f"### DEBUG - Output after Fully Connected Layer ###\n{x[:5]}")  
        #print("-"*100)

        return x
    
#######################################################################################

"""

class HyperspectralCNN(nn.Module):
    def __init__(self, input_dim=110, num_classes=3, kernel_size=6, start_filters=6):
        super(HyperspectralCNN, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=start_filters, kernel_size=kernel_size, padding="same")
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=start_filters, out_channels=start_filters*2, kernel_size=kernel_size, padding="same")
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=start_filters*2, out_channels=start_filters*3, kernel_size=kernel_size, padding="same")
        self.mp3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4
        self.conv4 = nn.Conv1d(in_channels=start_filters*3, out_channels=start_filters*4, kernel_size=kernel_size, padding="same")
        self.mp4 = nn.MaxPool1d(kernel_size=2)

        # Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(start_filters*4, num_classes)  # Final classification layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)

        x = F.relu(self.conv3(x))
        x = self.mp3(x)

        x = F.relu(self.conv4(x))
        x = self.mp4(x)

        x = self.global_avg_pool(x)  # Reduce to (batch, channels, 1)
        x = x.view(x.shape[0], -1)  # Flatten to (batch, channels)

        x = self.fc(x)  # Fully connected layer for classification
        return x
"""

#######################################################################################