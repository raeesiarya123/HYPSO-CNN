import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################
"""
Initializes the CNN model with six convolutional layers, batch normalization, 
global average pooling, dropout, and a fully connected output layer.

Parameters:
- input_dim (int): The input feature dimension (default: 598).
- num_classes (int): Number of output classes (default: 3 for land, sea and cloud classification).

The model consists of:
- Six 1D convolutional layers, each followed by batch normalization.
- Global average pooling to reduce dimensionality and prevent overfitting.
- Dropout to improve generalization.
- A fully connected layer for final classification.
"""

class cnn_1d(nn.Module):
    def __init__(self, input_dim=110, num_classes=3,
                conv1_filters=64,
                conv2_filters=128,
                conv3_filters=256,
                conv4_filters=512,
                ):
        super(cnn_1d, self).__init__()
        torch.backends.cudnn.benchmark = True

        # Layer 1
        # 1) Convolution
        # 2) Batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_filters, kernel_size=7, stride=1, padding="same")
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2
        # 1) Convolution
        # 2) Batch normalization
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=5, stride=1, padding="same")
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3
        # 1) Convolution
        # 2) Batch normalization
        self.conv3 = nn.Conv1d(in_channels=conv2_filters, out_channels=conv3_filters, kernel_size=3, stride=1, padding="same")
        self.bn3 = nn.BatchNorm1d(conv3_filters)
        self.mp3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4
        # 1) Convolution
        # 2) Batch normalization
        self.conv4 = nn.Conv1d(in_channels=conv3_filters, out_channels=conv4_filters, kernel_size=3, stride=1, padding="same")
        self.bn4 = nn.BatchNorm1d(conv4_filters)
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
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.mp1(x)

        #print(f"DEBUG - After Conv1: {x.shape}")  

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.mp2(x)
        #print(f"DEBUG - After Conv2: {x.shape}")  

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.mp3(x)
        #print(f"DEBUG - After Conv3: {x.shape}") 

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
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


"""
class cnn_1d(nn.Module):
    def __init__(self, input_dim=598, num_classes=3):
        super(cnn_1d, self).__init__()
        torch.backends.cudnn.benchmark = True

        # Activation function (LeakyReLU with negative_slope=0.01)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=120, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        self.bn5 = nn.BatchNorm1d(512)

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same')
        self.bn6 = nn.BatchNorm1d(1024)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

        # Fully Connected Layer
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.leaky_relu(self.bn1(self.conv1(x)))
        if x.shape[-1] > 1:
            x = self.pool1(x)

        x = self.leaky_relu(self.bn2(self.conv2(x)))
        if x.shape[-1] > 1:
            x = self.pool2(x)

        x = self.leaky_relu(self.bn3(self.conv3(x)))

        x = self.leaky_relu(self.bn4(self.conv4(x)))
        if x.shape[-1] > 1:
            x = self.pool3(x)

        x = self.leaky_relu(self.bn5(self.conv5(x)))

        x = self.leaky_relu(self.bn6(self.conv6(x)))

        x = self.global_avg_pool(x).squeeze(-1)

        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
"""