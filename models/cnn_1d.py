import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class cnn_1d(nn.Module):
    def __init__(self, input_dim=598, num_classes=3,
                conv1_filters=64,
                conv2_filters=128,
                conv3_filters=256,
                conv4_filters=512):
        super(cnn_1d, self).__init__()
        torch.backends.cudnn.benchmark = True

        """
        Initializes the CNN model with four convolutional layers, batch normalization, 
        global average pooling, dropout, and a fully connected output layer.

        Parameters:
        - input_dim (int): The input feature dimension (default: 598).
        - num_classes (int): Number of output classes (default: 3 for land, sea and cloud classification).
        - conv1_filters (int): Number of filters in the first convolutional layer (default: 64).
        - conv2_filters (int): Number of filters in the second convolutional layer (default: 128).
        - conv3_filters (int): Number of filters in the third convolutional layer (default: 256).
        - conv4_filters (int): Number of filters in the fourth convolutional layer (default: 512).

        The model consists of:
        - Four 1D convolutional layers, each followed by batch normalization.
        - Global average pooling to reduce dimensionality and prevent overfitting.
        - Two dropout layers to improve generalization.
        - A fully connected layer for final classification.
        """

        # Layer 1
        # 1) Convolution
        # 2) Batch normalization
        self.conv1 = nn.Conv1d(in_channels=120, out_channels=conv1_filters, kernel_size=7, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1_filters)

        # Layer 2
        # 1) Convolution
        # 2) Batch normalization
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=5, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2_filters)

        # Layer 3
        # 1) Convolution
        # 2) Batch normalization
        self.conv3 = nn.Conv1d(in_channels=conv2_filters, out_channels=conv3_filters, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(conv3_filters)

        # Layer 4
        # 1) Convolution
        # 2) Batch normalization
        self.conv4 = nn.Conv1d(in_channels=conv3_filters, out_channels=conv4_filters, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(conv4_filters)

        # Global Average Pooling (compresses data from (batch, 128, 598) → (batch, 128, 1))
        # Prevents overfitting and reduces the number of parameters --> less overtraining, faster training
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout1 = nn.Dropout(p=0.3) # Dropout to prevent overfitting
        self.dropout2 = nn.Dropout(p=0.5) # Dropout to prevent overfitting

        # Fully connected layer
        self.fc = nn.Linear(conv4_filters, num_classes)

    def forward(self, x):
        
        """
        Defines the forward pass of the 1D CNN model.

        The input tensor `x` passes through multiple convolutional layers with batch normalization 
        and SiLU activation, followed by dropout layers to prevent overfitting. 
        After feature extraction, the model applies global average pooling, 
        flattens the output, and passes it through a fully connected layer 
        for classification into land, sea, or cloud categories.

        Returns:
        - Log softmax probabilities for each class (log-probabilities used due to CrossEntropyLoss).
        """

        # Layer 1 - Convolution + Batch normalization + Activation function (SiLu)
        x = F.silu(self.bn1(self.conv1(x)))

        # Layer 2 - Convolution + Batch normalization + Activation function (SiLu)
        x = F.silu(self.bn2(self.conv2(x)))

        # Layer 3 - Convolution + Batch normalization + Activation function (SiLu)
        x = F.silu(self.bn3(self.conv3(x)))

        # Layer 4 - Convolution + Batch normalization + Activation function (SiLu)
        x = F.silu(self.bn4(self.conv4(x)))

        x = self.dropout1(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten
        x = x.view(x.shape[0], -1)

        # Dropout to prevent overfitting (randomly activate/deactivate neurons during training)
        x = self.dropout2(x)

        # Fully connected layer
        # Classifies the pixel as land, sea or cloud
        x = self.fc(x)

        # Softmax for classification (log probability because of CrossEntropyLoss)
        # CrossEntropyLoss: Crossentropy between prediction and truth
        return F.log_softmax(x, dim=1)