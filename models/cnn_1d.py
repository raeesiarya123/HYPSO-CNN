import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class cnn_1d(nn.Module):
    def __init__(self, input_dim=598, num_classes=3,
                conv1_filters=32,
                conv2_filters=64,
                conv3_filters=128):
        super(cnn_1d, self).__init__()
    
        # Lag 1
        # 1) Konvolusjonslag
        # 2) Batch normalisering
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_filters, kernel_size=7, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1_filters)

        # Antall nevroner: 598*32 = 19136

        # Lag 2
        # 1) Konvolusjonslag
        # 2) Batch normalisering
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=5, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2_filters)

        # Antall nevroner: 598*64 = 38272

        # Lag 3
        # 1) Konvolusjonslag
        # 2) Batch normalisering
        self.conv3 = nn.Conv1d(in_channels=conv2_filters, out_channels=conv3_filters, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(conv3_filters)

        # Antall nevroner: 598*128 = 76544

        # Global Average Pooling (komprimerer data fra (batch, 128, 598) → (batch, 128, 1))
        # Forhindrer overtilpasning og reduserer antall parametere --> mindre overtrening, raskere trening
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # Før GAP: (batch, 128, 598) --> 76554 verdier totalt
        # Etter GAP: (batch, 128, 1) --> 128 verdier totalt (1 per kanal)

        # Fullt tilkoblet lag
        self.fc = nn.Linear(conv3_filters, num_classes)

    def forward(self, x):
        # Lag 1 - Konvolusjon + Batch normalisering + Aktivering (ReLU)
        x = F.relu(self.bn1(self.conv1(x)))

        # Lag 2 - Konvolusjon + Batch normalisering + Aktivering (ReLU)
        x = F.relu(self.bn2(self.conv2(x)))

        # Lag 3 - Konvolusjon + Batch normalisering + Aktivering (ReLU)
        x = F.relu(self.bn3(self.conv3(x)))

        # Global Average Pooling
        # Komprimerer data fra (batch, 128, 598) → (batch, 128, 1)
        x = self.global_avg_pool(x)  # (batch, channels, 1)

        # Flatten
        # Gjør om data fra (batch, 128, 1) → (batch, 128)
        # Hvorfor? Fullt tilkoblet lag forventer input av form (batch, 128)
        x = x.view(x.shape[0], -1)

        # Fullt tilkoblet lag
        # Klassfiserer pikselen i land, sjø eller sky
        x = self.fc(x)

        # Softmax for klassifisering (log sannsynlighet pga CrossEntropyLoss)
        # CrossEntropyLoss: Kryssentropi mellom prediksjon og sannhet
        return F.log_softmax(x, dim=1)