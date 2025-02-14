import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

# Time complexity: O(n*k*d**2)
# Space complexity: O(n+k*d**2)
class JustoUNetSimple(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JustoUNetSimple, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(12, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(6, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x

# Time complexity: O(n*k*d**2)
# Space complexity: O(n+k*d**2)
class JustoCUNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JustoCUNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.sepconv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.sepconv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upsample1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        self.upsample2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.sepconv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.sepconv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.upsample1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = F.relu(self.upsample2(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        
        x = self.final(x)
        return F.softmax(x, dim=1)

# Wrapper around JustoCUNet (same complexity)
class JustoCUNetPlusPlus(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JustoCUNetPlusPlus, self).__init__()
        
        self.base_model = JustoCUNet(input_channels, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Time complexity: O(n*k*d**2)
# Space complexity: O(n+k*d**2)
class JustoFAUBAI(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JustoFAUBAI, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.final(x)
        return F.softmax(x, dim=1)
