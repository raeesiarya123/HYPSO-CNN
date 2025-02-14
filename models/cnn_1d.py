import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

# Time complexity: O(n*k*d)
# Space complexity: O(n+k*d)
class JustoLiuNet(nn.Module):
    def __init__(self, num_features, num_classes, kernel_size, starting_num_kernels):
        super(JustoLiuNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=starting_num_kernels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=starting_num_kernels, out_channels=starting_num_kernels*2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=starting_num_kernels*2, out_channels=starting_num_kernels*3, kernel_size=kernel_size)
        self.conv4 = nn.Conv1d(in_channels=starting_num_kernels*3, out_channels=starting_num_kernels*4, kernel_size=kernel_size)
        
        self.fc = nn.Linear(starting_num_kernels*4, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Time complexity: O(n*k*d)
# Space complexity: O(n+k*d)
class JustoHuNet(nn.Module):
    def __init__(self, num_features, num_classes, kernel_size, num_kernels, activation_conv, neurons_dense, activation_dense):
        super(JustoHuNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_kernels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=int((num_features - kernel_size + 1) / 35))
        self.fc1 = nn.Linear(num_kernels, neurons_dense)
        self.fc2 = nn.Linear(neurons_dense, num_classes)
        
        self.activation_conv = nn.Tanh() if activation_conv == 'tanh' else nn.ReLU()
        self.activation_dense = nn.Tanh() if activation_dense == 'tanh' else nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(self.activation_conv(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.activation_dense(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
# Time complexity: O(n*k*d)
# Space complexity: O(n+k*d)
class JustoLucasCNN(nn.Module):
    def __init__(self, num_features, num_classes, kernel_size, num_kernels_conv, activation_conv, activation_dense, neurons_first_dense, neurons_second_dense):
        super(JustoLucasCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_kernels_conv, kernel_size=kernel_size, padding='valid')
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(num_kernels_conv, neurons_first_dense)
        self.fc2 = nn.Linear(neurons_first_dense, neurons_second_dense)
        self.fc3 = nn.Linear(neurons_second_dense, num_classes)
        
        self.activation_conv = nn.Tanh() if activation_conv == 'tanh' else nn.ReLU()
        self.activation_dense = nn.Tanh() if activation_dense == 'tanh' else nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(self.activation_conv(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.activation_dense(self.fc1(x))
        x = self.activation_dense(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x