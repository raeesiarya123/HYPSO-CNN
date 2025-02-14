import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class HyperspectralDataset(Dataset):
    def __init__(self, data_path, labels_path=None, transform=None):
        """
        Args:
            data_path (str): Path to the numpy file containing hyperspectral data.
            labels_path (str, optional): Path to the numpy file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = np.load(data_path)
        self.labels = np.load(labels_path) if labels_path else None
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long) if self.labels is not None else None
        
        if self.transform:
            sample = self.transform(sample)
        
        return (sample, label) if label is not None else sample