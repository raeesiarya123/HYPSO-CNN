import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class hyperspectral_dataset(Dataset):
    def __init__(self, top_folder_name, label_path):
        """ Leser inn hyperspektral .bip@-data og tilsvarende labels """
        
        # Leser inn hyperspektral data fra .bip@
        with open(top_folder_name, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
        
        # Hardkodede dimensjoner (må sjekkes mot dataene)
        HEIGHT, WIDTH, BANDS = 598, 1092, 120  # Tilpass etter datasettene
        self.image_data = raw_data.reshape((BANDS, HEIGHT, WIDTH))
        
        # Endrer dimensjonene - (bands, height, width) -> (height, width, bands)
        self.image_data = np.transpose(self.image_data, (1, 2, 0))

        # Leser labels
        HEIGHT, WIDTH = 598, 1092  # Juster om nødvendig
        self.labels = np.fromfile(label_path, dtype=np.uint8).reshape((HEIGHT, WIDTH))


        # Sjekk om dimensjonene matcher
        if self.labels.shape != (self.image_data.shape[0], self.image_data.shape[1]):
            raise ValueError("Mismatch mellom labels og bildet! Sjekk label-filen.")
        
        # Flat ut bildet slik at hver piksel blir et datapunkt
        self.image_data = self.image_data.reshape(-1, self.image_data.shape[-1])  # (598*1092, bands)
        self.labels = self.labels.flatten()  # (598*1092,)

        # Konverter til PyTorch tensors
        self.image_data = torch.tensor(self.image_data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"Datasettet inneholder {len(self)} piksler.")
        print(f"Dimensjoner på bildedata: {self.image_data.shape}")
        print(f"Dimensjoner på labels: {self.labels.shape}")

    def __len__(self):
        """ Returnerer antall piksler i datasetet. """
        return len(self.image_data)

    def __getitem__(self, idx):
        """ Henter én piksel og dens label. """
        return self.image_data[idx], self.labels[idx]
   
    
"""
# Last inn dataset
dataset = hyperspectral_dataset("raw_data/aquawatchmoreton/aquawatchmoreton_2025-01-22T00-11-33Z/aquawatchmoreton_2025-01-22T00-11-33Z.bip@",
                                 "training_data/bluenile/bluenile_2025-01-25T08-23-16Z-l1a_products_dn_class.dat")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test lesing av en batch
batch = next(iter(dataloader))
sample_x, sample_y = batch

# Returner informasjon om batch
sample_x.shape, sample_y.shape, sample_x.dtype, sample_y.dtype

print("Eksempel labelverdier:", torch.unique(sample_y))  # Sjekker hvilke klasser som finnes
print("Eksempel spektralverdier:", sample_x[0])  # Sjekker en tilfeldig piksel
print("Maks intensitet i batch:", sample_x.max().item())
print("Min intensitet i batch:", sample_x.min().item())

# Sjekk alle unike verdier i sample_y
unique_labels = torch.unique(sample_y)
print("Alle unike labelverdier i batch:", sample_y)
"""