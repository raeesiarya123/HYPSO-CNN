import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class hyperspectral_dataset(Dataset):
    def __init__(self, top_folder_name, label_path=None, augment_factor=10, apply_augment=False):
        """ Leser inn hyperspektral .bip@-data og tilsvarende labels """

        self.augment_factor = augment_factor
        self.apply_augment = apply_augment
        
        # Leser inn hyperspektral data fra .bip@
        with open(top_folder_name, 'rb') as f:
            raw_data = cp.fromfile(f, dtype=cp.uint16)
            raw_data = cp.asnumpy(raw_data)
        
        # Hardkodede dimensjoner (må sjekkes mot dataene)
        HEIGHT, WIDTH, BANDS = 598, 1092, 120  # Tilpass etter datasettene
        self.image_data = raw_data.reshape((BANDS, HEIGHT, WIDTH))
        
        # Endrer dimensjonene - (bands, height, width) -> (height, width, bands)
        self.image_data = np.transpose(self.image_data, (1, 2, 0))

        if label_path is not None:
            # Leser labels
            HEIGHT, WIDTH = 598, 1092  # Juster om nødvendig
            self.labels = np.fromfile(label_path, dtype=np.uint8).reshape((HEIGHT, WIDTH))
            # Endre labels fra {1, 2, 3} → {0, 1, 2}
            self.labels = self.labels - 1

            # Sjekk om dimensjonene matcher
            if self.labels.shape != (self.image_data.shape[0], self.image_data.shape[1]):
                raise ValueError("Mismatch mellom labels og bildet! Sjekk label-filen.")
            
            # Flat ut bildet slik at hver piksel blir et datapunkt
            self.image_data = self.image_data.reshape(-1, self.image_data.shape[-1])
            self.image_data = torch.from_numpy(self.image_data).float().contiguous()
            self.labels = self.labels.flatten()  # (598*1092,)

            # Konverter til PyTorch tensors
            self.labels = torch.from_numpy(self.labels).long()
        else:
            self.labels = None

    def __len__(self):
        """ Returnerer antall piksler i datasetet. """
        return len(self.image_data) * (self.augment_factor + 1) if self.apply_augment else len(self.image_data)

    def __getitem__(self, idx):
        """ Henter én piksel og dens label. """

        base_idx = idx % len(self.image_data)
        pixel = self.image_data[base_idx]

        if self.apply_augment and idx >= len(self.image_data):
            noise_std = rd.uniform(0.01, 0.075)
            noise = torch.normal(mean=0, std=noise_std, size=pixel.shape)
            pixel = pixel + noise

        if self.labels is not None:
            return pixel, self.labels[base_idx]
        return pixel # For inference