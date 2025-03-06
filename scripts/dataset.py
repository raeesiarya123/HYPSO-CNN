import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

class hyperspectral_dataset(Dataset):
    def __init__(self, top_folder_name, label_path=None, augment_factor=10, apply_augment=False):
        self.augment_factor = augment_factor
        self.apply_augment = apply_augment
        
        # Read hyperspectral data
        with open(top_folder_name, 'rb') as f:
            raw_data = cp.fromfile(f, dtype=cp.uint16)
            raw_data = cp.asnumpy(raw_data)
        
        # Image Dimensions
        HEIGHT, WIDTH, BANDS = 598, 1092, 120
        self.image_data = raw_data.reshape((BANDS, HEIGHT, WIDTH))
        
        # Change Dimensions - (bands, height, width) -> (height, width, bands)
        self.image_data = np.transpose(self.image_data, (1, 2, 0))

        if label_path is not None:
            # Read labels
            HEIGHT, WIDTH = 598, 1092
            self.labels = np.fromfile(label_path, dtype=np.uint8).reshape((HEIGHT, WIDTH))
            # Change labels from {1, 2, 3} → {0, 1, 2}
            self.labels = self.labels - 1
            
            # Create base version of image and labels - add flipped versions later
            self.base_images = [self.image_data]
            self.base_labels = [self.labels]

            # Vertical Flip
            flip_v_image = np.flip(self.image_data, axis=0)
            flip_v_labels = np.flip(self.labels, axis=0)
            self.base_images.append(flip_v_image)
            self.base_labels.append(flip_v_labels)
                
            # Horizontal Flip
            flip_h_image = np.flip(self.image_data, axis=1)
            flip_h_labels = np.flip(self.labels, axis=1)
            self.base_images.append(flip_h_image)
            self.base_labels.append(flip_h_labels)
            
            # Flatten image data
            self.base_images = [img.reshape(-1, img.shape[-1]) for img in self.base_images]
            self.base_labels = [lbl.flatten() for lbl in self.base_labels]

            # Convert to PyTorch tensors
            self.base_images = [torch.from_numpy(img).float().contiguous() for img in self.base_images]
            self.base_labels = [torch.from_numpy(lbl).long() for lbl in self.base_labels]

        else:
            self.labels = None

    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        Each original image has 3 variations (original, flipped vertical, flipped horizontal).
        Each variation has augment_factor augmentations, making the total:
        3 * (1 original + augment_factor augmentations).
        For instance if augment_factor=10, the total number of images is 3 * 11 = 33 from one image.
        """
        base_size = len(self.base_images[0])
        if self.apply_augment:
            return base_size * (self.augment_factor + 1) * len(self.base_images)
        return base_size * len(self.base_images)

    def __getitem__(self, idx):
        """
        Returns one pixel and its corresponding label.
        The augmentations are applied probabilistically.
        The augmentations only happen after the original image and its flips.
        """
        base_size = len(self.base_images[0])
        base_idx = idx % len(self.base_images[0])
        image_idx = (idx // len(self.base_images[0])) % len(self.base_images)
        pixel = self.base_images[image_idx][base_idx]

        if self.apply_augment and idx >= len(self.base_images[0])*len(self.base_images):
            # Augmentation index - 0: original, 1 to augment_factor: augmentations
            aug_idx = (idx // base_size) % (self.augment_factor + 1)

            if aug_idx > 0:
                # Gaussian Noise
                if rd.random() < 0.1:
                    noise_std = rd.uniform(0.01, 0.03)
                    noise = torch.normal(mean=0, std=noise_std, size=pixel.shape)
                    pixel = pixel + noise

                # Spectral Scaling
                elif rd.random() < 0.1:
                    scale_factor_std = rd.uniform(0.01, 0.03)
                    scale_factor = torch.normal(mean=1, std=scale_factor_std, size=pixel.shape)
                    pixel = pixel * scale_factor

        if self.labels is not None:
            return pixel, self.base_labels[image_idx][base_idx]
        return pixel # For inference