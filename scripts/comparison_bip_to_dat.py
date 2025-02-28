import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from read_bip import read_bip

#######################################################################################
#######################################################################################
#######################################################################################

# Filbaner
top_folder_name = "raw_data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z-hsi0"
label_path = "data/gulf_of_california/gulfofcalifornia_2025-01-14T18-19-49Z-l1a_products_dn_class.dat"

# Les inn BIP-fil
image, bip_data = read_bip(top_folder_name)

# Les inn DAT-fil
dat_labels = np.fromfile(label_path, dtype=np.uint8)  # Sørg for at dtype stemmer
dat_labels = dat_labels.reshape((bip_data.shape[1], bip_data.shape[2]))  # (H, W)

# 📌 Sjekk dimensjoner
print("BIP shape:", bip_data.shape)  # (Bands, Height, Width)
print("DAT shape:", dat_labels.shape)  # (Height, Width)

# 📌 Sjekk om pikselantall matcher
assert bip_data.shape[1] * bip_data.shape[2] == dat_labels.size, "FEIL: Antall piksler i .bip og .dat matcher ikke!"

# 📌 Visualiser et tilfeldig spektralbilde vs. labelene
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Vis et tilfeldig spektralbånd fra BIP-bildet
band_idx = 10  # Velg et tilfeldig spektralbånd
ax[0].imshow(bip_data[band_idx, :, :], cmap="gray")
ax[0].set_title(f"BIP - Band {band_idx}")

# Vis tilhørende labels
ax[1].imshow(dat_labels, cmap="jet")
ax[1].set_title("Labels fra DAT-fil")

plt.show()
