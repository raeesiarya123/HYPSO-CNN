import numpy as np
from spectral import open_image
import matplotlib.pyplot as plt

def load_envi_file(dat_path, hdr_path):
    """ Laster inn ENVI .dat og .hdr filer og printer informasjon om dataene. """
    # Åpne hyperspektralt bilde
    img = open_image(hdr_path)
    data = img.load()
    
    # Konverter til uint8 for samsvar med .hdr
    data = data.astype(np.uint8)
    
    # Sjekk metadata
    print("Data shape:", data.shape)  # (Height, Width, Bands)
    print("Data type:", data.dtype)
    print("Header info:")
    with open(hdr_path, 'r') as f:
        for line in f.readlines():
            print(line.strip())
    
    # Lese binærdata direkte fra .dat
    raw_data = np.fromfile(dat_path, dtype=np.uint8)  # Juster dtype om nødvendig
    print("Raw .dat file size:", raw_data.shape)
    
    # Reshape til forventet dimensjon
    height, width, bands = data.shape
    reshaped_data = raw_data.reshape((height, width, bands))
    print("Reshaped .dat file shape:", reshaped_data.shape)
    
    # Fjern unclassified (0), hvis den finnes
    unique_values = np.unique(reshaped_data)
    unique_values = unique_values[unique_values > 0]
    print("Filtered unique values in .dat file:", unique_values)
    
    # Mapping av klasser
    class_mapping = {1: "Cloud", 2: "Sea", 3: "Land"}
    class_labels = [class_mapping[val] for val in unique_values if val in class_mapping]
    print("Mapped classes:", class_labels)

    plt.imshow(data[:, :, 0], cmap="jet")
    plt.colorbar()
    plt.title("Visualisering av klassifikasjonsdata")
    plt.savefig("classification_data.png")
    plt.show()
    
    return reshaped_data

# Eksempelbruk
hdr_file = "data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z-l1a_products_dn_class.hdr"
dat_file = "data/aeronetgalata/aeronetgalata_2025-01-02T08-52-34Z-l1a_products_dn_class.dat"
data = load_envi_file(dat_file, hdr_file)