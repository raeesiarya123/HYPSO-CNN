import numpy as np

label_path = "training_data/bluenile/bluenile_2025-01-25T08-23-16Z-l1a_products_dn_class.dat"
labels = np.fromfile(label_path, dtype=np.uint8)  # Prøv np.uint16 hvis nødvendig

print("Unike verdier i labels:", np.unique(labels))  # Sjekker hvilke klasser som finnes
