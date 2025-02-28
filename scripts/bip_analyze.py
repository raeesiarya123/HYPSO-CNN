import os
import numpy as np
import matplotlib.pyplot as plt

# 📂 Rot-mappen der filene ligger
root_dir = "raw_data"

# 🔍 Finn alle .bip@- og .bip-filer
bip_files = []
for location in os.listdir(root_dir):
    loc_path = os.path.join(root_dir, location)
    if os.path.isdir(loc_path):  # Sjekk at det er en mappe
        for capture in os.listdir(loc_path):
            cap_path = os.path.join(loc_path, capture)
            if os.path.isdir(cap_path):  # Sjekk at det er en submappe
                for file in os.listdir(cap_path):
                    if file.endswith(".bip@") or file.endswith(".bip"):
                        bip_files.append(os.path.join(cap_path, file))

# 🎯 Analyser hver .bip@-fil
def read_bip_file(file_path, height, width, bands, dtype=np.uint16):
    """ Leser en .bip@-fil og reshaper den til (Bands, Height, Width) """
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype)
    return raw_data.reshape((bands, height, width))

for bip_path in bip_files:
    print(f"📂 Analysere: {bip_path}")
    
    # 📏 Sett riktige dimensjoner (disse må tilpasses etter dataset)
    HEIGHT, WIDTH, BANDS = 598, 1092, 120  # Juster etter filene dine
    
    # Les inn BIP-data
    bip_data = read_bip_file(bip_path, HEIGHT, WIDTH, BANDS)

    # 📏 Sjekk dimensjoner
    print(f"Dimensjoner (Bands, Height, Width): {bip_data.shape}")

    # 🎨 Plot spektralprofil for én tilfeldig piksel
    h, w = bip_data.shape[1], bip_data.shape[2]
    rand_x, rand_y = np.random.randint(0, w), np.random.randint(0, h)
    spectrum = bip_data[:, rand_y, rand_x]

    plt.figure(figsize=(8, 5))
    plt.plot(spectrum, marker="o")
    plt.title(f"Spektralprofil for tilfeldig piksel ({rand_x}, {rand_y}) i {os.path.basename(bip_path)}")
    plt.xlabel("Bånd")
    plt.ylabel("Intensitet")
    plt.grid()
    plt.savefig(f"{bip_path}_spectral_analysis.png")
    plt.show()

    # 📊 Histogram for første bånd
    plt.figure(figsize=(8, 5))
    plt.hist(bip_data[0, :, :].flatten(), bins=50, color="blue", alpha=0.7)
    plt.title(f"Histogram av bånd 1 i {os.path.basename(bip_path)}")
    plt.xlabel("Intensitet")
    plt.ylabel("Antall piksler")
    plt.grid()
    plt.savefig(f"{bip_path}_histogram.png")
    plt.show()

print("✅ Analyse av .bip@-filer fullført!")