import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from models.cnn_1d import cnn_1d

#######################################################################################
#######################################################################################
#######################################################################################

# 🔥 **SETUP**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 📥 **LAST INN MODELLEN**
model = cnn_1d(input_dim=120, num_classes=3).to(device)
checkpoint_path = "models/best_model.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Filen {checkpoint_path} ble ikke funnet!")

print(f"📂 Laster inn sjekkpunkt fra {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# 🔄 **FIX `_orig_mod` OM NØDVENDIG**
if "model_state_dict" not in checkpoint:
    raise KeyError("❌ Nøkkel 'model_state_dict' mangler i checkpoint-filen!")

if any("_orig_mod" in key for key in checkpoint['model_state_dict'].keys()):
    print("🔄 Fixing `_orig_mod` issue in state_dict!")
    new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint['model_state_dict'])

print("✅ Modell vektet lastet inn!")
model.eval()

# 📊 **FUNKSJON FOR Å HENTE DATA FRA MODELLEN**
def data_from_model(model):
    print("\n📢 === Modellens detaljer ===")
    print(model)  # Printer modellarkitekturen

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Totalt antall parametere: {total_params:,}")

    for name, param in model.named_parameters():
        param = param.cpu().detach().numpy()
        param_shape = param.shape

        print(f"\n🔍 Layer: {name}")
        print(f"   📏 Shape: {param_shape}")
        print(f"   🔹 Min: {param.min()}, Max: {param.max()}, Mean: {param.mean()}")
        print(f"   🏗️ Antall parametere: {param.size}")

        if "weight" in name:
            print("   🔥 Dette er en vektmatrise.")
        elif "bias" in name:
            print("   🎯 Dette er en biasvektor.")

def plot_model_parameters(model):
    for name, param in model.named_parameters():
        param = param.cpu().detach().numpy()
        
        if "weight" in name and len(param.shape) > 1:  # Weight matrixes
            plt.figure(figsize=(10, 5))
            plt.imshow(param.mean(axis=0), cmap="inferno", aspect="auto")
            plt.colorbar()
            plt.title(f"{name} - Weight Matrix (Averaged)")
            plt.savefig(f"plots/bias_plots/weights_{name}.png")
            print(f"Saved weights plot as bias_{name}.png")
            plt.show()

        elif "bias" in name:  # Biases
            plt.figure(figsize=(6, 3))
            plt.hist(param, bins=30, color='purple', alpha=0.7)
            plt.title(f"{name} - Bias Distribution")
            plt.xlabel("Bias Value")
            plt.ylabel("Frequency")
            plt.savefig(f"plots/bias_plots/bias_{name}.png")
            print(f"Saved bias plot as bias_{name}.png")
            plt.show()

data_from_model(model)
plot_model_parameters(model)

print("\n✅ Ferdig! Modellens data er analysert!")
