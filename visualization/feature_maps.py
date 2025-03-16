import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from config import *
from models.cnn_1d import cnn_1d
from scripts.dataset import hyperspectral_dataset
from scripts.data_management import read_csv_file

#######################################################################################
#######################################################################################
#######################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = cnn_1d(input_dim=120, num_classes=3).to(device)
checkpoint_path = "models/best_model.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"File {checkpoint_path} not found!")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Retrieve paths from train_files.csv
train_bip_files, train_dat_files = read_csv_file("train_files.csv")

# Load the dataset using the first training file
dataset = hyperspectral_dataset(top_folder_name=train_bip_files[0], label_path=train_dat_files[0], apply_augment=False)

# Get the first image as input (adding batch dimension)
image_input = dataset.base_images[0].T[:120, :2000].unsqueeze(0).to(device)

def get_feature_maps(model, x):
    torch.cuda.empty_cache()
    activations = []
    hooks = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv1d):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(x.to(device))
    
    for hook in hooks:
        hook.remove()
    
    return activations

def plot_feature_maps(activations, num_layers=6):
    for i, activation in enumerate(activations[:num_layers]):
        num_feature_maps = activation.shape[1]
        
        fig, axes = plt.subplots(1, min(num_feature_maps, 6), figsize=(15, 5))
        fig.suptitle(f"Layer {i+1} - Feature Maps")
        plt.subplots_adjust(wspace=1.2)  # Add more space between subplots
        
        for j in range(min(num_feature_maps, 6)):
            ax = axes[j] if num_feature_maps > 1 else axes
            img = ax.imshow(activation[0, j, :].reshape(1, -1), cmap="inferno", aspect="auto")
            ax.set_xlabel("Pixel Index")
            ax.set_ylabel("Feature Index")
            ax.set_xticks(range(0, activation.shape[2], activation.shape[2] // 5))
            ax.set_yticks([0])
            fig.colorbar(img, ax=ax)
            ax.axis("on")
        
        plt.savefig(f"plots/feature_map/feature_map_real_data_layer_{i+1}.png")
        plt.show()

def plot_feature_maps(activations, num_layers=6):
    for i, activation in enumerate(activations[:num_layers]):
        num_feature_maps = activation.shape[1]
        
        fig, axes = plt.subplots(1, min(num_feature_maps, 6), figsize=(18, 6))
        fig.suptitle(f"Layer {i+1} - Feature Maps", fontsize=14)
        plt.subplots_adjust(wspace=1.2)  # Øk mellomrom mellom subplottene

        for j in range(min(num_feature_maps, 6)):
            ax = axes[j] if num_feature_maps > 1 else axes
            img = ax.imshow(activation[0, j, :].reshape(1, -1), cmap="inferno", aspect="auto")

            ax.set_xlabel("Pixel Index", fontsize=12)
            ax.set_ylabel("Feature Index", fontsize=12)

            # Juster x-ticks for bedre lesbarhet
            x_ticks = range(0, activation.shape[2], max(1, activation.shape[2] // 5))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, rotation=45, ha="right")

            # Sett y-ticks
            ax.set_yticks([0])

            # Legg til colorbar og øk skriftstørrelsen
            cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

            ax.axis("on")
        
        plt.savefig(f"plots/feature_map/feature_map_real_data_layer_{i+1}.png", bbox_inches="tight")
        plt.show()


# Extract feature maps from real training images
activations = get_feature_maps(model, image_input)

# Plot feature maps with better visualization
plot_feature_maps(activations)