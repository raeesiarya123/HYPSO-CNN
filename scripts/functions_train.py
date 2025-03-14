import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

# Get an earlier trained model for shorter training time
def get_earlier_model(device, model, path="models/best_model.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)

        if any("_orig_mod" in key for key in checkpoint['model_state_dict'].keys()):
            print("Fixing `_orig_mod` issue in state_dict!")
            new_state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model_state_dict'].items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resuming training from earlier training round! Best accuracy: {best_accuracy:.2f}%")
    else:
        best_accuracy = 0.0
        print(f"No earlier model found, starting with best_accuracy = {best_accuracy}.")

    return best_accuracy

#######################################################################################

# Plot pixel spectral accuracy
def plot_pixel_spectral_accuracy(prediction_list, label_list, epoch_num):
    if isinstance(prediction_list, torch.Tensor):
        prediction_list = prediction_list.cpu().numpy()
    if isinstance(label_list, torch.Tensor):
        label_list = label_list.cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    plt.plot(prediction_list, label='Predictions', color='blue')
    plt.plot(label_list, label='Labels', color='red')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Pixel Spectral Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/train/prediction_train_EPOCH{epoch_num}")
    plt.show()

#######################################################################################

# Normalize spectrum
def normalize_spectrum(spectrum):
    spectrum = spectrum.float().cuda()
    min_vals = spectrum.min(dim=1, keepdim=True)[0]
    max_vals = spectrum.max(dim=1, keepdim=True)[0]
    
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8

    return (spectrum - min_vals) / range_vals