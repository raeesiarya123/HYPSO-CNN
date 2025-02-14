import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *

#######################################################################################
#######################################################################################
#######################################################################################

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation loss and accuracy.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()


def visualize_predictions(model, dataloader, class_names, device):
    """
    Visualize model predictions on validation/test images.
    """
    model.eval()
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot images with predicted labels
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(10):
        img = images[i].cpu().numpy().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Pred: {class_names[preds[i]]} | True: {class_names[labels[i]]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model, image, layers_to_visualize=[0, 2]):
    """
    Visualize feature maps from convolutional layers.
    """
    model.eval()
    activation_maps = []

    def hook_fn(module, input, output):
        activation_maps.append(output)

    # Hook to store feature maps
    hooks = []
    for i, layer in enumerate(model.children()):
        if i in layers_to_visualize:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Plot feature maps
    for idx, feature_map in enumerate(activation_maps):
        num_features = feature_map.shape[1]
        fig, axes = plt.subplots(1, min(8, num_features), figsize=(15, 5))
        for i in range(min(8, num_features)):
            axes[i].imshow(feature_map[0, i].cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
        plt.suptitle(f"Feature Map - Layer {layers_to_visualize[idx]}")
        plt.show()