import matplotlib.pyplot as plt
import re

def plot_training_graphs(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = f.read()

    # Extract metrics using regex
    train_losses = [float(x) for x in re.findall(r'Loss:\s*([0-9.]+)', log_data)]
    train_ious = [float(x) for x in re.findall(r'\[Training\] IOU:\s*([0-9.]+)', log_data)]
    val_ious = [float(x) for x in re.findall(r'\[Validation\] IOU:\s*([0-9.]+)', log_data)]

    epochs = range(len(train_losses))

    # Create the plots
    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', linewidth=2, label='Training Loss')
    plt.title('Model Optimization (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot 2: Intersection over Union (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ious, 'b--', alpha=0.5, label='Train IoU')
    plt.plot(epochs, val_ious, 'g-', linewidth=2, label='Validation IoU')
    plt.title('Segmentation Accuracy (IoU)')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_graphs.png', dpi=300)
    print("Graph successfully saved as 'training_graphs.png'!")

if __name__ == '__main__':
    plot_training_graphs('training_log.txt')