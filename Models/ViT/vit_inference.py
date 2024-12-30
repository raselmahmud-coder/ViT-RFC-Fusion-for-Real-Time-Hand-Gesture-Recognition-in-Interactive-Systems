import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torchvision.datasets import ImageFolder

# Set the directory paths
data_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_split"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"
test_dir = f"{data_dir}/test"  

# Load the pretrained model from checkpoint directory
checkpoint_dir = "./vit_results/checkpoint-1582"  
model = ViTForImageClassification.from_pretrained(checkpoint_dir)

# Ensure the model is in evaluation mode
model.eval()

# Define the preprocessing pipeline manually (since ViTImageProcessor might not be available)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT input size typically 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Load datasets for validation, training, and testing
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Function to load and visualize `trainer_state.json`
def visualize_trainer_state(json_path):
    with open(json_path, 'r') as f:
        trainer_state = json.load(f)

    epochs = []
    eval_accuracy = []
    eval_loss = []
    eval_precision = []
    eval_recall = []

    for log in trainer_state['log_history']:
        if 'eval_accuracy' in log:
            epochs.append(log['epoch'])
            eval_accuracy.append(log['eval_accuracy'])
            eval_loss.append(log['eval_loss'])
            eval_precision.append(log['eval_precision'])
            eval_recall.append(log['eval_recall'])

    # Plot evaluation metrics
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, eval_accuracy, marker='o', label='Accuracy')
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, eval_loss, marker='o', label='Loss', color='orange')
    plt.title('Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, eval_precision, marker='o', label='Precision', color='green')
    plt.title('Evaluation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, eval_recall, marker='o', label='Recall', color='red')
    plt.title('Evaluation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main inference logic
def main():
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Visualize trainer state
    trainer_state_path = "./vit_results/checkpoint-1582/trainer_state.json" 
    visualize_trainer_state(trainer_state_path)

if __name__ == "__main__":
    main()
