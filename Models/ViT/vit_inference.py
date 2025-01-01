# vit_inference.py
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import confusion_matrix, accuracy_score
from datasets import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, pivot_class=None):
    """
    Plots a confusion matrix with optional pivot highlighting.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        class_names (list): List of class names.
        pivot_class (str, optional): Class name to highlight as pivot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(14, 12))

    # Heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # If pivot_class is specified, highlight its row and column
    if pivot_class and pivot_class in class_names:
        pivot_index = class_names.index(pivot_class)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i == pivot_index or j == pivot_index:
                    plt.gca().get_children()[0].get_children()[i*len(class_names)+j].set_facecolor('yellow')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Plot the normalized confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix')

    if pivot_class and pivot_class in class_names:
        pivot_index = class_names.index(pivot_class)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i == pivot_index or j == pivot_index:
                    plt.gca().get_children()[0].get_children()[i*len(class_names)+j].set_facecolor('yellow')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Function to visualize trainer summary and metrics
def visualize_trainer_state(json_path, model, val_loader, device, class_names):
    with open(json_path, 'r') as f:
        trainer_state = json.load(f)

    # Extract scalar properties
    best_metric = trainer_state.get('best_metric', 'N/A')
    num_train_epochs = trainer_state.get('num_train_epochs', 'N/A')
    total_flos = trainer_state.get('total_flos', 'N/A')
    train_batch_size = trainer_state.get('train_batch_size', 'N/A')

    # Display Trainer Summary
    print("----- Trainer Summary -----")
    print(f"Best Metric: {best_metric}")
    print(f"Total Training Epochs: {num_train_epochs}")
    print(f"Total FLOPs: {total_flos}")
    print(f"Training Batch Size: {train_batch_size}")
    print("---------------------------\n")

    # Initialize lists to store evaluation metrics
    eval_accuracy = []
    eval_loss = []
    eval_precision = []
    eval_recall = []
    eval_epochs = []

    # Iterate through log_history to extract evaluation logs
    for log in trainer_state['log_history']:
        if 'eval_accuracy' in log:
            eval_epochs.append(log['epoch'])
            eval_accuracy.append(log['eval_accuracy'])
            eval_loss.append(log['eval_loss'])
            eval_precision.append(log['eval_precision'])
            eval_recall.append(log['eval_recall'])

    # Plot Combined Evaluation Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, eval_accuracy, marker='o', label='Accuracy', color='red')
    plt.plot(eval_epochs, eval_precision, marker='o', label='Precision', color='green')
    plt.plot(eval_epochs, eval_recall, marker='o', label='Recall', color='purple')
    plt.plot(eval_epochs, eval_loss, marker='o', label='Loss', color='blue')
    plt.title('Evaluation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Perform Inference on Validation Set to Compute Confusion Matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Verify that the number of predictions matches the number of labels
    assert len(all_preds) == len(all_labels), "Mismatch between number of predictions and labels."

    # Compute accuracy from confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    computed_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Computed Accuracy from Confusion Matrix: {computed_accuracy:.4f}")
    print(f"Reported Best Accuracy: {best_metric:.4f}\n")

    # Ensure that computed accuracy matches the reported best metric
    if np.isclose(computed_accuracy, best_metric, atol=1e-4):
        print("Success: Computed accuracy matches the reported best metric.")
    else:
        print("Warning: Computed accuracy does not match the reported best metric.")
        print("Possible reasons:")
        print("- The best metric might have been achieved on a different epoch.")
        print("- There might be discrepancies in how accuracy is calculated.")
        print("- The confusion matrix reflects the final epoch's performance, which might differ from the best epoch.")

    # Optionally, specify a pivot class to highlight
    pivot_class = None  # Replace with a specific class name if desired, e.g., 'A'

    # Plot Confusion Matrix with Pivot (if specified)
    plot_confusion_matrix(all_labels, all_preds, class_names, pivot_class=pivot_class)

    # Debugging: Print a few predictions vs true labels
    print("Sample Predictions vs True Labels:")
    for i in range(min(10, len(all_labels))):
        true_label = class_names[all_labels[i]]
        pred_label = class_names[all_preds[i]]
        print(f"True: {true_label}, Predicted: {pred_label}")

if __name__ == "__main__":
    def main():
        # Set the directory paths
        data_dir = Path('../Datasets/Chinese_Hand_Gestures_Dataset/aug_imgs_split')
        val_dir = data_dir / "val"

        # Load the pretrained model from checkpoint directory
        checkpoint_dir = Path("../Trained_Results/vit_results/checkpoint-1582")  
        model = ViTForImageClassification.from_pretrained(checkpoint_dir)

        # Ensure the model is in evaluation mode
        model.eval()

        # Load the feature extractor used during training
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)

        # Prepare ImageFolder datasets
        val_image_folder = ImageFolder(root=val_dir)

        def get_image_paths_and_labels(image_folder):
            paths = [str(path) for path, _ in image_folder.imgs]
            labels = [label for _, label in image_folder.imgs]
            return {"image_path": paths, "label": labels}

        val_data = get_image_paths_and_labels(val_image_folder)

        # Convert to HuggingFace Dataset
        val_dataset_hf = Dataset.from_dict(val_data)

        # Define preprocessing function (same as in training)
        def preprocess_function(examples):
            images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
            processed = feature_extractor(images=images, return_tensors="pt")
            # Flatten the batch dimension
            pixel_values = processed["pixel_values"]
            return {"pixel_values": pixel_values, "label": examples["label"]}

        # Apply preprocessing
        val_dataset_hf = val_dataset_hf.map(preprocess_function, batched=True, batch_size=32, remove_columns=["image_path"])

        # Set format for PyTorch
        val_dataset_hf.set_format(type="torch", columns=["pixel_values", "label"])

        # Create DataLoader for validation set with num_workers=0
        val_loader = DataLoader(val_dataset_hf, batch_size=16, shuffle=False, num_workers=0)

        # Check if GPU is available and move the model to GPU if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model is loaded on device: {device}\n")

        # Path to trainer_state.json
        trainer_state_path = Path("../Trained_Results/vit_results/checkpoint-1582/trainer_state.json")
        
        if not trainer_state_path.exists():
            print(f"Trainer state file not found at {trainer_state_path}")
            return

        # Retrieve class names from validation dataset
        class_names = val_image_folder.classes

        # Visualize trainer state and plot confusion matrix
        visualize_trainer_state(trainer_state_path, model, val_loader, device, class_names)

    main()
