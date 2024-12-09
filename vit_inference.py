import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ViT
])

# Load datasets for validation, training, and testing
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Helper function to perform inference
def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_images = []  
    all_confidences = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Move images and labels to the device (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs.logits

            # Get predictions (argmax of logits)
            preds = torch.argmax(logits, dim=1)
            confidences = torch.softmax(logits, dim=1) 
            max_confidence = torch.max(confidences, dim=1).values

            # Collect predictions, true labels, and images
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            all_confidences.extend(max_confidence.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_images), np.array(all_confidences)


# Function to plot sample images with predictions and confidence scores
def plot_sample_images_with_confidence(images, labels, preds, confidences, class_names, num_samples=5):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        
        # Reverse the normalization for visualization
        img = img * np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :] + np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        
        # Clip to valid range [0, 1] for imshow
        img = np.clip(img, 0, 1)
        
        # Prepare the predicted label and confidence score
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence_score = confidences[i] * 100  # Convert to percentage

        # Plot the image with title (True label vs Predicted label with confidence)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence_score:.2f}%)")
        plt.axis('off')
    plt.show()


# Function to plot the comparison
def plot_comparison(metrics):
    labels = ['Validation', 'Test']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Accuracy
    axes[0, 0].bar(labels, metrics['Accuracy'], color=['blue', 'green'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')

    # Plot F1 Score
    axes[0, 1].bar(labels, metrics['F1 Score'], color=['blue', 'green'])
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')

    # Plot Precision
    axes[1, 0].bar(labels, metrics['Precision'], color=['blue', 'green'])
    axes[1, 0].set_title('Precision Comparison')
    axes[1, 0].set_ylabel('Precision')

    # Plot Recall
    axes[1, 1].bar(labels, metrics['Recall'], color=['blue', 'green'])
    axes[1, 1].set_title('Recall Comparison')
    axes[1, 1].set_ylabel('Recall')

    plt.tight_layout()
    plt.show()


# Function to compute evaluation metrics
def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return accuracy, f1, precision, recall

# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot sample images
def plot_sample_images(images, labels, preds, class_names, num_samples=5):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        plt.axis('off')
    plt.show()


# Main inference logic
def main():
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # type: ignore

    # Get class names from the dataset
    class_names = val_dataset.classes

    # Evaluate on the validation dataset
    val_labels, val_preds, val_images, val_confidences = evaluate(model, val_loader, device)

    # Compute and print evaluation metrics for validation
    val_accuracy, val_f1, val_precision, val_recall = compute_metrics(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    plot_sample_images_with_confidence(val_images, val_labels, val_preds, val_confidences, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds, class_names)

    # Plot sample images with true vs predicted labels
    plot_sample_images(val_images, val_labels, val_preds, class_names)

    # Evaluate on the test dataset
    test_labels, test_preds, test_images, test_confidences = evaluate(model, test_loader, device)

    # Compute and print evaluation metrics for test
    test_accuracy, test_f1, test_precision, test_recall = compute_metrics(test_labels, test_preds)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    # Plot sample images with true vs predicted labels and confidence for test
    plot_sample_images_with_confidence(test_images, test_labels, test_preds, test_confidences, class_names)
    # Store the metrics for easy comparison
    metrics = {
        'Accuracy': [val_accuracy, test_accuracy],
        'F1 Score': [val_f1, test_f1],
        'Precision': [val_precision, test_precision],
        'Recall': [val_recall, test_recall]
    }
    # Call the function to plot comparison
    plot_comparison(metrics)

    


if __name__ == "__main__":
    main()




