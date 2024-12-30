import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import mediapipe as mp

# Load the pretrained ViT model
checkpoint_dir = "./vit_trained_model"
model = ViTForImageClassification.from_pretrained(checkpoint_dir)
model.eval()

# Define the preprocessing pipeline for the ViT model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for ViT input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Standard normalization
])

# Path to validation dataset
val_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_split/train"

# Load the dataset using ImageFolder
val_dataset = ImageFolder(root=val_dir, transform=transform)

# Create a DataLoader with shuffling enabled
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Function to preprocess the detected hand region and get the model prediction


def process_and_predict(image_tensor):
    # Run the image through the model
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))  # Add batch dimension
        logits = outputs.logits
        confidences = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = confidences[0, predicted_class].item()
    return predicted_class, confidence

# Function to set the window position and size


def configure_window(window_name, width=800, height=600, center=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.resizeWindow(window_name, width, height)  # Set custom width and height

    if center:
        # Get screen resolution
        screen_width = cv2.getWindowImageRect(window_name)[2]
        screen_height = cv2.getWindowImageRect(window_name)[3]

        # Calculate center position
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Position the window at the center
        cv2.moveWindow(window_name, x, y)


# Iterate over the DataLoader to get random batches of images
for batch_idx, (images, labels) in enumerate(val_loader):
    for idx in range(images.size(0)):
        img_tensor = images[idx]  # Single image tensor
        label = labels[idx]       # Corresponding label

        # Convert image tensor to NumPy array for visualization
        # Convert (C, H, W) to (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + \
            np.array([0.485, 0.456, 0.406])  # Denormalize
        # Ensure pixel values are in range [0, 1]
        img_np = np.clip(img_np, 0, 1)
        # Convert to [0, 255] range for OpenCV
        img_np = (img_np * 255).astype(np.uint8)

        frame_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Predict the digit using the pretrained model
        predicted_class, confidence = process_and_predict(img_tensor)

        # Add text for actual label and prediction
        # Adjust for 1-based indexing
        predicted_class_str = f"{predicted_class + 1}"
        confidence_str = f"{confidence * 100:.2f}%"
        # Adjust for 1-based indexing
        actual_label_str = f"Actual Number: {label + 1}"
        color = (0, 0, 255) if label != predicted_class else (0, 207, 13)

        cv2.putText(frame_rgb, actual_label_str, (3, 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 156), 1)
        cv2.putText(frame_rgb, f"Predict: {predicted_class_str} Conf.{confidence_str}", (
            3, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)

        # Display the image with annotations
        window_name = "Hand Digit Recognition"
        configure_window(window_name, width=700, height=550, center=True)
        cv2.imshow(window_name, frame_rgb)
        key = cv2.waitKey(0)  # Wait for key press to move to the next image

        if key & 0xFF == ord('q'):
            break  # Exit loop if 'q' is pressed

    else:
        continue
    break

cv2.destroyAllWindows()
