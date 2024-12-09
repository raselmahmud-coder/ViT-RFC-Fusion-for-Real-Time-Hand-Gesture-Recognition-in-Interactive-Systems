import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# Load the trained model (adjust path as needed)
checkpoint_dir = "./vit_results/checkpoint-1582"
model = ViTForImageClassification.from_pretrained(checkpoint_dir)
model.eval()

# Define the preprocessing pipeline for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to process image and get prediction
def process_and_predict(image_path):
    # Open the image using PIL
    img = Image.open(image_path).convert('RGB')

    # Apply the transformations (no resizing, as it's already 224x224)
    img_tensor = transform(img).unsqueeze(0)  # type: ignore

# Run the image through the model
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits
        confidences = torch.softmax(logits, dim=1)  # Confidence scores
        preds = torch.argmax(logits, dim=1)  # Predicted class
        return preds.item(), confidences[0, preds.item()].item() # type: ignore

    # Get predicted class (argmax of logits)
    preds = torch.argmax(logits, dim=1)

    return preds.item()

# Set the root directory for validation images
val_root_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_split/val"

# Loop over each subdirectory (representing classes) in the validation set
for class_dir in os.listdir(val_root_dir):
    class_path = os.path.join(val_root_dir, class_dir)
    
    if os.path.isdir(class_path):
        # Loop through the images in the directory (one image per class for testing)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Process and predict the class
            predicted_class, confidence = process_and_predict(image_path)
            predicted_class += 1  # Adjust to 1-based index
            
            # Plot the image with actual and predicted class as the title
            img = Image.open(image_path).convert('RGB')
            plt.imshow(np.array(img))
            plt.title(f"Actual Digit: {class_dir}, Predicted Digit: {predicted_class:02}, Confidence: {confidence*100:.2f}%")  # Format predicted class as 01, 02, ...
            plt.axis('off')
            plt.show()

            # Break after the first image for each class (since we want only one per class)
            break
