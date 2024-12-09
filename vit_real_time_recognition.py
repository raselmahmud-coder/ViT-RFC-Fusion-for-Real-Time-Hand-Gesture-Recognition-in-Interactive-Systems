import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# Load the trained model (adjust path as needed)
checkpoint_dir = "./vit_results/checkpoint-1582"
model = ViTForImageClassification.from_pretrained(checkpoint_dir)
model.eval()

# Define the preprocessing pipeline for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for ViT input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Function to process image and get prediction
def process_and_predict(image):
    # Convert the image to PIL format for transformation
    img = Image.fromarray(image).convert('RGB')

    # Apply the transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits
        confidences = torch.softmax(logits, dim=1)  # Confidence scores
        preds = torch.argmax(logits, dim=1)  # Predicted class
        return preds.item(), confidences[0, preds.item()].item()

# Function to detect if a hand is in the frame
def detect_hand(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding to detect hand contours (simple method)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, it indicates a hand is present
    if contours:
        # Find the largest contour (most likely the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 5000:  # Area threshold to filter out small contours
            return True  # Hand detected
    return False  # No hand detected

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Detect if a hand is present in the frame
    hand_detected = detect_hand(frame)

    if hand_detected:
        # If hand is detected, get prediction for the current frame
        predicted_class, confidence = process_and_predict(frame)

        # Format the predicted class with leading zeros (e.g., 01, 02, 10)
        predicted_class_str = f"{predicted_class+1}" 
        confidence_str = f"Confidence: {confidence*100:.2f}%"

        # Put the predicted class and confidence on the frame
        cv2.putText(frame, f"Predicted Digit: {predicted_class_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, confidence_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # If no hand is detected, don't display any text
        cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Hand Digit Recognition', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
