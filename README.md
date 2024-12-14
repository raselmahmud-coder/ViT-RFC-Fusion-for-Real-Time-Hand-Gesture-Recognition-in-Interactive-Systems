# Real-Time Hand Digit Recognition using ViT (Vision Transformer)

This project implements a real-time hand digit recognition system using the Vision Transformer (ViT) model and OpenCV for webcam input. The model is trained to recognize hand gestures representing digits (0-9) from the Chinese Hand Gesture Number Recognition Dataset.

## Features
- Real-time hand gesture recognition using a webcam
- Digit/Number recognition using a Vision Transformer (ViT) model
- Confidence score for each prediction
- Displays predicted digit and confidence on the video feed
- Hand detection ensures prediction is only made when a hand is detected

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgements](#acknowledgements)

## Project Overview
This project demonstrates the application of a Vision Transformer (ViT) model for real-time hand digit recognition. The model was trained on a custom dataset of approximately 5000 containing images of hand gestures representing digits (1-10) and can classify them based on webcam input. The system detects 87% accurately in this tiny dataset.

The goal is to develop an interactive, real-time system that can recognize hand signs and provide accurate predictions, making it applicable for applications such as virtual sign language recognition, gesture-based control systems, accessible people use-case and educational tools.

## Video Demo
<video controls width="60%">
  <source src="https://github.com/raselmahmud-coder/real-time_sign_recognition/blob/main/Results/Video_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
Or [Download the demo video](./results/demo.mp4)

## Requirements
Before running the project, ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.9+
- OpenCV 4.5+
- Pillow
- transformers
- numpy
- matplotlib

### Please check out the requirement.txt file

You can install the necessary libraries using pip:

```bash
pip install torch torchvision opencv-python pillow transformers numpy matplotlib
```

## Setup Instructions

### 1. Clone the Repository
Clone the project repository to your local machine:

```
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Download or Train the Model
**Option 1:** Use the pre-trained model from the checkpoint directory (you should already have the model trained). Place your trained model in the directory `./vit_results/checkpoint-1582/`.

**Option 2:** Train the model on the dataset and save the checkpoint. The model should be a Vision Transformer (ViT) model fine-tuned for your specific hand gesture dataset.

### 3. Dataset Structure
Ensure your dataset is organized as follows:

```
/Datasets
    /Chinese Hand Gestures Number Recognition Dataset
        /aug_imgs_split
            /train
                /01
                /02
                /03
                ...
            /val
                /01
                /02
                /03
                ...
            /test
                /01
                /02
                /03
                ...
```

The dataset should be divided into training, validation, and testing folders, each containing subfolders for each digit class (e.g., 01, 02, ..., 10).

## Usage

### 1. Real-Time Hand Gesture Recognition with Webcam
To run the real-time digit recognition system using your webcam after training with your dataset, execute the following command:

```
python vit_real_time_recognition.py
```

Once you run the script, a window will open displaying the webcam feed. If a hand is detected in the frame, the model will predict the digit and display it along with the confidence score. If no hand is detected, it will display "No Hand Detected."

### 2. Display Predicted Digit and Confidence
The predicted digit will be shown on the video feed with the format:

```
Predicted Digit: 01
Confidence: ~95.12%
```

- **Predicted Digit:** XX shows the predicted digit, formatted with leading zeros.
- **Confidence:** XX% shows the confidence level for the prediction.

- **Validation Accuracy:** 87%
- **Test Accuracy:** 86%


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- **Vision Transformer (ViT):** For digit recognition.
- **OpenCV:** For real-time image processing and webcam input.
- **Chinese Hand Gesture Number Recognition Dataset:** For training the model.