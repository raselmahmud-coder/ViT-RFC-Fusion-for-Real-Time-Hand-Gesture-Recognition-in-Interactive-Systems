# ViT-RFC Fusion for Real-Time Hand Gesture Recognition in Interactive Systems

Hand gesture recognition is crucial for facilitating seamless human-computer interaction, particularly in applications involving sign languages and numerical representations. This study focuses on the real-time recognition of Chinese Sign Language (CSL) for numbers 1 to 10 and American Sign Language (ASL) hand gestures for letters A to Z. Two distinct datasets were utilized, comprising over 5,000 CSL images and 31,200 ASL images. The models were trained using Vision Transformer (ViT) and Random Forest Classifiers (RFC).

For the CSL dataset, the ViT model achieved <b> an accuracy of 93.60%, while the RFC model achieved 95.63%. In the ASL dataset, the ViT model reached an accuracy of 87.60%, and the RFC model achieved an impressive 99.89%</b>. These results demonstrate the effectiveness of traditional machine learning architectures in handling complex gesture recognition tasks for these datasets.

## Features
- Real-time hand gesture recognition using a webcam
- Digit/Number recognition using a Vision Transformer (ViT) model
- Confidence score for each prediction
- Displays predicted digit and confidence on the video feed
- Hand detection ensures prediction is only made when a hand is detected

## Video Demo

[![Watch the video](https://img.youtube.com/vi/qjcZRC0SM1I/hqdefault.jpg)](https://www.youtube.com/watch?v=qjcZRC0SM1I)


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
