# preprocess_data.py

import os
import pickle
import mediapipe as mp 
import cv2
from pathlib import Path

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# Configure Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = Path('../Datasets/Chinese_Hand_Gestures_Dataset/aug_imgs_vit_model')
directories = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

data = []
labels = []

for dir_path in directories: 
    label_name = dir_path.name
    print(f"Processing directory: {label_name}")
    files = [f.name for f in dir_path.iterdir() if f.is_file()]

    # Check if directory exists (redundant since we iterated over directories)
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        continue

    for img_name in sorted(files):  # You can limit with [:1] for testing
        full_img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(full_img_path)
        data_aux = []
        x_ = []
        y_ = []
        
        if img is None:
            print(f"Failed to read image: {full_img_path}")
            continue
        
        # Convert to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and extract hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Normalize coordinates by subtracting the minimum to ensure positive values
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(label_name)

# Close Mediapipe Hands
hands.close()

# Verify data and labels lengths
print(f"\nTotal samples collected: {len(data)}")
print(f"Total labels collected: {len(labels)}")

# Save the data and labels to a pickle file using a context manager
pickle_file_path = Path('../Trained_Results/Models/Chinese_digit_RFC.pickle')
with open(pickle_file_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Successfully saved the pickle file to {pickle_file_path}")
