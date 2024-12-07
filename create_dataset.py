import os
import pickle

import mediapipe as mp # type: ignore
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Datasets/American Sign Language Digits Dataset'

data = []
labels = []
for label in range(10): 
    img_dir = os.path.join(DATA_DIR, str(label), "Input Images - Sign " + str(label))

    # Check if directory exists
    if not os.path.exists(img_dir):
        print(f"Directory not found: {img_dir}")
        continue

    for img_path in os.listdir(img_dir):
            full_img_path = os.path.join(img_dir, img_path)
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(full_img_path)
            if img is None:
                print(f"Failed to read image: {full_img_path}")
                continue
            # Convert to RGB for displaying
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(img_path)





# print(data, labels, "hello data and labels")
        # Draft for testing land mark detection
# Iterate through labels 0 to 9
    # for label in range(10): 
    #     img_dir = os.path.join(DATA_DIR, str(label), "Input Images - Sign " + str(label))
    #     print(img_dir, "hello dir")

    #     # Check if directory exists
    #     if not os.path.exists(img_dir):
    #         print(f"Directory not found: {img_dir}")
    #         continue

    #     # Read the first image in the directory
    #     for img_path in os.listdir(img_dir)[:1]:
    #         img_full_path = os.path.join(img_dir, img_path)
    #         img = cv2.imread(img_full_path)

    #         if img is None:
    #             print(f"Failed to read image: {img_full_path}")
    #             continue

    #         # Convert to RGB for displaying
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         # Process the image with Mediapipe
    #         results = hands.process(img_rgb)
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 mp_drawing.draw_landmarks(
    #                     img_rgb,
    #                     hand_landmarks,
    #                     mp_hands.HAND_CONNECTIONS,
    #                     mp_drawing_styles.get_default_hand_landmarks_style(),
    #                     mp_drawing_styles.get_default_hand_connections_style()
    #                 )

    #         # Display the processed image
    #         plt.figure()
    #         plt.imshow(img_rgb)
    #         plt.title(f"Label: {label}")

    # # Show all images
    # plt.show()




f = open('./Datasets/Models/data_sign.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
print("Successfully saved the file", pickle.ADDITEMS)
f.close()
