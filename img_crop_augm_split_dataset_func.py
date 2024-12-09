import os
import cv2
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def preprocess_images(input_dir, output_dir, target_size=(64, 64)):
    """
    Preprocess images: Crop to the same size, convert .png to .jpg, and rename files.
    
    Args:
    - input_dir: Directory containing the digit folders with images.
    - output_dir: Directory to save the processed images.
    - target_size: Tuple (width, height) to resize images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each digit folder
    for digit in os.listdir(input_dir):
        digit_path = os.path.join(input_dir, digit)
        
        if os.path.isdir(digit_path):  # Only process directories
            # Create corresponding output directory
            digit_output_path = os.path.join(output_dir, digit)
            if not os.path.exists(digit_output_path):
                os.makedirs(digit_output_path)
            
            # Process each image in the digit folder
            for i, file_name in enumerate(os.listdir(digit_path)):
                file_path = os.path.join(digit_path, file_name)
                
                # Skip non-image files
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Load image using OpenCV
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Error loading {file_path}, skipping.")
                    continue
                
                # Resize image to target size
                resized_img = cv2.resize(img, target_size)
                
                # Convert to .jpg if the file is .png
                if file_name.lower().endswith('.png'):
                    new_file_name = f"digit_{digit}_{i+1:03d}.jpg"
                else:
                    new_file_name = f"digit_{digit}_{i+1:03d}.jpg"
                
                # Save the image in the output directory
                output_path = os.path.join(digit_output_path, new_file_name)
                cv2.imwrite(output_path, resized_img)
                
                print(f"Processed: {file_path} -> {output_path}")


input_directory = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/train"
output_directory = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/crop_img"
target_image_size = (224, 224)  # Change to desired size

# preprocess_images(input_directory, output_directory, target_image_size)




def augment_images(input_dir, output_dir, augmentations_per_image=45):
    """
    Perform augmentation on images and save the augmented images to a new directory.

    Args:
    - input_dir: Directory containing the original images.
    - output_dir: Directory to save the augmented images.
    - augmentations_per_image: Number of augmented images to generate per original image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define augmentation pipeline
    augmentation_pipeline = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% chance to flip horizontally # type: ignore
        iaa.Affine(rotate=(-90, 90)),  # Random rotation between -25 and 25 degrees
        iaa.Multiply((0.8, 1.2)),  # Change brightness
        iaa.Crop(percent=(0, 0.1)),  # Random crop up to 10%
        iaa.LinearContrast((0.8, 1.2))  # Adjust contrast
    ])

    # Process each folder of images
    for digit in os.listdir(input_dir):
        digit_path = os.path.join(input_dir, digit)

        if os.path.isdir(digit_path):  # Only process directories
            digit_output_path = os.path.join(output_dir, digit)
            if not os.path.exists(digit_output_path):
                os.makedirs(digit_output_path)

            # Process each image in the folder
            for file_name in os.listdir(digit_path):
                file_path = os.path.join(digit_path, file_name)

                # Load image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Error loading {file_path}, skipping.")
                    continue

                # Convert to RGB (optional, for color images)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Generate augmented images
                images_augmented = augmentation_pipeline(images=[img] * augmentations_per_image)

                # Save augmented images
                base_name, ext = os.path.splitext(file_name)
                for i, aug_img in enumerate(images_augmented): # type: ignore
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
                    output_path = os.path.join(digit_output_path, f"{base_name}_aug_{i+1:03d}.jpg")
                    cv2.imwrite(output_path, aug_img_bgr)

                print(f"Augmented {file_path} -> {digit_output_path}")

# Example Usage
input_directory = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/crop_img"  # Original images
output_directory = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_vit_model"  # Augmented images
augmentations_per_image = 47  # Number of augmentations per image

# augment_images(input_directory, output_directory, augmentations_per_image)





def split_dataset(input_dir, output_dir, val_split=0.2, test_split=0.1):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
    - input_dir: Path to the input dataset.
    - output_dir: Path to save the split dataset.
    - val_split: Fraction of data for validation.
    - test_split: Fraction of data for testing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_label in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_label)
        if not os.path.isdir(class_path):
            continue

        # Get all images in the class directory
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        train, val_test = train_test_split(images, test_size=val_split + test_split, random_state=42)
        val, test = train_test_split(val_test, test_size=test_split / (val_split + test_split), random_state=42)

        # Save splits
        for split, split_data in zip(["train", "val", "test"], [train, val, test]):
            split_dir = os.path.join(output_dir, split, class_label)
            os.makedirs(split_dir, exist_ok=True)
            for img_path in split_data:
                shutil.copy(img_path, os.path.join(split_dir, os.path.basename(img_path)))

dataset_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_vit_model" 
split_dataset_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_split" 
# Example usage
split_dataset(dataset_dir, split_dataset_dir)
