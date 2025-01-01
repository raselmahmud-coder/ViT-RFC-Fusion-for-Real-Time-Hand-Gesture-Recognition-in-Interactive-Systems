import os
import cv2
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path


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




def is_image(file_path):
    """Check if a file is an image by attempting to read it with cv2."""
    img = cv2.imread(file_path)
    return img is not None

def augment_images(input_dir, output_dir, target_total_image=1200):
    """
    Perform augmentation on images and save the augmented images to a new directory,
    ensuring all folders have the same number of images by augmenting to match the target.

    Args:
    - input_dir: Directory containing the original images.
    - output_dir: Directory to save the augmented images.
    - target_total_image: The desired number of images for each folder after augmentation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define augmentation pipeline
    augmentation_pipeline = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% chance to flip horizontally
        iaa.Affine(rotate=(-90, 90)),  # Random rotation between -90 and 90 degrees
        iaa.Multiply((0.8, 1.5)),  # Change brightness
        iaa.Crop(percent=(0, 0.1)),  # Random crop up to 10%
        iaa.LinearContrast((0.6, 1.5))  # Adjust contrast
    ])

    # Iterate through each folder in the input directory
    for digit in os.listdir(input_dir):
        digit_path = os.path.join(input_dir, digit)

        if os.path.isdir(digit_path):
            digit_output_path = os.path.join(output_dir, digit)
            if not os.path.exists(digit_output_path):
                os.makedirs(digit_output_path)

            # Get list of image files in the current folder
            image_files = [f for f in os.listdir(digit_path)
                           if os.path.isfile(os.path.join(digit_path, f)) and is_image(os.path.join(digit_path, f))]

            current_num_images = len(image_files)
            print(f"{digit}: {current_num_images} images")

            if current_num_images == 0:
                print(f"Skipping {digit} as it has no images.")
                continue

            # Copy original images to the output directory
            for file_name in image_files:
                src_path = os.path.join(digit_path, file_name)
                dst_path = os.path.join(digit_output_path, file_name)
                if not os.path.exists(dst_path):  # Avoid overwriting if already exists
                    img = cv2.imread(src_path)
                    if img is not None:
                        cv2.imwrite(dst_path, img)
                    else:
                        print(f"Failed to copy {src_path}.")

            # Calculate how many additional images are needed
            if current_num_images >= target_total_image:
                print(f"{digit} already has {current_num_images} images, which meets or exceeds the target.")
                continue

            required_augmentations = target_total_image - current_num_images
            print(f"{digit} needs {required_augmentations} additional images. Generating them...")

            # Calculate augmentations per image
            augmentations_per_image = required_augmentations // current_num_images
            remainder = required_augmentations % current_num_images
            print(f"Each image will be augmented {augmentations_per_image} times with {remainder} images augmented once more.")

            # Generate augmented images
            for idx, file_name in enumerate(image_files):
                src_path = os.path.join(digit_path, file_name)
                img = cv2.imread(src_path)
                if img is None:
                    print(f"Error loading {src_path}, skipping.")
                    continue

                # Convert to RGB for augmentation
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Determine number of augmentations for this image
                num_augmentations = augmentations_per_image + (1 if idx < remainder else 0)

                if num_augmentations <= 0:
                    continue  # No augmentation needed for this image

                # Generate augmented images
                images_augmented = augmentation_pipeline(images=[img_rgb] * num_augmentations)

                # Save augmented images
                base_name, ext = os.path.splitext(file_name)
                for i, aug_img in enumerate(images_augmented):
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
                    augmented_file_name = f"{base_name}_aug_{i+1:03d}.jpg"
                    output_path = os.path.join(digit_output_path, augmented_file_name)
                    success = cv2.imwrite(output_path, aug_img_bgr)
                    if success:
                        print(f"Saved augmented image: {output_path}")
                    else:
                        print(f"Failed to save augmented image: {output_path}")

    print("Augmentation complete.")

# Example Usage
if __name__ == "__main__":
    DATA_DIR = Path('../Datasets/ASL_Data')
    input_directory = DATA_DIR  # Original images
    output_directory = Path('../Datasets/ASL_Aug_Data')  # Augmented images
    target_total_image = 1200  # Desired number of images per folder
    os.makedirs(output_directory, exist_ok=True)

    # augment_images(input_directory, output_directory, target_total_image)





def split_dataset(input_dir, output_dir, val_split=0.2):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
    - input_dir: Path to the input dataset.
    - output_dir: Path to save the split dataset.
    - val_split: Fraction of data for validation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_label in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_label)
        if not os.path.isdir(class_path):
            continue

        # Get all images in the class directory
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        train, val_test = train_test_split(images, test_size=val_split, random_state=42)
        val, test = train_test_split(val_test, test_size= val_split, random_state=42)

        # Save splits
        for split, split_data in zip(["train", "val"], [train, val]):
            split_dir = os.path.join(output_dir, split, class_label)
            os.makedirs(split_dir, exist_ok=True)
            for img_path in split_data:
                shutil.copy(img_path, os.path.join(split_dir, os.path.basename(img_path)))

dataset_dir = Path("../Datasets/ASL_Aug_Data_ViT")
split_dataset_dir = Path("../Datasets/ASL_ViT_Split_Data") 
# Example usage
split_dataset(dataset_dir, split_dataset_dir)
