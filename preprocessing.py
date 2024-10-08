import os
import cv2
import fnmatch
import random
import numpy as np
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Define the paths to the folders containing different classes of breast cancer images
folders = {
    "Benign": "Dataset_BUSI_with_GT/benign",
    "Malignant": "Dataset_BUSI_with_GT/malignant",
    "Normal": "Dataset_BUSI_with_GT/normal"
}

# Function to load an image in grayscale mode
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image

# Function to display an image using OpenCV
def display_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the image window after key press

# Function to merge multiple mask images into a single mask
def merge_masks(masks):
    merged_mask = masks[0].copy()  # Start with the first mask
    # Iterate through the remaining masks and merge them using bitwise OR
    for mask in masks[1:]:
        merged_mask = cv2.bitwise_or(merged_mask, mask)
    return merged_mask

# Function to apply rotation to an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]  # Get the height and width of the image
    center = (w // 2, h // 2)  # Define the center of the image for rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Compute the rotation matrix
    rotated = cv2.warpAffine(image, M, (w, h))  # Apply the rotation
    return rotated

# Function to apply horizontal flip to an image
def flip_image(image):
    return cv2.flip(image, 1)  # Flip the image horizontally

# Function to apply additional augmentation (e.g., Gaussian blur) for 'Normal' class
def additional_augmentation(image):
    return cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur

# Function to save an image and its corresponding mask to the specified folder
def save_image_and_ground_truth(image, mask, image_class, folder_path):
    global unique_id
    image_name = f'{image_class}_{unique_id}.png'
    mask_name = f'{image_class}_{unique_id}_mask.png'
    cv2.imwrite(os.path.join(folder_path, image_name), image)
    cv2.imwrite(os.path.join(folder_path, mask_name), mask)
    unique_id += 1  # Increment the unique ID for naming the next image

# Main function to process and store images with optional augmentation
def process_images(breast_image, final_mask, folder_name, final_folder_name='final_dataset', augment_data=False):
    # Resize images and masks to a standard size (256x256)
    breast_image = cv2.resize(breast_image, (256, 256))
    final_mask = cv2.resize(final_mask, (256, 256))
    
    # Create the final dataset folder if it doesn't exist
    os.makedirs(final_folder_name, exist_ok=True)
    
    # Create subfolder for the specific type (Benign, Malignant, Normal)
    folder_path = os.path.join(final_folder_name, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Process 'Benign' class: simply store without augmentation
    if folder_name == 'Benign':
        save_image_and_ground_truth(breast_image, final_mask, folder_name, folder_path)

    # Process 'Malignant' class: duplicate with either random rotation or horizontal flip
    elif folder_name == 'Malignant':
        save_image_and_ground_truth(breast_image, final_mask, folder_name, folder_path)
        if augment_data:
            if random.choice([True, False]):  # Randomly choose rotation or flip
                angle = random.randint(1, 360)  # Random rotation angle
                augmented_image = rotate_image(breast_image.copy(), angle)
                augmented_mask = rotate_image(final_mask.copy(), angle)
            else:
                augmented_image = flip_image(breast_image.copy())
                augmented_mask = flip_image(final_mask.copy())
            save_image_and_ground_truth(augmented_image, augmented_mask, folder_name, folder_path)

    # Process 'Normal' class: quadruply the data with random augmentations
    elif folder_name == 'Normal':
        save_image_and_ground_truth(breast_image, final_mask, folder_name, folder_path)
        
        if augment_data:
            # Random rotation
            angle = random.randint(1, 360)
            augmented_image = rotate_image(breast_image.copy(), angle)
            augmented_mask = rotate_image(final_mask.copy(), angle)
            save_image_and_ground_truth(augmented_image, augmented_mask, folder_name, folder_path)

            # Horizontal flip
            augmented_image = flip_image(breast_image.copy())
            augmented_mask = flip_image(final_mask.copy())
            save_image_and_ground_truth(augmented_image, augmented_mask, folder_name, folder_path)

            # Additional augmentation (Gaussian blur)
            augmented_image = additional_augmentation(breast_image.copy())
            save_image_and_ground_truth(augmented_image, final_mask.copy(), folder_name, folder_path)

# Function to process masks and images and prepare them for further use
def process_masks_and_images(image_path, folder_path, folder_name, final_folder_name, augment_data):
    # Extract the base name of the image file (without extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create the mask pattern to match files like "image_name_mask.png", "image_name_mask_1.png", etc.
    mask_pattern = f"{image_name}_mask*.png"
    
    masks = []
    # Search for all files that match the mask pattern in the same folder
    for filename in os.listdir(folder_path):
        if fnmatch.fnmatch(filename, mask_pattern):
            mask_path = os.path.join(folder_path, filename)
            masks.append(load_image(mask_path))

    # If there are multiple masks, merge them; otherwise, use the single mask
    if len(masks) > 1:
        final_mask = merge_masks(masks)
    else:
        final_mask = masks[0]
    
    breast_image = load_image(image_path)  # Load the breast ultrasound image
    process_images(breast_image, final_mask, folder_name, final_folder_name, augment_data)  # Process the image and mask

# Define the final folder name to store intermediate data
final_folder_name = 'intermediate_dataset'
# Iterate through each folder and process the images sequentially
for folder_name, folder_path in folders.items():
    print(f"Processing images from folder: {folder_name}")
    unique_id = 0  # Initialize unique ID for image naming
    for filename in os.listdir(folder_path):
        # Process only images, ignoring mask files
        if "mask" not in filename and filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            process_masks_and_images(image_path, folder_path, folder_name, final_folder_name, False)

# Define directories for splitting the dataset
data_dir = f'./{final_folder_name}'
train_dir = './train'
val_dir = './validation'
test_dir = './test'

print("Splitting dataset into training, validation, and testing sets ...")

# Function to copy images and their corresponding masks to a destination directory
def copy_images_from_directory(dst_directory, image):
    src = os.path.join(class_path, image)
    dst = os.path.join(dst_directory, class_folder, image)
    mask_name = image.split('.')[0] + '_mask.png'

    src_mask = os.path.join(class_path, mask_name) 
    dst_mask = os.path.join(dst_directory, class_folder, mask_name)

    os.makedirs(os.path.dirname(dst), exist_ok=True)  # Create directories if they don't exist
    copyfile(src, dst)  # Copy the image from source to destination
    copyfile(src_mask, dst_mask)  # Copy the corresponding mask

# Loop through the classes to split data into training, validation, and test sets
for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        # Load images in the class folder
        images = [f for f in os.listdir(class_path) if f.endswith('.png') and 'mask' not in f]

        # Split the images into training, validation, and testing sets using a fixed random seed
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=24686103)
        val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=24686103)

        # Copy images to their respective directories
        for image in train_images:
            copy_images_from_directory(train_dir, image)
        for image in val_images:
            copy_images_from_directory(val_dir, image)
        for image in test_images:
            copy_images_from_directory(test_dir, image)

# AUGMENT TRAINING DATA
# Define the paths to the folders for training data augmentation
folders = {
    "Benign": "./train/Benign",
    "Malignant": "./train/Malignant",
    "Normal": "./train/Normal"
}

final_folder_name = 'augmented_training_dataset'
# Iterate through each folder and augment the training images
for folder_name, folder_path in folders.items():
    print(f"Processing images from folder: {folder_name}")
    unique_id = 0  # Reinitialize unique ID for naming augmented images
    for filename in os.listdir(folder_path):
        # Process only images, ignoring mask files
        if "mask" not in filename and filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            process_masks_and_images(image_path, folder_path, folder_name, final_folder_name, True)