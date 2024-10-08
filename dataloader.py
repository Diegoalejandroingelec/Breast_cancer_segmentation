import os  # Import the os module for file and directory management
import tensorflow as tf  # Import TensorFlow for deep learning utilities
import numpy as np  # Import NumPy for array manipulations

# Custom data generator class for segmentation tasks, inheriting from Keras Sequence for compatibility with Keras model training
class SegmentationDataGenerator(tf.keras.utils.Sequence):
    # Initialization method
    def __init__(self, image_dir, batch_size, target_size=(256, 256), shuffle=True):
        self.image_dir = image_dir  # Directory where the images are stored
        self.batch_size = batch_size  # Number of samples per batch
        self.target_size = target_size  # Target size for resizing the images
        self.shuffle = shuffle  # Whether to shuffle the dataset after each epoch

        # Get all image file paths in the directory (excluding mask files) and store them in a sorted list
        self.image_paths = sorted([os.path.join(root, f) for root, _, files in os.walk(image_dir) for f in files if not f.endswith('_mask.png')])

        # Call this method at the end of each epoch to shuffle the data if required
        self.on_epoch_end()

    # Define the length of the dataset in terms of batches (total images divided by batch size)
    def __len__(self):
        return len(self.image_paths) // self.batch_size  # The floor division returns the number of full batches

    # This method retrieves a batch of data (images and masks) for the given index
    def __getitem__(self, index):
        # Get the paths for the current batch of images based on the index
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize empty lists to store images, masks, and class labels
        images, masks, classes = [], [], []

        # Loop through each image in the batch and load its corresponding mask and classification label
        for image_path in batch_image_paths:
            # Generate the corresponding mask path by replacing the image filename suffix with '_mask'
            mask_path = image_path.replace('.png', '_mask.png')  

            # Load the image and mask, and resize them to the target size
            image, mask = self.load_image_mask(image_path, mask_path, self.target_size)

            # Append the loaded image and mask to the respective lists
            images.append(image)
            masks.append(mask)

            # Assign class labels based on the filename (Benign, Malignant, or Normal)
            if "Benign" in image_path:
                classes.append([1, 0, 0])  # One-hot encoding for the 'Benign' class
            elif "Malignant" in image_path:
                classes.append([0, 1, 0])  # One-hot encoding for the 'Malignant' class
            else:
                classes.append([0, 0, 1])  # One-hot encoding for the 'Normal' class

        # Return the batch as a tuple of the image data and a dictionary containing the segmentation masks and class labels
        return np.array(images), {"segmentation_output": np.array(masks), "classification_output": np.array(classes)}

    # This method is called at the end of each epoch to shuffle the dataset if required
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)  # Shuffle the image paths to randomize the order of data for the next epoch

    # Utility function to load and preprocess the image and its corresponding mask
    def load_image_mask(self, image_path, mask_path, target_size=(256, 256)):
        # Load the image in grayscale mode and resize to the target size
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=target_size)
        # Convert the image to a NumPy array and normalize the pixel values to the range [0, 1]
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0

        # Load the mask in grayscale mode and resize to the target size
        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode="grayscale", target_size=target_size)
        # Convert the mask to a NumPy array but keep the values unnormalized for correct mask representation
        mask = tf.keras.preprocessing.image.img_to_array(mask) / 255  # Keep mask in [0, 1] range

        # Return the preprocessed image and mask
        return image, mask