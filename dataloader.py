import os
import tensorflow as tf
import numpy as np

class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, batch_size, target_size=(256, 256), shuffle=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.image_paths = sorted([os.path.join(root, f) for root, _, files in os.walk(image_dir) for f in files if not f.endswith('_mask.png')])
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks, classes = [], [], []

        for image_path in batch_image_paths:
            mask_path = image_path.replace('.png', '_mask.png')  # Get corresponding mask path
            image, mask = self.load_image_mask(image_path, mask_path, self.target_size)
            images.append(image)
            masks.append(mask)
            if "Benign" in image_path:
                classes.append([1,0,0])
            elif "Malignant" in image_path:
                classes.append([0,1,0])
            else:
                classes.append([0,0,1])


        # Return a dictionary of outputs for segmentation and classification
        return np.array(images), {"segmentation_output": np.array(masks), "classification_output": np.array(classes)}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)


    def load_image_mask(self, image_path, mask_path, target_size=(256, 256)):
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize image

        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode="grayscale", target_size=target_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask) / 255  # Mask should not be normalized (keep integer values)
        #mask = mask.astype(np.uint8)

        return image, mask