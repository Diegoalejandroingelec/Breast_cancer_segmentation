from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from architecture import unet_model_with_classification
import numpy as np
import matplotlib.pyplot as plt
from dataloader import SegmentationDataGenerator

# Define the directory for the test dataset and target size for images
test_dir = 'test'
target_size = (256, 256)

# Create a data generator for the test set
test_generator = SegmentationDataGenerator(test_dir, batch_size=1, target_size=target_size)

# Load the pre-trained model weights
model_file = 'unet_classifier_best_weights'
inputs = tf.keras.layers.Input((256, 256, 1))
Unet = unet_model_with_classification(inputs, dropouts=0.07)
Unet.load_weights(model_file)


# Save the weights in the Keras-compatible .h5 format
Unet.save_weights('unet_weights_converted.h5')
