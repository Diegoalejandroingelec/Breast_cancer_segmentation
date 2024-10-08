import tensorflow as tf  # Import TensorFlow library for building and training the model
from tensorflow import keras  # Import Keras (high-level API) for model creation
from tensorflow.keras import layers  # Import layers from Keras to build neural network layers
import tensorflow_datasets as tfds  # Import TensorFlow Datasets for loading and preparing datasets
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import NumPy for numerical operations


# This block defines a function that performs two consecutive convolution operations
def Conv2dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
    # First convolution layer
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)
    
    # Apply batch normalization if specified
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    # Apply ReLU activation function
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second convolution layer
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    
    # Apply batch normalization if specified
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    # Apply ReLU activation function
    x = tf.keras.layers.Activation('relu')(x)
    
    # Return the processed tensor
    return x


# This function defines a U-Net model with an additional classification head
def unet_model_with_classification(inputImage, numFilters=16, dropouts=0.1, doBatchNorm=True, num_classes=3):
    # Encoder path (downsampling)
    
    # First encoding block
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)  # Apply convolution block
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)  # Downsampling using max pooling
    p1 = tf.keras.layers.Dropout(dropouts)(p1)  # Apply dropout to prevent overfitting
    
    # Second encoding block
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropouts)(p2)
    
    # Third encoding block
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropouts)(p3)
    
    # Fourth encoding block
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropouts)(p4)
    
    # Bottleneck layer (deepest part of the U-Net)
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # Decoder path (upsampling for segmentation)
    
    # First upsampling block
    u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(c5)  # Upsampling
    u6 = tf.keras.layers.concatenate([u6, c4])  # Skip connection from encoder
    u6 = tf.keras.layers.Dropout(dropouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    
    # Second upsampling block
    u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    
    # Third upsampling block
    u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    
    # Fourth upsampling block
    u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    
    # Output layer for segmentation (1 channel, sigmoid activation for binary segmentation)
    segmentation_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(c9)

    # Classification head (parallel to the segmentation head)
    
    # Apply global average pooling to reduce spatial dimensions
    classification_pool = tf.keras.layers.GlobalAveragePooling2D()(c5)  # Use the bottleneck layer as input
    
    # Fully connected dense layers for classification
    classification_dense1 = tf.keras.layers.Dense(512, activation='relu')(classification_pool)  # First dense layer
    classification_dropout1 = tf.keras.layers.Dropout(0.5)(classification_dense1)  # Dropout for regularization
    
    classification_dense2 = tf.keras.layers.Dense(256, activation='relu')(classification_dropout1)  # Second dense layer
    classification_dropout2 = tf.keras.layers.Dropout(0.5)(classification_dense2)  # Dropout for regularization
    
    # Output layer for classification (number of classes, softmax activation for multi-class classification)
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(classification_dropout2)

    # Define the model with both segmentation and classification outputs
    model = tf.keras.Model(inputs=[inputImage], outputs=[segmentation_output, classification_output])

    # Return the complete U-Net model with classification head
    return model
