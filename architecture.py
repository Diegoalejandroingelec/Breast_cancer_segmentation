import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

###########################################################################
#                                Model   Defination                       #
###########################################################################


# this block essentially performs 2 convolution

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


#
# def unet_model(inputImage, numFilters = 16, droupouts = 0.1, doBatchNorm = True):
#     # defining encoder Path
#     c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
#     p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
#     p1 = tf.keras.layers.Dropout(droupouts)(p1)
    
#     c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
#     p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
#     p2 = tf.keras.layers.Dropout(droupouts)(p2)
    
#     c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
#     p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
#     p3 = tf.keras.layers.Dropout(droupouts)(p3)
    
#     c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
#     p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
#     p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
#     c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
#     # defining decoder path
#     u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
#     u6 = tf.keras.layers.concatenate([u6, c4])
#     u6 = tf.keras.layers.Dropout(droupouts)(u6)
#     c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
#     u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    
#     u7 = tf.keras.layers.concatenate([u7, c3])
#     u7 = tf.keras.layers.Dropout(droupouts)(u7)
#     c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
#     u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
#     u8 = tf.keras.layers.concatenate([u8, c2])
#     u8 = tf.keras.layers.Dropout(droupouts)(u8)
#     c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
#     u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#     u9 = tf.keras.layers.concatenate([u9, c1])
#     u9 = tf.keras.layers.Dropout(droupouts)(u9)
#     c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
#     output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
#     model = tf.keras.Model(inputs = [inputImage], outputs = [output])
#     return model


def unet_model_with_classification(inputImage, numFilters=16, dropouts=0.1, doBatchNorm=True, num_classes=3):
    # Encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # Decoder Path (for segmentation)
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    
    segmentation_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(c9)

    # Classification Head (parallel to the segmentation head)
    classification_pool = tf.keras.layers.GlobalAveragePooling2D()(c5)  # Use the bottleneck layer

    # First dense layer with dropout
    classification_dense1 = tf.keras.layers.Dense(512, activation='relu')(classification_pool)
    classification_dropout1 = tf.keras.layers.Dropout(0.5)(classification_dense1)  # Dropout added

    # Second dense layer with dropout
    classification_dense2 = tf.keras.layers.Dense(256, activation='relu')(classification_dropout1)
    classification_dropout2 = tf.keras.layers.Dropout(0.5)(classification_dense2)  # Dropout added

    # Output layer
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(classification_dropout2)

    # Define the model with both outputs
    model = tf.keras.Model(inputs=[inputImage], outputs=[segmentation_output, classification_output])

    
    return model
