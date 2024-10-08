from architecture import unet_model_with_classification
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Import learning rate reduction callback
import matplotlib.pyplot as plt
from dataloader import SegmentationDataGenerator  # Custom data loader for segmentation

# Function to plot loss and metric curves for both segmentation and classification tasks
def plot_loss_metric_curves(loss, metric, epochNo, metric_name):
    
    # Create two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the training and validation losses for both segmentation and classification
    ax1.plot(list(range(epochNo)), loss[0], label="Training (segmentation_output_loss)")
    ax1.plot(list(range(epochNo)), loss[1], label="Training (classification_output_loss)")
    ax1.plot(list(range(epochNo)), loss[2], label="Validation (val_segmentation_output_loss)")
    ax1.plot(list(range(epochNo)), loss[3], label="Validation (val_classification_output_loss)")

    # Set labels and title for the loss plot
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.legend()

    # Plot the training and validation accuracy for both segmentation and classification
    ax2.plot(list(range(epochNo)), metric[0], label="Training (segmentation_output_accuracy)")
    ax2.plot(list(range(epochNo)), metric[1], label="Training (classification_output_accuracy)")
    ax2.plot(list(range(epochNo)), metric[2], label="Validation (val_segmentation_output_accuracy)")
    ax2.plot(list(range(epochNo)), metric[3], label="Validation (val_classification_output_accuracy)")

    # Set labels and title for the accuracy plot
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(f'{metric_name}')
    ax2.set_title(f'{metric_name} vs Epochs')
    ax2.legend()

    # Adjust layout to prevent overlap
    fig.tight_layout(h_pad=10)
    plt.show()


# Directories for training and validation datasets
train_dir = 'augmented_training_dataset'
val_dir = 'validation'

# Hyperparameters
batch_size = 8  # Number of samples per gradient update
target_size = (256, 256)  # Target size for the input images (height, width)

# Create train and validation data generators using the custom SegmentationDataGenerator
train_generator = SegmentationDataGenerator(train_dir, batch_size=batch_size, target_size=target_size)
validation_generator = SegmentationDataGenerator(val_dir, batch_size=batch_size, target_size=target_size)

# Define input shape for the U-Net model (256x256 grayscale images)
inputs = tf.keras.layers.Input((256, 256, 1))

# Instantiate the U-Net model with dropout regularization
Unet = unet_model_with_classification(inputs, dropouts=0.07)

# Display the model summary
Unet.summary()

# Plot the model architecture and save as a PNG file
plot_model(Unet, to_file='unet_plot.png', show_shapes=True, show_layer_names=True)

# Compile the model with two outputs (segmentation and classification) and different losses for each output
Unet.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), 
             loss={'segmentation_output': 'binary_crossentropy', 'classification_output': 'categorical_crossentropy'},
             metrics={'segmentation_output': 'accuracy', 'classification_output': 'accuracy'})

# Number of epochs to train the model
epochNo = 80

# File to save the best model weights
model_file = 'unet_classifier_best_weights'

# Implement learning rate reduction callback to reduce the learning rate when validation accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_classification_output_accuracy',  # Monitor validation accuracy for classification
    factor=0.1,  # Reduce learning rate by a factor of 0.1
    patience=5,  # Wait for 5 epochs with no improvement before reducing learning rate
    min_lr=0.0000001  # Set the minimum learning rate
)

# Define a callback to save the best model weights based on validation accuracy
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_file,  # Filepath to save model weights
    save_weights_only=True,  # Save only the weights, not the entire model
    monitor='val_classification_output_accuracy',  # Monitor validation accuracy for classification
    mode='max',  # Save weights when validation accuracy is maximized
    save_best_only=True  # Save only the best model weights
)

# Train the model with training and validation data generators
history = Unet.fit(
    train_generator,  # Training data generator
    steps_per_epoch=len(train_generator),  # Number of steps per epoch (batches per epoch)
    epochs=epochNo,  # Total number of epochs
    verbose=1,  # Verbosity mode (1 = progress bar)
    validation_data=validation_generator,  # Validation data generator
    validation_steps=len(validation_generator),  # Number of validation steps
    callbacks=[model_checkpoint_callback, reduce_lr]  # Callbacks for learning rate reduction and model checkpoint
)

# Plot the training and validation loss/accuracy curves using the custom plotting function
plot_loss_metric_curves(
    [history.history['segmentation_output_loss'], history.history['classification_output_loss'], 
     history.history['val_segmentation_output_loss'], history.history['val_classification_output_loss']],
    [history.history['segmentation_output_accuracy'], history.history['classification_output_accuracy'], 
     history.history['val_segmentation_output_accuracy'], history.history['val_classification_output_accuracy']],
    epochNo,
    'accuracy'
)