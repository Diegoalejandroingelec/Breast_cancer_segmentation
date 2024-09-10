from architecture import unet_model
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Import learning rate reduction callback
import matplotlib.pyplot as plt
from dataloader import SegmentationDataGenerator

def plot_loss_metric_curves(loss,metric,training,epochNo,metric_name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)


    #plot epochs vs Loss for training set
    ax1.plot(list(range(epochNo)),loss[0],label="training")
    #plot epochs vs Loss
    ax1.plot(list(range(epochNo)),loss[1],label="testing")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (binary crossentropy)')
    ax1.set_title('loss vs Epochs')
    ax1.legend()
    
    #plot epochs vs metric for training set
    ax2.plot(list(range(epochNo)),metric[0],label="training")
    #plot epochs vs metric for testing set
    ax2.plot(list(range(epochNo)),metric[1],label="testing")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(f'{metric_name}')
    ax2.set_title(f'{metric_name} vs Epochs')
    ax2.legend()
    fig.tight_layout(h_pad=10)
    plt.show() 




# Directories
train_dir = 'augmented_training_dataset'
val_dir = 'validation'


# Hyperparameters
batch_size = 8
target_size = (256,256)

# Create train and validation generators
train_generator = SegmentationDataGenerator(train_dir, batch_size=batch_size, target_size=target_size)
validation_generator = SegmentationDataGenerator(val_dir, batch_size=batch_size, target_size=target_size)


print(train_generator[0][0][1].shape)
plt.figure(figsize = (10, 7))
plt.subplot(1,2,1)
plt.imshow(train_generator[0][0][1],cmap='gray', vmin=0, vmax=1)
plt.title('Ultra Sound Image')
plt.subplot(1,2,2)
plt.imshow(train_generator[0][1][1])
plt.title('Mask for Tumour')
plt.show()




inputs = tf.keras.layers.Input((256, 256, 1))
Unet = unet_model(inputs, droupouts= 0.07)


Unet.summary()
plot_model(Unet, to_file='unet_plot.png', show_shapes=True, show_layer_names=True)
Unet.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


epochNo = 80
model_file = 'unet_best_weights'

#Implement learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Monitor validation accuracy
    factor=0.1,  # Reduce learning rate by a factor of 0.1 when triggered
    patience=5,  # Number of epochs with no improvement before reducing learning rate
    min_lr=0.0000001  # Minimum learning rate threshold
)

# Define a callback function to check after each epoch the accuracy in the testing set and save the best weights
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_file,  # Filepath to save the model weights
    save_weights_only=True,  # Save only the model weights, not the entire model
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max',  # Save the weights when the monitored metric is maximized
    save_best_only=True  # Save only the best model weights based on the monitored metric
)

history = Unet.fit(
    train_generator,  # Training data generator
    steps_per_epoch=len(train_generator),  # Number of steps per epoch (length of training generator)
    epochs=epochNo,  # Number of epochs to train
    verbose=1,  # Verbosity mode (1 for progress bar)
    validation_data=validation_generator,  # Validation data generator
    validation_steps=len(validation_generator),  # Number of steps for validation (length of validation generator)
    callbacks=[model_checkpoint_callback, reduce_lr]  # Callbacks for learning rate reduction and model checkpoint
)


# Plot the training curves
plot_loss_metric_curves([history.history['loss'],history.history['val_loss']],
                        [history.history['accuracy'],history.history['val_accuracy']],
                          True,
                          epochNo,
                          'accuracy')













