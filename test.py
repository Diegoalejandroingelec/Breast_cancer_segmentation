from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from architecture import unet_model
import numpy as np
import matplotlib.pyplot as plt
from dataloader import SegmentationDataGenerator

test_dir =  'test'
target_size = (256,256)
test_generator = SegmentationDataGenerator(test_dir, batch_size=1, target_size=target_size)



model_file = 'unet_best_weights'
inputs = tf.keras.layers.Input((256, 256, 1))
Unet = unet_model(inputs, droupouts= 0.07)
Unet.load_weights(model_file)


# Function to calculate IoU
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Function to calculate Dice Coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

# Function to calculate Precision, Recall, and Accuracy
def calculate_precision_recall_accuracy(y_true, y_pred):
    y_true_f = y_true.flatten().astype(np.uint8)
    y_pred_f = y_pred.flatten().astype(np.uint8)
    
    precision = precision_score(y_true_f, y_pred_f, zero_division=1)
    recall = recall_score(y_true_f, y_pred_f, zero_division=1)
    accuracy = accuracy_score(y_true_f, y_pred_f)
    
    return precision, recall, accuracy


def compute_performance_metrics(data_generator):
    # Variables to accumulate metrics
    total_iou = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    num_samples = 0

    # Loop through the test dataset
    for n, batch in enumerate(data_generator):
        X = batch[0]
        Y = batch[1]  # Ground truth mask

        prediction=Unet(X)
        prediction = np.array(prediction)

        # Apply threshold to convert predictions to binary masks (0 or 1)
        prediction = (prediction > 0.1).astype(np.float32)

        if(n%10==0):
            plt.figure(figsize = (10, 7))

            plt.subplot(1,3,1)
            plt.imshow(X[0],cmap='gray')
            plt.title('Ultra Sound Image')

            plt.subplot(1,3,2)
            plt.imshow(Y[0])
            plt.title('Ground Truth Mask for Tumour')

            plt.subplot(1,3,3)
            plt.imshow(prediction[0])
            plt.title('Segmentation for Tumour')
            plt.show()


        
        # Calculate metrics for the current image
        iou = iou_metric(Y[0], prediction[0])
        dice = dice_coefficient(Y[0], prediction[0])
        precision, recall, accuracy = calculate_precision_recall_accuracy(Y[0], prediction[0])

        # Accumulate the metrics
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy
        num_samples += 1

    # Calculate the average metrics
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_accuracy = total_accuracy / num_samples

    # Print the average performance metrics
    print(f"Average IoU (Jaccard Index): {avg_iou:.4f}")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")


compute_performance_metrics(test_generator)