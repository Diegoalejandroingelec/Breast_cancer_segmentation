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
model_file = 'unet_weights_converted.h5'
inputs = tf.keras.layers.Input((256, 256, 1))
Unet = unet_model_with_classification(inputs, dropouts=0.07)
Unet.load_weights(model_file)

# Function to calculate Intersection over Union (IoU) metric
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

# Function to calculate Precision, Recall, and Accuracy for segmentation
def calculate_precision_recall_accuracy(y_true, y_pred):
    y_true_f = y_true.flatten().astype(np.uint8)
    y_pred_f = y_pred.flatten().astype(np.uint8)
    
    precision = precision_score(y_true_f, y_pred_f, zero_division=1)
    recall = recall_score(y_true_f, y_pred_f, zero_division=1)
    accuracy = accuracy_score(y_true_f, y_pred_f)
    
    return precision, recall, accuracy

# Function to calculate Precision, Recall, Accuracy, and F1-score for classification
def calculate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    
    return precision, recall, accuracy, f1

# Function to get the class name from the class index
def get_class_name(class_idx):
    if class_idx == 0:
        class_name = "Benign"
    elif class_idx == 1:
        class_name = "Malignant"
    else:
        class_name = "Normal"
    return class_name

# Function to compute and display performance metrics
def compute_performance_metrics(data_generator):
    # Variables to accumulate metrics
    total_iou = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_f1 = 0  # For classification F1-score
    num_samples = 0



    # Lists to store ground truth and predictions for classification
    all_classification_ground_truths = []
    all_classification_predictions = []

    # Loop through the test dataset
    for n, batch in enumerate(data_generator):
        X = batch[0]  # Input image
        Y = batch[1]  # Ground truth for both segmentation and classification
        segmentation_ground_truth = Y['segmentation_output'][0]
        classification_ground_truth = np.argmax(Y['classification_output'][0])

        # Get model predictions for segmentation and classification
        prediction = Unet(X)
        segmentation_prediction = np.array(prediction[0])
        classification_prediction = np.array(prediction[1])

        # Apply threshold to convert predictions to binary masks (0 or 1)
        segmentation_prediction = (segmentation_prediction > 0.2).astype(np.float32)

        # Argmax to get the class with the highest probability for classification
        classification_prediction = np.argmax(classification_prediction)

        # Append ground truth and prediction to classification lists
        all_classification_ground_truths.append(classification_ground_truth)
        all_classification_predictions.append(classification_prediction)

        # Visualization every 10th image
        if (n % 10 == 0):
            classification_prediction_name = get_class_name(classification_prediction)
            classification_ground_truth_name = get_class_name(classification_ground_truth)

            plt.figure(figsize=(10, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(X[0], cmap='gray')
            plt.title(f'Ultra Sound Image\n Ground Truth Class: {classification_ground_truth_name}')

            plt.subplot(1, 3, 2)
            plt.imshow(segmentation_ground_truth)
            plt.title(f'Ground Truth Mask for Tumour\n Ground Truth Class: {classification_ground_truth_name}')

            plt.subplot(1, 3, 3)
            plt.imshow(segmentation_prediction[0])
            plt.title(f'Segmentation for Tumour\n Predicted Class: {classification_prediction_name}')
            plt.show()

        # Calculate metrics for segmentation
        iou = iou_metric(segmentation_ground_truth, segmentation_prediction[0])
        dice = dice_coefficient(segmentation_ground_truth, segmentation_prediction[0])
        precision, recall, accuracy = calculate_precision_recall_accuracy(segmentation_ground_truth, segmentation_prediction[0])



        # Accumulate the segmentation metrics
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy

        num_samples += 1


        # Calculate metrics for classification
    classification_precision, classification_recall, classification_accuracy, classification_f1 = calculate_classification_metrics(
        all_classification_ground_truths, all_classification_predictions)
    
    
    # Calculate the average segmentation metrics
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_accuracy = total_accuracy / num_samples


    # Print the average segmentation performance metrics
    print(f"Average IoU (Jaccard Index): {avg_iou:.4f}")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Precision (Segmentation): {avg_precision:.4f}")
    print(f"Average Recall (Segmentation): {avg_recall:.4f}")
    print(f"Average Accuracy (Segmentation): {avg_accuracy:.4f}")

    # Print the average classification performance metrics
    print(f"Average Precision (Classification): {classification_precision:.4f}")
    print(f"Average Recall (Classification): {classification_recall:.4f}")
    print(f"Average Accuracy (Classification): {classification_accuracy:.4f}")
    print(f"Average F1-Score (Classification): {classification_f1:.4f}")

    # Compute the confusion matrix for classification predictions
    cm = confusion_matrix(all_classification_ground_truths, all_classification_predictions)

    # Define the class labels
    class_labels = ['Benign', 'Malignant', 'Normal']

    # Display the confusion matrix using a heatmap with class names
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Classification')
    plt.show()

# Compute performance metrics for the test dataset
compute_performance_metrics(test_generator)