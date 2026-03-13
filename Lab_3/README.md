**IT549: Deep Learning - Lab 3
Image-Based AQI Classification using CNN and Pretrained Models**

Assignment Title : Image-Based AQI Classification using CNN and Pretrained Models
Course           : IT549 - Deep Learning
Name             : Khushali Mandalia
Student ID       : 202511025

**PROJECT OVERVIEW**

This project builds a complete deep learning pipeline to classify
Air Quality Index (AQI) categories from images of locations.
Two models are implemented and compared:

  1. BasicCNN        - A CNN trained from scratch
  2. ResNet18        - A pretrained model using transfer learning

**DATASET**

Source         : https://drive.google.com/drive/folders/1usBxgNB67GfhCQ2f7xRkDlF6fgIZZrP
Files Used     : data.csv (image_path, AQI_Class), sampled_images/
Train Split    : 70%
Validation     : 15%
Test Split     : 15%

**CODE STRUCTURE (What each Task does)**

Task 1 - Data Preparation
         Load CSV, encode labels, resize images to 224x224,
         normalize pixel values, split into train/val/test sets

Task 2 - BasicCNN Model
         4 convolutional blocks + fully connected layers
         trained from scratch on AQI dataset

Task 3 - Pretrained ResNet18
         Loaded with ImageNet weights, final layer replaced
         with Linear(512, NUM_CLASSES), fine-tuned on AQI dataset

Task 4 - Evaluation
         Accuracy, Precision, Recall, F1-Score
         Confusion matrix for both models on test set

Task 5 - Training Curves
         Epoch vs Train Loss, Val Loss, Train Accuracy, Val Accuracy
         plotted for both models side by side

Task 6 - Misclassification Analysis
         Top 10 misclassified images visualized with true label,
         predicted label, confidence score, and reason for error

**RESULTS SUMMARY**

  Metric       BasicCNN     ResNet18 (Pretrained)
  -------------------------------------------------
  Accuracy     0.7122       0.9578
  Precision    0.7236       0.9581
  Recall       0.7122       0.9578
  F1-Score     0.7015       0.9576
  -------------------------------------------------

Key Observation:
  ResNet18 outperformed BasicCNN because it was pretrained on
  1.2 million ImageNet images and already understood visual
  features like edges, textures, and haze patterns which are
  directly useful for AQI classification. BasicCNN trained from
  scratch required more epochs and still achieved lower accuracy
  due to the limited size of the AQI dataset.

**MISCLASSIFICATION SUMMARY**
Most common reasons for wrong predictions:

  1. Low contrast images - haze makes all AQI classes look similar
  2. Adjacent classes confused - Good vs Moderate look nearly identical
  3. Bright or dark images - hide important sky details
  4. Reddish tone - sunset lighting looks similar to heavy pollution
  5. Low model confidence - image sits between two classes visually
