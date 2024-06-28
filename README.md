# ProjectAssignmentFS_5

# Team FS_5

| Name | Student ID | Handle |
| ---- | ---------- | ------ |
| Jose Semaan | 40244141 | @jozesemaan |
| Nour Hassoun | 40233077 | @NourHadieHassoun |
| Harout Kayabalian | 40209920 | @Harkay99 |

Python Scripts:

Data Visualization and Cleaning.py:

This script is designed to organize a dataset of images into training, testing, and validation sets, and then visualize the distribution and characteristics of the images within these sets. The script performs the following key tasks:
1.	Splits the dataset into training, testing, and validation sets.
2.	Loads and processes images from the cleaned dataset.
3.	Visualizes the class distribution of the images.
4.	Generates pixel intensity histograms for each class.
5.	Displays sample images and their pixel intensity distributions for each class.
6.  Names the images properly and saves it in a new data/fulldataset_cleaned directory

- - - - - - - - - - - - - -- - - - -- - - -- - - - -- - - -- - - - -- 
All models folder     

Each folder contains one model with small differences in between:

- Main Model: Original CNN Architecture (3x3 Kernel size and 2 convolutional layers)
  
- Variant 1: Added Convolutional Layer (3x3 kernel size and 3 convolutional layers)
  
- Variant 2: Changed Kernel Size in Convolutional Layers (5x5 kernel size and 2 convolutional layers)

Each version contain a train_model.py and evaluate_model.py


- - - - - - - - - - - - - -- - - - -- - - -- - - - -- - - -- - - - -- 
train_model.py:

Step 1: Define the SimpleCNN Architecture
A simple CNN model is defined with the following layers:
- Two convolutional layers with ReLU activation and MaxPooling.
- Two fully connected layers with ReLU activation and Dropout.
- Output layer with four classes.

Step 2: Define Image Transformations
- Images are transformed to grayscale, resized to 48x48 pixels, and converted to tensors for input into the model.

Step 3: Define Custom Dataset Class
- A custom dataset class is defined to load images and their corresponding labels.

Step 4: Load Datasets
- The load_datasets function loads training and testing datasets from the specified directory.

Step 5: Initialize Data Loaders
- DataLoader instances are created for the training and testing datasets with a batch size of 32.

Step 6: Training Function with Early Stopping and Minimum Epochs
- The train_model function trains the model using early stopping criteria:
- Early Stopping: Stops training if the validation loss does not improve for a specified number of epochs (patience).
- Minimum Epochs: Ensures the model is trained for at least a specified number of epochs (min_epochs).

Step 7: Evaluation Function
- The evaluate_model function evaluates the model on a given dataset, printing performance metrics including accuracy, precision, recall, F1-score, and validation loss. It also displays the confusion matrix.

Step 8: Initialize Model, Optimizer, and Loss Function
- Model: An instance of SimpleCNN.
- Criterion: CrossEntropyLoss.
- Optimizer: Adam optimizer with a learning rate of 0.001.

Step 9: Train the Model
- The model is trained using the train_model function, with early stopping and minimum epochs specified.

- - - - - - - - - - - - - -- - - - -- - - -- - - - -- - - -- - - - -- 
Evaluate_model.py: 
- The goal of this script is to evaluate the best_model.pth file on the validation set that was generated using the train_model.py, the train_model generates runs for a certain number of epochs, tests each one on the test set and then chooses the best_model.
The evaluate_model takes that same best model and tests it on the validation set.

---------------------------------------------------------------------

FOR PART 3
----------
- Readme.py : A script that's used to rename the images properly inside each folder

Dataset for PART3:

- The full dataset non-cleaned is in fulldataset-part3
- The full cleaned dataset is in data/fulldataset_cleaned_part3
- The dataset for Part3 has been augmented from Part2

Gender_classification 
- Gender classificationfolder contains the classified folders male,female,young,middle_aged,old
- Inside each classification folder there's the 4 main classes [Angry,Engaged,Neutral,Happy] and inside these directories are the images that are used for testing and collecting the metrics to fill the tables
- It's evaluated based on the best_model that was generated with the validation/test/train split using the augmented dataset that's in the ALl models-part3 folder

Bias

- The bias folder contains 3 folders [ 10-15 percent,20-30 percent, 40-50 ]
- Each folder contains the 4 folders for main classes [Angry, Engaged, Happy, Neutral]
- Inside each folder the data is classified so that there's a bias for female vs male and there's exactly 50 images per folder

- The 10-15 percent folder contains 28 females vs 22 males
- The 20-30 percent folder contains 35 females vs 15 males
- The 40-50 percent folder contains 40 females vs 10 males 

