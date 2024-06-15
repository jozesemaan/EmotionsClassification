import os
import cv2
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Paths for dataset
base_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\fulldataset'
output_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\data\fulldataset_cleaned'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to get the proper filename
def get_new_filename(class_name, train_test_name, idx):
    return f'{class_name}_{train_test_name}_{idx}.png'

# Step 1: Denoise images and save them to a new directory structure with proper renaming
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)
    
    for train_test_name in os.listdir(class_path):
        train_test_path = os.path.join(class_path, train_test_name)
        train_test_output_path = os.path.join(class_output_path, train_test_name)
        os.makedirs(train_test_output_path, exist_ok=True)
        
        # Initialize index for proper renaming
        idx = 1
        
        for img_name in os.listdir(train_test_path):
            img_path = os.path.join(train_test_path, img_name)
            
            # Read image using OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Denoise image using Non-Local Means Denoising
            denoised_img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Construct path for cleaned image in output directory
            new_img_name = get_new_filename(class_name, train_test_name, idx)
            cleaned_img_path = os.path.join(train_test_output_path, new_img_name)
            
            # Save cleaned image
            cv2.imwrite(cleaned_img_path, denoised_img)
            
            # Increment index for next image
            idx += 1
            
print("Cleaning and saving images complete.")

# Step 2: Load cleaned images and perform visualization
class_names = ['Angry', 'Happy', 'Neutral', 'Engaged']
train_test_names = ['train', 'test']
X = []
y = []

# Iterate through each class and each train/test folder in the cleaned dataset
for class_name in class_names:
    class_path = os.path.join(output_dir, class_name)
    for train_test_name in train_test_names:
        train_test_path = os.path.join(class_path, train_test_name)
        for img_name in os.listdir(train_test_path):
            img_path = os.path.join(train_test_path, img_name)
            img = Image.open(img_path).convert('L')  # ensure grayscale
            img = img.resize((48, 48))  # resize to 48x48
            X.append(np.array(img))
            y.append(class_names.index(class_name))

X = np.array(X)
y = np.array(y)

# Class Distribution
unique, counts = np.unique(y, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(unique, counts, tick_label=class_names)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()

# Pixel Intensity Distribution per Class
X_flatten = X.reshape(X.shape[0], -1)
plt.figure(figsize=(15, 10))
for i in unique:
    plt.subplot(2, 2, i+1)
    plt.hist(X_flatten[y == i].ravel(), bins=256, color='blue', alpha=0.7)
    plt.title(f'Class {class_names[i]}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Sample Images and Pixel Intensity Distribution for Each Class
for i in unique:
    class_indices = np.where(y == i)[0]
    sample_indices = random.sample(list(class_indices), min(15, len(class_indices)))  # Randomly select up to 15 samples

    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 20))
    fig.suptitle(f'Sample Images and Pixel Intensity Histograms - {class_names[i]}', fontsize=16)

    for idx, sample_idx in enumerate(sample_indices):
        row, col = divmod(idx, 3)

        # Plot the sample image
        axes[row, col * 2].imshow(X[sample_idx], cmap='gray')
        axes[row, col * 2].set_title(f'{class_names[i]} Image {idx+1}')
        axes[row, col * 2].axis('off')

        # Plot the pixel intensity histogram
        axes[row, col * 2 + 1].hist(X[sample_idx].ravel(), bins=256, color='green', alpha=0.7)
        axes[row, col * 2 + 1].set_title(f'Pixel Intensity {idx+1}')
        axes[row, col * 2 + 1].set_xlabel('Pixel Intensity')
        axes[row, col * 2 + 1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
