import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

base_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\fulldataset'  

class_names = ['Angry', 'Happy', 'Neutral', 'Engaged']
train_test_names = ['train', 'test']
X = []
y = []

for class_name in class_names:
    class_path = os.path.join(base_dir, class_name, class_name)
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
    sample_indices = random.sample(list(class_indices), 15)  # Randomly select 15 samples

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
