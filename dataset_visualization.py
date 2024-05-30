import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

base_dir = r'C:\Users\jozes\Desktop\COMP472\Project\DataSetFinal'  

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
plt.figure(figsize=(15, 10))
for i in unique:
    # Sample Images
    plt.subplot(2, 4, i+1)
    sample_img_idx = np.where(y == i)[0][0]  # Get the index of the first image of the class
    plt.imshow(X[sample_img_idx], cmap='gray')
    plt.title(f'Sample Image - {class_names[i]}')
    plt.axis('off')

    # Pixel Intensity Distribution for Sample Image
    plt.subplot(2, 4, i+5)
    plt.hist(X[sample_img_idx].ravel(), bins=256, color='green', alpha=0.7)
    plt.title(f'Pixel Intensity - {class_names[i]} Sample Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
