import os
import cv2

base_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\fulldataset'
output_dir = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\fulldataset_cleaned'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to get the proper filename
def get_new_filename(class_name, train_test_name, idx):
    return f'{class_name}_{train_test_name}_{idx}.png'

# Example: Denoising images and saving to a new directory structure with proper renaming
for class_idx, class_name in enumerate(os.listdir(base_dir)):
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
