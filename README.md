# ProjectAssignmentFS_5

# Team FS_5

| Name | Student ID | Handle |
| ---- | ---------- | ------ |
| Jose Semaan | 40244141 | @jozesemaan |
| Nour Hassoun | 40233077 | @NourHadieHassoun |
| Harout Kayabalian | 40209920 | @Harkay99 |

Python Scripts:

Data Visualization and Cleaning.py

This script is designed to organize a dataset of images into training, testing, and validation sets, and then visualize the distribution and characteristics of the images within these sets. The script performs the following key tasks:
1.	Splits the dataset into training, testing, and validation sets.
2.	Loads and processes images from the cleaned dataset.
3.	Visualizes the class distribution of the images.
4.	Generates pixel intensity histograms for each class.
5.	Displays sample images and their pixel intensity distributions for each class.


grayscale_image.py:
1- Arrangement of Folders for Input and Output:
The input_folder and output_folder variables indicate the locations of the input images and where the grayscale images will be stored, respectively.
2- Function convert_to_grayscale:
A function that changes an image to grayscale.
Verifies the existence of the output folder and generates it if it is not present (os.makedirs(output_folder, exist_ok=True)).
Obtains a directory of files in the specified input folder by using os.listdir(input_folder).
3- Goes through every file in sequence:
Uses cv2.imread to open and view the image.
Uses cv2.cvtColor to change the image to grayscale.
Stores the black and white image in the destination directory with cv2.imwrite.
Displays a message of success for every image that has been converted.

rename_images.py:
The raw string literal in Python is indicated by the r prefix in front of the directory path (r"C:\Users\pierh\Downloads\fer2013\train\neutral_new"), and it helps to avoid the need for escaping backslashes.
The script presupposes that there are precisely 528 PNG files in the designated folder. An error message will be displayed if there are a different number of files.
The files are renamed in sequence as neutral1.png, neutral2.png, up to neutral528.png during the renaming process.

resize_image.py:
Parsing arguments is the process of analyzing input data to extract relevant information.
The argparse module in the script is utilized for managing command-line arguments. The function requires two parameters: input_image (location of the original image) and output_image (location to store the resized image).
Function to resize an image (resize_image):
Use PIL.Image.open to access the input image.
Displays the image's initial dimensions.
Resizes the original image to a specified size of 48x48 pixels using the resize method.
Displays the new size of the image.
Resizes the image and saves it to the specified output path by calling resized_image.save(output_image).
Primary Execution:
Interprets the input parameters provided in the command line (input_image and output_image).
Specifies the size to be (48, 48).
Executes the resize_image function with the provided image path, destination image path, and specified size.
