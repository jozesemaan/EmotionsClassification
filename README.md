# ProjectAssignmentFS_5

# Team FS_5

| Name | Student ID | Handle |
| ---- | ---------- | ------ |
| Jose Semaan | 40244141 | @jozesemaan |
| Nour Hassoun | 40233077 | @NourHadieHassoun |
| Harout Kayabalian | 40209920 | @Harkay99 |

Python Scripts:

dataset_visualization.py: 
1-Uploading and preparing images for use:
Pictures are transformed into grayscale, adjusted to 48x48 pixels, and changed into numpy arrays (X).
Labels (y) are created using the folder arrangement (class_names) and counting the images per class.

2-Distribution of the class:
A bar graph displays how images are divided among various categories (Angry, Happy, Neutral, Engaged).

3-Distribution of pixel intensity for each class:
Graphs are created for each category displaying the range of pixel values in the black-and-white images.

4-Examples of Images and Distribution of Pixel Intensity for Individual Class:
15 random sample images are chosen for each class and their pixel intensity histograms are shown next to them.

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
