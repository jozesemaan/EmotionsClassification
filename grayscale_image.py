import cv2
import os

def convert_to_grayscale(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all files in the input folder
    files = os.listdir(input_folder)

    # Process each file in the input folder
    for file in files:
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Write the grayscale image to the output folder
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, grayscale_image)

        print(f"{file} converted to grayscale")

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = "input_images"
    output_folder = "output_images"

    # Convert images to grayscale
    convert_to_grayscale(input_folder, output_folder)
