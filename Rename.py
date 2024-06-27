import os

# Set the base path where your folders are located
base_path = r'C:\Users\pierh\assignment1\ProjectAssignmentFS_5\Bias\10-15_percent_bias'

# Define the folder names
folders = ['Angry', 'Happy', 'Engaged', 'Neutral']

# Loop through each folder
for folder in folders:
    # Define the full path to the folder
    folder_path = os.path.join(base_path, folder)
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Sort the files to ensure consistent numbering
    files.sort()
    # Loop through each file in the folder
    for idx, filename in enumerate(files):
        # Define the new filename with 'img' prefix and extension
        new_filename = f"img{idx+1}{os.path.splitext(filename)[1]}"
        # Define the full old and new file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")
