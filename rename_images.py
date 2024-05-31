import os

# Set the directory containing the images
directory = r"C:\Users\pierh\Downloads\fer2013\train\neutral_new"

# Get a list of all files in the directory
files = os.listdir(directory)

# Filter only .png files and sort them
png_files = sorted([file for file in files if file.endswith('.png')])

# Ensure we have exactly 501 files
if len(png_files) != 528:
    print(f"Expected 529 .png files, but found {len(png_files)}.")
else:
    # Rename each file
    for i, filename in enumerate(png_files):
        new_name = f'neutral{i+1}.png'
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        os.rename(old_file, new_file)
    print("Renaming completed successfully!")
