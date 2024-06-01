from PIL import Image
import argparse

def resize_image(input_image, output_image, size):
    original_image = Image.open(input_image)
    width, height = original_image.size
    print(f"Original image size: {width}x{height}")

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print(f"Resized image size: {width}x{height}")

    resized_image.save(output_image)
    print(f"Resized image saved as {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image to 48x48 pixels.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_image", help="Path to save the output image.")
    args = parser.parse_args()

    input_image = args.input_image
    output_image = args.output_image
    size = (48, 48)

    resize_image(input_image, output_image, size)
