import cv2
import numpy as np
import cairosvg
import svgwrite


def svg_to_image(svg_path, output_image_path):
    """
    Converts an SVG file to a PNG file.
    """
    cairosvg.svg2png(url=svg_path, write_to=output_image_path)
    return cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)


def process_image_with_opencv(image):
    """
    Processes the image to change non-transparent content to red.
    """
    # Get dimensions
    height, width = image.shape[:2]

    # Split the channels (assumes PNG with alpha channel)
    bgr = image[:, :, :3]  # BGR channels
    alpha = image[:, :, 3]  # Alpha channel

    # Create a red image of the same size
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]  # Set non-transparent regions to red

    # Create a mask for red regions
    mask = np.all(red_image == [0, 0, 255], axis=-1)

    return red_image, mask


def image_to_svg_with_red(image, mask, output_svg_path):
    """
    Converts the processed image back to SVG by extracting contours.
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an SVG document
    height, width = image.shape[:2]
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Add the red-filled content as paths
    for contour in contours:
        path_data = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in contour])
        path = dwg.path(d=path_data, fill='red', stroke='none')
        dwg.add(path)

    # Save the SVG
    dwg.save()


def process_svg(svg_path, output_image_path, output_svg_path):
    """
    Main function to process an SVG and convert its content to red, saving the output as a new SVG.
    """
    # Step 1: Convert SVG to raster image
    image = svg_to_image(svg_path, output_image_path)

    # Step 2: Process the raster image with OpenCV
    red_image, mask = process_image_with_opencv(image)

    # Step 3: Convert the processed image back to SVG
    image_to_svg_with_red(red_image, mask, output_svg_path)

    print(f"SVG processed and saved to {output_svg_path}")


# Example Usage
process_svg("input_sticker.svg", "temp_image.png", "output_sticker_red.svg")