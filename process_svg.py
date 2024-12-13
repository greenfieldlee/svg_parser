import cv2
import numpy as np
import cairosvg
import logging
import svgwrite
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def svg_to_image(svg_path, output_image_path, dpi=300):
    """
    Converts an SVG file to a PNG image at the specified DPI.
    """
    try:
        cairosvg.svg2png(url=svg_path, write_to=output_image_path, dpi=dpi)
        logger.info(f"Successfully converted SVG to image: {output_image_path}")
        return cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.error(f"Error converting SVG to image: {str(e)}")
        raise

def calculate_scale_factor(original_width, original_height, grow):
    """
    Calculates the scale factor based on the grow value.
    """
    scale_factor_width = (original_width + grow) / original_width
    scale_factor_height = (original_height + grow) / original_height
    # Use the larger scale factor to ensure uniform scaling
    return [max(scale_factor_width, scale_factor_height), scale_factor_width, scale_factor_height]

def resize_image(image, grow):
    """
    Resize the image based on the grow value.
    """
    try:
        original_height, original_width = image.shape[:2]
        scale_factor = calculate_scale_factor(original_width, original_height, grow)
        
        new_width = int(original_width * scale_factor[1])
        new_height = int(original_height * scale_factor[2])
        
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        logger.info(f"Resized image to {new_width}x{new_height} (scale factor: {scale_factor})")
        return resized_image
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise

def process_image_with_opencv(image):
    """
    Processes the image to change non-transparent content to red.
    """
    try:
        height, width = image.shape[:2]
        bgr = image[:, :, :3]  # Extract BGR channels
        alpha = image[:, :, 3]  # Extract Alpha channel

        # Change non-transparent regions to red
        red_image = bgr.copy()
        red_image[alpha > 0] = [0, 0, 255]

        # Create a mask for the red regions
        mask = np.all(red_image == [0, 0, 255], axis=-1)

        logger.info("Processed image with OpenCV")
        return red_image, mask
    except Exception as e:
        logger.error(f"Error processing image with OpenCV: {str(e)}")
        raise

def smooth_contours(contours, epsilon=2):
    """
    Smooths the contours using polygon approximation.
    """
    try:
        smoothed_contours = []
        for contour in contours:
            smoothed = cv2.approxPolyDP(contour, epsilon, True)
            if len(smoothed) > 8:  # Keep circular contours less simplified
                smoothed_contours.append(smoothed)
            else:
                smoothed_contours.append(contour)
        logger.info("Smoothed contours")
        return smoothed_contours
    except Exception as e:
        logger.error(f"Error smoothing contours: {str(e)}")
        raise

def image_to_svg_path(image, mask):
    """
    Converts the processed image into SVG path data by extracting contours.
    """
    try:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed_contours = smooth_contours(contours, epsilon=2)

        # Adjust path coordinates to account for the image growth
        path_data = []
        for contour in contours:
            contour_path = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in contour])
            path_data.append(contour_path)

        full_path = ' '.join(path_data)
        logger.info("Generated SVG path data")
        return full_path
    except Exception as e:
        logger.error(f"Error converting image to SVG path: {str(e)}")
        raise

def generate_full_svg(svg_path_data, width, height):
    """
    Generates the full SVG content (including the <svg> container and path data).
    """
    try:
        full_svg = f'<?xml version="1.0" encoding="UTF-8"?>\n' \
                  f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n' \
                  f'    <path d="{svg_path_data}" fill="none" stroke="black" stroke-width="1" />\n' \
                  f'</svg>'
        logger.info("Generated full SVG content")
        return full_svg
    except Exception as e:
        logger.error(f"Error generating full SVG content: {str(e)}")
        raise

def save_svg_content(full_svg_content, output_svg_path):
    """
    Save the full SVG content (including the <svg> container) to a file.
    """
    try:
        with open(output_svg_path, 'w') as f:
            f.write(full_svg_content)
        logger.info(f"Full SVG content saved successfully to {output_svg_path}")
    except Exception as e:
        logger.error(f"Error saving full SVG content: {str(e)}")
        raise

def process_svg(svg_path, output_image_path, output_svg_path, dpi=300, grow=10):
    """
    Main function to process SVG: convert to image, apply transformations, generate and save the full SVG content.
    """
    try:
        logger.info(f"Processing SVG: {svg_path}")

        # Step 1: Convert the SVG to PNG
        image = svg_to_image(svg_path, output_image_path, dpi)
        logger.info(f"Converted SVG to image: {output_image_path}")

        # Step 2: Resize the image based on grow value
        resized_image = resize_image(image, grow)

        # Step 3: Process the resized image with OpenCV
        red_image, mask = process_image_with_opencv(resized_image)
        logger.info("Processed resized image with OpenCV")

        # Step 4: Generate SVG path data from the processed image
        svg_path_data = image_to_svg_path(red_image, mask)
        logger.info("Generated SVG path data")

        # Step 5: Generate the full SVG content
        height, width = resized_image.shape[:2]
        full_svg_content = generate_full_svg(svg_path_data, width, height)

        # Step 6: Save the full SVG content to a file
        save_svg_content(full_svg_content, output_svg_path)
        logger.info(f"Full SVG content saved to: {output_svg_path}")

        # Return both the full SVG content and the SVG path data
        return full_svg_content, svg_path_data
    except Exception as e:
        logger.error(f"Error in process_svg: {str(e)}")
        raise
