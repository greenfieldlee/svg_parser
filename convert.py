import cv2
import numpy as np
import svgwrite

def convert_image_to_red_and_svg_with_border(image_path, output_svg_path):
    # Step 1: Load the image with alpha channel (transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    # Step 2: Get the dimensions of the image
    height, width = image.shape[:2]

    # Step 3: Split the image into 4 channels (BGR + Alpha)
    bgr = image[:, :, :3]  # BGR channels (without alpha)
    alpha = image[:, :, 3]  # Alpha channel (transparency)

    # Step 4: Replace non-transparent areas with red (0, 0, 255 in BGR)
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]  # Set red color in non-transparent areas

    # Step 5: Create a binary mask of red areas (where the content was filled with red)
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)

    # Step 6: Find contours in the red areas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Create an SVG document with width and height
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Step 8: Convert contours to SVG paths for the red-filled content
    for contour in contours:
        path_data = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in contour])
        path = dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1)
        dwg.add(path)

    # Step 9: Extract the outermost border (largest contour)
    # If there are multiple contours, choose the one with the largest area (outermost boundary)
    outer_contour = max(contours, key=cv2.contourArea)

    # Step 10: Convert the outermost contour to an SVG path
    border_path_data = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in outer_contour])
    border_path = dwg.path(d=border_path_data, fill='none', stroke='black', stroke_width=2)  # Black border
    dwg.add(border_path)

    # Step 11: Save the SVG file
    dwg.save()

# Example: Convert the image to red, extract the border, and save as SVG
convert_image_to_red_and_svg_with_border("sticker.png", "output_sticker_with_border.svg")
