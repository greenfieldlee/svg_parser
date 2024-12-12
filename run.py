import cv2
import numpy as np
import svgwrite
import cairosvg

def convert_svg_to_red_and_smooth_svg(input_svg_path, output_svg_path):
    # Step 1: Render the SVG to an image (PNG) using CairoSVG
    temp_png_path = "temp_rendered.png"
    cairosvg.svg2png(url=input_svg_path, write_to=temp_png_path)

    # Step 2: Load the rendered image (with alpha channel)
    image = cv2.imread(temp_png_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]

    # Step 3: Separate channels
    bgr = image[:, :, :3]  # BGR channels
    alpha = image[:, :, 3]  # Alpha channel

    # Step 4: Create a red version of the image where content exists
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]  # Set red color for visible areas

    # Step 5: Create a binary mask of red areas
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)

    # Step 6: Apply Gaussian blur to smooth the mask
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (5, 5), 0)

    # Step 7: Find contours on the smoothed mask
    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Create an SVG document with width and height
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Step 9: Add contours as SVG paths (smoothed with approxPolyDP)
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust epsilon for more or less smoothing
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert the smoothed contour to SVG path data
        path_data = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in smoothed_contour]) + ' Z'
        path = dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1)
        dwg.add(path)

    # Step 10: Extract the outermost contour and add as a border path
    if contours:
        outer_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(outer_contour, True)
        smoothed_outer_contour = cv2.approxPolyDP(outer_contour, epsilon, True)

        # Convert the outermost smoothed contour to SVG path data
        border_path_data = 'M ' + ' '.join([f"{point[0][0]},{point[0][1]}" for point in smoothed_outer_contour]) + ' Z'
        border_path = dwg.path(d=border_path_data, fill='none', stroke='black', stroke_width=2)
        dwg.add(border_path)

    # Step 11: Save the SVG file
    dwg.save()

# Example Usage
convert_svg_to_red_and_smooth_svg("sticker.svg", "output_sticker_with_smooth_border.svg")
