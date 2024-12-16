import sys
import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
import svgwrite
import cairosvg


def convert_svg_to_png(svg_file_path, output_png_path):
    """
    Converts an SVG file to a PNG file using cairosvg.
    """
    try:
        # Read the SVG content
        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()
        
        # Convert to PNG and save
        cairosvg.svg2png(bytestring=svg_content, write_to=output_png_path)
        print(f"Successfully converted {svg_file_path} to {output_png_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def add_smooth_grow_outline(input_path, output_path, border_color=(255, 0, 0, 255), grow_pixels=10, blur_radius=5):
    # Load the original image
    original = Image.open(input_path).convert("RGBA")

    # Determine padding size (can adjust as needed)
    padding = grow_pixels + blur_radius

    # Create a new, larger image with transparent background
    width, height = original.size
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    padded_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    # Paste the original image into the center
    padded_image.paste(original, (padding, padding))

    # Now work with the padded image
    image = padded_image
    alpha = image.getchannel("A")  # Extract the alpha channel

    # Step 1: Expand the alpha channel using a larger Gaussian Blur for smoother edges
    expanded_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=grow_pixels + blur_radius))

    # Step 2: Threshold the blurred alpha to create a solid outline with anti-aliasing
    outline_alpha = expanded_alpha.point(lambda p: 255 if p > 10 else 0)

    # Step 3: Create the border mask
    outline_alpha_np = np.array(outline_alpha)
    original_alpha_np = np.array(alpha)

    # Create the outline by subtracting the original alpha from the expanded alpha
    border_mask_np = np.clip(outline_alpha_np - original_alpha_np, 0, 255)
    border_mask = Image.fromarray(border_mask_np.astype("uint8"), mode="L")

    # Step 4: Apply a slight blur to the border mask
    border_mask = border_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Step 5: Apply the border color
    border_image = Image.new("RGBA", image.size, border_color)
    border_image.putalpha(border_mask)

    # Step 6: Composite the border image with the original (padded) image
    composite = Image.alpha_composite(border_image, image)

    # Optional: Trim the image to remove excess padding if desired
    bbox = composite.getbbox()
    if bbox:
        composite = composite.crop(bbox)

    # Save the result
    composite.save(output_path, "PNG")


def convert_image_to_red_and_svg_with_border(image_path, output_svg_path):
    # Step 1: Load the image with alpha channel (transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 2: Get the dimensions of the image
    height, width = image.shape[:2]

    # Step 3: Split the image into 4 channels (BGR + Alpha)
    bgr = image[:, :, :3]  # BGR channels (without alpha)
    alpha = image[:, :, 3]  # Alpha channel

    # Step 4: Replace non-transparent areas with red (0, 0, 255)
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]

    # Step 5: Create a binary mask of red areas
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)

    # Step 6: Find contours in the red areas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")

    # Step 7: Create an SVG document
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Step 8: Convert all contours to red-filled SVG paths
    for contour in contours:
        contour = contour.reshape(-1, 2)
        closed_contour = np.vstack([contour, contour[0]])
        path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_contour])
        path = dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1)
        dwg.add(path)

    # Step 9: Extract the outermost border (largest contour)
    outer_contour = max(contours, key=cv2.contourArea)

    # Step 10: Convert the outermost contour to an SVG path (black border)
    outer_contour = outer_contour.reshape(-1, 2)
    closed_outer_contour = np.vstack([outer_contour, outer_contour[0]])
    border_path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_outer_contour])
    border_path = dwg.path(d=border_path_data, fill='none', stroke='black', stroke_width=2)
    dwg.add(border_path)

    # Step 11: Save the SVG
    dwg.save()


if __name__ == "__main__":
    # Usage example:
    # python run.py input.svg intermediate.png outlined.png final.svg
    if len(sys.argv) < 5:
        print("Usage: python run.py <input_svg> <intermediate_png> <outlined_png> <final_svg>")
        sys.exit(1)

    input_svg = sys.argv[1]
    intermediate_png = sys.argv[2]
    outlined_png = sys.argv[3]
    final_svg = sys.argv[4]

    # Step 1: Convert SVG to PNG
    convert_svg_to_png(input_svg, intermediate_png)

    # Step 2: Add smooth grow outline to the PNG
    add_smooth_grow_outline(
        input_path=intermediate_png,
        output_path=outlined_png,
        border_color=(255, 0, 0, 255),  # Red border
        grow_pixels=10,
        blur_radius=1
    )

    # Step 3: Convert the outlined PNG to SVG
    convert_image_to_red_and_svg_with_border(
        image_path=outlined_png,
        output_svg_path=final_svg
    )

    print("Conversion complete!")
