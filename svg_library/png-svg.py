from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import svgwrite
import sys
import os

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
    alpha = image[:, :, 3]  # Alpha channel (transparency)

    # Step 4: Replace non-transparent areas with red (0, 0, 255 in BGR)
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]  # Set red color in non-transparent areas

    # Step 5: Create a binary mask of red areas
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)

    # Step 6: Find contours in the red areas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")

    # Step 7: Create an SVG document with width and height
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Step 8: Convert contours to SVG paths for the red-filled content
    for contour in contours:
        # Ensure the contour is closed by adding the first point at the end
        contour = contour.reshape(-1, 2)
        closed_contour = np.vstack([contour, contour[0]])

        # Build the SVG path data
        path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_contour])

        # Create the SVG path for the contour
        path = dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1)
        dwg.add(path)

    # Step 9: Extract the outermost border (largest contour)
    outer_contour = max(contours, key=cv2.contourArea)

    # Step 10: Ensure the outer contour is closed
    outer_contour = outer_contour.reshape(-1, 2)
    closed_outer_contour = np.vstack([outer_contour, outer_contour[0]])

    # Convert the outermost contour to an SVG path
    border_path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_outer_contour])
    border_path = dwg.path(d=border_path_data, fill='none', stroke='black', stroke_width=2)
    dwg.add(border_path)

    # Step 11: Save the SVG file
    dwg.save()

if __name__ == "__main__":
    # Example usage from command line:
    # python combined_script.py input.png outlined_output.png final_output.svg
    if len(sys.argv) < 4:
        print("Usage: python combined_script.py <input_png> <outlined_png> <output_svg>")
        sys.exit(1)

    input_png = sys.argv[1]
    outlined_png = sys.argv[2]
    output_svg = sys.argv[3]

    # Add smooth grow outline
    add_smooth_grow_outline(
        input_path=input_png,
        output_path=outlined_png,
        border_color=(255, 0, 0, 255),  # Red border
        grow_pixels=10,
        blur_radius=1
    )

    # Convert outlined PNG to SVG
    convert_image_to_red_and_svg_with_border(
        image_path=outlined_png,
        output_svg_path=output_svg
    )

    print(f"SVG conversion complete: {output_svg}")
