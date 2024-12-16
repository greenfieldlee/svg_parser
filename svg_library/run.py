import sys
import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
import svgwrite
import cairosvg
from scipy.interpolate import splprep, splev


def convert_svg_to_png(svg_file_path, output_png_path):
    """
    Converts an SVG file to a PNG file using cairosvg.
    """
    try:
        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()
        
        cairosvg.svg2png(bytestring=svg_content, write_to=output_png_path)
        print(f"Successfully converted {svg_file_path} to {output_png_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def add_smooth_grow_outline(input_path, output_path, border_color=(255, 0, 0, 255), grow_pixels=10, blur_radius=5):
    # Load the original image
    original = Image.open(input_path).convert("RGBA")

    # Increase padding to avoid cutting off the outline
    padding = (grow_pixels + blur_radius) * 2

    width, height = original.size
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    padded_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    padded_image.paste(original, (padding, padding))

    image = padded_image
    alpha = image.getchannel("A")

    # Expand alpha channel and create an outline
    expanded_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=grow_pixels + blur_radius))
    outline_alpha = expanded_alpha.point(lambda p: 255 if p > 10 else 0)

    outline_alpha_np = np.array(outline_alpha)
    original_alpha_np = np.array(alpha)
    border_mask_np = np.clip(outline_alpha_np - original_alpha_np, 0, 255)
    border_mask = Image.fromarray(border_mask_np.astype("uint8"), mode="L")

    # Slight blur for smoother edge
    border_mask = border_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Apply border color
    border_image = Image.new("RGBA", image.size, border_color)
    border_image.putalpha(border_mask)

    composite = Image.alpha_composite(border_image, image)

    # Optionally crop with a buffer
    bbox = composite.getbbox()
    if bbox:
        buffer = 10
        left, upper, right, lower = bbox
        left = max(left - buffer, 0)
        upper = max(upper - buffer, 0)
        right = min(right + buffer, composite.width)
        lower = min(lower + buffer, composite.height)
        composite = composite.crop((left, upper, right, lower))

    composite.save(output_path, "PNG")


def smooth_contour_with_bspline(contour, smoothing_factor=0.001, num_points=200):
    """
    Smooth a contour using B-spline interpolation.

    :param contour: The contour as an Nx2 numpy array.
    :param smoothing_factor: Higher values produce smoother curves.
    :param num_points: Number of points in the smoothed output contour.
    """
    if len(contour) < 3:
        return contour  # Not enough points to smooth meaningfully

    contour = contour.astype(np.float32)
    x = contour[:, 0]
    y = contour[:, 1]

    t = np.linspace(0, 1, len(x))
    # Using s proportional to the length of the contour for tuning smoothness
    tck, u = splprep([x, y], s=smoothing_factor * len(x), per=True)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)

    smoothed_contour = np.column_stack((x_fine, y_fine)).astype(np.float32)
    return smoothed_contour


def convert_image_to_red_and_svg_with_smooth_border(image_path, output_svg_path, smoothing_factor=0.001, num_points=200):
    # Load the image with alpha
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    # Separate channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Replace non-transparent areas with red
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]

    # Create binary mask
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)
    mask = (mask.astype(np.uint8) * 255)

    # Morphological smoothing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Slight blur and re-threshold for smoother edges
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask_blurred, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in the image.")

    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Convert all contours to red-filled SVG paths, smoothing them and closing with 'Z'
    for contour in contours:
        contour = contour.reshape(-1, 2)
        smoothed = smooth_contour_with_bspline(contour, smoothing_factor=smoothing_factor, num_points=num_points)
        path_data = 'M ' + ' '.join([f"{pt[0]},{pt[1]}" for pt in smoothed]) + ' Z'
        dwg.add(dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1))

    # Extract the largest contour for the border and smooth it as well
    outer_contour = max(contours, key=cv2.contourArea)
    outer_contour = outer_contour.reshape(-1, 2)
    outer_smoothed = smooth_contour_with_bspline(outer_contour, smoothing_factor=smoothing_factor, num_points=num_points)
    outer_path_data = 'M ' + ' '.join([f"{pt[0]},{pt[1]}" for pt in outer_smoothed]) + ' Z'
    dwg.add(dwg.path(d=outer_path_data, fill='none', stroke='black', stroke_width=2))

    dwg.save()


if __name__ == "__main__":
    # Usage:
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
        border_color=(255, 0, 0, 255),
        grow_pixels=10,
        blur_radius=1
    )

    # Step 3: Convert the outlined PNG to a final SVG with B-spline smoothed border
    # Ensuring the paths are closed with 'Z'
    convert_image_to_red_and_svg_with_smooth_border(
        image_path=outlined_png,
        output_svg_path=final_svg,
        smoothing_factor=0.001,
        num_points=200
    )

    print("Conversion complete!")
