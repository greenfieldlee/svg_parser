import os
import uuid
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import cairosvg
import numpy as np
import cv2
import svgwrite
from PIL import Image, ImageFilter
from xml.etree import ElementTree as ET
from scipy.interpolate import splprep, splev

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup CORS middleware
origins = [
    "http://localhost:3000",
    "https://sticker.getonnet.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)

# Ensure temporary directories exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("temp_outputs", exist_ok=True)


def convert_svg_to_png(svg_file_path, output_png_path):
    """
    Converts an SVG file to a PNG file using cairosvg.
    """
    try:
        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()

        cairosvg.svg2png(bytestring=svg_content, write_to=output_png_path)
        logger.info(f"Successfully converted {svg_file_path} to {output_png_path}")
    except Exception as e:
        logger.error(f"Error during SVG to PNG conversion: {e}")
        raise


def add_smooth_grow_outline(input_path, output_path, border_color=(255, 0, 0, 255), grow_pixels=10, blur_radius=1):
    # Load the original image
    original = Image.open(input_path).convert("RGBA")

    # Determine padding size
    # You can increase this if the outline appears cut off.
    padding = grow_pixels + blur_radius

    # Create a new, larger image with transparent background
    width, height = original.size
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    padded_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    padded_image.paste(original, (padding, padding))

    image = padded_image
    alpha = image.getchannel("A")

    # Expand alpha channel and create outline
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

    # Crop if needed
    bbox = composite.getbbox()
    if bbox:
        # Add a buffer if needed
        # buffer = 10
        # left, upper, right, lower = bbox
        # left = max(left - buffer, 0)
        # upper = max(upper - buffer, 0)
        # right = min(right + buffer, composite.width)
        # lower = min(lower + buffer, composite.height)
        # composite = composite.crop((left, upper, right, lower))
        composite = composite.crop(bbox)

    composite.save(output_path, "PNG")


def smooth_contour_with_bspline(contour, smoothing_factor=0.001, num_points=200):
    """
    Smooth a contour using B-spline interpolation.
    """
    if len(contour) < 3:
        return contour

    contour = contour.astype(np.float32)
    x = contour[:, 0]
    y = contour[:, 1]

    t = np.linspace(0, 1, len(x))
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


def extract_svg_paths(svg_file_path):
    """
    Parse the final SVG and extract all path 'd' attributes.
    """
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    namespace = '{http://www.w3.org/2000/svg}'
    paths = []
    for elem in root.findall(f'{namespace}path'):
        d_attr = elem.get('d', '')
        if d_attr:
            paths.append(d_attr)
    return paths


@app.get("/process-svg/")
async def process_svg_info():
    return JSONResponse(content={
        "message": "This endpoint processes SVG files.",
        "usage": "Send a POST request to this endpoint with 'file' (SVG file) and 'grow' (float) in the form data."
    })


@app.post("/process-svg/")
async def process_svg_endpoint(file: UploadFile = File(...), grow: float = Form(0.0)):
    # Generate unique filenames
    temp_svg = f"temp_uploads/{uuid.uuid4()}.svg"
    temp_intermediate_png = f"temp_outputs/{uuid.uuid4()}.png"
    temp_outlined_png = f"temp_outputs/{uuid.uuid4()}.png"
    temp_result_svg = f"temp_outputs/{uuid.uuid4()}.svg"

    try:
        # Save uploaded file
        with open(temp_svg, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {temp_svg}")

        # Step 1: Convert the input SVG to PNG
        convert_svg_to_png(temp_svg, temp_intermediate_png)

        # Step 2: Add smooth grow outline to the PNG
        # Adjusting grow parameter if needed
        add_smooth_grow_outline(
            input_path=temp_intermediate_png,
            output_path=temp_outlined_png,
            border_color=(255, 0, 0, 255),
            grow_pixels=int(grow / 5),
            blur_radius=1
        )

        # Step 3: Convert the outlined PNG to final SVG with smoothing
        # You can adjust smoothing_factor and num_points as needed
        convert_image_to_red_and_svg_with_smooth_border(
            image_path=temp_outlined_png,
            output_svg_path=temp_result_svg,
            smoothing_factor=0.001,
            num_points=200
        )

        # Extract path data from final SVG
        svg_paths = extract_svg_paths(temp_result_svg)

        logger.info("Processing complete.")
        return JSONResponse(content={"svg_paths": svg_paths})

    except Exception as e:
        logger.error(f"Error processing SVG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files
        for temp_file in [temp_svg, temp_intermediate_png, temp_outlined_png]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
