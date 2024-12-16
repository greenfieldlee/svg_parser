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
        # Read the SVG content
        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()

        # Convert to PNG and save
        cairosvg.svg2png(bytestring=svg_content, write_to=output_png_path)
        logger.info(f"Successfully converted {svg_file_path} to {output_png_path}")
    except Exception as e:
        logger.error(f"Error during SVG to PNG conversion: {e}")
        raise


def add_smooth_grow_outline(input_path, output_path, border_color=(255, 0, 0, 255), grow_pixels=10, blur_radius=0.5):
    # Load the original image
    original = Image.open(input_path).convert("RGBA")

    # Determine padding size
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

    # Step 1: Expand the alpha channel
    expanded_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=grow_pixels + blur_radius))

    # Step 2: Threshold the blurred alpha
    outline_alpha = expanded_alpha.point(lambda p: 255 if p > 10 else 0)

    # Create the outline by subtracting the original alpha from the expanded alpha
    outline_alpha_np = np.array(outline_alpha)
    original_alpha_np = np.array(alpha)
    border_mask_np = np.clip(outline_alpha_np - original_alpha_np, 0, 255)
    border_mask = Image.fromarray(border_mask_np.astype("uint8"), mode="L")

    # Step 4: Apply a slight blur to the border mask
    border_mask = border_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Step 5: Apply the border color
    border_image = Image.new("RGBA", image.size, border_color)
    border_image.putalpha(border_mask)

    # Step 6: Composite the border image with the original (padded) image
    composite = Image.alpha_composite(border_image, image)

    # Trim the image to remove excess padding
    bbox = composite.getbbox()
    if bbox:
        composite = composite.crop(bbox)

    # Save the result
    composite.save(output_path, "PNG")


def convert_image_to_red_and_svg_with_border(image_path, output_svg_path):
    # Load the image with alpha channel (transparency)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]

    # Split channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Replace non-transparent areas with red
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]

    # Create a binary mask of red areas
    mask = np.all(red_image == [0, 0, 255], axis=-1) & (alpha > 0)

    # Find contours in the red areas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")

    # Create an SVG document
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{width}px", f"{height}px"))

    # Convert all contours to SVG paths (red fill)
    for contour in contours:
        contour = contour.reshape(-1, 2)
        closed_contour = np.vstack([contour, contour[0]])
        path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_contour])
        path = dwg.path(d=path_data, fill='red', stroke='none', stroke_width=1)
        dwg.add(path)

    # Extract outermost border (largest contour)
    outer_contour = max(contours, key=cv2.contourArea)
    outer_contour = outer_contour.reshape(-1, 2)
    closed_outer_contour = np.vstack([outer_contour, outer_contour[0]])
    border_path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in closed_outer_contour])
    border_path = dwg.path(d=border_path_data, fill='none', stroke='black', stroke_width=2)
    dwg.add(border_path)

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
        # Using grow as grow_pixels for demonstration
        add_smooth_grow_outline(
            input_path=temp_intermediate_png,
            output_path=temp_outlined_png,
            border_color=(255, 0, 0, 255),
            grow_pixels=int(grow / 5),
            blur_radius=1
        )

        # Step 3: Convert the outlined PNG to final SVG
        convert_image_to_red_and_svg_with_border(temp_outlined_png, temp_result_svg)

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
