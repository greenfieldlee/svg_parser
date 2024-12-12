import cv2
import numpy as np
import cairosvg
import svgwrite
from scipy.interpolate import splprep, splev

def svg_to_image(svg_path, output_image_path, scale=4):
    """
    Converts an SVG file to a PNG file with increased resolution.
    """
    width, height = get_svg_dimensions(svg_path)
    cairosvg.svg2png(url=svg_path, write_to=output_image_path, output_width=width*scale, output_height=height*scale)
    return cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)

def get_svg_dimensions(svg_path):
    """
    Extract width and height from SVG file.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = int(float(root.attrib['width'].rstrip('px')))
    height = int(float(root.attrib['height'].rstrip('px')))
    return width, height

def process_image_with_opencv(image):
    """
    Processes the image to change non-transparent content to red.
    """
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    red_image = bgr.copy()
    red_image[alpha > 0] = [0, 0, 255]
    mask = np.all(red_image == [0, 0, 255], axis=-1)
    return red_image, mask

def smooth_contour(contour, k=3, s=0):
    """
    Smooth the contour using spline interpolation.
    """
    x, y = contour.T
    # Convert from numpy arrays to lists
    x = x.tolist()[0]
    y = y.tolist()[0]

    # Append the starting point to the end of the list of points
    x.append(x[0])
    y.append(y[0])

    # Fit splines to x=f(u) and y=g(u), treating both as periodic. Also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = splprep([x, y], u=None, s=s, per=True)

    # Evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = splev(np.linspace(0, 1, 1000), tck)
    
    return np.column_stack((xi, yi)).astype(int)

def image_to_svg_with_red(image, mask, output_svg_path, original_width, original_height):
    """
    Converts the processed image back to SVG by extracting and smoothing contours.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{original_width}px", f"{original_height}px"))
    
    scale_factor = original_width / image.shape[1]
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            smooth_cont = smooth_contour(contour)
            scaled_cont = (smooth_cont * scale_factor).astype(int)
            path_data = 'M ' + ' '.join([f"{point[0]},{point[1]}" for point in scaled_cont])
            path = dwg.path(d=path_data, fill='red', stroke='none')
            dwg.add(path)
    
    dwg.save()

def process_svg(svg_path, output_image_path, output_svg_path):
    """
    Main function to process an SVG and convert its content to red, saving the output as a new SVG.
    """
    original_width, original_height = get_svg_dimensions(svg_path)
    image = svg_to_image(svg_path, output_image_path)
    red_image, mask = process_image_with_opencv(image)
    image_to_svg_with_red(red_image, mask, output_svg_path, original_width, original_height)
    print(f"SVG processed and saved to {output_svg_path}")

# Example Usage
process_svg("input_sticker.svg", "temp_image.png", "output_sticker_red.svg")

