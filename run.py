import cv2
import numpy as np
import cairosvg
import svgwrite
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
from shapely.ops import unary_union

def svg_to_image(svg_path, output_image_path, scale=2):
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

def simplify_polygon(polygon, tolerance=1.0):
    """
    Simplify a polygon using Shapely's simplify method.
    """
    poly = Polygon(polygon)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    return np.array(simplified.exterior.coords).astype(int)

def fit_bezier(points, num_control_points=4):
    """
    Fit a Bezier curve to a set of points.
    """
    x, y = points.T
    t = np.linspace(0, 1, len(x))
    
    # Fit x(t) and y(t) cubic Bezier curves
    cx = np.polyfit(t, x, num_control_points - 1)
    cy = np.polyfit(t, y, num_control_points - 1)
    
    # Generate Bezier curve points
    t_new = np.linspace(0, 1, 100)
    x_bezier = np.polyval(cx, t_new)
    y_bezier = np.polyval(cy, t_new)
    
    return np.column_stack((x_bezier, y_bezier)).astype(int)

def image_to_svg_with_red(image, mask, output_svg_path, original_width, original_height):
    """
    Converts the processed image back to SVG by extracting, simplifying, and smoothing contours.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny', size=(f"{original_width}px", f"{original_height}px"))
    
    scale_factor = original_width / image.shape[1]
    
    # Merge overlapping contours
    polygons = [Polygon(cont.reshape(-1, 2)) for cont in contours if cv2.contourArea(cont) > 50]
    merged = unary_union(polygons)
    
    if isinstance(merged, Polygon):
        merged = [merged]
    
    for poly in merged:
        contour = np.array(poly.exterior.coords).astype(int)
        simplified = simplify_polygon(contour, tolerance=1.0)
        smooth_cont = fit_bezier(simplified)
        scaled_cont = (smooth_cont * scale_factor).astype(int)
        
        # Convert to SVG path
        path_data = f"M {scaled_cont[0][0]},{scaled_cont[0][1]}"
        for i in range(1, len(scaled_cont), 3):
            if i + 2 < len(scaled_cont):
                path_data += f" C {scaled_cont[i][0]},{scaled_cont[i][1]} {scaled_cont[i+1][0]},{scaled_cont[i+1][1]} {scaled_cont[i+2][0]},{scaled_cont[i+2][1]}"
        
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

