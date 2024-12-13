from PIL import Image, ImageFilter, ImageOps
import numpy as np

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
    # This crops the image to the smallest bounding box that includes all non-transparent pixels.
    bbox = composite.getbbox()
    if bbox:
        composite = composite.crop(bbox)

    # Save the result
    composite.save(output_path, "PNG")

# Example usage:
add_smooth_grow_outline("input.png", "output_with_smooth_grow_outline.png", border_color=(255, 0, 0, 255), grow_pixels=10, blur_radius=1)
