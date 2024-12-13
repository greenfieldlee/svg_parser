import cairosvg

def convert_svg_to_png(svg_file_path, output_png_path):
    """
    Converts an SVG file to a PNG file.

    :param svg_file_path: Path to the input SVG file.
    :param output_png_path: Path to the output PNG file.
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

# Example usage
if __name__ == "__main__":
    input_svg = "sample.svg"  # Replace with your SVG file path
    output_png = "sticker.png"  # Replace with your desired PNG output path
    convert_svg_to_png(input_svg, output_png)
