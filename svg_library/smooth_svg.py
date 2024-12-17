import sys
import math
from svgpathtools import svg2paths, wsvg, Path, CubicBezier
import numpy as np
from scipy.interpolate import CubicSpline

def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points
    first_pt = np.array(points[0])
    last_pt = np.array(points[-1])

    dist_max = 0
    index = 0
    for i in range(1, len(points)-1):
        pt = np.array(points[i])
        dist = point_line_distance(pt, first_pt, last_pt)
        if dist > dist_max:
            dist_max = dist
            index = i

    if dist_max > epsilon:
        results1 = douglas_peucker(points[:index+1], epsilon)
        results2 = douglas_peucker(points[index:], epsilon)
        return results1[:-1] + results2
    else:
        return [points[0], points[-1]]

def point_line_distance(p, a, b):
    ap = p - a
    ab = b - a
    cross_prod = np.cross(ab, ap)
    dist = np.linalg.norm(cross_prod) / np.linalg.norm(ab)
    return dist

def sample_path(path, num_samples=200):
    length = path.length()
    distances = np.linspace(0, length, num_samples)
    sampled_points = [path.point(d) for d in distances]
    return [(p.real, p.imag) for p in sampled_points]

def fit_cubic_spline(points):
    points = np.array(points)
    xs, ys = points[:,0], points[:,1]
    t = np.linspace(0, 1, len(points))

    cs_x = CubicSpline(t, xs)
    cs_y = CubicSpline(t, ys)

    dense_t = np.linspace(0, 1, 20)
    dense_points = np.column_stack((cs_x(dense_t), cs_y(dense_t)))

    segments = []
    for i in range(0, len(dense_points)-3, 3):
        p0 = dense_points[i]
        p1 = dense_points[i+1]
        p2 = dense_points[i+2]
        p3 = dense_points[i+3]

        seg = CubicBezier(
            complex(p0[0], p0[1]),
            complex(p1[0], p1[1]),
            complex(p2[0], p2[1]),
            complex(p3[0], p3[1])
        )
        segments.append(seg)

    return Path(*segments)

def smooth_path(original_path, epsilon=1.0):
    sampled = sample_path(original_path, num_samples=300)
    simplified = douglas_peucker(sampled, epsilon=epsilon)
    new_path = fit_cubic_spline(simplified)
    return new_path

def main(input_file, output_file, epsilon=1.0):
    # Try loading with three outputs first:
    try:
        paths, attributes, svg_attributes = svg2paths(input_file)
    except ValueError:
        # If older version of svgpathtools, fallback
        svg_attributes = {}
        paths, attributes = svg2paths(input_file)
    
    new_paths = []
    for p in paths:
        new_p = smooth_path(p, epsilon=epsilon)
        new_paths.append(new_p)

    # Try writing out
    # wsvg(new_paths, filename=output_file, attributes=attributes, svg_attributes=svg_attributes)
    print("New paths:", new_paths)
    # wsvg(new_paths, filename=output_file)

    svg_content = wsvg(paths=new_paths)  # no filename, returns string
    with open(output_file, "w") as f:
        f.write(svg_content)
    
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python smooth_svg.py test.svg output.svg [epsilon]")
        sys.exit(1)

    input_svg = sys.argv[1]
    output_svg = sys.argv[2]
    if len(sys.argv) > 3:
        epsilon_val = float(sys.argv[3])
    else:
        epsilon_val = 1.0

    main(input_svg, output_svg, epsilon=epsilon_val)
