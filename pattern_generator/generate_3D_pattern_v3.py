import numpy as np
import trimesh
from trimesh.creation import cylinder, box
from scipy.spatial.distance import cdist
from itertools import combinations
from math import acos, degrees
from tqdm import tqdm
import argparse

def too_many_close_points(set1, set2, threshold=0.3, max_allowed=1):
    # check if two sets have overlapping points
    # Compute all pairwise distances
    dists = cdist(set1, set2)  # Shape: (5, 5)
    
    # Count how many distances are below threshold
    num_close_pairs = np.sum(dists < threshold)

    return num_close_pairs > max_allowed

def compute_edges(triangle):
    # Returns sorted edge lengths
    a = np.linalg.norm(triangle[0] - triangle[1])
    b = np.linalg.norm(triangle[1] - triangle[2])
    c = np.linalg.norm(triangle[2] - triangle[0])
    return np.sort([a, b, c])

def compute_angles(edges):
    # Edges: sorted lengths [a, b, c]
    a, b, c = edges
    # Use Law of Cosines
    angles = []
    try:
        angles.append(degrees(acos((b**2 + c**2 - a**2) / (2 * b * c))))
        angles.append(degrees(acos((a**2 + c**2 - b**2) / (2 * a * c))))
        angles.append(degrees(acos((a**2 + b**2 - c**2) / (2 * a * b))))
    except ValueError:
        return None  # Invalid triangle
    return np.sort(angles)

def triangles_similar(tri1, tri2, angle_thresh=5, edge_thresh=0.3):
    edges1 = compute_edges(tri1)
    edges2 = compute_edges(tri2)
    angles1 = compute_angles(edges1)
    angles2 = compute_angles(edges2)
    if angles1 is None or angles2 is None:
        return False
    # Compare sorted edge lengths and angles
    if np.all(np.abs(edges1 - edges2) < edge_thresh) and np.all(np.abs(angles1 - angles2) < angle_thresh):
        return True
    return False

def export_stl(pillar_pos_array, grid_size=8, cell_spacing=1.5, pillar_radius = 0.3, base_thickness = 0.5, id=0):
     
    # pillar_pos_array range: (grid_size - 2) * cell_spacing
    
    # --- Step 2: Create the base plate ---
    base_size = [grid_size * cell_spacing, grid_size * cell_spacing, base_thickness]
    base = box(extents=base_size)
    base.apply_translation([base_size[0] / 2, base_size[1] / 2, base_thickness / 2])

    pillars = []
    # two pillars diagonal
    height = 0.5
    cyl = cylinder(radius=pillar_radius, height=height, sections=32)
    x = (0 + 0.5) * cell_spacing
    y = (0 + 0.5) * cell_spacing
    z = base_thickness + height / 2
    cyl.apply_translation([x, y, z])

    pillars.append(cyl)

    height = 1.8
    cyl = cylinder(radius=pillar_radius, height=height, sections=32)
    x = (7 + 0.5) * cell_spacing
    y = (7 + 0.5) * cell_spacing
    z = base_thickness + height / 2
    cyl.apply_translation([x, y, z])

    pillars.append(cyl)

    # --- Step 3: Create cylindrical pillars ---
    for pp in pillar_pos_array:
        height = np.random.uniform(0.3, 2.3)
        cyl = cylinder(radius=pillar_radius, height=height, sections=32)
        x = cell_spacing + pp[0]
        y = cell_spacing + pp[1]

        z = base_thickness + height / 2
        cyl.apply_translation([x, y, z])

        pillars.append(cyl)    

    # --- Step 4: Combine base + pillars ---
    all_meshes = [base] + pillars
    combined = trimesh.util.concatenate(all_meshes)

    # --- Step 5: Export to STL ---
    file_exp_name = f'./pillar_model_{id}.stl'
    combined.export(file_exp_name)
    # print(f"STL model exported to '{file_exp_name}'")

def generate_position(n_points, grid_size=8, cell_spacing=1.5, pillar_radius = 0.3, max_tries=1000):
    points = []
    tries = 0
    region_size = (grid_size - 2) * cell_spacing
    min_dist = pillar_radius * 2 + 0.5

    while len(points) < n_points and tries < max_tries:
        candidate = np.random.uniform(0, region_size, size=2)
        if all(np.linalg.norm(candidate - p) >= min_dist for p in points):
            points.append(candidate)
        tries += 1

    if len(points) < n_points:
        raise ValueError(f"Could not generate {n_points} points with min distance {min_dist} after {max_tries} tries.")

    return np.array(points)

def points_similar(points1, points2, dist_threshold):
    # compare triangle similarity
    indices = list(combinations(range(5), 3))
    for idx1 in indices:
        tri1 = points1[list(idx1)]
        for idx2 in indices:
            tri2 = points2[list(idx2)]
            if triangles_similar(tri1, tri2):
                return True

    # compare point similarity
    if too_many_close_points(points1, points2, dist_threshold):
        return True
    
    return False

def generate_cylinder_collections(n_samples, n_points, grid_size=8, cell_spacing=1.5, pillar_radius = 0.3, max_tries=1000):
    generated_points = []
    dist_threshold = pillar_radius

    for i in range(n_samples):
        points = generate_position(n_points, grid_size, cell_spacing, pillar_radius, max_tries)

        if len(generated_points) > 0:
            try_count = 0
            check_passed = True
            pbar = tqdm(total=None, desc=f"Checking {i}-th point set", unit="tries")

            # get different pattern with Monte Carlo process
            while (try_count < max_tries):
                pbar.update(1)
                for point_set in generated_points:
                    if points_similar(point_set, points, dist_threshold):
                        # generate new set
                        points = generate_position(n_points, grid_size, cell_spacing, pillar_radius, max_tries)
                        check_passed = False

                        break

                if not check_passed:
                    check_passed = True
                    try_count += 1
                else:
                    break

            if try_count >= max_tries:
                raise ValueError(f"Could not generate {i}-th point set after {max_tries} tries.")
                

        generated_points.append(points)

    return generated_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Generate a collection of pattern containing the same number of points.\n" + 
                                     "The height of each point is randomized."
                                     )
    parser.add_argument('-p', '--n_points', type=int, default=5, help='Number of points in pattern')
    parser.add_argument('-n', '--n_sets', type=int, default=20, help='Number of sets to generate')
    parser.add_argument('-m', '--max_tries', type=int, default=1000, help='Maximum number of tries in generation')

    args = parser.parse_args()

    m_try = args.max_tries
    repeat = args.n_sets
    n_points = args.n_points
    point_collection = generate_cylinder_collections(repeat, n_points, max_tries=m_try)

    for i, ps in enumerate(point_collection):
        export_stl(ps, id=i)
