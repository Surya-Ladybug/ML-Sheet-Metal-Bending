#!/usr/bin/env python3
"""
V-BEND ANALYSIS SCRIPT
- Inner Bend Radius: curvature-based
- Bend Angle: edge-based linear fits
- Tool nodes explicitly ignored (silent)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import sys


# --------------------------------------------------
# EXPLICIT TOOL NODE BLACKLIST (OPTION 1)
# --------------------------------------------------
IGNORED_NODE_IDS = {
    13656, 13655, 13654, 13653, 13652,
    13757, 13758, 13759, 13760, 13761
}


# --------------------------------------------------
# NODE PARSER
# --------------------------------------------------
def parse_nodes(filename):
    nodes = {}
    in_node_section = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('*NODE'):
                in_node_section = True
                continue
            if line.startswith('*') and in_node_section and not line.startswith('*NODE'):
                in_node_section = False
                continue
            if in_node_section and line and not line.startswith('*'):
                try:
                    p = line.split()
                    nid = int(p[0])
                    if nid not in IGNORED_NODE_IDS:
                        nodes[nid] = np.array([
                            float(p[1]),
                            float(p[2]),
                            float(p[3])
                        ])
                except:
                    pass
    return nodes


# --------------------------------------------------
# RADIUS CALCULATION (UNCHANGED)
# --------------------------------------------------
def calculate_curvature_radius(x, z, smoothing):
    spl = UnivariateSpline(x, z, k=3, s=smoothing)
    dz_dx = spl.derivative(1)(x)
    d2z_dx2 = spl.derivative(2)(x)

    curvature = np.abs(d2z_dx2) / (1 + dz_dx**2)**(3/2)
    radii = 1 / (curvature + 1e-10)

    return radii


# --------------------------------------------------
# ANGLE CALCULATION (FROM CODE 2)
# --------------------------------------------------
def calculate_bend_angle_correctly(x, z):
    n = len(x)
    left_end = n // 3
    right_start = 2 * n // 3

    if left_end < 3 or (n - right_start) < 3:
        return None

    left_fit = np.polyfit(x[:left_end], z[:left_end], 1)
    right_fit = np.polyfit(x[right_start:], z[right_start:], 1)

    angle_left = np.degrees(np.arctan(left_fit[0]))
    angle_right = np.degrees(np.arctan(right_fit[0]))

    return abs(angle_right - angle_left)


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------
def analyze_bend(neutral_file, deformed_file):

    print("\n" + "="*70)
    print("V-BEND RADIUS + ANGLE ANALYSIS")
    print("="*70)

    neutral_nodes = parse_nodes(neutral_file)
    deformed_nodes = parse_nodes(deformed_file)

    # --- TOP EDGE SELECTION
    edge_nodes = [
        (nid, c) for nid, c in deformed_nodes.items()
        if abs(c[1] - 100) < 0.5
    ]

    if len(edge_nodes) < 5:
        print("❌ Not enough edge nodes found")
        return None

    edge_nodes.sort(key=lambda x: x[1][0])

    node_ids = np.array([nid for nid, _ in edge_nodes])
    x = np.array([c[0] for _, c in edge_nodes])
    z = np.array([c[2] for _, c in edge_nodes])

    print(f"\n✓ Using {len(node_ids)} valid surface nodes")

    # --- RADIUS (STABILITY VIA MULTI-SMOOTHING)
    results = []
    for s in [0.01, 0.05, 0.1, 0.2]:
        try:
            r = calculate_curvature_radius(x, z, s)
            idx = np.argmin(r)
            if 0.1 < r[idx] < 1000:
                results.append((s, r, idx))
        except:
            pass

    if not results:
        print("❌ Radius calculation failed")
        return None

    smoothing, radii, min_idx = results[len(results) // 2]

    min_radius = radii[min_idx]
    min_node = node_ids[min_idx]
    min_x = x[min_idx]
    min_z = z[min_idx]

    # --- ANGLE
    bend_angle = calculate_bend_angle_correctly(x, z)

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Inner Bend Radius : {min_radius:.4f} mm")
    print(f"Primary Node ID   : {min_node}")
    print(f"Location          : X = {min_x:.2f} mm")

    if bend_angle is not None:
        print(f"Bend Angle        : {bend_angle:.2f}°")
    else:
        print("Bend Angle        : Could not compute")

    # --------------------------------------------------
    # NODE DETAILS
    # --------------------------------------------------
    print("\n" + "="*70)
    print("NODES USED IN RADIUS CALCULATION")
    print("="*70)
    print(f"{'Node ID':<12} {'X (mm)':<12} {'Z (mm)':<12} {'Local R (mm)':<15} {'Role':<20}")
    print("-"*70)

    neighbor_range = 2
    start = max(0, min_idx - neighbor_range)
    end = min(len(node_ids), min_idx + neighbor_range + 1)

    for i in range(start, end):
        if i == min_idx:
            role = "⭐ MIN RADIUS"
        elif abs(i - min_idx) == 1:
            role = "Adjacent"
        else:
            role = "Nearby"

        print(f"{node_ids[i]:<12} "
              f"{x[i]:<12.4f} "
              f"{z[i]:<12.4f} "
              f"{radii[i]:<15.4f} "
              f"{role:<20}")

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70 + "\n")

    return {
        "radius": min_radius,
        "angle": bend_angle,
        "node": min_node,
        "location": min_x
    }


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    neutral = input("Enter neutral file path: ").strip().strip('"').strip("'")
    deformed = input("Enter deformed file path: ").strip().strip('"').strip("'")

    analyze_bend(neutral, deformed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
