#!/usr/bin/env python
"""
Calibrate camera GPS coordinates from VeRiCameraNetwork travel times.

The realistic_timestamps.pkl was generated using VeRiCameraNetwork's hardcoded
travel times. We need GPS coordinates that make V2 TCM (Haversine + tortuosity +
speed profiles) produce time windows compatible with those travel times.

Approach:
1. Extract pairwise (T_min, T_max) from VeRiCameraNetwork
2. Back-compute target Haversine distances from T_min and V2 speed profiles
3. Use scipy optimization to find 2D coordinates matching those distances
4. Convert to lat/lon and print updated VERI_CAMERA_POSITIONS
"""

import sys
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '.')
from create_realistic_veri_simulation import VeRiCameraNetwork

TAU = 1.4  # tortuosity

# V2 TCM speed profiles (km/h → m/s)
ROAD_TYPES = [
    # (name, max_dist_m, v_max_kmh, v_min_kmh)
    ('parking',     200,   20,  1),
    ('residential', 1000,  40,  5),
    ('arterial',    5000,  60, 10),
    ('highway',     1e9,  120, 40),
]


def get_speed_for_distance(d_driving):
    """Return (v_max_ms, v_min_ms) for a given driving distance."""
    for name, max_d, vmax, vmin in ROAD_TYPES:
        if d_driving < max_d:
            return vmax * 1000 / 3600, vmin * 1000 / 3600
    return 120 * 1000 / 3600, 40 * 1000 / 3600


def compute_target_haversine(t_min_target):
    """
    Find the Haversine distance that makes V2 TCM's T_min = t_min_target.
    Iterative because road type depends on D_driving which depends on D_hav.
    """
    d_driving = t_min_target * 11.11  # initial guess
    for _ in range(20):
        v_max_ms, _ = get_speed_for_distance(d_driving)
        d_new = t_min_target * v_max_ms
        if abs(d_new - d_driving) < 0.01:
            break
        d_driving = d_new
    return d_driving / TAU


def smacof_stress(coords_flat, target_dist_mat, n, weights):
    """SMACOF stress function for optimization."""
    coords = coords_flat.reshape(n, 2)
    stress = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d_target = target_dist_mat[i, j]
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d_actual = np.sqrt(dx*dx + dy*dy)
            w = weights[i, j]
            stress += w * (d_actual - d_target) ** 2
    return stress


def main():
    net = VeRiCameraNetwork()
    
    cams = list(range(1, 21))
    n = len(cams)
    
    # Step 1: Get VeRiCameraNetwork travel times for all pairs
    target_tmin = np.zeros((n, n))
    target_tmax = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mn, mx = net.get_travel_time(cams[i], cams[j])
                target_tmin[i, j] = mn
                target_tmax[i, j] = mx
    
    # Step 2: Compute target Haversine distances
    # Camera 20 (index 19) has no edges → default (300,600). Handle separately.
    target_dhav = np.zeros((n, n))
    weights = np.ones((n, n))  # optimization weights
    
    print("Back-computing target Haversine distances...\n")
    
    for i in range(n):
        for j in range(i + 1, n):
            t_mn = target_tmin[i, j]
            d_hav = compute_target_haversine(t_mn)
            target_dhav[i, j] = d_hav
            target_dhav[j, i] = d_hav
            
            # Lower weight for cam 20 (no real edges) and very long distances
            if cams[i] == 20 or cams[j] == 20:
                weights[i, j] = 0.01
                weights[j, i] = 0.01
            elif target_tmin[i, j] >= 200:
                weights[i, j] = 0.5  # distant pairs less important
                weights[j, i] = 0.5
    
    dists = target_dhav[np.triu_indices(n, k=1)]
    print(f"Target Haversine distances: min={dists.min():.1f}m, max={dists.max():.1f}m, mean={dists.mean():.1f}m")
    
    # Step 3: Optimize 2D coordinates to match target distances
    # Use multiple restarts to find the best embedding
    print("\nOptimizing 2D coordinates (weighted SMACOF)...")
    
    best_stress = np.inf
    best_coords = None
    
    for seed in range(50):
        rng = np.random.RandomState(seed)
        # Initialize with spread proportional to max target distance
        x0 = rng.randn(n * 2) * np.mean(dists)
        
        result = minimize(smacof_stress, x0, args=(target_dhav, n, weights),
                          method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12})
        
        if result.fun < best_stress:
            best_stress = result.fun
            best_coords = result.x.reshape(n, 2)
    
    # Evaluate quality (on weighted pairs only)
    errors_close = []
    errors_far = []
    for i in range(n):
        for j in range(i + 1, n):
            if cams[i] == 20 or cams[j] == 20:
                continue
            d_recon = np.sqrt((best_coords[i, 0] - best_coords[j, 0])**2 +
                              (best_coords[i, 1] - best_coords[j, 1])**2)
            d_target = target_dhav[i, j]
            if d_target > 0:
                rel_err = abs(d_recon - d_target) / d_target
                if target_tmin[i, j] < 50:
                    errors_close.append(rel_err)
                else:
                    errors_far.append(rel_err)
    
    print(f"Close pairs (<50s) error: mean={np.mean(errors_close):.3f}, max={np.max(errors_close):.3f}")
    print(f"Far pairs (>=50s) error:  mean={np.mean(errors_far):.3f}, max={np.max(errors_far):.3f}")
    
    # Step 4: Convert 2D meters to lat/lon
    ref_lat = 40.7600
    ref_lon = -73.9800
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(ref_lat))
    
    positions = {}
    for i in range(n):
        lat = ref_lat + best_coords[i, 1] / meters_per_deg_lat
        lon = ref_lon + best_coords[i, 0] / meters_per_deg_lon
        positions[cams[i]] = (round(lat, 6), round(lon, 6))
    
    # Step 5: Verify against ACTUAL same-vehicle timestamp data
    print("\n\nVerification against realistic_timestamps.pkl...")
    
    import re
    from collections import defaultdict
    
    with open('data/VeRi/realistic_timestamps.pkl', 'rb') as f:
        ts = pickle.load(f)
    
    pattern = re.compile(r'([-\d]+)_c(\d+)_(\d+)_\d+\.jpg')
    vehicle_obs = defaultdict(list)
    for fname, timestamp in ts.items():
        m = pattern.match(fname)
        if m:
            pid = int(m.group(1))
            camid = int(m.group(2))
            vehicle_obs[pid].append((camid, timestamp))
    
    # Build V2 TCM with new positions
    from utils.rule_based_tcm_v2 import RuleBasedTCMV2
    tcm = RuleBasedTCMV2(positions)
    
    # Check: for every same-vehicle cross-camera pair, does V2 accept it?
    accepted = 0
    rejected = 0
    total_pairs = 0
    rejected_examples = []
    
    for pid, obs in vehicle_obs.items():
        for i in range(len(obs)):
            for j in range(len(obs)):
                if i != j:
                    c1, t1 = obs[i]
                    c2, t2 = obs[j]
                    if c1 != c2:
                        total_pairs += 1
                        dt = abs(t2 - t1)
                        
                        # Get V2 time window (use query timestamp for alpha)
                        t_min_v2, t_max_v2 = tcm.get_travel_time(c1, c2, query_time=t1)
                        
                        if t_min_v2 <= dt <= t_max_v2:
                            accepted += 1
                        else:
                            rejected += 1
                            if len(rejected_examples) < 10:
                                rejected_examples.append(
                                    f"  pid={pid} C{c1}->C{c2}: dt={dt:.0f}s, "
                                    f"window=[{t_min_v2:.0f},{t_max_v2:.0f}]")
    
    recall = 100 * accepted / total_pairs if total_pairs > 0 else 0
    print(f"True match pairs tested: {total_pairs}")
    print(f"Accepted: {accepted} ({recall:.2f}%)")
    print(f"Rejected: {rejected} ({100-recall:.2f}%)")
    
    if rejected_examples:
        print(f"\nSample rejections:")
        for ex in rejected_examples:
            print(ex)
    
    # Step 6: Compute V2 distance stats  
    all_dists = []
    for i in range(n):
        for j in range(i+1, n):
            c1, c2 = cams[i], cams[j]
            d = tcm.distance_matrix.get((c1, c2), tcm.distance_matrix.get((c2, c1), 0))
            if d and d < float('inf'):
                all_dists.append(d)
    all_dists = np.array(all_dists)
    print(f"\nV2 driving distance range: {all_dists.min():.1f}m - {all_dists.max():.1f}m (mean {all_dists.mean():.1f}m)")
    
    # Step 7: Print the new positions
    print("\n\n# === PASTE THIS INTO utils/veri_camera_config.py ===")
    print("VERI_CAMERA_POSITIONS = {")
    for cam_id in cams:
        lat, lon = positions[cam_id]
        print(f"    {cam_id}: ({lat}, {lon}),")
    print("}")


if __name__ == '__main__':
    main()
