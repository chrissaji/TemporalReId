#!/usr/bin/env python
"""
Analyze VeRi-776 dataset to understand actual temporal patterns
This will help create realistic camera network configuration
"""

import os
import re
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

def analyze_veri_temporal_patterns():
    """Analyze actual VeRi-776 data to understand temporal patterns"""
    
    # VeRi data paths
    query_dir = 'data/VeRi/image_query'
    gallery_dir = 'data/VeRi/image_test'
    
    # Pattern to extract: vehicleID_cameraID_timestamp_frameID.jpg
    pattern = re.compile(r'([-\d]+)_c(\d+)_(\d+)_\d+\.jpg')
    
    # Collect data
    vehicle_sightings = defaultdict(list)  # {vehicle_id: [(cam, timestamp, path), ...]}
    camera_pairs = defaultdict(list)  # {(cam1, cam2): [time_diffs]}
    camera_vehicle_counts = defaultdict(set)  # {cam: set(vehicle_ids)}
    
    print("Analyzing VeRi-776 temporal patterns...")
    
    # Process all images
    all_images = []
    for directory in [query_dir, gallery_dir]:
        if os.path.exists(directory):
            all_images.extend(glob.glob(os.path.join(directory, '*.jpg')))
    
    print(f"Found {len(all_images)} images")
    
    # Extract temporal information
    for img_path in all_images:
        match = pattern.search(os.path.basename(img_path))
        if match:
            vehicle_id = int(match.group(1))
            camera_id = int(match.group(2))
            timestamp = int(match.group(3))
            
            if vehicle_id == -1:  # Skip junk
                continue
                
            vehicle_sightings[vehicle_id].append((camera_id, timestamp, img_path))
            camera_vehicle_counts[camera_id].add(vehicle_id)
    
    print(f"\nFound {len(vehicle_sightings)} unique vehicles")
    print(f"Found {len(camera_vehicle_counts)} cameras")
    
    # Analyze cross-camera temporal patterns
    print("\nAnalyzing cross-camera patterns...")
    
    for vehicle_id, sightings in vehicle_sightings.items():
        # Sort by timestamp
        sightings.sort(key=lambda x: x[1])
        
        # Look at consecutive sightings
        for i in range(len(sightings) - 1):
            cam1, time1, _ = sightings[i]
            cam2, time2, _ = sightings[i + 1]
            
            time_diff = time2 - time1
            
            # Record time difference between camera pairs
            cam_pair = tuple(sorted([cam1, cam2]))
            camera_pairs[cam_pair].append(time_diff)
    
    # Analyze results
    print("\n" + "="*60)
    print("CAMERA PAIR ANALYSIS")
    print("="*60)
    
    # Camera pair statistics
    for (cam1, cam2), time_diffs in sorted(camera_pairs.items()):
        if len(time_diffs) >= 5:  # Only show pairs with enough samples
            time_diffs = np.array(time_diffs)
            print(f"\nCamera {cam1} <-> Camera {cam2}:")
            print(f"  Transitions: {len(time_diffs)}")
            print(f"  Time diff - Min: {time_diffs.min():.0f}, Max: {time_diffs.max():.0f}")
            print(f"  Time diff - Mean: {time_diffs.mean():.0f}, Median: {np.median(time_diffs):.0f}")
            print(f"  Time diff - Std: {time_diffs.std():.0f}")
    
    # Camera statistics
    print("\n" + "="*60)
    print("CAMERA STATISTICS")
    print("="*60)
    
    for cam_id in sorted(camera_vehicle_counts.keys()):
        vehicle_count = len(camera_vehicle_counts[cam_id])
        print(f"Camera {cam_id:2d}: {vehicle_count:3d} unique vehicles")
    
    # Overall temporal statistics
    print("\n" + "="*60)
    print("OVERALL TEMPORAL STATISTICS")
    print("="*60)
    
    all_time_diffs = []
    for time_diffs in camera_pairs.values():
        all_time_diffs.extend(time_diffs)
    
    all_time_diffs = np.array(all_time_diffs)
    print(f"\nTotal cross-camera transitions: {len(all_time_diffs)}")
    print(f"Time diff range: {all_time_diffs.min():.0f} to {all_time_diffs.max():.0f}")
    print(f"Time diff mean: {all_time_diffs.mean():.0f}")
    print(f"Time diff median: {np.median(all_time_diffs):.0f}")
    print(f"Time diff std: {all_time_diffs.std():.0f}")
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95, 99]
    print(f"\nTime diff percentiles:")
    for p in percentiles:
        print(f"  {p:2d}th: {np.percentile(all_time_diffs, p):.0f}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR TCM")
    print("="*60)
    
    min_reasonable = np.percentile(all_time_diffs, 1)
    max_reasonable = np.percentile(all_time_diffs, 99)
    
    print(f"\nSuggested min_time_diff: {min_reasonable:.0f}")
    print(f"Suggested max_time_diff: {max_reasonable:.0f}")
    print(f"\nNote: These are based on actual VeRi-776 vehicle transitions")
    print(f"Frame numbers appear to be continuous counters per camera")
    
    return {
        'camera_pairs': camera_pairs,
        'camera_vehicle_counts': camera_vehicle_counts,
        'all_time_diffs': all_time_diffs
    }

if __name__ == '__main__':
    results = analyze_veri_temporal_patterns()
