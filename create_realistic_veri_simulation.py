#!/usr/bin/env python
"""
Create Realistic Temporal Simulation for VeRi-776
Overlay a realistic camera network with proper travel times on existing VeRi data
"""

import os
import re
import glob
import numpy as np
import networkx as nx
from collections import defaultdict
import pickle

class VeRiCameraNetwork:
    """Realistic camera network topology for VeRi-776"""
    
    def __init__(self):
        # Define realistic camera network based on actual VeRi camera relationships
        # From our analysis: cameras 3-4, 6-10-11, 8-9, 14-15-16 are close
        
        self.G = nx.Graph()
        
        # Add all 20 VeRi cameras
        for cam_id in range(1, 21):
            self.G.add_node(cam_id)
        
        # Define edges with realistic travel times (in seconds)
        # Format: (cam1, cam2, min_time, max_time) based on speed limits and distance
        
        edges = [
            # Cluster 1: Highway entrance area (cameras 1, 2, 13)
            (1, 2, 20, 60),
            (1, 13, 15, 45),
            (2, 13, 25, 70),
            
            # Cluster 2: Central intersection (cameras 3, 4, 9, 10, 11, 6)
            (3, 4, 5, 15),    # Very close - adjacent lanes
            (3, 9, 10, 30),
            (3, 10, 20, 50),
            (4, 9, 8, 25),
            (4, 10, 15, 40),
            (6, 10, 5, 20),   # Very close
            (6, 11, 5, 15),   # Very close
            (10, 11, 3, 12),  # Adjacent cameras
            (9, 8, 8, 25),
            
            # Cluster 3: Parking/residential area (cameras 14, 15, 16, 19)
            (14, 15, 5, 20),  # Very close
            (14, 16, 15, 40),
            (15, 16, 12, 35),
            (15, 19, 20, 55),
            (16, 19, 18, 50),
            
            # Cluster 4: Secondary roads (cameras 5, 7, 8)
            (5, 7, 8, 30),
            (5, 8, 25, 70),
            (7, 8, 20, 60),
            
            # Cluster 5: Outer area (cameras 12, 17, 18)
            (12, 13, 30, 90),
            (12, 17, 20, 60),
            (17, 18, 10, 35),
            (17, 19, 15, 45),
            
            # Long-distance connections
            (2, 6, 40, 120),
            (6, 11, 5, 18),
            (11, 2, 35, 100),
            (1, 12, 60, 180),
            (13, 17, 50, 150),
            (8, 16, 70, 200),
        ]
        
        for cam1, cam2, min_time, max_time in edges:
            self.G.add_edge(cam1, cam2, min_time=min_time, max_time=max_time)
    
    def get_travel_time(self, cam1, cam2):
        """Get realistic travel time between two cameras"""
        if cam1 == cam2:
            return 0, 0  # Same camera
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.G, cam1, cam2)
            
            # Sum up travel times along path
            total_min = 0
            total_max = 0
            
            for i in range(len(path) - 1):
                edge_data = self.G[path[i]][path[i+1]]
                total_min += edge_data['min_time']
                total_max += edge_data['max_time']
            
            return total_min, total_max
            
        except nx.NetworkXNoPath:
            # No path between cameras - very distant
            return 300, 600  # 5-10 minutes
    
    def is_temporally_feasible(self, cam1, time1, cam2, time2):
        """Check if transition is physically possible"""
        if cam1 == cam2:
            return True  # Always feasible for same camera
        
        time_diff = abs(time2 - time1)
        min_time, max_time = self.get_travel_time(cam1, cam2)
        
        # Add some buffer (traffic variability)
        min_time = max(0, min_time - 10)
        max_time = max_time + 30
        
        return min_time <= time_diff <= max_time


def create_realistic_timestamps():
    """Create realistic timestamps for VeRi-776 based on camera network"""
    
    print("="*70)
    print("CREATING REALISTIC TEMPORAL SIMULATION FOR VeRi-776")
    print("="*70)
    
    network = VeRiCameraNetwork()
    
    query_dir = 'data/VeRi/image_query'
    gallery_dir = 'data/VeRi/image_test'
    
    pattern = re.compile(r'([-\d]+)_c(\d+)_(\d+)_\d+\.jpg')
    
    # Collect all vehicle sightings
    vehicle_sightings = defaultdict(list)
    
    for directory in [query_dir, gallery_dir]:
        if os.path.exists(directory):
            for img_path in glob.glob(os.path.join(directory, '*.jpg')):
                match = pattern.search(os.path.basename(img_path))
                if match:
                    vid = int(match.group(1))
                    cam = int(match.group(2))
                    frame = int(match.group(3))
                    
                    if vid != -1:
                        vehicle_sightings[vid].append({
                            'camera': cam,
                            'original_frame': frame,
                            'path': img_path
                        })
    
    print(f"\nFound {len(vehicle_sightings)} vehicles")
    
    # Re-assign timestamps to be consistent with network topology
    realistic_timestamps = {}
    
    for vid, sightings in vehicle_sightings.items():
        # Sort by original frame number (preserves relative ordering)
        sightings.sort(key=lambda x: x['original_frame'])
        
        # Start at random time (simulate different vehicles at different times of day)
        current_time = np.random.randint(0, 86400)  # Random second in a day
        
        for i, sight in enumerate(sightings):
            # Store timestamp for this image
            img_name = os.path.basename(sight['path'])
            realistic_timestamps[img_name] = current_time
            
            # If there's a next sighting, calculate realistic travel time
            if i < len(sightings) - 1:
                next_sight = sightings[i + 1]
                
                cam1 = sight['camera']
                cam2 = next_sight['camera']
                
                # Get realistic travel time
                min_time, max_time = network.get_travel_time(cam1, cam2)
                
                # Add random travel time within realistic bounds
                travel_time = np.random.randint(min_time, max_time + 1)
                
                # Add some dwell time if same camera
                if cam1 == cam2:
                    dwell_time = np.random.randint(5, 30)  # 5-30 seconds
                    current_time += dwell_time
                else:
                    current_time += travel_time
    
    print(f"Generated realistic timestamps for {len(realistic_timestamps)} images")
    
    # Save the mapping
    output_file = 'data/VeRi/realistic_timestamps.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(realistic_timestamps, f)
    
    print(f"\nSaved to: {output_file}")
    
    # Analyze the simulation
    print("\n" + "="*70)
    print("SIMULATION STATISTICS")
    print("="*70)
    
    # Check temporal feasibility of generated data
    feasible_count = 0
    total_count = 0
    
    for vid, sightings in list(vehicle_sightings.items())[:50]:
        sightings.sort(key=lambda x: realistic_timestamps[os.path.basename(x['path'])])
        
        for i in range(len(sightings) - 1):
            sight1 = sightings[i]
            sight2 = sightings[i + 1]
            
            cam1 = sight1['camera']
            cam2 = sight2['camera']
            time1 = realistic_timestamps[os.path.basename(sight1['path'])]
            time2 = realistic_timestamps[os.path.basename(sight2['path'])]
            
            if network.is_temporally_feasible(cam1, time1, cam2, time2):
                feasible_count += 1
            total_count += 1
    
    print(f"\nTemporal feasibility check: {feasible_count}/{total_count} ({100*feasible_count/total_count:.1f}%)")
    
    # Show some examples
    print("\n" + "="*70)
    print("EXAMPLE VEHICLE TRAJECTORIES")
    print("="*70)
    
    for vid in list(vehicle_sightings.keys())[:3]:
        sightings = vehicle_sightings[vid]
        sightings.sort(key=lambda x: realistic_timestamps[os.path.basename(x['path'])])
        
        print(f"\nVehicle {vid}:")
        prev_time = None
        prev_cam = None
        
        for sight in sightings[:5]:
            cam = sight['camera']
            time = realistic_timestamps[os.path.basename(sight['path'])]
            
            if prev_time is not None:
                time_diff = time - prev_time
                min_t, max_t = network.get_travel_time(prev_cam, cam)
                feasible = "✓" if min_t <= time_diff <= max_t + 30 else "✗"
                print(f"  Camera {cam:2d} at {time:6d}s (Δ{time_diff:3d}s from cam {prev_cam}, range: {min_t}-{max_t}s) {feasible}")
            else:
                print(f"  Camera {cam:2d} at {time:6d}s")
            
            prev_time = time
            prev_cam = cam
    
    return realistic_timestamps, network


if __name__ == '__main__':
    timestamps, network = create_realistic_timestamps()
    
    print("\n" + "="*70)
    print("✓ Realistic temporal simulation created!")
    print("="*70)
    print("\nNow you can test TCM with:")
    print("  python test_rule_based_tcm_simulated.py")
