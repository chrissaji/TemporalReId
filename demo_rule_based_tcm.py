#!/usr/bin/env python
"""
Simple Demo: Rule-Based TCM for Vehicle Re-ID
Shows how the rule-based temporal consistency filtering works
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.veri_camera_config import create_veri_tcm, get_camera_distance_analysis

def create_mock_reid_results():
    """Create mock Re-ID results for demonstration"""
    # Simulate 50 query images and 200 gallery images
    n_queries = 50
    n_gallery = 200
    
    # Random feature similarity scores (0 = identical, 1 = completely different)
    np.random.seed(42)  # For reproducible results
    distance_matrix = np.random.rand(n_queries, n_gallery)
    
    # Mock camera and timestamp data
    query_cameras = np.random.randint(1, 21, n_queries)  # Cameras 1-20
    gallery_cameras = np.random.randint(1, 21, n_gallery)
    
    # Mock timestamps (in seconds from some reference point)
    base_time = 10000  # Starting timestamp
    query_timestamps = base_time + np.random.randint(0, 3600, n_queries)  # Within 1 hour
    gallery_timestamps = base_time + np.random.randint(0, 3600, n_gallery)
    
    # Ensure some matches are temporally impossible for demonstration
    for i in range(10):  # Make first 10 queries have some impossible matches
        for j in range(20):  # In first 20 gallery images
            # Create impossible scenarios: very different cameras, very short time
            if query_cameras[i] != gallery_cameras[j]:
                time_diff = abs(gallery_timestamps[j] - query_timestamps[i])
                if time_diff < 30:  # Less than 30 seconds between distant cameras
                    gallery_timestamps[j] = query_timestamps[i] + 10  # 10 second gap
    
    return {
        'distance_matrix': distance_matrix,
        'query_cameras': query_cameras,
        'gallery_cameras': gallery_cameras, 
        'query_timestamps': query_timestamps,
        'gallery_timestamps': gallery_timestamps
    }

def apply_tcm_filtering(data, tcm):
    """Apply rule-based TCM filtering to mock results"""
    distance_matrix = data['distance_matrix'].copy()
    query_cameras = data['query_cameras']
    gallery_cameras = data['gallery_cameras']
    query_timestamps = data['query_timestamps']
    gallery_timestamps = data['gallery_timestamps']
    
    n_queries, n_gallery = distance_matrix.shape
    filtered_count = 0
    feasible_count = 0
    
    print("Applying TCM filtering...")
    
    for q_idx in range(n_queries):
        q_cam = query_cameras[q_idx]
        q_time = query_timestamps[q_idx]
        
        for g_idx in range(n_gallery):
            g_cam = gallery_cameras[g_idx]
            g_time = gallery_timestamps[g_idx]
            
            # Skip same camera (always feasible)
            if q_cam == g_cam:
                feasible_count += 1
                continue
            
            # Check temporal feasibility
            feasible, confidence, details = tcm.is_temporally_feasible(q_cam, q_time, g_cam, g_time)
            
            if feasible:
                feasible_count += 1
            else:
                # Apply penalty to impossible matches
                penalty = 2.0 * (1 - confidence)
                distance_matrix[q_idx, g_idx] += penalty
                filtered_count += 1
                
                # Show example of filtering
                if filtered_count <= 5:  # Show first 5 filtered matches
                    time_diff = abs(g_time - q_time)
                    print(f"  Filtered match {filtered_count}: Query Cam{q_cam} -> Gallery Cam{g_cam}")
                    print(f"    Time difference: {time_diff:.1f}s, Distance: {details.get('distance_meters', 0):.1f}m")
                    print(f"    Confidence: {confidence:.2f}, Penalty: {penalty:.2f}")
    
    total_matches = n_queries * n_gallery
    filter_percentage = (filtered_count / total_matches) * 100
    feasible_percentage = (feasible_count / total_matches) * 100
    
    print(f"\nTCM Filtering Results:")
    print(f"  Total matches: {total_matches}")
    print(f"  Feasible matches: {feasible_count} ({feasible_percentage:.1f}%)")
    print(f"  Filtered matches: {filtered_count} ({filter_percentage:.1f}%)")
    
    return distance_matrix, filter_percentage

def compute_mock_metrics(distance_matrix, suffix=""):
    """Compute mock Re-ID metrics for demonstration"""
    n_queries, n_gallery = distance_matrix.shape
    
    # Mock ground truth: each query has 5 positive matches in gallery
    rank_1_correct = 0
    rank_5_correct = 0
    rank_10_correct = 0
    
    for q_idx in range(n_queries):
        # Get sorted indices by distance (ascending)
        sorted_indices = np.argsort(distance_matrix[q_idx])
        
        # Mock: assume indices 0-4 in gallery are positives for this query
        # (In real scenario, you'd have ground truth labels)
        positive_indices = [(q_idx * 4 + i) % n_gallery for i in range(4)]
        
        # Check if any positive is in top-k
        top_1 = sorted_indices[:1]
        top_5 = sorted_indices[:5]
        top_10 = sorted_indices[:10]
        
        if any(idx in positive_indices for idx in top_1):
            rank_1_correct += 1
        if any(idx in positive_indices for idx in top_5):
            rank_5_correct += 1  
        if any(idx in positive_indices for idx in top_10):
            rank_10_correct += 1
    
    # Calculate CMC
    rank_1_acc = rank_1_correct / n_queries
    rank_5_acc = rank_5_correct / n_queries
    rank_10_acc = rank_10_correct / n_queries
    
    # Mock mAP (simplified)
    mock_map = np.mean([rank_1_acc, rank_5_acc * 0.8, rank_10_acc * 0.6])
    
    print(f"\nResults{suffix}:")
    print(f"  mAP: {mock_map:.4f}")
    print(f"  CMC rank-1: {rank_1_acc:.4f}")
    print(f"  CMC rank-5: {rank_5_acc:.4f}")
    print(f"  CMC rank-10: {rank_10_acc:.4f}")
    
    return {
        'mAP': mock_map,
        'rank_1': rank_1_acc,
        'rank_5': rank_5_acc,
        'rank_10': rank_10_acc
    }

def main():
    print("="*60)
    print("RULE-BASED TCM DEMO FOR VEHICLE RE-IDENTIFICATION")
    print("="*60)
    
    # Analyze VeRi camera network
    print("\n" + "="*40)
    print("VERI CAMERA NETWORK ANALYSIS")
    print("="*40)
    stats = get_camera_distance_analysis()
    print(f"Number of cameras: 20")
    print(f"Camera pairs analyzed: {len(stats['camera_pairs'])}")
    print(f"Distance range: {stats['min_distance_m']:.1f}m - {stats['max_distance_m']:.1f}m")
    print(f"Mean inter-camera distance: {stats['mean_distance_m']:.1f}m")
    
    # Initialize TCM
    tcm = create_veri_tcm()
    print(f"\nRule-based TCM initialized with VeRi-776 configuration")
    
    # Create mock Re-ID results
    print(f"\n" + "="*40)
    print("GENERATING MOCK RE-ID RESULTS")
    print("="*40)
    data = create_mock_reid_results()
    print(f"Mock dataset: {data['distance_matrix'].shape[0]} queries, {data['distance_matrix'].shape[1]} gallery")
    print(f"Camera range: 1-20 (VeRi-776)")
    print(f"Time range: {data['query_timestamps'].min():.0f} - {data['query_timestamps'].max():.0f} seconds")
    
    # Compute baseline results
    print(f"\n" + "="*40)
    print("BASELINE RESULTS (No Temporal Filtering)")
    print("="*40)
    baseline_results = compute_mock_metrics(data['distance_matrix'], " (Baseline)")
    
    # Apply TCM filtering
    print(f"\n" + "="*40)
    print("APPLYING RULE-BASED TCM FILTERING")
    print("="*40)
    tcm_distance_matrix, filter_rate = apply_tcm_filtering(data, tcm)
    
    # Compute TCM-enhanced results
    print(f"\n" + "="*40)
    print("TCM-ENHANCED RESULTS")
    print("="*40)
    tcm_results = compute_mock_metrics(tcm_distance_matrix, " (TCM-Enhanced)")
    
    # Final comparison
    print(f"\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"Baseline mAP:       {baseline_results['mAP']:.4f}")
    print(f"TCM-Enhanced mAP:   {tcm_results['mAP']:.4f}")
    print(f"mAP Improvement:    {tcm_results['mAP'] - baseline_results['mAP']:+.4f}")
    print(f"")
    print(f"Baseline Rank-1:    {baseline_results['rank_1']:.4f}")
    print(f"TCM-Enhanced Rank-1: {tcm_results['rank_1']:.4f}")
    print(f"Rank-1 Improvement: {tcm_results['rank_1'] - baseline_results['rank_1']:+.4f}")
    print(f"")
    print(f"Temporal Filter Rate: {filter_rate:.1f}% of matches penalized")
    
    # TCM behavior analysis
    print(f"\n" + "="*40)
    print("TCM BEHAVIOR ANALYSIS")
    print("="*40)
    
    print("Testing specific scenarios:")
    
    # Test cases for different temporal scenarios
    test_scenarios = [
        (1, 10000, 5, 10020, "Nearby cameras, 20 second gap"),
        (1, 10000, 15, 10120, "Distant cameras, 120 second gap"),
        (3, 10000, 3, 10010, "Same camera, 10 second gap"),
        (2, 10000, 18, 10030, "Medium distance, 30 second gap"),
        (10, 10000, 20, 10015, "Very close cameras, 15 second gap"),
    ]
    
    for i, (q_cam, q_time, g_cam, g_time, desc) in enumerate(test_scenarios):
        feasible, confidence, details = tcm.is_temporally_feasible(q_cam, q_time, g_cam, g_time)
        time_diff = abs(g_time - q_time)
        
        print(f"\nScenario {i+1}: {desc}")
        print(f"  Query: Camera {q_cam} at time {q_time}")
        print(f"  Gallery: Camera {g_cam} at time {g_time}")
        print(f"  Time difference: {time_diff} seconds")
        
        if q_cam != g_cam and 'distance_meters' in details:
            print(f"  Physical distance: {details['distance_meters']:.1f}m")
            print(f"  Minimum travel time: {details['min_travel_time_seconds']:.1f}s")
        
        status = "✅ FEASIBLE" if feasible else "❌ IMPOSSIBLE"
        print(f"  Result: {status} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()