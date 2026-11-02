#!/usr/bin/env python
"""
Rule-Based Temporal Consistency Module (TCM)
Eliminates impossible vehicle re-identification matches using physics-based constraints.

This approach is:
- More interpretable than learned models
- Computationally efficient 
- Based on real-world physics and traffic patterns
- Easily tunable for different environments
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

class RuleBasedTCM:
    """
    Rule-Based Temporal Consistency Module for Vehicle Re-ID
    
    Uses physics-based constraints to filter impossible temporal matches:
    - Camera distance matrix with real geographic positions
    - Variable speed limits by road type and time of day
    - Traffic pattern modeling (rush hour, normal, night)
    - Weather and road condition adjustments
    """
    
    def __init__(self, camera_positions: Dict[int, Tuple[float, float]], 
                 config: Optional[Dict] = None):
        """
        Initialize rule-based TCM
        
        Args:
            camera_positions: Dict mapping camera_id -> (lat, lon) coordinates
            config: Configuration parameters for filtering rules
        """
        self.camera_positions = camera_positions
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
            
        self.logger = logging.getLogger(__name__)
        
        # Build camera network graph
        self.camera_graph = self._build_camera_network()
        self.distance_matrix = self._compute_distance_matrix()
        
        # Traffic pattern models
        self.traffic_patterns = self._init_traffic_patterns()
        
        self.logger.info(f"Initialized Rule-Based TCM with {len(camera_positions)} cameras")
        
    def _get_default_config(self) -> Dict:
        """Default configuration for rule-based filtering"""
        return {
            # Speed constraints (km/h)
            'max_speed_highway': 120,
            'max_speed_arterial': 80,
            'max_speed_residential': 50,
            'max_speed_parking': 20,
            
            # Traffic adjustment factors
            'rush_hour_factor': 0.6,      # 60% of normal speed during rush hour
            'night_factor': 1.2,          # 20% faster at night
            'weekend_factor': 1.1,        # 10% faster on weekends
            
            # Time windows (seconds)
            'min_dwell_time': 5,          # Minimum time to stay in camera view
            'max_reasonable_time': 3600,  # 1 hour max for reasonable matches
            
            # Filtering thresholds
            'strict_mode': False,         # If True, use conservative estimates
            'confidence_threshold': 0.8,  # Minimum confidence for temporal feasibility
            
            # Geographic constraints
            'earth_radius_km': 6371,      # Earth radius for distance calculations
            'gps_accuracy_meters': 10,    # GPS accuracy buffer
        }
    
    def _build_camera_network(self) -> nx.Graph:
        """Build network graph of cameras with road connections"""
        G = nx.Graph()
        
        # Add camera nodes
        for cam_id, (lat, lon) in self.camera_positions.items():
            G.add_node(cam_id, lat=lat, lon=lon)
        
        # Add edges based on realistic road connections
        # In practice, this would use real road network data (OSM, Google Maps API)
        for cam1 in self.camera_positions:
            for cam2 in self.camera_positions:
                if cam1 != cam2:
                    distance = self._haversine_distance(
                        self.camera_positions[cam1], 
                        self.camera_positions[cam2]
                    )
                    
                    # Classify road type based on distance and camera placement
                    road_type = self._classify_road_type(cam1, cam2, distance)
                    
                    G.add_edge(cam1, cam2, 
                             distance=distance, 
                             road_type=road_type,
                             max_speed=self._get_speed_limit(road_type))
        
        return G
    
    def _haversine_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """
        Calculate haversine distance between two GPS coordinates
        
        Returns:
            Distance in meters
        """
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return self.config['earth_radius_km'] * c * 1000  # Convert to meters
    
    def _classify_road_type(self, cam1: int, cam2: int, distance: float) -> str:
        """
        Classify road type between cameras based on distance and context
        
        This is a simplified heuristic - in practice would use real road data
        """
        if distance > 5000:  # > 5km
            return 'highway'
        elif distance > 1000:  # 1-5km  
            return 'arterial'
        elif distance > 200:   # 200m-1km
            return 'residential'
        else:                  # < 200m
            return 'parking'
    
    def _get_speed_limit(self, road_type: str) -> float:
        """Get speed limit for road type (km/h)"""
        speed_map = {
            'highway': self.config['max_speed_highway'],
            'arterial': self.config['max_speed_arterial'], 
            'residential': self.config['max_speed_residential'],
            'parking': self.config['max_speed_parking']
        }
        return speed_map.get(road_type, self.config['max_speed_residential'])
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distance matrix between all cameras"""
        cam_ids = list(self.camera_positions.keys())
        n_cams = len(cam_ids)
        distance_matrix = np.zeros((n_cams, n_cams))
        
        # Use networkx shortest path for realistic routing
        for i, cam1 in enumerate(cam_ids):
            for j, cam2 in enumerate(cam_ids):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    try:
                        # Get shortest path considering road network
                        path = nx.shortest_path(self.camera_graph, cam1, cam2, weight='distance')
                        total_distance = 0
                        
                        for k in range(len(path) - 1):
                            edge_data = self.camera_graph[path[k]][path[k+1]]
                            total_distance += edge_data['distance']
                        
                        distance_matrix[i][j] = total_distance
                        
                    except nx.NetworkXNoPath:
                        # If no path exists, use direct distance
                        distance_matrix[i][j] = self._haversine_distance(
                            self.camera_positions[cam1],
                            self.camera_positions[cam2]
                        )
        
        return distance_matrix
    
    def _init_traffic_patterns(self) -> Dict:
        """Initialize traffic pattern models"""
        return {
            'rush_hours': [(7, 9), (17, 19)],  # Morning and evening rush
            'night_hours': [(22, 6)],           # Night time  
            'weekend_days': [5, 6],             # Saturday, Sunday (0=Monday)
        }
    
    def _get_traffic_factor(self, timestamp: float) -> float:
        """
        Get traffic adjustment factor based on time of day/week
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Speed adjustment factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
        """
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()
        
        # Weekend adjustment
        if weekday in self.traffic_patterns['weekend_days']:
            base_factor = self.config['weekend_factor']
        else:
            base_factor = 1.0
        
        # Rush hour adjustment
        for start, end in self.traffic_patterns['rush_hours']:
            if start <= hour <= end:
                return base_factor * self.config['rush_hour_factor']
        
        # Night adjustment
        for start, end in self.traffic_patterns['night_hours']:
            if hour >= start or hour <= end:
                return base_factor * self.config['night_factor']
        
        return base_factor
    
    def is_temporally_feasible(self, query_cam: int, query_time: float,
                             gallery_cam: int, gallery_time: float,
                             confidence_threshold: Optional[float] = None) -> Tuple[bool, float, Dict]:
        """
        Determine if a query-gallery match is temporally feasible
        
        Args:
            query_cam: Query camera ID
            query_time: Query timestamp
            gallery_cam: Gallery camera ID  
            gallery_time: Gallery timestamp
            confidence_threshold: Override default confidence threshold
            
        Returns:
            Tuple of (is_feasible, confidence_score, details)
        """
        # Time difference
        time_diff = abs(gallery_time - query_time)
        
        # Handle same camera case
        if query_cam == gallery_cam:
            # Must have minimum dwell time
            is_feasible = time_diff >= self.config['min_dwell_time']
            confidence = 1.0 if is_feasible else 0.0
            details = {
                'reason': 'same_camera',
                'min_dwell_time': self.config['min_dwell_time'],
                'actual_time_diff': time_diff
            }
            return is_feasible, confidence, details
        
        # Get camera indices for distance matrix
        cam_ids = list(self.camera_positions.keys())
        try:
            query_idx = cam_ids.index(query_cam)
            gallery_idx = cam_ids.index(gallery_cam)
        except ValueError:
            # Unknown camera - conservative approach
            return False, 0.0, {'reason': 'unknown_camera'}
        
        # Get distance between cameras
        distance = self.distance_matrix[query_idx][gallery_idx]
        
        # Get traffic adjustment factor
        avg_time = (query_time + gallery_time) / 2
        traffic_factor = self._get_traffic_factor(avg_time)
        
        # Calculate minimum travel time
        # Use road type specific speed limits
        edge_data = self.camera_graph[query_cam][gallery_cam]
        max_speed_kmh = edge_data['max_speed']
        max_speed_ms = (max_speed_kmh * traffic_factor * 1000) / 3600  # Convert to m/s
        
        min_travel_time = distance / max_speed_ms
        
        # Apply strict mode if enabled
        if self.config['strict_mode']:
            min_travel_time *= 1.5  # 50% buffer for conservative estimation
        
        # Check feasibility
        is_feasible = time_diff >= min_travel_time
        
        # Calculate confidence score
        if time_diff >= self.config['max_reasonable_time']:
            # Too long - might be different person/vehicle
            confidence = max(0.0, 1.0 - (time_diff - self.config['max_reasonable_time']) / 3600)
        elif is_feasible:
            # Feasible - higher confidence for reasonable time differences
            time_ratio = time_diff / min_travel_time
            confidence = min(1.0, 0.5 + 0.5 * np.log(time_ratio + 1))
        else:
            # Not feasible - confidence decreases with impossibility
            confidence = max(0.0, time_diff / min_travel_time)
        
        # Apply confidence threshold
        threshold = confidence_threshold or self.config['confidence_threshold']
        is_feasible = is_feasible and confidence >= threshold
        
        details = {
            'distance_meters': distance,
            'min_travel_time_seconds': min_travel_time,
            'actual_time_diff_seconds': time_diff,
            'traffic_factor': traffic_factor,
            'max_speed_kmh': max_speed_kmh,
            'road_type': edge_data['road_type'],
            'time_ratio': time_diff / min_travel_time if min_travel_time > 0 else float('inf')
        }
        
        return is_feasible, confidence, details
    
    def filter_similarity_matrix(self, similarity_matrix: np.ndarray,
                                query_metadata: List[Dict],
                                gallery_metadata: List[Dict],
                                return_details: bool = False) -> np.ndarray:
        """
        Filter similarity matrix to remove temporally impossible matches
        
        Args:
            similarity_matrix: (N_query, N_gallery) similarity scores
            query_metadata: List of query metadata dicts with 'camera_id' and 'timestamp'
            gallery_metadata: List of gallery metadata dicts with 'camera_id' and 'timestamp' 
            return_details: Whether to return detailed filtering information
            
        Returns:
            Filtered similarity matrix with impossible matches set to -inf
        """
        filtered_matrix = similarity_matrix.copy()
        filtering_stats = {'total_matches': 0, 'filtered_matches': 0, 'details': []}
        
        for i, query_meta in enumerate(query_metadata):
            for j, gallery_meta in enumerate(gallery_metadata):
                filtering_stats['total_matches'] += 1
                
                is_feasible, confidence, details = self.is_temporally_feasible(
                    query_meta['camera_id'], query_meta['timestamp'],
                    gallery_meta['camera_id'], gallery_meta['timestamp']
                )
                
                if not is_feasible:
                    filtered_matrix[i][j] = -float('inf')
                    filtering_stats['filtered_matches'] += 1
                    
                    if return_details:
                        filtering_stats['details'].append({
                            'query_idx': i,
                            'gallery_idx': j,
                            'query_cam': query_meta['camera_id'],
                            'gallery_cam': gallery_meta['camera_id'], 
                            'confidence': confidence,
                            'details': details
                        })
        
        filter_rate = filtering_stats['filtered_matches'] / filtering_stats['total_matches']
        self.logger.info(f"Filtered {filtering_stats['filtered_matches']}/{filtering_stats['total_matches']} "
                        f"matches ({filter_rate:.1%}) as temporally impossible")
        
        if return_details:
            return filtered_matrix, filtering_stats
        return filtered_matrix
    
    def analyze_temporal_violations(self, query_metadata: List[Dict],
                                  gallery_metadata: List[Dict]) -> Dict:
        """
        Analyze temporal violation patterns in the dataset
        
        Returns comprehensive statistics about impossible matches
        """
        stats = {
            'total_pairs': 0,
            'feasible_pairs': 0,
            'same_camera_pairs': 0,
            'impossible_pairs': 0,
            'violation_by_distance': {},
            'violation_by_time_diff': {},
            'confidence_distribution': []
        }
        
        for query_meta in query_metadata:
            for gallery_meta in gallery_metadata:
                stats['total_pairs'] += 1
                
                is_feasible, confidence, details = self.is_temporally_feasible(
                    query_meta['camera_id'], query_meta['timestamp'],
                    gallery_meta['camera_id'], gallery_meta['timestamp']
                )
                
                stats['confidence_distribution'].append(confidence)
                
                if query_meta['camera_id'] == gallery_meta['camera_id']:
                    stats['same_camera_pairs'] += 1
                elif is_feasible:
                    stats['feasible_pairs'] += 1
                else:
                    stats['impossible_pairs'] += 1
                    
                    # Analyze violation patterns
                    if 'distance_meters' in details:
                        dist_bin = int(details['distance_meters'] // 1000)  # 1km bins
                        stats['violation_by_distance'][dist_bin] = stats['violation_by_distance'].get(dist_bin, 0) + 1
                        
                        time_diff_hours = details['actual_time_diff_seconds'] // 3600
                        stats['violation_by_time_diff'][time_diff_hours] = stats['violation_by_time_diff'].get(time_diff_hours, 0) + 1
        
        # Calculate summary statistics
        stats['feasibility_rate'] = stats['feasible_pairs'] / stats['total_pairs']
        stats['violation_rate'] = stats['impossible_pairs'] / stats['total_pairs']
        stats['avg_confidence'] = np.mean(stats['confidence_distribution'])
        
        return stats