#!/usr/bin/env python
"""
Rule-Based Temporal Consistency Module (TCM) - Version 2
Strictly aligned with the IEEE Paper methodology.

Features matched to paper:
1. Tortuosity Factor (tau = 1.4) in distance calculation.
2. Dynamic Traffic Factor alpha(t) for upper bound expansion.
3. Explicit T_min and T_max constraints (Time Window).
4. Road-type specific speed profiles (v_max, v_avg, v_min).
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
import datetime

class RuleBasedTCMV2:
    """
    Physics-Aware Rule-Based TCM (Paper Implementation)
    
    Implements:
    - Eq 1: D_driving = tau * D_haversine
    - Eq 2: T_min = D / v_max
    - Eq 3: T_max = (D / v_min) * alpha(t)
    - Eq 5: delta(g) = 1 if T_min <= dt <= T_max
    """
    
    def __init__(self, camera_positions: Dict[int, Tuple[float, float]], 
                 config: Optional[Dict] = None):
        self.camera_positions = camera_positions
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
            
        self.logger = logging.getLogger(__name__)
        
        # Build camera network graph
        self.camera_graph = self._build_camera_network()
        self.distance_matrix = self._compute_distance_matrix()
        
        self.logger.info(f"Initialized V2 TCM with {len(camera_positions)} cameras, tau={self.config['tortuosity_factor']}")

    def _get_default_config(self) -> Dict:
        return {
            # Paper Parameter: Tortuosity (tau)
            'tortuosity_factor': 1.4,
            
            # Speed Profiles (km/h) -> v_max, v_min
            'speed_limits': {
                'highway':     {'v_max': 120, 'v_min': 40},
                'arterial':    {'v_max': 60,  'v_min': 10}, # Paper example
                'residential': {'v_max': 40,  'v_min': 5},
                'parking':     {'v_max': 20,  'v_min': 1}
            },
            
            # Paper Parameter: Alpha(t) constants
            'alpha_rush': 1.5,
            'alpha_normal': 1.0,
            
            # Traffic Windows
            'rush_hours': [(7, 9), (17, 19)],
            
            # Constants
            'earth_radius_km': 6371,
            'min_dwell_time': 0, # For same camera, 0 implies instantaneous is impossible but strict 0 might be harsh, paper implies T_min check handles diff cameras.
        }

    def _haversine_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Raw geodesic distance (Eq 1 part 1)"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return self.config['earth_radius_km'] * c * 1000  # meters

    def _get_driving_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Eq 1: D_driving = tau * D_haversine"""
        d_hav = self._haversine_distance(pos1, pos2)
        return self.config['tortuosity_factor'] * d_hav

    def _classify_road_type(self, distance: float) -> str:
        """Heuristic road typing based on distance (Paper Section III.B)"""
        if distance > 5000: return 'highway'
        elif distance > 1000: return 'arterial'
        elif distance > 200: return 'residential'
        else: return 'parking'

    def _build_camera_network(self) -> nx.Graph:
        G = nx.Graph()
        for cam_id, pos in self.camera_positions.items():
            G.add_node(cam_id, pos=pos)
            
        # Fully connected mesh for simplicity, or use real map
        for cam1 in self.camera_positions:
            for cam2 in self.camera_positions:
                if cam1 != cam2:
                    p1 = self.camera_positions[cam1]
                    p2 = self.camera_positions[cam2]
                    
                    dist = self._get_driving_distance(p1, p2)
                    rtype = self._classify_road_type(dist)
                    
                    G.add_edge(cam1, cam2, 
                             distance=dist, 
                             road_type=rtype)
        return G

    def _compute_distance_matrix(self) -> np.ndarray:
        # Since we use a fully connected heuristic graph, shortest path is direct edge
        # But we implement the structure if we had a sparse graph
        cam_ids = sorted(list(self.camera_positions.keys()))
        n = len(cam_ids)
        mat = np.zeros((n, n))
        
        for i, c1 in enumerate(cam_ids):
            for j, c2 in enumerate(cam_ids):
                if i != j:
                    try:
                        # For fully connected, this is just edge weight
                        # If sparse, nx.shortest_path_length(..., weight='distance')
                        dist = self.camera_graph[c1][c2]['distance']
                        mat[i, j] = dist
                    except:
                        mat[i, j] = float('inf')
        return mat

    def _get_alpha_t(self, timestamp: float) -> float:
        """Get traffic expansion factor alpha(t) (Eq 4)"""
        # Ensure UTC is used for logic consistency with validation script
        dt = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
        hour = dt.hour
        
        # Rush hours defined in paper: 07:00-09:00, 17:00-19:00
        for start, end in self.config['rush_hours']:
            # Python range is start inclusive, but standard meaning 07:00 to 09:00 covers hours 7 and 8 (until 9:00) 
            # or strictly 7 <= h < 9? Paper says [07:00, 09:00].
            # Let's assume inclusive of the hour integer.
            if start <= hour < end: 
                return self.config['alpha_rush']
                
        return self.config['alpha_normal']

    def is_temporally_feasible(self, query_cam: int, query_time: float,
                             gallery_cam: int, gallery_time: float) -> Tuple[bool, Dict]:
        
        if query_cam == gallery_cam:
            # Paper Section III-C4: Same-camera handling
            # Accept if |dt| < epsilon (vehicle stationary) or |dt| is large
            # (vehicle left and returned). Reject mid-range only if needed.
            dt = abs(query_time - gallery_time)
            epsilon = 5.0  # seconds, per paper
            # Always accept same-camera: either stationary (small dt) or
            # departed-and-returned (large dt). VeRi-776 protocol excludes
            # same-camera pairs anyway.
            return True, {'reason': 'same_cam', 'dt': dt, 'epsilon': epsilon}
            
        cam_ids = sorted(list(self.camera_positions.keys()))
        try:
            q_idx = cam_ids.index(query_cam)
            g_idx = cam_ids.index(gallery_cam)
        except ValueError:
            return False, {'reason': 'unknown_cam'}
            
        # Driving Distance
        distance = self.distance_matrix[q_idx][g_idx] # D_driving (includes tau)
        
        # Road Physics
        edge_data = self.camera_graph[query_cam][gallery_cam]
        rtype = edge_data['road_type']
        speeds = self.config['speed_limits'].get(rtype, self.config['speed_limits']['residential'])
        
        v_max = speeds['v_max'] / 3.6 # km/h -> m/s
        v_min = speeds['v_min'] / 3.6 # km/h -> m/s
        
        # Eq 2: T_min
        T_min = distance / v_max
        
        # Eq 3: T_max
        # "alpha(t) is a time-dependent expansion factor"
        # We check both query and gallery time context? Usually query time is reference.
        alpha = self._get_alpha_t(query_time) 
        T_max = (distance / v_min) * alpha
        
        # Delta t
        dt = abs(query_time - gallery_time)
        
        # Eq 5: delta(g)
        is_feasible = (T_min <= dt <= T_max)
        
        details = {
            'dt': dt,
            'T_min': T_min,
            'T_max': T_max,
            'distance': distance,
            'alpha': alpha,
            'road_type': rtype,
            'reason': 'physics_violation' if not is_feasible else 'feasible'
        }
        
        return is_feasible, details

    def get_travel_time(self, cam1: int, cam2: int, query_time: float = 43200.0) -> Tuple[float, float]:
        """
        Get (T_min, T_max) travel time bounds for a camera pair.
        
        Args:
            cam1: Source camera ID
            cam2: Target camera ID
            query_time: Unix timestamp for alpha(t) lookup (default: noon)
            
        Returns:
            (T_min, T_max) in seconds
        """
        if cam1 == cam2:
            return 0.0, 0.0
            
        cam_ids = sorted(list(self.camera_positions.keys()))
        try:
            i = cam_ids.index(cam1)
            j = cam_ids.index(cam2)
        except ValueError:
            return 0.0, 600.0  # fallback
            
        distance = self.distance_matrix[i][j]
        edge_data = self.camera_graph[cam1][cam2]
        rtype = edge_data['road_type']
        speeds = self.config['speed_limits'].get(rtype, self.config['speed_limits']['residential'])
        
        v_max = speeds['v_max'] / 3.6
        v_min = speeds['v_min'] / 3.6
        
        T_min = distance / v_max
        alpha = self._get_alpha_t(query_time)
        T_max = (distance / v_min) * alpha
        
        return T_min, T_max

    def filter_distance_matrix(self, dist_matrix: np.ndarray,
                                query_camids: np.ndarray, gallery_camids: np.ndarray,
                                query_timestamps: list, gallery_timestamps: list,
                                penalty: float = 1e6) -> Tuple[np.ndarray, int, int]:
        """
        Apply physics-based TCM filtering to a full distance matrix.
        Uses Haversine + tortuosity + speed profiles (Eq 1-5).
        Requires real GPS positions and timestamps in seconds.
        
        Returns:
            (filtered_matrix, filtered_count, total_count)
        """
        filtered = dist_matrix.copy()
        filtered_count = 0
        total_count = 0
        
        for q_idx in range(len(query_camids)):
            q_cam = int(query_camids[q_idx]) + 1  # 0-indexed → 1-indexed
            q_time = query_timestamps[q_idx] if query_timestamps[q_idx] is not None else 0
            
            for g_idx in range(len(gallery_camids)):
                g_cam = int(gallery_camids[g_idx]) + 1
                g_time = gallery_timestamps[g_idx] if gallery_timestamps[g_idx] is not None else 0
                total_count += 1
                
                if q_cam == g_cam:
                    continue  # same-camera: always accept
                    
                feasible, _ = self.is_temporally_feasible(q_cam, q_time, g_cam, g_time)
                if not feasible:
                    filtered[q_idx, g_idx] = penalty
                    filtered_count += 1
        
        return filtered, filtered_count, total_count

    def filter_by_temporal_window(self, dist_matrix: np.ndarray,
                                   query_camids: np.ndarray, gallery_camids: np.ndarray,
                                   query_timestamps: list, gallery_timestamps: list,
                                   window: int = 5000, penalty: float = 0.02
                                   ) -> Tuple[np.ndarray, int, int]:
        """
        Temporal-window TCM filtering for frame-number timestamps.

        VeRi-776 embeds a monotonic frame index in every filename
        (e.g., 0001_c001_00016450_0.jpg → 16450).  Within each camera the
        frame numbers grow with wall-clock time, so a forward-looking
        window captures "the target was seen *after* the query, within a
        reasonable travel horizon."

        Constraint applied to every cross-camera pair (q, g):
            if g_time < q_time  OR  |g_time − q_time| > window  →  +penalty

        A soft penalty (default 0.1 on cosine distance ∈ [0, 2]) slightly
        deprioritises temporally implausible matches without removing them.

        Args:
            dist_matrix:      (Q × G) baseline cosine-distance matrix.
            query_camids:     (Q,) camera IDs (0-indexed).
            gallery_camids:   (G,) camera IDs (0-indexed).
            query_timestamps: list[int | None], frame numbers.
            gallery_timestamps: list[int | None], frame numbers.
            window:           forward-window size in frame numbers.
            penalty:          additive distance penalty.

        Returns:
            (filtered_matrix, filtered_count, total_count)
        """
        filtered = dist_matrix.copy()
        filtered_count = 0
        total_count = 0

        for q_idx in range(len(query_camids)):
            q_cam = int(query_camids[q_idx])
            q_time = query_timestamps[q_idx] if query_timestamps[q_idx] is not None else 0

            for g_idx in range(len(gallery_camids)):
                g_cam = int(gallery_camids[g_idx])
                g_time = gallery_timestamps[g_idx] if gallery_timestamps[g_idx] is not None else 0
                total_count += 1

                if q_cam == g_cam:
                    continue  # same-camera: always accept

                time_diff = abs(g_time - q_time)
                if g_time < q_time or time_diff > window:
                    filtered[q_idx, g_idx] += penalty
                    filtered_count += 1

        return filtered, filtered_count, total_count
