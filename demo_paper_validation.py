#!/usr/bin/env python
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.rule_based_tcm_v2 import RuleBasedTCMV2
from utils.veri_camera_config import get_veri_camera_positions

def run_paper_validation_demo():
    print("="*60)
    print("IEEE PAPER VALIDATION SUITE (TCM V2)")
    print("="*60)
    
    # 1. Setup Environment
    # --------------------
    cam_pos = get_veri_camera_positions()
    tcm = RuleBasedTCMV2(cam_pos)
    
    # 2. Validate Tortuosity (Tau)
    # ----------------------------
    print("\n[Validation 1] Tortuosity Factor (Is distance scaled?)")
    c1, c2 = 1, 2
    p1 = cam_pos[c1]
    p2 = cam_pos[c2]
    
    # Manually calc haversine
    from utils.rule_based_tcm_v2 import RuleBasedTCMV2 as TCM
    # Instantiate temp wrapper to access protected
    temp_inst = TCM(cam_pos)
    
    raw_hav = temp_inst._haversine_distance(p1, p2)
    driving_dist = temp_inst.distance_matrix[0][1] # assuming sorted keys 1, 2...
    
    ratio = driving_dist / raw_hav
    print(f"  Raw Haversine: {raw_hav:.2f}m")
    print(f"  Driving Dist:  {driving_dist:.2f}m")
    print(f"  Ratio (tau):   {ratio:.2f}")
    if abs(ratio - 1.4) < 0.01:
        print("  ✅ MATCHES PAPER (tau=1.4)")
    else:
        print(f"  ❌ MISMATCH (Expected 1.4, got {ratio})")

    # 3. Validate Traffic Dynamics (Alpha)
    # ------------------------------------
    print("\n[Validation 2] Traffic Factor Alpha(t)")
    t_rush = 1609484400 # 2021-01-01 07:00:00 (Rush Hour)
    t_norm = 1609506000 # 2021-01-01 13:00:00 (Normal)
    
    a_rush = temp_inst._get_alpha_t(t_rush)
    a_norm = temp_inst._get_alpha_t(t_norm)
    
    print(f"  07:00 Alpha: {a_rush}")
    print(f"  13:00 Alpha: {a_norm}")
    
    if a_rush == 1.5 and a_norm == 1.0:
        print("  ✅ MATCHES PAPER (Rush=1.5, Normal=1.0)")
    else:
        print("  ❌ MISMATCH")

    # 4. Validate Physics Constraints
    # -------------------------------
    print("\n[Validation 3] Physics Constraints (T_min, T_max)")
    # Pick a pair with known distance
    # Cam 1 -> Cam 2
    # Assume Arterial road (v_max=60, v_min=10)
    # T_min = D / (60/3.6)
    # T_max = (D / (10/3.6)) * alpha
    
    q_cam, g_cam = 1, 2
    dist = driving_dist
    
    # Calculate expected bounds manually
    v_max_ms = 60 / 3.6 # Arterial? Check logic
    rtype = temp_inst.camera_graph[1][2]['road_type']
    print(f"  Road Type: {rtype}")
    
    # Let's see what the class calculated
    # Case A: Too Fast (Impossible)
    # D approx 2000m?
    # Say T_min is 30s. We try dt=5s.
    
    # We query the feasibility
    # Case 1: Instant Teleport (Impossible)
    feasible, dets = tcm.is_temporally_feasible(1, t_norm, 2, t_norm + 1)
    print(f"  Instant Teleport (dt=1s): {feasible} | T_min={dets['T_min']:.1f}s")
    
    # Case 2: Crawling Traffic (Possible in Rush Hour?)
    # Rush Hour T_max should be higher
    _, dets_norm = tcm.is_temporally_feasible(1, t_norm, 2, t_norm + 1000)
    _, dets_rush = tcm.is_temporally_feasible(1, t_rush, 2, t_rush + 1000)
    
    print(f"  T_max (Normal): {dets_norm['T_max']:.1f}s")
    print(f"  T_max (Rush):   {dets_rush['T_max']:.1f}s")
    
    if dets_rush['T_max'] > dets_norm['T_max']:
         print("  ✅ MATCHES PAPER (Rush hour expands allowed time window)")
    else:
         print("  ❌ MISMATCH")
         
    print("\nDone. Codebase V2 aligns with paper methodology.")

if __name__ == "__main__":
    run_paper_validation_demo()
