import matplotlib.pyplot as plt
import numpy as np
import time
import os
import random

def visualize_tcm_logic():
    print("\n" + "="*60)
    print("TCM VISUAL DEMO: Physics-Based Filtering Logic")
    print("="*60)
    
    # Scenario Setup
    query_time = 100  # seconds
    cam_a = 1
    cam_b = 2
    min_travel = 20   # seconds
    max_travel = 60   # seconds
    
    valid_start = query_time + min_travel
    valid_end = query_time + max_travel
    
    print(f"\nScenario: Vehicle seen at Camera {cam_a} at t={query_time}s")
    print(f"Target:   Checking Camera {cam_b}")
    print(f"Physics:  Travel time is {min_travel}-{max_travel}s")
    print(f"Window:   Valid matches must be between t={valid_start}s and t={valid_end}s")
    print("-" * 60)
    
    # Generate synthetic gallery data
    num_samples = 50
    gallery_times = np.sort(np.random.randint(query_time - 20, query_time + 100, num_samples))
    
    accepted = []
    rejected = []
    
    # ASCII Animation
    print("\nProcessing Gallery Candidates:")
    print("Time (s) | Status | Visualization")
    print("-" * 45)
    
    for t in gallery_times:
        is_valid = valid_start <= t <= valid_end
        
        marker = "✅ MATCH" if is_valid else "❌ REJECT"
        bar = " " * int((t - (query_time-20))/2) + ("🟢" if is_valid else "🔴")
        
        if is_valid:
            accepted.append(t)
        else:
            rejected.append(t)
            
        # Slow down slightly for "demo" effect
        if random.random() < 0.1:
            time.sleep(0.05)
            
        # Only print some lines to avoid cluttering terminal too much, or print all if small
        if num_samples <= 20 or random.random() < 0.3:
             print(f"t={t:3d}s  | {marker} | {bar}")

    print("-" * 45)
    print(f"\nResults:")
    print(f"Total Candidates: {num_samples}")
    print(f"Filtered Out:     {len(rejected)} ({(len(rejected)/num_samples)*100:.1f}%)")
    print(f"Kept for Re-ID:   {len(accepted)}")
    
    # --- Matplotlib Visualization ---
    print("\nGenerating visual plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot timeline
    plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    
    # Plot Query
    plt.scatter([query_time], [1], color='blue', s=200, label='Query (Cam A)', zorder=10)
    plt.text(query_time, 1.1, f'Query\nt={query_time}s', ha='center', color='blue')
    
    # Highlight Valid Window
    plt.axvspan(valid_start, valid_end, color='green', alpha=0.2, label='Feasible Window')
    plt.text((valid_start+valid_end)/2, 0.8, f'Travel Time\n{min_travel}-{max_travel}s', ha='center', color='green')
    
    # Plot Gallery Points
    plt.scatter(rejected, [1]*len(rejected), color='red', marker='x', s=100, label='Rejected (Impossible)', alpha=0.6)
    plt.scatter(accepted, [1]*len(accepted), color='green', marker='o', s=100, label='Accepted (Feasible)', zorder=5)
    
    # Formatting
    plt.title(f'TCM Logic Visualization: Camera {cam_a} $\\to$ Camera {cam_b}', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.yticks([])
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save
    output_path = 'tcm_logic_visualization.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Visualization saved to: {os.path.abspath(output_path)}")
    print("Open this image to show the concept visually!")

if __name__ == "__main__":
    visualize_tcm_logic()
