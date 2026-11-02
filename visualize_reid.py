def visualize_reid_cross_camera(
    query_img_path, query_pid, query_camid,
    gallery_list,
    output_path,
    layout='horizontal',
    tile_size=256
):
    """
    Visualize only gallery matches from different cameras than the query.
    gallery_list: list of dicts with keys: img_path, pid, camid, score, correct, same_cam
    """
    import matplotlib.patches as patches
    # Filter gallery to only different cameras
    cross_cam_gallery = [g for g in gallery_list if g['camid'] != query_camid]
    n = len(cross_cam_gallery)
    if n == 0:
        print("No cross-camera matches to visualize.")
        return
    if layout == 'horizontal':
        fig, axes = plt.subplots(1, n + 1, figsize=(tile_size/50*(n+1), tile_size/50))
    else:
        fig, axes = plt.subplots(n + 1, 1, figsize=(tile_size/50, tile_size/50*(n+1)))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    # Query image
    img = Image.open(query_img_path).resize((tile_size, tile_size))
    axes[0].imshow(img)
    axes[0].set_title(f'QUERY\npid={query_pid} cam={query_camid}', fontsize=10)
    axes[0].axis('off')
    rect = patches.Rectangle((0,0),tile_size,tile_size,linewidth=6,edgecolor='black',facecolor='none')
    axes[0].add_patch(rect)
    # Gallery images (cross-camera only)
    for i, g in enumerate(cross_cam_gallery):
        ax = axes[i+1] if layout=='horizontal' else axes[i+1]
        img = Image.open(g['img_path']).resize((tile_size, tile_size))
        ax.imshow(img)
        color = 'green' if g['correct'] else 'red'
        rect = patches.Rectangle((0,0),tile_size,tile_size,linewidth=6,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
        label = f"Rank {i+1}\npid={g['pid']} cam={g['camid']}\n"
        label += f"{'sim' if g.get('sim',None) is not None else 'dist'}: {g['score']:.2f}"
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Example usage: python visualize_reid.py <query_img_index> <output_image_path>
# This script assumes you have already run inference and have access to the query and gallery features, pids, camids, and image paths.

def visualize_reid(
    query_img_path, query_pid, query_camid,
    top_gallery,
    output_path,
    layout='horizontal',
    tile_size=256
):
    """
    top_gallery: list of dicts with keys: img_path, pid, camid, score, correct, same_cam
    """
    import matplotlib.patches as patches
    n = len(top_gallery)
    if layout == 'horizontal':
        fig, axes = plt.subplots(1, n + 1, figsize=(tile_size/50*(n+1), tile_size/50))
    else:
        fig, axes = plt.subplots(n + 1, 1, figsize=(tile_size/50, tile_size/50*(n+1)))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    # Query image
    img = Image.open(query_img_path).resize((tile_size, tile_size))
    axes[0].imshow(img)
    axes[0].set_title(f'QUERY\npid={query_pid} cam={query_camid}', fontsize=10)
    axes[0].axis('off')
    # Add border to query (black)
    rect = patches.Rectangle((0,0),tile_size,tile_size,linewidth=6,edgecolor='black',facecolor='none')
    axes[0].add_patch(rect)
    # Gallery images
    for i, g in enumerate(top_gallery):
        ax = axes[i+1] if layout=='horizontal' else axes[i+1]
        img = Image.open(g['img_path']).resize((tile_size, tile_size))
        ax.imshow(img)
        # Border color
        if g['correct']:
            color = 'green'
        elif g['same_cam']:
            color = 'yellow'
        else:
            color = 'red'
        rect = patches.Rectangle((0,0),tile_size,tile_size,linewidth=6,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
        # Title and label
        label = f"Rank {i+1}\npid={g['pid']} cam={g['camid']}\n"
        label += f"{'sim' if g.get('sim',None) is not None else 'dist'}: {g['score']:.2f}"
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# The following is a template for how you would use this in your pipeline:
# 1. Extract features for all query and gallery images (already done in test.py/processor.py)
# 2. Select a query index (e.g., 0 for the first query image)
# 3. Compute distances and get top-N gallery matches
# 4. Call visualize_reid with the paths and scores

# You will need to adapt this template to your actual feature extraction and data loading pipeline.

if __name__ == "__main__":
    import sys
    # Usage: python visualize_reid.py <query_img_index> <output_image_path> [output_dir] [--no-same-cam]
    if len(sys.argv) < 3:
        print("Usage: python visualize_reid.py <query_img_index> <output_image_path> [output_dir] [--no-same-cam]")
        sys.exit(1)
    query_idx = int(sys.argv[1])
    output_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else "./logs"
    no_same_cam = '--no-same-cam' in sys.argv

    # Load distmat, img_path_list, num_query, pids, camids
    distmat_path = os.path.join(output_dir, 'distmat.npy')
    img_path_list_path = os.path.join(output_dir, 'img_path_list.npy')
    num_query_path = os.path.join(output_dir, 'num_query.npy')
    pids_path = os.path.join(output_dir, 'pids.npy')
    camids_path = os.path.join(output_dir, 'camids.npy')

    if not (os.path.exists(distmat_path) and os.path.exists(img_path_list_path) and os.path.exists(num_query_path)):
        print(f"Missing distmat/img_path_list/num_query in {output_dir}. Run test.py first.")
        sys.exit(1)

    distmat = np.load(distmat_path)
    img_path_list = np.load(img_path_list_path, allow_pickle=True)
    num_query = int(np.load(num_query_path)[0])
    # Try to load pids/camids, else fallback to default
    if os.path.exists(pids_path) and os.path.exists(camids_path):
        pids = np.load(pids_path)
        camids = np.load(camids_path)
    else:
        # fallback: try to extract from metrics if available
        print("Warning: pids/camids.npy not found, using dummy values.")
        pids = np.zeros(len(img_path_list), dtype=int)
        camids = np.zeros(len(img_path_list), dtype=int)

    query_img_paths = img_path_list[:num_query]
    gallery_img_paths = img_path_list[num_query:]
    query_pids = pids[:num_query]
    gallery_pids = pids[num_query:]
    query_camids = camids[:num_query]
    gallery_camids = camids[num_query:]

    # For the selected query, get distances to all gallery images
    dists = distmat[query_idx]
    q_pid = query_pids[query_idx]
    q_camid = query_camids[query_idx]

    # Optionally filter out same-camera gallery
    valid = np.ones(len(gallery_img_paths), dtype=bool)
    if no_same_cam:
        valid = gallery_camids != q_camid
    # Sort by distance
    sorted_idx = np.argsort(dists)
    sorted_idx = [i for i in sorted_idx if valid[i]]
    topk = 10
    # First 10
    topk_indices_1 = sorted_idx[:topk]
    # Next 10
    topk_indices_2 = sorted_idx[topk:topk*2]

    def build_top_gallery(indices):
        gallery = []
        for i in indices:
            correct = gallery_pids[i] == q_pid
            same_cam = gallery_camids[i] == q_camid
            gallery.append({
                'img_path': gallery_img_paths[i],
                'pid': gallery_pids[i],
                'camid': gallery_camids[i],
                'score': dists[i],
                'correct': correct,
                'same_cam': same_cam
            })
        return gallery

    top_gallery_1 = build_top_gallery(topk_indices_1)
    top_gallery_2 = build_top_gallery(topk_indices_2)

    # First 10 gallery images
    visualize_reid(query_img_paths[query_idx], q_pid, q_camid, top_gallery_1, output_path, layout='horizontal')
    # Next 10 gallery images
    output_path2 = output_path.replace('.png', '2.png')
    visualize_reid(query_img_paths[query_idx], q_pid, q_camid, top_gallery_2, output_path2, layout='horizontal')

    # Visualize only cross-camera matches (at most one per unique camera, up to 10)
    cross_camera_path = output_path.replace('.png', '_cross_camera.png')
    # Find all unique cameras in the gallery (excluding query camera)
    # These are candidate cameras where the query vehicle could potentially appear
    all_gallery_cams = set([cam for cam in gallery_camids if cam != q_camid])
    seen_cams = set()
    cross_cam_indices = []
    for i in sorted_idx:
        cam = gallery_camids[i]
        if cam != q_camid and cam not in seen_cams:
            cross_cam_indices.append(i)
            seen_cams.add(cam)
        if len(cross_cam_indices) == 10:
            break
    top_cross_cam_gallery = build_top_gallery(cross_cam_indices)
    visualize_reid_cross_camera(query_img_paths[query_idx], q_pid, q_camid, top_cross_cam_gallery, cross_camera_path, layout='horizontal')
    # failed_cams are candidate cameras where the query vehicle could potentially appear but was not matched in the top cross-camera results
    failed_cams = all_gallery_cams - seen_cams
    print(f"Saved visualizations to {output_path}, {output_path2}, and {cross_camera_path} (showing {len(top_cross_cam_gallery)} unique cross-camera matches)")
    print(f"Candidate cameras not matched in top cross-camera results: {len(failed_cams)} -> {sorted(list(failed_cams)) if failed_cams else 'None'}")

    # Visualize failed cameras: for each failed camera, show the closest match from that camera where the same vehicle (pid) is present
    if failed_cams:
        failed_indices = []
        for cam in failed_cams:
            # Find all gallery indices for this camera and same pid as query
            cam_indices = [i for i in sorted_idx if gallery_camids[i] == cam and gallery_pids[i] == q_pid]
            if cam_indices:
                # Take the closest (lowest distance)
                failed_indices.append(cam_indices[0])
        if failed_indices:
            failed_gallery = build_top_gallery(failed_indices)
            failed_cams_path = output_path.replace('.png', '_failed_cams.png')
            visualize_reid_cross_camera(query_img_paths[query_idx], q_pid, q_camid, failed_gallery, failed_cams_path, layout='horizontal')
            print(f"Saved failed camera visualization to {failed_cams_path} (showing {len(failed_gallery)} failed cameras with correct pid)")
        else:
            print("No failed cameras where the same vehicle (pid) is present but not identified.")
