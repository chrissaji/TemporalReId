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
    topk_indices = sorted_idx[:topk]

    top_gallery = []
    for i in topk_indices:
        correct = gallery_pids[i] == q_pid
        same_cam = gallery_camids[i] == q_camid
        top_gallery.append({
            'img_path': gallery_img_paths[i],
            'pid': gallery_pids[i],
            'camid': gallery_camids[i],
            'score': dists[i],
            'correct': correct,
            'same_cam': same_cam
        })

    # Wide (horizontal)
    visualize_reid(query_img_paths[query_idx], q_pid, q_camid, top_gallery, output_path, layout='horizontal')
    # Stacked (vertical)
    out2 = output_path.replace('.png', '_stacked.png')
    visualize_reid(query_img_paths[query_idx], q_pid, q_camid, top_gallery, out2, layout='vertical')
    print(f"Saved visualizations to {output_path} and {out2}")
