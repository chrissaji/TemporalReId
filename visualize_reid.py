import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Example usage: python visualize_reid.py <query_img_index> <output_image_path>
# This script assumes you have already run inference and have access to the query and gallery features, pids, camids, and image paths.

def visualize_reid(query_img_path, top_gallery_paths, scores, output_path):
    fig, axes = plt.subplots(1, len(top_gallery_paths) + 1, figsize=(15, 5))
    # Show query image
    axes[0].imshow(Image.open(query_img_path))
    axes[0].set_title('Query')
    axes[0].axis('off')
    # Show top gallery matches
    for i, (img_path, score) in enumerate(zip(top_gallery_paths, scores)):
        axes[i+1].imshow(Image.open(img_path))
        axes[i+1].set_title(f'Rank {i+1}\nScore: {score:.2f}')
        axes[i+1].axis('off')
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
    # Usage: python visualize_reid.py <query_img_index> <output_image_path> [output_dir]
    if len(sys.argv) < 3:
        print("Usage: python visualize_reid.py <query_img_index> <output_image_path> [output_dir]")
        sys.exit(1)
    query_idx = int(sys.argv[1])
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    else:
        output_dir = "./logs"  # default output dir, change if needed

    # Load distmat, img_path_list, num_query
    distmat_path = os.path.join(output_dir, 'distmat.npy')
    img_path_list_path = os.path.join(output_dir, 'img_path_list.npy')
    num_query_path = os.path.join(output_dir, 'num_query.npy')

    if not (os.path.exists(distmat_path) and os.path.exists(img_path_list_path) and os.path.exists(num_query_path)):
        print(f"Missing distmat/img_path_list/num_query in {output_dir}. Run test.py first.")
        sys.exit(1)

    distmat = np.load(distmat_path)
    img_path_list = np.load(img_path_list_path, allow_pickle=True)
    num_query = int(np.load(num_query_path)[0])

    # Split image paths into query and gallery
    query_img_paths = img_path_list[:num_query]
    gallery_img_paths = img_path_list[num_query:]

    # For the selected query, get distances to all gallery images
    dists = distmat[query_idx]
    topk = 5
    topk_indices = np.argsort(dists)[:topk]
    top_gallery_paths = [gallery_img_paths[i] for i in topk_indices]
    top_scores = dists[topk_indices]

    visualize_reid(query_img_paths[query_idx], top_gallery_paths, top_scores, output_path)
    print(f"Saved visualization to {output_path}")
