#!/usr/bin/env python
"""
Generate a clean, publication-quality Camera Network Graph
for the VeRi-776 Temporal Consistency Module paper/presentation.

Version 2 – spring-layout for clarity, cleaner typography.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

# ── Edges with realistic min/max travel times (seconds) ──────────────────
EDGES = [
    # Cluster 1 – Highway Entrance
    (1,  2,  20,  60),
    (1,  13, 15,  45),
    (2,  13, 25,  70),
    # Cluster 2 – Central Intersection
    (3,  4,   5,  15),
    (3,  9,  10,  30),
    (3,  10, 20,  50),
    (4,  9,   8,  25),
    (4,  10, 15,  40),
    (6,  10,  5,  20),
    (6,  11,  5,  15),
    (10, 11,  3,  12),
    (9,  8,   8,  25),
    # Cluster 3 – Parking / Residential
    (14, 15,  5,  20),
    (14, 16, 15,  40),
    (15, 16, 12,  35),
    (15, 19, 20,  55),
    (16, 19, 18,  50),
    # Cluster 4 – Secondary Roads
    (5,  7,   8,  30),
    (5,  8,  25,  70),
    (7,  8,  20,  60),
    # Cluster 5 – Outer Area
    (12, 13, 30,  90),
    (12, 17, 20,  60),
    (17, 18, 10,  35),
    (17, 19, 15,  45),
    # Long-distance connections
    (2,  6,  40, 120),
    (11, 2,  35, 100),
    (1,  12, 60, 180),
    (13, 17, 50, 150),
    (8,  16, 70, 200),
]

# ── Cluster definitions ──────────────────────────────────────────────────
CLUSTERS = {
    "Highway Entrance":      [1, 2, 13],
    "Central Intersection":  [3, 4, 6, 9, 10, 11],
    "Parking / Residential": [14, 15, 16, 19],
    "Secondary Roads":       [5, 7, 8],
    "Outer Area":            [12, 17, 18, 20],
}

CLUSTER_COLOURS = {
    "Highway Entrance":      "#2980b9",
    "Central Intersection":  "#c0392b",
    "Parking / Residential": "#27ae60",
    "Secondary Roads":       "#e67e22",
    "Outer Area":            "#8e44ad",
}

# ── Helper: edge colour by min travel time ────────────────────────────────
def edge_colour(min_t):
    if min_t < 20:   return "#27ae60"   # green  – fast
    if min_t <= 50:  return "#f39c12"   # amber  – medium
    return "#e74c3c"                     # red    – slow


def main():
    np.random.seed(42)

    # ── Build graph ───────────────────────────────────────────────────────
    G = nx.Graph()
    for cluster_cams in CLUSTERS.values():
        for c in cluster_cams:
            G.add_node(c)

    for c1, c2, mn, mx in EDGES:
        G.add_edge(c1, c2, min_time=mn, max_time=mx)

    # Use Kamada-Kawai layout – tends to separate clusters nicely
    pos = nx.kamada_kawai_layout(G, weight="min_time")

    # ── Node → cluster colour map ────────────────────────────────────────
    cam_to_colour = {}
    cam_to_cluster = {}
    for cname, members in CLUSTERS.items():
        for cam in members:
            cam_to_colour[cam] = CLUSTER_COLOURS[cname]
            cam_to_cluster[cam] = cname

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # light grid
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#dee2e6")
    ax.set_axisbelow(True)

    # ── Draw cluster background patches (convex hulls) ────────────────────
    from scipy.spatial import ConvexHull

    for cname, members in CLUSTERS.items():
        if len(members) < 3:
            continue   # can't draw hull for < 3 points
        pts = np.array([pos[c] for c in members])
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            # close the polygon
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            # expand slightly
            centroid = pts.mean(axis=0)
            hull_pts_exp = centroid + 1.25 * (hull_pts - centroid)
            ax.fill(hull_pts_exp[:, 0], hull_pts_exp[:, 1],
                    color=CLUSTER_COLOURS[cname], alpha=0.06, zorder=0)
            ax.plot(hull_pts_exp[:, 0], hull_pts_exp[:, 1],
                    color=CLUSTER_COLOURS[cname], alpha=0.20,
                    linewidth=1.2, linestyle="--", zorder=0)
        except Exception:
            pass

    # ── Draw edges ────────────────────────────────────────────────────────
    for (u, v, data) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        col = edge_colour(data["min_time"])
        lw  = 1.8 if data["min_time"] < 20 else 1.2
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=lw,
                alpha=0.45, zorder=1, solid_capstyle="round")

    # ── Draw edge labels (midpoint) ───────────────────────────────────────
    drawn_labels = []  # track positions to avoid overlap
    for (u, v, data) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2

        # quick nudge to avoid dense overlap
        for (ox, oy) in drawn_labels:
            if abs(mx - ox) < 0.04 and abs(my - oy) < 0.03:
                mx += 0.015
                my += 0.012
                break

        label = f"{data['min_time']}–{data['max_time']}s"
        ax.text(mx, my, label, fontsize=5.5, color="#555555",
                ha="center", va="center", fontfamily="sans-serif",
                zorder=2,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.85))
        drawn_labels.append((mx, my))

    # ── Draw nodes ────────────────────────────────────────────────────────
    node_size = 420
    for n in G.nodes():
        x, y = pos[n]
        col = cam_to_colour.get(n, "#95a5a6")

        # soft glow
        glow = plt.Circle((x, y), 0.062, color=col, alpha=0.12, zorder=3)
        ax.add_patch(glow)

        # node body
        ax.scatter(x, y, s=node_size, c=col, edgecolors="white",
                   linewidths=2.0, zorder=4)

        # label
        ax.text(x, y, f"C{n}", fontsize=8, fontweight="bold",
                color="white", ha="center", va="center", zorder=5,
                fontfamily="sans-serif",
                path_effects=[pe.withStroke(linewidth=1.5, foreground=col)])

    # ── Legends ───────────────────────────────────────────────────────────
    cluster_handles = [
        mpatches.Patch(facecolor=col, edgecolor="white", linewidth=0.8,
                       label=name)
        for name, col in CLUSTER_COLOURS.items()
    ]
    edge_handles = [
        plt.Line2D([0], [0], color="#27ae60", lw=2.5, label="Fast (< 20 s)"),
        plt.Line2D([0], [0], color="#f39c12", lw=2.5, label="Medium (20–50 s)"),
        plt.Line2D([0], [0], color="#e74c3c", lw=2.5, label="Slow (> 50 s)"),
    ]

    leg1 = ax.legend(handles=cluster_handles, title="Camera Clusters",
                     loc="upper left", fontsize=8, title_fontsize=9,
                     framealpha=0.95, edgecolor="#dee2e6", fancybox=True)
    ax.add_artist(leg1)
    ax.legend(handles=edge_handles, title="Travel-Time Band",
              loc="lower left", fontsize=8, title_fontsize=9,
              framealpha=0.95, edgecolor="#dee2e6", fancybox=True)

    # ── Stats box ─────────────────────────────────────────────────────────
    info = (f"Cameras: {G.number_of_nodes()}   ·   "
            f"Links: {G.number_of_edges()}   ·   "
            f"Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}")
    ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="#555555", fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#dee2e6",
                      alpha=0.95))

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title("VeRi-776 Camera Network Graph",
                 fontsize=16, fontweight="bold", fontfamily="sans-serif",
                 pad=16, color="#212529")
    ax.text(0.5, 1.01,
            "Simulated urban surveillance topology  ·  20 cameras  ·  "
            "edge labels = min–max travel time (seconds)",
            transform=ax.transAxes, fontsize=8.5, ha="center", va="bottom",
            color="#6c757d", fontfamily="sans-serif")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    out_path = "docs/figures/camera_network_clean.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    print(f"✓  Saved  →  {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
