"""
Regenerate all frontend plot images with updated experimental numbers.
Sources: latest logs from logs/tcm_comparison/2026-03-06_15-10-07/ and logs/ablation/2026-03-06_15-53-04/

Generates into: logs/tcm_speedup_test/
"""

import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("logs/tcm_speedup_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Actual experimental numbers (from latest logs 2026-03-06) ──────────────

BASELINE = {"mAP": 82.14, "rank1": 97.44, "rank5": 98.75, "rank10": 99.34}
TCM      = {"mAP": 82.53, "rank1": 97.38, "rank5": 98.69, "rank10": 99.40}

SEQ = {"total": 126063, "filtered": 125690, "retained": 373, "reduction": 99.70, "speedup": "338×"}
FG  = {"total": 19429562, "filtered": 17300557, "retained": 2129005, "reduction": 89.0}

ABLATION_TAU = [
    {"tau": "≈1.0", "W": 3000, "mAP": 82.5, "rank1": 97.3, "recall": 27.8, "filt": 96.59},
    {"tau": "≈1.2", "W": 5000, "mAP": 82.5, "rank1": 97.4, "recall": 32.7, "filt": 94.47, "ours": True},
    {"tau": "≈1.4", "W": 7000, "mAP": 82.5, "rank1": 97.4, "recall": 34.8, "filt": 92.37},
    {"tau": "≈1.6", "W": 10000, "mAP": 82.4, "rank1": 97.3, "recall": 36.7, "filt": 89.26},
]
ABLATION_ALPHA = [
    {"mode": "Static (α=1.0, pen=0.05)", "mAP": 82.6, "rank1": 96.8, "recall": 32.7, "filt": 94.47},
    {"mode": "Dynamic α(t) (pen=0.02)", "mAP": 82.5, "rank1": 97.4, "recall": 32.7, "filt": 94.47, "ours": True},
]

# ── Colour palette ─────────────────────────────────────────────────────────

C_BASELINE = "#3498db"
C_TCM      = "#e74c3c"
C_GREEN    = "#27ae60"
C_ORANGE   = "#f39c12"
C_PURPLE   = "#9b59b6"
C_TEAL     = "#1abc9c"
C_BG       = "#0f172a"
C_CARD     = "#1e293b"
C_TEXT     = "#e2e8f0"
C_ACCENT   = "#22d3ee"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": C_CARD,
    "figure.facecolor": C_BG,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "axes.edgecolor": "#334155",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
})

# ═══════════════════════════════════════════════════════════════════════════
# 1. performance_comparison.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_performance_comparison():
    metrics = ["mAP", "Rank-1", "Rank-5", "Rank-10"]
    bl_vals = [BASELINE["mAP"], BASELINE["rank1"], BASELINE["rank5"], BASELINE["rank10"]]
    tcm_vals = [TCM["mAP"], TCM["rank1"], TCM["rank5"], TCM["rank10"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs TCM-Enhanced Performance on VeRi-776",
                 fontsize=18, fontweight="bold", color=C_ACCENT)

    for idx, (name, bv, tv) in enumerate(zip(metrics, bl_vals, tcm_vals)):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(2)
        bars = ax.bar(x, [bv, tv], color=[C_BASELINE, C_TCM], alpha=0.85,
                       edgecolor="white", linewidth=1.2, width=0.55)
        for bar, val in zip(bars, [bv, tv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.2f}%", ha="center", va="bottom", fontweight="bold",
                    fontsize=12, color=C_TEXT)
        diff = tv - bv
        sign = "+" if diff >= 0 else ""
        ax.set_title(f"{name}  ({sign}{diff:.2f}%)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "TCM-Enhanced"])
        ax.set_ylim(min(bv, tv) - 1.5, max(bv, tv) + 1.5)
        ax.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUTPUT_DIR / "performance_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. search_space_reduction.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_search_space_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("TCM Search Space Reduction", fontsize=18, fontweight="bold", color=C_ACCENT)

    # Left: Sequential tracking
    ax = axes[0]
    labels = ["Full Gallery", "Filtered", "Retained"]
    vals   = [SEQ["total"], SEQ["filtered"], SEQ["retained"]]
    colours = [C_BASELINE, C_TCM, C_GREEN]
    bars = ax.bar(labels, vals, color=colours, alpha=0.85, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11, color=C_TEXT)
    ax.set_title(f"Sequential Tracking  ({SEQ['speedup']} speedup, {SEQ['reduction']}% eliminated)",
                 fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Number of pairs (log scale)")
    ax.grid(axis="y")

    # Right: Full gallery
    ax2 = axes[1]
    fg_labels = ["Total Pairs", "Filtered", "Retained"]
    fg_vals   = [FG["total"], FG["filtered"], FG["retained"]]
    bars2 = ax2.bar(fg_labels, fg_vals, color=colours, alpha=0.85, edgecolor="white", linewidth=1)
    for bar, val in zip(bars2, fg_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=10, color=C_TEXT)
    ax2.set_title(f"Full Gallery Evaluation  ({FG['reduction']}% reduction)",
                  fontsize=13, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylabel("Number of pairs (log scale)")
    ax2.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "search_space_reduction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ablation_study.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_study():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Ablation Study — Sensitivity Analysis (Paper Table VIII)",
                 fontsize=18, fontweight="bold", color=C_ACCENT)

    # Left: Tortuosity sweep
    ax = axes[0]
    taus = [r["tau"] for r in ABLATION_TAU]
    maps = [r["mAP"] for r in ABLATION_TAU]
    filts = [r["filt"] for r in ABLATION_TAU]
    recalls = [r["recall"] for r in ABLATION_TAU]

    x = np.arange(len(taus))
    w = 0.25
    b1 = ax.bar(x - w, maps, w, label="mAP (%)", color=C_BASELINE, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,     [r/10 + 73 for r in recalls], w, label="Recall (scaled)", color=C_GREEN, alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + w, [f/10 + 73 for f in filts], w, label="Filter Rate (scaled)", color=C_ORANGE, alpha=0.85, edgecolor="white")

    for bars, vals, fmt in [(b1, maps, "{:.1f}%"), (b2, recalls, "{:.1f}%"), (b3, filts, "{:.1f}%")]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    fmt.format(val), ha="center", va="bottom", fontsize=8, color=C_TEXT)

    # Highlight "ours"
    for i, r in enumerate(ABLATION_TAU):
        if r.get("ours"):
            ax.annotate("★ Ours", (x[i] - w, maps[i] + 0.3),
                        fontsize=10, fontweight="bold", color=C_ACCENT, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"τ{t}\n(W={r['W']})" for t, r in zip(taus, ABLATION_TAU)])
    ax.set_ylabel("mAP (%) / scaled metrics")
    ax.set_title("Temporal Window Sensitivity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(75, 84)
    ax.grid(axis="y")

    # Right: Penalty strength
    ax2 = axes[1]
    modes = [r["mode"] for r in ABLATION_ALPHA]
    alpha_maps = [r["mAP"] for r in ABLATION_ALPHA]
    alpha_r1   = [r["rank1"] for r in ABLATION_ALPHA]

    x2 = np.arange(len(modes))
    w2 = 0.30
    b4 = ax2.bar(x2 - w2/2, alpha_maps, w2, label="mAP (%)", color=C_BASELINE, alpha=0.85, edgecolor="white")
    b5 = ax2.bar(x2 + w2/2, alpha_r1,  w2, label="Rank-1 (%)", color=C_TCM, alpha=0.85, edgecolor="white")
    for bars, vals in [(b4, alpha_maps), (b5, alpha_r1)]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color=C_TEXT)
    for i, r in enumerate(ABLATION_ALPHA):
        if r.get("ours"):
            ax2.annotate("★ Ours", (x2[i], max(r["mAP"], r["rank1"]) + 0.4),
                         fontsize=10, fontweight="bold", color=C_ACCENT, ha="center")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([m.replace(" (pen", "\n(pen") for m in modes], fontsize=9)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Penalty Strength Sensitivity", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(95, 98.5)
    ax2.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "ablation_study.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. filtering_pipeline.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_filtering_pipeline():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    fig.suptitle("TCM Filtering Pipeline — Sequential Tracking",
                 fontsize=18, fontweight="bold", color=C_ACCENT)

    steps = [
        ("Full Gallery", f"{SEQ['total']:,} candidate pairs", C_BASELINE),
        ("Speed-Limit Check (Tmin)", "Δt < Tmin → vehicle too fast → impossible", C_ORANGE),
        ("Congestion Window (Tmax·α)", "Δt > Tmax·α(t) → vehicle too slow → implausible", C_ORANGE),
        (f"Valid Candidates: {SEQ['retained']:,}", f"{SEQ['speedup']} speedup  •  {SEQ['reduction']}% eliminated", C_GREEN),
    ]
    arrows = [
        f"↓  Tmin filter  (too fast = physically impossible)",
        f"↓  Tmax filter  (too slow = implausible match)",
        f"↓  {SEQ['reduction']}% eliminated",
    ]

    y = 0.92
    dy_box = 0.17
    dy_arrow = 0.05

    for i, (title, desc, color) in enumerate(steps):
        # Draw box
        box_h = 0.10
        rect = plt.Rectangle((0.15, y - box_h), 0.7, box_h, transform=ax.transAxes,
                              facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.18, y - box_h / 2 + 0.015, title, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="center", color=C_TEXT)
        ax.text(0.18, y - box_h / 2 - 0.02, desc, transform=ax.transAxes,
                fontsize=10, va="center", color="#94a3b8")

        # Draw arrow between steps
        if i < len(arrows):
            arrow_y = y - box_h - dy_arrow
            ax.annotate("", xy=(0.5, arrow_y - 0.02), xytext=(0.5, arrow_y + 0.02),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2))
            ax.text(0.52, arrow_y, arrows[i], transform=ax.transAxes,
                    fontsize=9, va="center", color="#94a3b8", style="italic")

        y -= dy_box

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "filtering_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. temporal_constraints.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_temporal_constraints():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Temporal Constraint Model — Road-Type Speed Profiles",
                 fontsize=18, fontweight="bold", color=C_ACCENT)

    # Left: Speed profiles by road type
    ax = axes[0]
    road_types = ["Highway\n(>5km)", "Arterial\n(1–5km)", "Residential\n(200m–1km)", "Parking\n(<200m)"]
    vmax = [120, 60, 40, 20]
    vavg = [80,  35, 20, 10]
    vmin = [40,  10,  5,  1]

    x = np.arange(len(road_types))
    w = 0.22
    b1 = ax.bar(x - w, vmax, w, label="Vmax (km/h)", color=C_TCM,     alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,     vavg, w, label="Vavg (km/h)", color=C_ORANGE,  alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + w, vmin, w, label="Vmin (km/h)", color=C_BASELINE, alpha=0.85, edgecolor="white")

    for bars, vals in [(b1, vmax), (b2, vavg), (b3, vmin)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=9, color=C_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(road_types)
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Profiles by Road Type", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y")

    # Right: Valid time windows vs distance
    ax2 = axes[1]
    distances = np.linspace(100, 1200, 50)
    # Residential range ~200m–1km: vmax=40, vmin=5 km/h → m/s: 11.11, 1.39
    t_min = distances / 11.11   # fastest (vmax residential)
    t_max = distances / 1.39    # slowest (vmin residential)
    # Rush-hour extension
    t_max_rush = t_max * 1.5

    ax2.fill_between(distances, t_min, t_max, alpha=0.25, color=C_GREEN, label="Normal window")
    ax2.fill_between(distances, t_max, t_max_rush, alpha=0.15, color=C_ORANGE, label="Rush-hour extension")
    ax2.plot(distances, t_min, "-", color=C_GREEN, lw=2, label="Tmin (vmax)")
    ax2.plot(distances, t_max, "-", color=C_TCM, lw=2, label="Tmax (vmin)")
    ax2.plot(distances, t_max_rush, "--", color=C_ORANGE, lw=2, label="Tmax·α(rush)")

    ax2.set_xlabel("Camera Distance (m)")
    ax2.set_ylabel("Travel Time (s)")
    ax2.set_title("Valid Time Windows vs Camera Distance", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "temporal_constraints.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. improvement_summary.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_improvement_summary():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    fig.suptitle("Executive Summary — Key Results", fontsize=20, fontweight="bold", color=C_ACCENT)

    results = [
        (SEQ["speedup"],                    "Sequential\nSpeedup",       C_GREEN),
        (f"{SEQ['reduction']}%",            "Search Space\nEliminated",  C_TEAL),
        (f"+{TCM['mAP']-BASELINE['mAP']:.2f}%",  "mAP\nImprovement",   C_ORANGE),
        (f"{FG['reduction']}%",             "Full Gallery\nReduction",   C_PURPLE),
        ("O(1)",                            "Per-Pair\nOverhead",        C_BASELINE),
    ]

    n = len(results)
    for i, (value, label, color) in enumerate(results):
        cx = (i + 0.5) / n
        # Card background
        card_w, card_h = 0.16, 0.55
        rect = plt.Rectangle((cx - card_w/2, 0.22), card_w, card_h,
                              transform=ax.transAxes, facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=2, clip_on=False)
        ax.add_patch(rect)
        # Value
        ax.text(cx, 0.58, str(value), transform=ax.transAxes,
                fontsize=26, fontweight="bold", ha="center", va="center", color=color)
        # Label
        ax.text(cx, 0.33, label, transform=ax.transAxes,
                fontsize=11, ha="center", va="center", color=C_TEXT, linespacing=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = OUTPUT_DIR / "improvement_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Regenerating frontend plot images with updated numbers…\n")
    plot_performance_comparison()
    plot_search_space_reduction()
    plot_ablation_study()
    plot_filtering_pipeline()
    plot_temporal_constraints()
    plot_improvement_summary()
    print(f"\n✅  All 6 plots regenerated in {OUTPUT_DIR}/")
