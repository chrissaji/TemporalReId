"""
Re-ID Visualization Frontend Server
Simple HTTP server that serves the frontend and provides API endpoints
for browsing query images and finding cross-camera matches.

Usage:
    python frontend/server.py

Then open http://localhost:8765 in your browser.
"""

import http.server
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = PROJECT_ROOT / "data" / "VeRi" / "image_query"
GALLERY_DIR = PROJECT_ROOT / "data" / "VeRi" / "image_test"
TRAIN_DIR = PROJECT_ROOT / "data" / "VeRi" / "image_train"
LOGS_DIR = PROJECT_ROOT / "logs" / "tcm_speedup_test"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

PORT = 8765

# Parse VeRi filename: {vehicleID}_c{cameraID}_{frameID}_{index}.jpg
FILENAME_RE = re.compile(r"^(\d+)_c(\d+)_(\d+)_(\d+)\.jpg$")


def parse_veri_filename(fname):
    """Parse a VeRi-776 filename into components."""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    return {
        "vehicle_id": int(m.group(1)),
        "camera_id": int(m.group(2)),
        "frame_id": int(m.group(3)),
        "index": int(m.group(4)),
        "filename": fname,
    }


def build_vehicle_index():
    """Build an index of vehicles from query images."""
    index = {}  # vehicle_id -> list of image info dicts
    if not QUERY_DIR.exists():
        return index
    for fname in sorted(os.listdir(QUERY_DIR)):
        info = parse_veri_filename(fname)
        if info:
            vid = info["vehicle_id"]
            if vid not in index:
                index[vid] = []
            index[vid].append(info)
    return index


def find_gallery_matches(vehicle_id):
    """Find gallery images for the same vehicle ID (simulating Re-ID matches)."""
    matches = []
    # Check both gallery and query dirs for cross-camera images
    for search_dir, label in [(GALLERY_DIR, "gallery"), (QUERY_DIR, "query")]:
        if not search_dir.exists():
            continue
        for fname in sorted(os.listdir(search_dir)):
            info = parse_veri_filename(fname)
            if info and info["vehicle_id"] == vehicle_id:
                info["source"] = label
                info["path"] = f"/images/{label}/{fname}"
                matches.append(info)
    return matches


# Pre-build index
print("Building vehicle index...")
VEHICLE_INDEX = build_vehicle_index()
VEHICLE_IDS = sorted(VEHICLE_INDEX.keys())
print(f"Found {len(VEHICLE_IDS)} unique vehicles with {sum(len(v) for v in VEHICLE_INDEX.values())} query images")


class ReIDHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, filepath, content_type=None):
        if not filepath.exists():
            self.send_error(404, f"File not found: {filepath}")
            return
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(filepath))
            content_type = content_type or "application/octet-stream"
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)

        # API endpoints
        if path == "/api/vehicles":
            # List all vehicle IDs with counts
            page = int(qs.get("page", ["1"])[0])
            per_page = int(qs.get("per_page", ["20"])[0])
            start = (page - 1) * per_page
            end = start + per_page
            vehicle_list = []
            for vid in VEHICLE_IDS[start:end]:
                imgs = VEHICLE_INDEX[vid]
                vehicle_list.append({
                    "vehicle_id": vid,
                    "num_images": len(imgs),
                    "cameras": sorted(set(i["camera_id"] for i in imgs)),
                    "thumbnail": f"/images/query/{imgs[0]['filename']}",
                })
            self.send_json({
                "vehicles": vehicle_list,
                "total": len(VEHICLE_IDS),
                "page": page,
                "per_page": per_page,
                "total_pages": (len(VEHICLE_IDS) + per_page - 1) // per_page,
            })
            return

        if path.startswith("/api/vehicle/"):
            # Get details + matches for a specific vehicle
            try:
                vid = int(path.split("/")[-1])
            except ValueError:
                self.send_json({"error": "Invalid vehicle ID"}, 400)
                return
            query_images = VEHICLE_INDEX.get(vid, [])
            if not query_images:
                self.send_json({"error": "Vehicle not found"}, 404)
                return
            matches = find_gallery_matches(vid)
            # Group by camera
            by_camera = {}
            for m in matches:
                cid = m["camera_id"]
                if cid not in by_camera:
                    by_camera[cid] = []
                by_camera[cid].append(m)
            self.send_json({
                "vehicle_id": vid,
                "query_images": [{
                    **img,
                    "path": f"/images/query/{img['filename']}",
                } for img in query_images],
                "all_matches": matches,
                "by_camera": {str(k): v for k, v in sorted(by_camera.items())},
                "num_cameras": len(by_camera),
                "total_matches": len(matches),
            })
            return

        if path == "/api/metrics":
            # Return paper metrics
            self.send_json({
                "baseline": {
                    "mAP": 82.1,
                    "rank1": 97.4,
                    "rank5": 98.8,
                    "rank10": 99.3,
                },
                "tcm_enhanced": {
                    "mAP": 82.5,
                    "rank1": 97.4,
                    "rank5": 98.7,
                    "rank10": 99.4,
                },
                "filtering": {
                    "sequential": {
                        "total_pairs": 126063,
                        "filtered": 125690,
                        "retained": 373,
                        "reduction_pct": 99.70,
                        "speedup": "338x",
                    },
                    "full_gallery": {
                        "total_pairs": 19429562,
                        "filtered": 17300557,
                        "retained": 2129005,
                        "reduction_pct": 89.0,
                    },
                },
                "ablation": {
                    "tortuosity": [
                        {"tau": 1.0, "mAP": 82.5, "rank1": 97.3, "recall": 27.8, "filter_rate": 96.59},
                        {"tau": 1.2, "mAP": 82.5, "rank1": 97.4, "recall": 32.7, "filter_rate": 94.47},
                        {"tau": 1.4, "mAP": 82.5, "rank1": 97.4, "recall": 34.8, "filter_rate": 92.37},
                        {"tau": 1.6, "mAP": 82.4, "rank1": 97.3, "recall": 36.7, "filter_rate": 89.26},
                    ],
                    "traffic": [
                        {"mode": "Static (α=1.0)", "mAP": 82.6, "rank1": 96.8, "recall": 32.7, "filter_rate": 94.47},
                        {"mode": "Dynamic α(t)", "mAP": 82.5, "rank1": 97.4, "recall": 32.7, "filter_rate": 94.47},
                    ],
                },
                "camera_network": {
                    "num_cameras": 20,
                    "min_distance_m": 87.9,
                    "max_distance_m": 1105.2,
                    "mean_distance_m": 483.1,
                    "tortuosity": 1.4,
                },
            })
            return

        # Serve images
        if path.startswith("/images/query/"):
            fname = path.split("/")[-1]
            self.send_file(QUERY_DIR / fname)
            return
        if path.startswith("/images/gallery/"):
            fname = path.split("/")[-1]
            self.send_file(GALLERY_DIR / fname)
            return
        if path.startswith("/images/plots/"):
            fname = path.split("/")[-1]
            self.send_file(LOGS_DIR / fname)
            return

        # Serve frontend files
        if path == "/" or path == "":
            self.send_file(FRONTEND_DIR / "index.html", "text/html")
            return
        
        # Serve other static frontend files
        safe_path = path.lstrip("/")
        filepath = FRONTEND_DIR / safe_path
        if filepath.exists() and filepath.is_file():
            self.send_file(filepath)
            return

        self.send_error(404, f"Not found: {path}")


def main():
    server = http.server.HTTPServer(("127.0.0.1", PORT), ReIDHandler)
    print(f"\n{'='*60}")
    print(f"  Re-ID Visualization Frontend")
    print(f"  Open in browser: http://localhost:{PORT}")
    print(f"{'='*60}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
