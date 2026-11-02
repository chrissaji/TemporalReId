#!/usr/bin/env python
"""
Parameter Sweep for TCM Window/Penalty Tuning

Extracts features ONCE, then sweeps over multiple (window, penalty)
combinations to find parameters that yield a positive mAP delta.
"""

import os
import sys
import time
import datetime
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# TransReID imports
from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func

# TCM imports (only for camera config, not used in vectorised sweep)
from utils.veri_camera_config import get_veri_camera_positions


def extract_features(cfg, model_path, logger):
    """Extract features once for all sweeps using proper num_query split."""
    device = torch.device(cfg.MODEL.DEVICE)

    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_classes, camera_num, view_num)
    model.load_param(model_path)
    model.eval()
    model.to(device)

    logger.info(f"Model loaded: {model_path}")
    logger.info(f"Dataset: {cfg.DATASETS.NAMES} with {num_classes} identities")

    all_features = []
    all_pids = []
    all_camids = []
    all_timestamps = []

    start_time = time.time()
    logger.info("Extracting features (one-time cost)...")

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids, camids_batch, viewids, img_paths, timestamps) in enumerate(tqdm(val_loader, desc="Feature extraction")):
            imgs = imgs.to(device)
            camids_batch = camids_batch.to(device)
            viewids = viewids.to(device)
            if cfg.TEST.FEAT_NORM == 'yes':
                features = torch.nn.functional.normalize(model(imgs, cam_label=camids_batch, view_label=viewids), dim=1, p=2)
            else:
                features = model(imgs, cam_label=camids_batch, view_label=viewids)
            all_features.append(features.cpu())
            all_pids.extend(pids)
            all_camids.extend(camids)
            all_timestamps.extend(timestamps if timestamps is not None else [None] * len(pids))

    all_features = torch.cat(all_features, dim=0)

    # Split by num_query (standard VeRi-776 protocol)
    nq = num_query

    logger.info(f"Feature extraction done in {time.time() - start_time:.1f}s")
    logger.info(f"Queries: {nq}, Gallery: {len(all_pids) - nq}")

    return {
        'query_features': all_features[:nq],
        'gallery_features': all_features[nq:],
        'query_pids': np.array(all_pids[:nq]),
        'gallery_pids': np.array(all_pids[nq:]),
        'query_camids': np.array(all_camids[:nq]),
        'gallery_camids': np.array(all_camids[nq:]),
        'query_timestamps': all_timestamps[:nq],
        'gallery_timestamps': all_timestamps[nq:],
    }


def compute_baseline(data, logger):
    """Compute baseline cosine distance matrix and metrics."""
    qf = torch.nn.functional.normalize(data['query_features'], dim=1, p=2)
    gf = torch.nn.functional.normalize(data['gallery_features'], dim=1, p=2)
    sim = torch.mm(qf, gf.t())
    dist = (1 - sim).numpy()

    cmc, mAP = eval_func(dist, data['query_pids'], data['gallery_pids'],
                          data['query_camids'], data['gallery_camids'])
    logger.info(f"Baseline  mAP={mAP:.4f}  R-1={cmc[0]:.4f}  R-5={cmc[4]:.4f}")
    return dist, mAP, cmc


def precompute_masks(data, logger):
    """Precompute numpy arrays for vectorised window sweeps."""
    q_cams = np.array(data['query_camids'])
    g_cams = np.array(data['gallery_camids'])
    q_ts = np.array([t if t is not None else 0 for t in data['query_timestamps']], dtype=np.int64)
    g_ts = np.array([t if t is not None else 0 for t in data['gallery_timestamps']], dtype=np.int64)

    # (Q, G) boolean: True where cameras differ
    cross_cam = q_cams[:, None] != g_cams[None, :]

    # (Q, G) signed time difference: g_time - q_time
    dt = g_ts[None, :].astype(np.int64) - q_ts[:, None].astype(np.int64)

    # (Q, G) absolute time difference
    abs_dt = np.abs(dt)

    # (Q, G) boolean: gallery frame is before query frame
    g_before_q = dt < 0

    total_count = q_cams.shape[0] * g_cams.shape[0]

    logger.info(f"Precomputed masks: {q_cams.shape[0]} queries x {g_cams.shape[0]} gallery = {total_count} pairs")
    return cross_cam, g_before_q, abs_dt, total_count


def sweep(dist_baseline, data, cross_cam, g_before_q, abs_dt, total_count,
          windows, penalties, logger):
    """Vectorised sweep: each combo takes ~0.1s instead of minutes."""
    results = []

    for window in windows:
        # Binary mask: cross-camera AND (g_before_q OR time_diff > window)
        penalty_mask = cross_cam & (g_before_q | (abs_dt > window))
        filt_cnt = int(penalty_mask.sum())
        filt_pct = 100.0 * filt_cnt / total_count if total_count else 0

        for penalty in penalties:
            t0 = time.time()
            filtered = dist_baseline + penalty * penalty_mask.astype(dist_baseline.dtype)
            cmc, mAP = eval_func(filtered,
                                  data['query_pids'], data['gallery_pids'],
                                  data['query_camids'], data['gallery_camids'])
            dt = time.time() - t0
            results.append({
                'window': window,
                'penalty': penalty,
                'mAP': mAP,
                'R1': cmc[0],
                'R5': cmc[4],
                'filter_pct': filt_pct,
                'filtered_count': filt_cnt,
                'total_count': total_count,
                'time_s': dt,
            })
            logger.info(
                f"  w={window:>6d}  p={penalty:.3f}  "
                f"mAP={mAP:.4f}  R-1={cmc[0]:.4f}  "
                f"Filter={filt_pct:.1f}%  ({dt:.1f}s)"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep TCM window/penalty parameters")
    parser.add_argument('--config_file', type=str,
                        default='configs/VeRi/vit_transreid_stride.yml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='logs/tcm_sweep')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("tcm_sweep", str(output_dir), if_train=False)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Model : {args.model_path}")

    # ── 1. Extract features (one-time) ────────────────────────────
    data = extract_features(cfg, args.model_path, logger)

    # ── 2. Baseline metrics ───────────────────────────────────────
    dist_baseline, baseline_mAP, baseline_cmc = compute_baseline(data, logger)

    # ── 3. Precompute masks (vectorised) ─────────────────────────
    cross_cam, g_before_q, abs_dt, total_count = precompute_masks(data, logger)

    # ── 4. Sweep grid ────────────────────────────────────────────
    windows   = [5000, 7000, 8000, 9000, 10000, 11000, 12000, 15000, 20000, 30000]
    penalties = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]

    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER SWEEP")
    logger.info("=" * 70)
    results = sweep(dist_baseline, data, cross_cam, g_before_q, abs_dt, total_count,
                    windows, penalties, logger)

    # ── 5. Summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TOP CONFIGURATIONS (positive mAP delta, sorted by mAP)")
    logger.info("=" * 70)

    positive = [r for r in results if r['mAP'] > baseline_mAP]
    positive.sort(key=lambda r: r['mAP'], reverse=True)

    if not positive:
        logger.info("  ** No configuration improved mAP over baseline **")
        # Show least-harmful instead
        results.sort(key=lambda r: r['mAP'], reverse=True)
        for r in results[:10]:
            delta = r['mAP'] - baseline_mAP
            logger.info(
                f"  w={r['window']:>6d}  p={r['penalty']:.3f}  "
                f"mAP={r['mAP']:.4f} ({delta:+.4f})  R-1={r['R1']:.4f}  "
                f"Filter={r['filter_pct']:.1f}%"
            )
    else:
        for r in positive[:15]:
            delta = r['mAP'] - baseline_mAP
            logger.info(
                f"  w={r['window']:>6d}  p={r['penalty']:.3f}  "
                f"mAP={r['mAP']:.4f} ({delta:+.4f})  R-1={r['R1']:.4f}  "
                f"Filter={r['filter_pct']:.1f}%"
            )

    # ── 6. Best result ────────────────────────────────────────────
    best = max(results, key=lambda r: r['mAP'])
    delta = best['mAP'] - baseline_mAP
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"  Window:   {best['window']}")
    logger.info(f"  Penalty:  {best['penalty']}")
    logger.info(f"  mAP:      {best['mAP']:.4f} ({delta:+.4f} vs baseline {baseline_mAP:.4f})")
    logger.info(f"  Rank-1:   {best['R1']:.4f}")
    logger.info(f"  Filter:   {best['filter_pct']:.1f}%")

    # Save all results
    np.save(output_dir / 'sweep_results.npy', {
        'baseline_mAP': baseline_mAP,
        'baseline_R1': baseline_cmc[0],
        'results': results,
    })
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
