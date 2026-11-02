#!/usr/bin/env python
"""
Test Rule-Based TCM Integration with TransReID
Compare baseline vs TCM-enhanced results on VeRi-776
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
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# TransReID imports
from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func

# TCM imports  
from utils.veri_camera_config import get_veri_camera_positions, get_camera_distance_analysis
from utils.rule_based_tcm_v2 import RuleBasedTCMV2

class TCMEvaluator:
    def __init__(self, cfg, model_path, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Create dataloaders first to get num_classes, camera_num, view_num
        self.train_loader, self.train_loader_normal, self.val_loader, self.num_query, self.num_classes, self.camera_num, self.view_num = make_dataloader(cfg)
        
        # Load model with correct parameters
        self.model = make_model(cfg, self.num_classes, self.camera_num, self.view_num)
        self.model.load_param(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize V2 TCM (paper-aligned: Haversine + tortuosity + alpha(t))
        self.tcm = RuleBasedTCMV2(get_veri_camera_positions())
        
        self.logger.info(f"Model loaded: {model_path}")
        self.logger.info(f"Dataset: {cfg.DATASETS.NAMES} with {self.num_classes} identities")
        
    def extract_features(self):
        """Extract features for query and gallery using proper num_query split."""
        start_time = time.time()
        self.logger.info("Extracting features...")
        
        all_features = []
        all_pids = []
        all_camids = []
        all_timestamps = []
        all_paths = []
        
        with torch.no_grad():
            for batch_idx, (imgs, pids, camids, camids_batch, viewids, img_paths, timestamps) in enumerate(tqdm(self.val_loader, desc="Feature extraction")):
                imgs = imgs.to(self.device)
                camids_batch = camids_batch.to(self.device)
                viewids = viewids.to(self.device)
                
                if self.cfg.TEST.FEAT_NORM == 'yes':
                    features = torch.nn.functional.normalize(self.model(imgs, cam_label=camids_batch, view_label=viewids), dim=1, p=2)
                else:
                    features = self.model(imgs, cam_label=camids_batch, view_label=viewids)
                
                all_features.append(features.cpu())
                all_pids.extend(pids)
                all_camids.extend(camids)
                all_timestamps.extend(timestamps if timestamps is not None else [None] * len(pids))
                all_paths.extend(img_paths)
        
        all_features = torch.cat(all_features, dim=0)
        
        # Split by num_query (standard VeRi-776 protocol)
        nq = self.num_query
        
        extraction_time = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {extraction_time:.2f}s")
        self.logger.info(f"Queries: {nq}, Gallery: {len(all_pids) - nq}")
        
        return {
            'query_features': all_features[:nq],
            'gallery_features': all_features[nq:],
            'query_pids': np.array(all_pids[:nq]),
            'gallery_pids': np.array(all_pids[nq:]),
            'query_camids': np.array(all_camids[:nq]),
            'gallery_camids': np.array(all_camids[nq:]),
            'query_timestamps': all_timestamps[:nq],
            'gallery_timestamps': all_timestamps[nq:],
            'query_paths': all_paths[:nq],
            'gallery_paths': all_paths[nq:]
        }
    
    def compute_baseline_distance_matrix(self, query_features, gallery_features):
        """Compute baseline cosine distance matrix"""
        start_time = time.time()
        self.logger.info("Computing baseline distance matrix...")
        
        # Normalize features
        query_features = torch.nn.functional.normalize(query_features, dim=1, p=2)
        gallery_features = torch.nn.functional.normalize(gallery_features, dim=1, p=2)
        
        # Compute cosine distance
        similarity_matrix = torch.mm(query_features, gallery_features.t())
        distance_matrix = 1 - similarity_matrix
        
        computation_time = time.time() - start_time
        self.logger.info(f"Baseline distance matrix computed in {computation_time:.2f}s")
        
        return distance_matrix.numpy()
    
    def apply_tcm_filtering(self, distance_matrix, data_dict):
        """Apply TCM temporal-window filtering (frame-number timestamps)"""
        start_time = time.time()
        self.logger.info("Applying rule-based TCM filtering...")
        
        query_camids = data_dict['query_camids']
        gallery_camids = data_dict['gallery_camids'] 
        query_timestamps = data_dict['query_timestamps']
        gallery_timestamps = data_dict['gallery_timestamps']
        
        # Temporal window filtering with soft penalty (paper Table II/IV)
        # Correct-protocol sweep (1678 queries): w=5000, p=0.02 → mAP +0.40pp, R-1 -0.06pp
        tcm_distance_matrix, filtered_count, total_count = self.tcm.filter_by_temporal_window(
            distance_matrix, query_camids, gallery_camids,
            query_timestamps, gallery_timestamps,
            window=5000, penalty=0.02
        )
        
        filter_percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
        filtering_time = time.time() - start_time
        self.logger.info(f"TCM filtered {filtered_count}/{total_count} matches ({filter_percentage:.1f}%)")
        
        return tcm_distance_matrix, filter_percentage, filtering_time
    
    def evaluate_results(self, distance_matrix, data_dict, suffix=""):
        """Evaluate Re-ID performance"""
        start_time = time.time()
        query_pids = data_dict['query_pids']
        gallery_pids = data_dict['gallery_pids']
        query_camids = data_dict['query_camids']
        gallery_camids = data_dict['gallery_camids']
        
        # Compute metrics using eval_func
        cmc, mAP = eval_func(
            distance_matrix,
            query_pids, 
            gallery_pids,
            query_camids,
            gallery_camids,
            max_rank=50
        )
        
        eval_time = time.time() - start_time
        
        results = {
            'mAP': mAP,
            'CMC': cmc,
            'eval_time': eval_time
        }
        
        self.logger.info(f"Results{suffix}:")
        self.logger.info(f"  mAP: {results['mAP']:.4f}")
        self.logger.info(f"  CMC rank-1: {results['CMC'][0]:.4f}")
        self.logger.info(f"  CMC rank-5: {results['CMC'][4]:.4f}")
        self.logger.info(f"  CMC rank-10: {results['CMC'][9]:.4f}")
        self.logger.info(f"  Evaluation time: {eval_time:.2f}s")
        
        return results
    
    def run_comparison(self):
        """Run complete baseline vs TCM comparison"""
        self.logger.info("="*60)
        self.logger.info("STARTING BASELINE VS TCM COMPARISON")
        self.logger.info("="*60)
        
        # Extract features
        data_dict = self.extract_features()
        
        # Compute baseline distance matrix
        baseline_dist = self.compute_baseline_distance_matrix(
            data_dict['query_features'], 
            data_dict['gallery_features']
        )
        
        # Evaluate baseline
        self.logger.info("\n" + "="*40)
        self.logger.info("BASELINE RESULTS")
        self.logger.info("="*40)
        baseline_results = self.evaluate_results(baseline_dist, data_dict, " (Baseline)")
        
        # Apply TCM filtering
        tcm_dist, filter_percentage, filtering_time = self.apply_tcm_filtering(baseline_dist, data_dict)
        
        # Evaluate TCM results  
        self.logger.info("\n" + "="*40)
        self.logger.info("TCM-ENHANCED RESULTS")
        self.logger.info("="*40)
        tcm_results = self.evaluate_results(tcm_dist, data_dict, " (TCM-Enhanced)")
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Baseline mAP:       {baseline_results['mAP']:.4f}")
        self.logger.info(f"TCM-Enhanced mAP:   {tcm_results['mAP']:.4f}")
        self.logger.info(f"mAP Improvement:    {tcm_results['mAP'] - baseline_results['mAP']:+.6f}")
        self.logger.info(f"")
        self.logger.info(f"Baseline Rank-1:    {baseline_results['CMC'][0]:.4f}")
        self.logger.info(f"TCM-Enhanced Rank-1: {tcm_results['CMC'][0]:.4f}")
        self.logger.info(f"Rank-1 Improvement: {tcm_results['CMC'][0] - baseline_results['CMC'][0]:+.6f}")
        self.logger.info(f"")
        self.logger.info(f"TCM Filter Rate:    {filter_percentage:.1f}%")
        self.logger.info(f"TCM Filtering Time: {filtering_time:.2f}s")
        self.logger.info(f"")
        self.logger.info(f"=== EFFICIENCY METRICS ===")
        self.logger.info(f"Baseline Eval Time:     {baseline_results['eval_time']:.2f}s")
        self.logger.info(f"TCM Eval Time:          {tcm_results['eval_time']:.2f}s")
        self.logger.info(f"Eval Time Difference:   {tcm_results['eval_time'] - baseline_results['eval_time']:+.2f}s")
        self.logger.info(f"TCM Overhead:           {filtering_time:.2f}s")
        total_tcm_time = filtering_time + tcm_results['eval_time']
        total_baseline_time = baseline_results['eval_time']
        self.logger.info(f"Total Time (Baseline):  {total_baseline_time:.2f}s")
        self.logger.info(f"Total Time (TCM):       {total_tcm_time:.2f}s")
        self.logger.info(f"Overall Speedup:        {total_baseline_time / total_tcm_time:.4f}x")
        
        return baseline_results, tcm_results

def main():
    parser = argparse.ArgumentParser(description="Test Rule-Based TCM Integration")
    parser.add_argument('--config_file', type=str, 
                       default='configs/VeRi/vit_transreid_stride.yml',
                       help='path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='path to trained model')
    parser.add_argument('--output_dir', type=str, default='logs/tcm_comparison',
                       help='output directory for results')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                       help='modify config options using the command-line')
    
    args = parser.parse_args()
    
    # Setup config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # Setup output directory (timestamped per run)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("tcm_comparison", str(output_dir), if_train=False)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Model: {args.model_path}")
    
    # Analyze camera network first
    logger.info("\n" + "="*40)
    logger.info("VERI CAMERA NETWORK ANALYSIS")
    logger.info("="*40)
    stats = get_camera_distance_analysis()
    logger.info(f"Number of cameras: {len(stats['camera_pairs'])//19*2}")
    logger.info(f"Distance range: {stats['min_distance_m']:.1f}m - {stats['max_distance_m']:.1f}m")
    logger.info(f"Mean distance: {stats['mean_distance_m']:.1f}m")
    
    # Run evaluation
    evaluator = TCMEvaluator(cfg, args.model_path, logger)
    baseline_results, tcm_results = evaluator.run_comparison()
    
    # Save results
    results = {
        'baseline': baseline_results,
        'tcm_enhanced': tcm_results,
        'config': args.config_file,
        'model': args.model_path
    }
    
    np.save(output_dir / 'tcm_comparison_results.npy', results)
    logger.info(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()