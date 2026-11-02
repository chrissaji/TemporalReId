# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TemporalReId** is a Vehicle Re-Identification research system built on TransReID (ICCV 2021), enhanced with a **Temporal Consistency Module (TCM)** — a physics-based post-processing filter that eliminates temporally impossible camera-to-camera matches without any ML training. The TCM reduces the search space by 99.70% in multi-camera tracking scenarios.

## Commands

### Training
```bash
# Single GPU
python train.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# Multi-GPU distributed (required for VehicleID)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 \
  train.py --config_file configs/VehicleID/vit_transreid_stride.yml MODEL.DIST_TRAIN True
```

### Evaluation
```bash
# Standard evaluation
python test.py --config_file configs/VeRi/vit_transreid_stride.yml \
  MODEL.DEVICE_ID "('0')" TEST.WEIGHT 'logs/veri_vit_transreid_stride/transformer_120.pth'

# VehicleID requires 10-trial averaging (handled automatically)
python test.py --config_file configs/VehicleID/vit_transreid_stride.yml TEST.WEIGHT '<model_path>'
```

### TCM Demos
```bash
python demo_visual_logic.py          # Quick TCM demo (~10 sec), outputs tcm_logic_visualization.png
python demo_rule_based_tcm.py        # Rule-based TCM with mock Re-ID results
python demo_tcm_presentation.py      # Full TransReID + TCM demo (~40–50 min, needs GPU + dataset)
python test_sequential_tracking.py   # Multi-camera sequential tracking analysis
```

### Visualization
```bash
python visualize_reid.py             # Query-gallery matching visualization
python create_realistic_veri_simulation.py  # Generate realistic VeRi timestamps
```

## Architecture

### Pipeline Flow
```
Config (YACS) → make_dataloader → make_model → Loss setup → processor.do_train / do_inference
```

1. **Config system** ([config/defaults.py](config/defaults.py)): YACS-based. Always merge then freeze:
   ```python
   cfg.merge_from_file(args.config_file)
   cfg.merge_from_list(args.opts)
   cfg.freeze()
   ```
   Dataset configs live in `configs/<DATASET>/<MODEL_TYPE>.yml`.

2. **Data loading** ([datasets/make_dataloader.py](datasets/make_dataloader.py)): All datasets inherit from `bases.ImageDataset`. Factory in `datasets/__init__.py` maps names to classes. Uses `RandomIdentitySampler` for triplet-based training. VeRi filenames encode timestamps for TCM use.

3. **Model** ([model/make_model.py](model/make_model.py)): ViT backbone with optional:
   - **JPM** (Jigsaw Patch Module) — local feature learning via patch shuffle/shift; requires `MODEL.RE_ARRANGE=True`
   - **SIE** (Side Information Embedding) — camera/view embeddings via `MODEL.SIE_CAMERA` / `MODEL.SIE_VIEW`
   - **BNNeck** — batch norm between feature and classifier; `TEST.NECK_FEAT` selects `'before'`/`'after'` for extraction

4. **Loss** ([loss/](loss/)): Combination of ID loss (softmax / arcface / cosface / amsoftmax / circle) + metric loss (triplet / center). JPM models return a list of scores `[global, local1, local2, ...]` for multi-branch training.

5. **Training loop** ([processor/processor.py](processor/processor.py)): AMP (Automatic Mixed Precision) enabled by default. Saves checkpoints every `CHECKPOINT_PERIOD` epochs.

### Temporal Consistency Module (TCM)

The core research contribution. All TCM logic is **post-processing** — it modifies the distance matrix after TransReID feature extraction, so it works with any Re-ID model.

| File | Role |
|------|------|
| [utils/rule_based_tcm.py](utils/rule_based_tcm.py) | Core `RuleBasedTCM` class: Haversine distance, NetworkX camera graph, traffic-aware travel time bounds, confidence-based penalty injection |
| [utils/rule_based_tcm_v2.py](utils/rule_based_tcm_v2.py) | Refined version with tuned parameters |
| [utils/veri_camera_config.py](utils/veri_camera_config.py) | GPS coordinates and speed limits for all 20 VeRi-776 cameras |

TCM applies **soft filtering** (penalty added to distance matrix) rather than hard elimination. Traffic factors: rush hour 0.6×, night 1.2×, weekend 1.1× road speed.

### Key Config Parameters

| Parameter | Effect |
|-----------|--------|
| `MODEL.TRANSFORMER_TYPE` | Architecture variant |
| `MODEL.STRIDE_SIZE` | Patch stride (stride version uses ~12GB GPU vs ~7GB) |
| `MODEL.JPM` | Enable Jigsaw Patch Module |
| `MODEL.SIE_CAMERA` / `MODEL.SIE_VIEW` | Enable side information embeddings |
| `TEST.FEAT_NORM` | Feature normalization (cosine vs Euclidean distance) |
| `TEST.NECK_FEAT` | Feature extraction point (`'before'`/`'after'` BN) |
| `MODEL.DIST_TRAIN` | Enable DistributedDataParallel |

### Adding a New Dataset

1. Implement class in `datasets/` inheriting `bases.ImageDataset`
2. Register it in the factory dict in `datasets/__init__.py`
3. Create a config YAML in `configs/<DATASET>/`
4. If the dataset has temporal data, extract timestamps in the dataset class (see [datasets/veri.py](datasets/veri.py))

### Evaluation Outputs

`test.py` saves numpy files to the log directory:
- `distmat.npy` — pairwise distance matrix (query × gallery)
- `img_path_list.npy`, `pids.npy`, `camids.npy` — metadata for visualization

Metrics reported: Rank-1, Rank-5, Rank-10, mAP.

## Dependencies

```
torch  torchvision  timm  yacs  opencv-python
```

Install into the `.venv` already present in the repo root, or a conda environment.
