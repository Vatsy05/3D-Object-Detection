#!/usr/bin/env python3
"""
Run Phase 8 VoteNet inference on a new point cloud scene.

Designed to run on a machine with CUDA (Kaggle / Colab / local NVIDIA GPU).
Reads a point cloud from disk, runs detection, saves predictions + viz.

Usage on Kaggle/Colab (after Cells A-D-recovery have been run interactively):

    !python predict_on_new_scene.py \\
        --scene-file /kaggle/input/my-new-scene/scene.npy \\
        --checkpoint /kaggle/input/.../votenet_27class_best.pt \\
        --out-dir /kaggle/working/inference

For a local NVIDIA GPU setup, same flags but absolute local paths.

The scene file can be:
  * (N, 3) numpy array — XYZ only
  * (N, 6) numpy array — XYZ + RGB
  * .npz file with key 'pc' or 'point_cloud'
  * .ply / .pcd file (requires Open3D)

By default the script assumes the scene is already in the correct coordinate
convention (Z-up, floor near Z=0). Use --y-up to convert from Y-up.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# === Configure these for your environment ===
# (When running interactively in a Kaggle notebook, sys.path is already set up.)
DEFAULT_VOTENET_REPO = '/kaggle/working/votenet'


def setup_paths(votenet_repo):
    for p in [votenet_repo, f'{votenet_repo}/utils',
              f'{votenet_repo}/models', f'{votenet_repo}/sunrgbd']:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_scene(scene_file, y_up=False):
    """Load a point cloud from various formats. Returns (N, 3) or (N, 6) numpy array."""
    p = Path(scene_file)
    if p.suffix == '.npy':
        pc = np.load(p)
    elif p.suffix == '.npz':
        with np.load(p) as f:
            key = 'pc' if 'pc' in f.files else f.files[0]
            pc = f[key]
    elif p.suffix in {'.ply', '.pcd'}:
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(cloud.points)
        cols = np.asarray(cloud.colors) if cloud.has_colors() else np.zeros_like(pts)
        pc = np.concatenate([pts, cols], axis=1)
    else:
        raise ValueError(f'Unsupported file type: {p.suffix}')
    pc = pc.astype(np.float32)
    if y_up:
        # Swap Y and Z to convert Y-up -> Z-up
        if pc.shape[1] >= 3:
            pc = pc[:, [0, 2, 1] + list(range(3, pc.shape[1]))]
    return pc


def preprocess_scene(pc, num_points=20000):
    """Subsample/pad to num_points, add height feature. Returns (num_points, 4)."""
    # Keep XYZ only (drop RGB if present)
    xyz = pc[:, :3]

    # Resample to exactly num_points
    if len(xyz) > num_points:
        idx = np.random.choice(len(xyz), num_points, replace=False)
    else:
        extra = np.random.choice(len(xyz), num_points - len(xyz), replace=True)
        idx = np.concatenate([np.arange(len(xyz)), extra])
    xyz = xyz[idx]

    # Add height feature: Z minus floor
    floor_z = np.percentile(xyz[:, 2], 0.99)
    height = (xyz[:, 2] - floor_z)[:, None]
    return np.concatenate([xyz, height], axis=1).astype(np.float32)


def build_model_and_load(checkpoint_path, dc, device):
    from votenet import VoteNet
    model = VoteNet(
        num_class=dc.num_class,
        num_heading_bin=dc.num_heading_bin,
        num_size_cluster=dc.num_size_cluster,
        mean_size_arr=dc.mean_size_arr,
        num_proposal=256,
        input_feature_dim=1,
        vote_factor=1,
        sampling='vote_fps',
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  loaded checkpoint epoch={ckpt.get("epoch","?")} val_loss={ckpt.get("val_loss",float("nan")):.4f}')
    return model


def run_inference(model, point_cloud_np, dc, device, score_threshold=0.30):
    """Run inference on a single scene. Returns list of (class_id, box_params, score)."""
    from ap_helper import parse_predictions

    pc_tensor = torch.from_numpy(point_cloud_np).unsqueeze(0).to(device)  # (1, N, 4)
    end_points = {'point_clouds': pc_tensor}

    with torch.no_grad():
        end_points = model(end_points)
        # parse_predictions also needs point_clouds in end_points
        end_points['point_clouds'] = pc_tensor

    config_dict = {
        'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': True, 'per_class_proposal': True,
        'conf_thresh': score_threshold, 'dataset_config': dc,
    }
    pred_map_cls = parse_predictions(end_points, config_dict)
    return pred_map_cls[0]   # batch size 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-file', required=True, help='Path to scene point cloud (.npy/.npz/.ply/.pcd)')
    ap.add_argument('--checkpoint', required=True, help='Path to votenet_27class_best.pt')
    ap.add_argument('--out-dir', default='/kaggle/working/inference', help='Output directory')
    ap.add_argument('--votenet-repo', default=DEFAULT_VOTENET_REPO,
                    help='Path to votenet source tree (must have pointnet2._ext compiled)')
    ap.add_argument('--num-points', type=int, default=20000)
    ap.add_argument('--score-threshold', type=float, default=0.30)
    ap.add_argument('--y-up', action='store_true',
                    help='Scene is in Y-up convention; convert to Z-up')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading scene from {args.scene_file}...')
    raw = load_scene(args.scene_file, y_up=args.y_up)
    print(f'  raw shape: {raw.shape}')
    pc = preprocess_scene(raw, num_points=args.num_points)
    print(f'  preprocessed shape: {pc.shape}')

    setup_paths(args.votenet_repo)

    # Need DC config — recreate it from CLASS_NAMES (matches Cell B)
    CLASS_NAMES = [
        'bed', 'table', 'sofa', 'chair', 'toilet',
        'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub',
        'ammo_box', 'binoculars', 'combat_knife', 'flashlight', 'gas_mask',
        'hand_grenade', 'helmet', 'magazine', 'military_radio', 'pistol',
        'rifle', 'rocket_launcher', 'shotgun', 'sniper_rifle',
        'tactical_backpack', 'tactical_vest', 'wire_cutter',
    ]

    # Try to read mean_size_arr from checkpoint metadata; else fallback to defaults
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'mean_size_arr' in ckpt:
        mean_size_arr = np.asarray(ckpt['mean_size_arr'])
    else:
        # rough defaults from CLASS_REAL_SIZE
        CLASS_REAL_SIZE = {
            'bed': 2.0, 'table': 1.5, 'sofa': 2.0, 'chair': 0.55, 'toilet': 0.6,
            'desk': 1.4, 'dresser': 1.0, 'night_stand': 0.55, 'bookshelf': 0.85, 'bathtub': 1.6,
            'ammo_box': 0.35, 'binoculars': 0.22, 'combat_knife': 0.30, 'flashlight': 0.18,
            'gas_mask': 0.28, 'hand_grenade': 0.12, 'helmet': 0.28, 'magazine': 0.18,
            'military_radio': 0.30, 'pistol': 0.22, 'rifle': 0.95, 'rocket_launcher': 1.20,
            'shotgun': 0.95, 'sniper_rifle': 1.20, 'tactical_backpack': 0.55,
            'tactical_vest': 0.50, 'wire_cutter': 0.25,
        }
        mean_size_arr = np.array([[CLASS_REAL_SIZE[c]*0.6,
                                    CLASS_REAL_SIZE[c]*0.4,
                                    CLASS_REAL_SIZE[c]*0.5] for c in CLASS_NAMES])

    class SyntheticDatasetConfig:
        num_class = 27
        num_heading_bin = 12
        num_size_cluster = 27
        type2class = {c: i for i, c in enumerate(CLASS_NAMES)}
        class2type = {i: c for i, c in enumerate(CLASS_NAMES)}
        type2onehotclass = dict(type2class)
        mean_size_arr_ = mean_size_arr.astype(np.float32)
        type_mean_size = {c: mean_size_arr_[i] for i, c in enumerate(CLASS_NAMES)}

        @property
        def mean_size_arr(self):
            return self.mean_size_arr_

        def angle2class(self, a):
            n = self.num_heading_bin
            a = a % (2*np.pi)
            per = 2*np.pi/n
            shifted = (a + per/2) % (2*np.pi)
            cls = int(shifted/per)
            return cls, shifted - (cls*per + per/2)

        def class2angle(self, cls, residual, to_label_format=True):
            per = 2*np.pi/self.num_heading_bin
            a = cls*per + residual
            if to_label_format and a > np.pi:
                a -= 2*np.pi
            return a

        def size2class(self, size, type_name):
            return self.type2class[type_name], size - self.type_mean_size[type_name]

        def class2size(self, cls, residual):
            return self.type_mean_size[self.class2type[cls]] + residual

        def param2obb(self, center, hc, hr, sc, sr):
            obb = np.zeros((7,))
            obb[0:3] = center
            obb[3:6] = self.class2size(int(sc), sr)
            obb[6] = -self.class2angle(hc, hr)
            return obb

    dc = SyntheticDatasetConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Building model + loading checkpoint...')
    model = build_model_and_load(args.checkpoint, dc, device)

    print(f'Running inference (score >= {args.score_threshold})...')
    detections = run_inference(model, pc, dc, device, score_threshold=args.score_threshold)
    print(f'  found {len(detections)} detections')

    # Serialize
    serialized = []
    for cls_id, box, score in detections:
        box_np = box.copy() if isinstance(box, np.ndarray) else np.array(box, dtype=np.float32)
        serialized.append({
            'class_id': int(cls_id),
            'class_name': CLASS_NAMES[int(cls_id)],
            'box': box_np.tolist(),
            'score': float(score),
        })

    # Save predictions JSON
    with open(out / 'detections.json', 'w') as f:
        json.dump({'scene_file': args.scene_file,
                   'n_points': int(len(pc)),
                   'score_threshold': args.score_threshold,
                   'detections': serialized}, f, indent=2)
    print(f'Saved {out/"detections.json"}')

    # Save pickle with the actual point cloud for visualization
    with open(out / 'scene_with_predictions.pkl', 'wb') as f:
        pickle.dump({
            'class_names': CLASS_NAMES,
            'scenes': [{
                'scan_idx': 0,
                'point_cloud': pc,
                'predictions': [{'class_id': d['class_id'],
                                 'box': np.array(d['box']),
                                 'score': d['score']} for d in serialized],
                'groundtruths': [],   # no GT for a new scene
            }],
        }, f)
    print(f'Saved {out/"scene_with_predictions.pkl"}')

    print('\n=== Detection summary ===')
    for d in sorted(serialized, key=lambda x: -x['score']):
        print(f'  {d["class_name"]:18s}  score={d["score"]:.3f}')


if __name__ == '__main__':
    main()
