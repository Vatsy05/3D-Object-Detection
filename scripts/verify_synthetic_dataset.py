#!/usr/bin/env python3
"""
Sanity-check the synthetic dataset before shipping to Kaggle.

Run:
    python scripts/verify_synthetic_dataset.py
"""

from collections import Counter
from pathlib import Path

import numpy as np

DATA_ROOT = Path('/Users/dosvatsky/3D Object Detection/data/synthetic_v2')

CLASS_NAMES = [
    'bed', 'table', 'sofa', 'chair', 'toilet',
    'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub',
    'ammo_box', 'binoculars', 'combat_knife', 'flashlight', 'gas_mask',
    'hand_grenade', 'helmet', 'magazine', 'military_radio', 'pistol',
    'rifle', 'rocket_launcher', 'shotgun', 'sniper_rifle',
    'tactical_backpack', 'tactical_vest', 'wire_cutter',
]


def _first_array(npz):
    """Return the first array in an npz file regardless of key name."""
    return npz[npz.files[0]]


def load_scene(scene_dir, idx):
    sid = f'{idx:06d}'
    with np.load(scene_dir / f'{sid}_pc.npz') as f:
        pc = _first_array(f)
    bbox = np.load(scene_dir / f'{sid}_bbox.npy')
    with np.load(scene_dir / f'{sid}_votes.npz') as f:
        votes = _first_array(f)
    return pc, bbox, votes


def main():
    train_dir = DATA_ROOT / 'train'
    val_dir = DATA_ROOT / 'val'

    train_files = sorted(train_dir.glob('*_pc.npz'))
    val_files = sorted(val_dir.glob('*_pc.npz'))

    print(f'Train scenes: {len(train_files)}')
    print(f'Val scenes:   {len(val_files)}')
    print()

    # === Sanity-check 10 random scenes ===
    print('=== Spot-check 10 random scenes ===')
    sample_idx = np.random.choice(len(train_files), 10, replace=False)
    pc_shapes = []
    bbox_counts = []
    fg_ratios = []
    point_ranges = []
    bbox_dim_ranges = []

    for i in sample_idx:
        pc, bbox, votes = load_scene(train_dir, int(i))
        pc_shapes.append(pc.shape)
        bbox_counts.append(len(bbox))
        fg_ratios.append(float((votes[:, 0] > 0.5).mean()))
        point_ranges.append((pc[:, :3].min(axis=0), pc[:, :3].max(axis=0)))
        if len(bbox) > 0:
            bbox_dim_ranges.append((bbox[:, 3:6].min(), bbox[:, 3:6].max()))

    print(f'  pc.shape examples: {pc_shapes[:3]}')
    print(f'  bbox counts: min={min(bbox_counts)}, max={max(bbox_counts)}, mean={np.mean(bbox_counts):.1f}')
    print(f'  fg ratio: min={min(fg_ratios):.3f}, max={max(fg_ratios):.3f}, mean={np.mean(fg_ratios):.3f}')
    pr_min = np.min([r[0] for r in point_ranges], axis=0)
    pr_max = np.max([r[1] for r in point_ranges], axis=0)
    print(f'  point coord range across samples: X[{pr_min[0]:.2f}, {pr_max[0]:.2f}]  '
          f'Y[{pr_min[1]:.2f}, {pr_max[1]:.2f}]  Z[{pr_min[2]:.2f}, {pr_max[2]:.2f}]')
    print(f'  bbox dim range: min={min(r[0] for r in bbox_dim_ranges):.3f}  '
          f'max={max(r[1] for r in bbox_dim_ranges):.3f}')

    # === Format check on first scene ===
    print('\n=== Format check on scene 0 ===')
    pc, bbox, votes = load_scene(train_dir, 0)
    print(f'  pc:    shape={pc.shape}, dtype={pc.dtype}')
    print(f'         per-col range: '
          f'X[{pc[:,0].min():.2f},{pc[:,0].max():.2f}] '
          f'Y[{pc[:,1].min():.2f},{pc[:,1].max():.2f}] '
          f'Z[{pc[:,2].min():.2f},{pc[:,2].max():.2f}] '
          f'RGB[{pc[:,3:].min():.2f},{pc[:,3:].max():.2f}]')
    print(f'  bbox:  shape={bbox.shape}, dtype={bbox.dtype}')
    print(f'         columns: [cx, cy, cz, l, h, w, heading, class_id]')
    if len(bbox) > 0:
        print(f'         first row: {bbox[0]}')
        cls_ids = bbox[:, 7].astype(int)
        print(f'         class IDs in scene 0: {[CLASS_NAMES[i] for i in cls_ids]}')
    print(f'  votes: shape={votes.shape}, dtype={votes.dtype}')
    print(f'         fg points: {int((votes[:,0] > 0.5).sum())}/{len(votes)}')

    # === Pass/fail criteria ===
    print('\n=== Pass criteria ===')
    checks = []
    checks.append(('pc has shape (20000, 6)', pc.shape == (20000, 6)))
    checks.append(('pc dtype is float32', pc.dtype == np.float32))
    checks.append(('bbox shape is (N, 8)', bbox.ndim == 2 and bbox.shape[1] == 8))
    checks.append(('votes shape is (20000, 10)', votes.shape == (20000, 10)))
    checks.append(('class IDs in [0, 26]', bool(((bbox[:, 7] >= 0) & (bbox[:, 7] < 27)).all())))
    checks.append(('all bbox dims positive', bool((bbox[:, 3:6] > 0).all())))
    checks.append(('fg ratio reasonable (0.1-0.9)',
                   0.1 < float((votes[:, 0] > 0.5).mean()) < 0.9))
    for name, ok in checks:
        print(f'  {"OK " if ok else "FAIL"}  {name}')

    # === Full class distribution sanity ===
    print('\n=== Class distribution across all train scenes ===')
    counter = Counter()
    for f in train_files:
        sid = f.stem.replace('_pc', '')
        bbox = np.load(train_dir / f'{sid}_bbox.npy')
        for box in bbox:
            counter[CLASS_NAMES[int(box[7])]] += 1

    total = sum(counter.values())
    print(f'Total instances in train: {total}')
    for cls in CLASS_NAMES:
        n = counter[cls]
        pct = 100 * n / total
        print(f'  {cls:18s} {n:6d}  ({pct:5.2f}%)')

    # Disk usage
    train_size = sum(f.stat().st_size for f in train_dir.iterdir()) / 1e6
    val_size = sum(f.stat().st_size for f in val_dir.iterdir()) / 1e6
    print(f'\nTrain disk: {train_size:.0f} MB')
    print(f'Val disk:   {val_size:.0f} MB')
    print(f'Total:      {(train_size + val_size) / 1024:.2f} GB')


if __name__ == '__main__':
    main()
