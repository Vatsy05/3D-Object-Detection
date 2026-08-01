#!/usr/bin/env python3
"""
Offline visualization for Phase 8 VoteNet predictions.

No GPU / no VoteNet needed — reads val_predictions.pkl produced by Cell F.

Run:
    python scripts/visualize_phase8_detections.py
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def compute_box_corners(center, size, heading):
    cx, cy, cz = center
    l, w, h = size
    corners = np.array([
        [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2],
    ])
    cosh, sinh = np.cos(heading), np.sin(heading)
    R = np.array([[cosh, -sinh, 0], [sinh, cosh, 0], [0, 0, 1]])
    return corners @ R.T + np.array([cx, cy, cz])


def to_corners(box):
    if box.shape == (8, 3):
        return box
    return compute_box_corners(box[:3], box[3:6], box[6])


def draw_box_3d(ax, corners, color, linewidth=1.5, alpha=1.0, linestyle='-'):
    for i, j in BOX_EDGES:
        ax.plot3D([corners[i, 0], corners[j, 0]],
                  [corners[i, 1], corners[j, 1]],
                  [corners[i, 2], corners[j, 2]],
                  color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='/Users/dosvatsky/3D Object Detection/checkpoints/val_predictions.pkl')
    ap.add_argument('--out-dir', default='/Users/dosvatsky/3D Object Detection')
    ap.add_argument('--n-scenes', type=int, default=6)
    ap.add_argument('--score-threshold', type=float, default=0.30)
    ap.add_argument('--top-k', type=int, default=15)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--scene-indices', default='', help='Comma-separated scene indices to use')
    args = ap.parse_args()

    print(f'Loading {args.pkl}...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']
    scenes = data['scenes']
    print(f'  loaded {len(scenes)} scenes, {len(class_names)} classes')

    # Choose which scenes to visualize
    if args.scene_indices:
        chosen = [int(x) for x in args.scene_indices.split(',')]
        chosen = [i for i in chosen if 0 <= i < len(scenes)]
    else:
        np.random.seed(args.seed)
        # Prefer scenes that have at least 3 GT objects and at least 1 prediction above threshold
        good = [i for i, s in enumerate(scenes)
                if len(s['groundtruths']) >= 3 and
                any(p['score'] >= args.score_threshold for p in s['predictions'])]
        if len(good) < args.n_scenes:
            good = list(range(len(scenes)))
        chosen = list(np.random.choice(good, args.n_scenes, replace=False))
    print(f'  chosen scene indices: {chosen}')

    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 3D view ===
    fig = plt.figure(figsize=(20, 14))
    for plot_idx, scene_idx in enumerate(chosen):
        s = scenes[scene_idx]
        pc = s['point_cloud'][:, :3]
        preds = sorted([p for p in s['predictions'] if p['score'] >= args.score_threshold],
                       key=lambda p: -p['score'])[:args.top_k]
        gts = s['groundtruths']

        ax = fig.add_subplot(2, 3, plot_idx + 1, projection='3d')
        bg = np.random.choice(len(pc), min(2500, len(pc)), replace=False)
        ax.scatter(pc[bg, 0], pc[bg, 1], pc[bg, 2], c='lightgray', s=0.4, alpha=0.5)

        for g in gts:
            corners = to_corners(g['box'])
            draw_box_3d(ax, corners, color='black', linewidth=2.0, linestyle='--', alpha=0.6)
            ax.text(corners[:, 0].mean(), corners[:, 1].mean(), corners[:, 2].max() + 0.05,
                    f'GT:{class_names[g["class_id"]]}', fontsize=6, color='black', alpha=0.7)

        for p in preds:
            corners = to_corners(p['box'])
            c = colors[p['class_id']]
            draw_box_3d(ax, corners, color=c, linewidth=1.8, alpha=0.95)
            ax.text(corners[:, 0].mean(), corners[:, 1].mean(), corners[:, 2].max() + 0.15,
                    f'{class_names[p["class_id"]]} {p["score"]:.2f}',
                    fontsize=7, color=c, weight='bold')

        ax.set_title(f"Scene {s['scan_idx']}  |  {len(gts)} GT, {len(preds)} pred (≥{args.score_threshold})",
                     fontsize=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=-60)

    handles = [Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Ground Truth'),
               Line2D([0], [0], color='steelblue', linewidth=2, label='Prediction (colored by class)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle(f'Phase 8 — VoteNet 27-class detections (score >= {args.score_threshold})',
                 fontsize=14, y=1.0)
    plt.tight_layout()
    out_3d = out_dir / 'phase8_detections_3d.png'
    plt.savefig(out_3d, dpi=130, bbox_inches='tight')
    print(f'Saved {out_3d}')
    plt.close(fig)

    # === Top-down view ===
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    for plot_idx, scene_idx in enumerate(chosen):
        ax = axes.flat[plot_idx]
        s = scenes[scene_idx]
        pc = s['point_cloud'][:, :3]
        preds = sorted([p for p in s['predictions'] if p['score'] >= args.score_threshold],
                       key=lambda p: -p['score'])[:args.top_k]
        gts = s['groundtruths']

        ax.scatter(pc[:, 0], pc[:, 1], c='lightgray', s=0.5, alpha=0.5)

        for g in gts:
            corners = to_corners(g['box'])
            bottom = corners[:4]
            order = [0, 1, 2, 3, 0]
            ax.plot(bottom[order, 0], bottom[order, 1], 'k--', linewidth=1.5, alpha=0.7)

        for p in preds:
            corners = to_corners(p['box'])
            bottom = corners[:4]
            order = [0, 1, 2, 3, 0]
            c = colors[p['class_id']]
            ax.plot(bottom[order, 0], bottom[order, 1], color=c, linewidth=2, alpha=0.95)
            ax.text(bottom[:, 0].mean(), bottom[:, 1].mean(),
                    f'{class_names[p["class_id"]]}\n{p["score"]:.2f}',
                    fontsize=7, color=c, ha='center', va='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

        ax.set_title(f"Scene {s['scan_idx']}  ({len(gts)} GT, {len(preds)} pred)")
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Top-down view — predicted (color) vs GT (black dashed)', fontsize=14)
    plt.tight_layout()
    out_top = out_dir / 'phase8_detections_topdown.png'
    plt.savefig(out_top, dpi=130, bbox_inches='tight')
    print(f'Saved {out_top}')
    plt.close(fig2)

    # Per-class quick summary printed to console
    print('\n=== Per-class prediction count summary (across all val scenes) ===')
    counts_pred = {c: 0 for c in class_names}
    counts_gt = {c: 0 for c in class_names}
    for s in scenes:
        for p in s['predictions']:
            if p['score'] >= args.score_threshold:
                counts_pred[class_names[p['class_id']]] += 1
        for g in s['groundtruths']:
            counts_gt[class_names[g['class_id']]] += 1
    print(f'{"class":18s} {"gt":>7s} {"pred":>7s}')
    for c in class_names:
        print(f'  {c:18s} {counts_gt[c]:>7d} {counts_pred[c]:>7d}')


if __name__ == '__main__':
    main()
