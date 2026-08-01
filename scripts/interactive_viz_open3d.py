#!/usr/bin/env python3
"""
Interactive Open3D viewer for Phase 8 VoteNet predictions.

Usage:
    python scripts/interactive_viz_open3d.py                 # random scene
    python scripts/interactive_viz_open3d.py --scene 778     # specific scene
    python scripts/interactive_viz_open3d.py --list          # list scenes with most objects

Controls inside the viewer window:
    Left-click + drag : rotate
    Right-click + drag: pan
    Mouse wheel       : zoom
    Q or Esc          : quit
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import open3d as o3d


PKL_PATH = Path('/Users/dosvatsky/3D Object Detection/checkpoints/val_predictions.pkl')

# 12 cube edges connecting 8 corners
BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
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
    box = np.asarray(box)
    if box.shape == (8, 3):
        return box
    return compute_box_corners(box[:3], box[3:6], box[6])


def make_lineset(corners, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    ls.colors = o3d.utility.Vector3dVector([color for _ in BOX_EDGES])
    return ls


def make_text_at(text, pos, color=(0, 0, 0)):
    """Returns a small sphere as a 'label marker' since Open3D's text rendering is limited."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.translate(pos)
    sphere.paint_uniform_color(color)
    return sphere


def visualize_scene(scene, class_names, score_threshold=0.30, top_k=15):
    pc = scene['point_cloud'][:, :3]
    preds = sorted(scene['predictions'], key=lambda p: -p['score'])
    preds = [p for p in preds if p['score'] >= score_threshold][:top_k]
    gts = scene['groundtruths']

    geoms = []

    # Point cloud (gray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd)

    # Color palette per class
    rng = np.random.default_rng(42)
    palette = rng.random((len(class_names), 3))

    # GT boxes — dark with thinner appearance (we simulate by using a darker, slightly transparent color)
    for g in gts:
        corners = to_corners(g['box'])
        geoms.append(make_lineset(corners, [0.0, 0.0, 0.0]))   # black

    # Predicted boxes — class-colored
    for p in preds:
        corners = to_corners(p['box'])
        c = palette[p['class_id']].tolist()
        geoms.append(make_lineset(corners, c))

    # Coordinate frame at origin
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    # === Print legend to console (Open3D doesn't render text well) ===
    print(f"\n=== Scene {scene['scan_idx']} ===")
    print(f'GROUND TRUTH ({len(gts)} objects, shown as BLACK boxes):')
    for g in gts:
        print(f'  {class_names[g["class_id"]]}')
    print(f'\nPREDICTIONS (top {len(preds)} above {score_threshold}, shown as COLORED boxes):')
    for p in preds:
        print(f'  {class_names[p["class_id"]]:18s} score={p["score"]:.2f}')

    # === Show ===
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f'Phase 8 — Scene {scene["scan_idx"]}  |  black=GT, colored=pred',
        width=1400, height=900,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=str(PKL_PATH))
    ap.add_argument('--scene', type=int, default=None,
                    help='Specific scan_idx to visualize')
    ap.add_argument('--score-threshold', type=float, default=0.30)
    ap.add_argument('--top-k', type=int, default=15)
    ap.add_argument('--list', action='store_true',
                    help='List scenes ranked by number of objects (then exit)')
    ap.add_argument('--n-scenes', type=int, default=1,
                    help='Number of random scenes to show (one at a time)')
    args = ap.parse_args()

    print(f'Loading {args.pkl} ...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']
    scenes = data['scenes']
    print(f'  {len(scenes)} scenes, {len(class_names)} classes')

    if args.list:
        ranked = sorted(scenes, key=lambda s: -len(s['groundtruths']))[:30]
        print('\nTop 30 scenes by GT object count:')
        for s in ranked:
            classes = ', '.join(class_names[g['class_id']] for g in s['groundtruths'])
            print(f"  Scene {s['scan_idx']:4d}  {len(s['groundtruths'])} GT  |  {classes}")
        return

    if args.scene is not None:
        match = [s for s in scenes if s['scan_idx'] == args.scene]
        if not match:
            print(f'ERROR: no scene with scan_idx={args.scene}')
            return
        visualize_scene(match[0], class_names,
                        score_threshold=args.score_threshold, top_k=args.top_k)
    else:
        # Show several random scenes one at a time
        np.random.seed(7)
        # Prefer scenes with at least 5 GT objects
        good = [s for s in scenes if len(s['groundtruths']) >= 5]
        chosen = np.random.choice(len(good), args.n_scenes, replace=False)
        for i in chosen:
            visualize_scene(good[i], class_names,
                            score_threshold=args.score_threshold, top_k=args.top_k)


if __name__ == '__main__':
    main()
