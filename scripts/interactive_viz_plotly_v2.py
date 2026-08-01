#!/usr/bin/env python3
"""
Improved Plotly interactive 3D viewer for Phase 8 VoteNet predictions.

Predictions are color-coded by CORRECTNESS, not by class:
  GREEN solid  = correct prediction (right class, IoU >= match_iou with a GT)
  ORANGE solid = wrong class (right location, predicted wrong class)
  RED solid    = false positive (no nearby GT)
  BLACK dashed = ground truth that was correctly found
  YELLOW dashed = ground truth that was MISSED (no correct prediction)

This way every box you see has obvious meaning.

Usage:
    python scripts/interactive_viz_plotly_v2.py --scene 778
    python scripts/interactive_viz_plotly_v2.py --scene 778 --score-threshold 0.6
    python scripts/interactive_viz_plotly_v2.py --scene 778 --match-iou 0.25 --top-k 10
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


PKL_PATH = Path('/Users/dosvatsky/3D Object Detection/checkpoints/val_predictions.pkl')

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


def aabb_from_corners(corners):
    return corners.min(axis=0), corners.max(axis=0)


def aabb_iou(c1, c2):
    """3D axis-aligned IoU between two boxes given by their 8 corners."""
    a_min, a_max = aabb_from_corners(c1)
    b_min, b_max = aabb_from_corners(c2)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.clip(inter_max - inter_min, 0, None)
    inter = inter_dims.prod()
    vol_a = (a_max - a_min).prod()
    vol_b = (b_max - b_min).prod()
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_predictions_to_gt(preds, gts, match_iou=0.25):
    """
    Returns:
      pred_status: list of 'correct' / 'wrong_class' / 'fp' per prediction
      gt_status:   list of 'found' / 'missed' per GT
    Each prediction is greedily matched to the GT with highest IoU above threshold.
    """
    pred_corners = [to_corners(p['box']) for p in preds]
    gt_corners = [to_corners(g['box']) for g in gts]

    pred_status = [None] * len(preds)
    gt_status = ['missed'] * len(gts)
    used_gt = set()

    # Match each pred to its best-IoU GT
    iou_matrix = np.zeros((len(preds), len(gts)))
    for i, pc in enumerate(pred_corners):
        for j, gc in enumerate(gt_corners):
            iou_matrix[i, j] = aabb_iou(pc, gc)

    # Greedy: sort (pred, gt) pairs by IoU desc, assign if both still free
    pairs = []
    for i in range(len(preds)):
        for j in range(len(gts)):
            if iou_matrix[i, j] >= match_iou:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    matched_pred = {}
    for iou, i, j in pairs:
        if i in matched_pred or j in used_gt:
            continue
        matched_pred[i] = j
        used_gt.add(j)
        # Mark GT as found regardless of class match
        if preds[i]['class_id'] == gts[j]['class_id']:
            pred_status[i] = 'correct'
            gt_status[j] = 'found'
        else:
            pred_status[i] = 'wrong_class'
            # GT class was wrong, so it's still considered missed
            gt_status[j] = 'missed_wrong_class'

    for i, st in enumerate(pred_status):
        if st is None:
            pred_status[i] = 'fp'

    return pred_status, gt_status


def box_traces(corners, color, name, width=5, dash=None, opacity=1.0):
    xs, ys, zs = [], [], []
    for i, j in BOX_EDGES:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        opacity=opacity,
        showlegend=False,
    )


_DATA_CACHE = {}

def load_data(pkl_path=PKL_PATH):
    pkl_path = str(pkl_path)
    if pkl_path not in _DATA_CACHE:
        with open(pkl_path, 'rb') as f:
            _DATA_CACHE[pkl_path] = pickle.load(f)
    return _DATA_CACHE[pkl_path]


PRED_COLOR = {
    'correct':     ('rgb(20, 180, 60)',  '✅ correct'),
    'wrong_class': ('rgb(255, 140, 0)',  '⚠️ wrong class'),
    'fp':          ('rgb(220, 30, 30)',  '❌ false positive'),
}
GT_COLOR = {
    'found':              ('rgb(10, 10, 10)',  '— ground truth (found)'),
    'missed':             ('rgb(240, 210, 0)', '— ground truth (MISSED)'),
    'missed_wrong_class': ('rgb(240, 210, 0)', '— ground truth (wrong class)'),
}


def show_scene(scan_idx, score_threshold=0.50, match_iou=0.25, top_k=20,
               save_html=None, view_preset='oblique', pkl_path=None):
    """Render scene with correctness-colored boxes."""
    data = load_data(pkl_path) if pkl_path else load_data()
    class_names = data['class_names']
    scenes = data['scenes']
    match = [s for s in scenes if s['scan_idx'] == scan_idx]
    if not match:
        raise ValueError(f'No scene with scan_idx={scan_idx}')
    scene = match[0]

    pc = scene['point_cloud'][:, :3]
    preds_all = sorted(scene['predictions'], key=lambda p: -p['score'])
    preds = [p for p in preds_all if p['score'] >= score_threshold][:top_k]
    gts = scene['groundtruths']

    # Match predictions to GTs
    pred_status, gt_status = match_predictions_to_gt(preds, gts, match_iou=match_iou)

    # Subsample pc for speed
    if len(pc) > 4000:
        idx = np.random.default_rng(0).choice(len(pc), 4000, replace=False)
        pc = pc[idx]

    traces = []

    # Point cloud
    traces.append(go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.45),
        name='point cloud',
        showlegend=False,
        hoverinfo='skip',
    ))

    # GT boxes
    for g, st in zip(gts, gt_status):
        corners = to_corners(g['box'])
        color, _ = GT_COLOR[st]
        traces.append(box_traces(
            corners, color,
            name=f'GT:{class_names[g["class_id"]]} ({st})',
            width=5, dash='dash', opacity=0.95 if st != 'found' else 0.55,
        ))
        center = corners.mean(axis=0)
        z_top = corners[:, 2].max()
        label = class_names[g['class_id']]
        if st == 'missed':
            label = f'MISSED: {label}'
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[z_top + 0.08],
            mode='text',
            text=[label],
            textfont=dict(color=color, size=11),
            showlegend=False, hoverinfo='skip',
        ))

    # Predicted boxes
    for p, st in zip(preds, pred_status):
        corners = to_corners(p['box'])
        color, _ = PRED_COLOR[st]
        traces.append(box_traces(
            corners, color,
            name=f'{class_names[p["class_id"]]} ({st}) {p["score"]:.2f}',
            width=6,
        ))
        center = corners.mean(axis=0)
        z_top = corners[:, 2].max()
        label = f'{class_names[p["class_id"]]} {p["score"]:.2f}'
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[z_top + 0.20],
            mode='text',
            text=[label],
            textfont=dict(color=color, size=11, family='Arial Black'),
            showlegend=False, hoverinfo='skip',
        ))

    # Legend traces (invisible boxes, just for the side legend)
    for status, (color, label) in PRED_COLOR.items():
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines', line=dict(color=color, width=6),
            name=label, showlegend=True,
        ))
    for status, (color, label) in list(GT_COLOR.items())[:2]:  # only "found" + "missed"
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines', line=dict(color=color, width=5, dash='dash'),
            name=label, showlegend=True,
        ))

    # Camera preset
    if view_preset == 'topdown':
        camera = dict(eye=dict(x=0, y=0, z=2.2),
                      up=dict(x=0, y=1, z=0),
                      center=dict(x=0, y=0, z=0))
    else:   # oblique 45-degree
        camera = dict(eye=dict(x=1.6, y=-1.6, z=1.2),
                      up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0))

    # Counts for title
    n_correct  = sum(1 for s in pred_status if s == 'correct')
    n_wrong    = sum(1 for s in pred_status if s == 'wrong_class')
    n_fp       = sum(1 for s in pred_status if s == 'fp')
    n_missed   = sum(1 for s in gt_status   if s in {'missed', 'missed_wrong_class'})

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f'Scene {scan_idx}  |  '
               f'<span style="color:rgb(20,180,60)">✅ {n_correct} correct</span>  '
               f'<span style="color:rgb(255,140,0)">⚠️ {n_wrong} wrong-class</span>  '
               f'<span style="color:rgb(220,30,30)">❌ {n_fp} false-positives</span>  '
               f'<span style="color:rgb(240,210,0)">⏷ {n_missed} missed GTs</span>  '
               f'(score ≥ {score_threshold}, top {top_k})'),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=camera,
        ),
        legend=dict(x=0.85, y=0.95, font=dict(size=12)),
        margin=dict(l=0, r=0, t=60, b=0),
        height=850,
    )

    if save_html:
        fig.write_html(save_html)
        print(f'Saved {save_html}')

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=str(PKL_PATH),
                    help='Path to predictions pkl (default: val_predictions.pkl)')
    ap.add_argument('--scene', type=int, default=778)
    ap.add_argument('--score-threshold', type=float, default=0.50,
                    help='Only show predictions with confidence >= this (default 0.50)')
    ap.add_argument('--match-iou', type=float, default=0.25,
                    help='IoU threshold for matching pred to GT (default 0.25)')
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--view', choices=['oblique', 'topdown'], default='oblique')
    ap.add_argument('--out', default='/Users/dosvatsky/3D Object Detection/phase8_scene_viewer.html')
    args = ap.parse_args()

    show_scene(args.scene, score_threshold=args.score_threshold,
               match_iou=args.match_iou, top_k=args.top_k,
               save_html=args.out, view_preset=args.view,
               pkl_path=args.pkl)
    print(f'\nOpen in browser: {args.out}')


if __name__ == '__main__':
    main()
