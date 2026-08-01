#!/usr/bin/env python3
"""
Plotly interactive 3D viewer for Phase 8 predictions.

Can be run as a script (opens browser tab) OR called from a notebook cell:

    from scripts.interactive_viz_plotly import show_scene
    show_scene(778)
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


def box_traces(corners, color, name, width=4, dash=None):
    """Returns list of plotly Scatter3d traces for a box's 12 edges."""
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
        showlegend=False,
    )


_DATA_CACHE = {}

def load_data(pkl_path=PKL_PATH):
    pkl_path = str(pkl_path)
    if pkl_path not in _DATA_CACHE:
        with open(pkl_path, 'rb') as f:
            _DATA_CACHE[pkl_path] = pickle.load(f)
    return _DATA_CACHE[pkl_path]


def show_scene(scan_idx, score_threshold=0.30, top_k=15, save_html=None):
    """Render scene as an interactive Plotly figure.
    Works in Jupyter (auto-displays) or as standalone HTML."""
    data = load_data()
    class_names = data['class_names']
    scenes = data['scenes']
    match = [s for s in scenes if s['scan_idx'] == scan_idx]
    if not match:
        raise ValueError(f'No scene with scan_idx={scan_idx}')
    scene = match[0]

    pc = scene['point_cloud'][:, :3]
    preds = sorted(scene['predictions'], key=lambda p: -p['score'])
    preds = [p for p in preds if p['score'] >= score_threshold][:top_k]
    gts = scene['groundtruths']

    # Subsample pc for fast browser rendering
    if len(pc) > 5000:
        idx = np.random.default_rng(0).choice(len(pc), 5000, replace=False)
        pc = pc[idx]

    rng = np.random.default_rng(42)
    palette = (rng.random((len(class_names), 3)) * 255).astype(int)

    traces = []

    # Point cloud
    traces.append(go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.5),
        name='points',
        showlegend=False,
        hoverinfo='skip',
    ))

    # GT boxes (black solid, thicker)
    for g in gts:
        corners = to_corners(g['box'])
        traces.append(box_traces(
            corners, 'rgb(20,20,20)',
            name=f'GT:{class_names[g["class_id"]]}',
            width=5,
        ))
        # Label
        center = corners.mean(axis=0)
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[corners[:, 2].max() + 0.05],
            mode='text', text=[f'GT:{class_names[g["class_id"]]}'],
            textfont=dict(color='black', size=10),
            showlegend=False, hoverinfo='skip',
        ))

    # Predicted boxes (colored)
    seen_classes = set()
    for p in preds:
        corners = to_corners(p['box'])
        c = palette[p['class_id']]
        rgb = f'rgb({c[0]},{c[1]},{c[2]})'
        name = class_names[p['class_id']]
        traces.append(box_traces(
            corners, rgb,
            name=name if name not in seen_classes else None,
            width=4,
        ))
        seen_classes.add(name)
        center = corners.mean(axis=0)
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[corners[:, 2].max() + 0.15],
            mode='text',
            text=[f'{name}<br>{p["score"]:.2f}'],
            textfont=dict(color=rgb, size=10),
            showlegend=False, hoverinfo='skip',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Scene {scan_idx}  |  {len(gts)} GT (black), {len(preds)} pred (colored, score >= {score_threshold})',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )

    if save_html:
        fig.write_html(save_html)
        print(f'Saved {save_html}')

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=int, default=778)
    ap.add_argument('--score-threshold', type=float, default=0.30)
    ap.add_argument('--top-k', type=int, default=15)
    ap.add_argument('--out',
                    default='/Users/dosvatsky/3D Object Detection/phase8_scene_viewer.html')
    args = ap.parse_args()

    fig = show_scene(args.scene, score_threshold=args.score_threshold,
                     top_k=args.top_k, save_html=args.out)
    print(f'\nOpen this file in your browser: {args.out}')


if __name__ == '__main__':
    main()
