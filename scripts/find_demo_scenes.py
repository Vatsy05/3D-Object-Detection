#!/usr/bin/env python3
"""
Scan val_predictions.pkl to find the best demo scenes in different categories.

Run:
    python scripts/find_demo_scenes.py
    python scripts/find_demo_scenes.py --score-threshold 0.5
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

PKL_PATH = Path('/Users/dosvatsky/3D Object Detection/checkpoints/val_predictions.pkl')

FURNITURE_CLASSES = {
    'bed', 'table', 'sofa', 'chair', 'toilet',
    'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub',
}
TINY_CLASSES = {
    'combat_knife', 'flashlight', 'hand_grenade', 'magazine',
    'wire_cutter', 'pistol', 'binoculars',
}


def aabb_iou_from_boxes(box1, box2):
    c1 = np.asarray(box1)
    c2 = np.asarray(box2)
    if c1.shape != (8, 3):
        return 0.0
    a_min, a_max = c1.min(0), c1.max(0)
    b_min, b_max = c2.min(0), c2.max(0)
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None).prod()
    vol1 = (a_max - a_min).prod()
    vol2 = (b_max - b_min).prod()
    u = vol1 + vol2 - inter
    return float(inter / u) if u > 0 else 0.0


def score_scene(scene, class_names, score_threshold=0.5, match_iou=0.25):
    preds = [p for p in scene['predictions'] if p['score'] >= score_threshold]
    gts = scene['groundtruths']

    # Greedy IoU matching
    iou_mat = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = aabb_iou_from_boxes(p['box'], g['box'])

    used_p, used_g = set(), set()
    correct = 0
    wrong_class = 0
    pairs = [(iou_mat[i, j], i, j)
             for i in range(len(preds)) for j in range(len(gts))
             if iou_mat[i, j] >= match_iou]
    pairs.sort(reverse=True)
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        if preds[i]['class_id'] == gts[j]['class_id']:
            correct += 1
        else:
            wrong_class += 1

    fp = len(preds) - correct - wrong_class
    missed = len(gts) - correct - wrong_class
    return {
        'scan_idx': scene['scan_idx'],
        'n_gt': len(gts),
        'n_pred': len(preds),
        'correct': correct,
        'wrong_class': wrong_class,
        'fp': fp,
        'missed': missed,
        'classes': [class_names[g['class_id']] for g in gts],
        'unique_classes': len({g['class_id'] for g in gts}),
        'precision': correct / max(1, len(preds)),
        'recall': correct / max(1, len(gts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=str(PKL_PATH))
    ap.add_argument('--score-threshold', type=float, default=0.5)
    ap.add_argument('--match-iou', type=float, default=0.25)
    args = ap.parse_args()

    print(f'Loading {args.pkl}...')
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    class_names = data['class_names']
    scenes = data['scenes']
    print(f'  {len(scenes)} scenes\n')

    print('Scoring scenes...')
    summaries = [score_scene(s, class_names, args.score_threshold, args.match_iou)
                 for s in scenes]

    def show(label, ranked, n=5, fmt=None):
        print(f'\n=== {label} ===')
        for s in ranked[:n]:
            classes = ', '.join(s['classes'])[:90]
            extra = fmt(s) if fmt else ''
            print(f"  Scene {s['scan_idx']:4d}  GT={s['n_gt']:2d}  pred={s['n_pred']:2d}  "
                  f"correct={s['correct']:2d}  missed={s['missed']:2d}  "
                  f"fp={s['fp']:2d}  {extra}")
            print(f"    classes: {classes}")

    # === Best performances (high precision + recall) ===
    ranked_best = sorted(summaries,
                        key=lambda s: -(s['precision'] * s['recall'] * np.log(1 + s['n_gt'])))
    show('BEST performance (high precision×recall, weighted by object count)', ranked_best)

    # === Densest scenes ===
    ranked_dense = sorted(summaries, key=lambda s: -s['n_gt'])
    show('DENSEST scenes (most ground truth objects)', ranked_dense)

    # === Most diverse classes ===
    ranked_diverse = sorted(summaries, key=lambda s: -s['unique_classes'])
    show('MOST DIVERSE (most unique classes)', ranked_diverse)

    # === Worst — most missed objects ===
    ranked_missed = sorted([s for s in summaries if s['n_gt'] >= 5],
                          key=lambda s: -s['missed'])
    show('WORST recall (most missed GT — useful for honest discussion)', ranked_missed)

    # === Furniture-only scenes ===
    def is_all_furniture(s):
        return all(c in FURNITURE_CLASSES for c in s['classes'])
    ranked_furn = sorted([s for s in summaries
                          if is_all_furniture(s) and s['n_gt'] >= 4],
                         key=lambda s: -(s['precision'] * s['recall']))
    show('FURNITURE-only scenes', ranked_furn)

    # === Military-heavy scenes ===
    def military_count(s):
        return sum(1 for c in s['classes'] if c not in FURNITURE_CLASSES)
    ranked_mil = sorted(summaries,
                       key=lambda s: -(military_count(s) - 0.5 * len(s['classes'])))
    show('MILITARY-heavy scenes', ranked_mil)

    # === Scenes with tiny objects ===
    def tiny_count(s):
        return sum(1 for c in s['classes'] if c in TINY_CLASSES)
    ranked_tiny = sorted(summaries,
                       key=lambda s: -tiny_count(s))
    show('TINY-object scenes (knife/grenade/flashlight/etc)', ranked_tiny)


if __name__ == '__main__':
    main()
