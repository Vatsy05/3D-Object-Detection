#!/usr/bin/env python3
"""
Phase 9 — Synthetic scene generator v3 (60 classes, small-object aware)

Inherits from v2 (Phase 8.4): 40,000 points, natural-extent bboxes
(size = pre-rotation extents, heading = yaw).

New in v3:
  1. 60-class taxonomy imported from class_registry_v2.py (flat folder layout,
     all .glb, data/mesh_dataset_v1/<class>/*.glb)
  2. Area-weighted point sampling with a 300-point minimum per object
     -> small objects are never point-starved
  3. Class-balanced composition: inverse-frequency class picking, and every
     second placement slot in room scenes is forced to a small class
  4. Two scene types:
       ROOM (70%):     floor layout as before, 8-14 objects
       TABLETOP (30%): a table/desk with 4-8 small military items ON its
                       surface + 1-3 large floor objects
  5. Objects get ~50% of the point budget

Output format is IDENTICAL to v1/v2 (pc/bbox/votes files), so the Phase 8
dataset class works unchanged apart from num_points=40000.

Run on the Mac M2:
    cd "/Users/dosvatsky/3D Object Detection"
    python scripts/generate_synthetic_dataset_v3.py --n-train 8000 --n-val 1500

Resumes automatically. Bounded memory (~2-3 GB).
"""

import argparse
import gc
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import trimesh
import psutil

sys.path.insert(0, str(Path(__file__).parent))
from class_registry_v2 import (
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, CLASS_REAL_SIZE,
    FURNITURE_CLASSES, MILITARY_CLASSES, FLAT_CLASSES, TALL_CLASSES,
    SMALL_CLASSES,
)

N_POINTS = 40000            # total points per scene
OBJ_POINT_SHARE = 0.5       # fraction of budget for object surfaces
MIN_PTS_PER_OBJ = 300       # hard floor so small objects are never starved
TABLETOP_FRACTION = 0.30    # fraction of scenes that are tabletop layouts

# classes small enough to sit on a table (longest dim <= 0.65 m)
TABLE_ITEMS = sorted(c for c in MILITARY_CLASSES if CLASS_REAL_SIZE[c] <= 0.65)
TABLE_SURFACES = ['table', 'desk']

# ============================================
# Bounded LRU mesh cache
# ============================================
MESH_CACHE_MAX = 40
_MESH_CACHE = OrderedDict()


def cache_get(key):
    if key in _MESH_CACHE:
        _MESH_CACHE.move_to_end(key)
        return _MESH_CACHE[key]
    return None


def cache_put(key, value):
    _MESH_CACHE[key] = value
    _MESH_CACHE.move_to_end(key)
    while len(_MESH_CACHE) > MESH_CACHE_MAX:
        _MESH_CACHE.popitem(last=False)


# ============================================
# Mesh registry — FLAT layout: data_root/<class>/*.glb
# ============================================
def build_mesh_registry(data_root):
    registry = {}
    for c in CLASS_NAMES:
        folder = data_root / c
        registry[c] = sorted(folder.glob('*.glb')) if folder.exists() else []
    return registry


# ============================================
# Mesh loading + orientation
# ============================================
def auto_orient_mesh(m, class_name):
    ext = m.extents.copy()
    longest_axis = int(np.argmax(ext))
    if class_name in FLAT_CLASSES:
        if longest_axis == 1:
            R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            m.apply_transform(R)
    elif class_name in TALL_CLASSES:
        if longest_axis == 0:
            R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
            m.apply_transform(R)
        elif longest_axis == 2:
            R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            m.apply_transform(R)
    return m


def load_normalized_mesh(path, class_name, decimate_threshold=150_000):
    cache_key = (str(path), class_name)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    m = trimesh.load(str(path), force='mesh', skip_materials=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))

    try:
        m.visual = trimesh.visual.ColorVisuals(mesh=m)
    except Exception:
        pass

    if len(m.vertices) > decimate_threshold:
        try:
            m = m.simplify_quadric_decimation(decimate_threshold // 2)
        except Exception:
            pass

    m = auto_orient_mesh(m, class_name)
    m.apply_translation(-m.centroid)

    target = CLASS_REAL_SIZE[class_name]
    longest = float(m.extents.max())
    if longest > 1e-6:
        m.apply_scale(target / longest)

    min_y = float(m.bounds[0, 1])
    m.apply_translation([0, -min_y, 0])

    cache_put(cache_key, m.copy())
    return m


# ============================================
# Class-balanced sampling
# ============================================
GLOBAL_CLASS_COUNTS = Counter()


def pick_class(candidates):
    """Inverse-frequency weighted class choice -> long-run balance."""
    weights = np.array([1.0 / (1.0 + GLOBAL_CLASS_COUNTS[c]) for c in candidates])
    weights /= weights.sum()
    return np.random.choice(candidates, p=weights)


# ============================================
# Placement helpers
# ============================================
def aabb_overlap_2d(a, b, buffer=0.05):
    return not (a[2] + buffer < b[0] or b[2] + buffer < a[0] or
                a[3] + buffer < b[1] or b[3] + buffer < a[1])


def try_place(base_mesh, class_name, region, floor_y, placed_aabbs,
              max_attempts=20, buffer=0.05):
    """Random yaw + position inside region=(x0,z0,x1,z1) at height floor_y.
    bbox size = NATURAL extents of base_mesh (pre-rotation) — the v2 fix.
    Returns (mesh, bbox_tuple, aabb) or None."""
    natural_size = base_mesh.extents.astype(np.float32)  # pre-yaw extents
    for _ in range(max_attempts):
        yaw = np.random.uniform(0, 2 * np.pi)
        mesh_try = base_mesh.copy()
        R = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
        mesh_try.apply_transform(R)
        mesh_try.apply_translation([0, floor_y - float(mesh_try.bounds[0, 1]), 0])

        half_x = (mesh_try.bounds[1, 0] - mesh_try.bounds[0, 0]) / 2
        half_z = (mesh_try.bounds[1, 2] - mesh_try.bounds[0, 2]) / 2
        x0, z0, x1, z1 = region
        if x0 + half_x >= x1 - half_x or z0 + half_z >= z1 - half_z:
            return None  # too big for region

        cx = np.random.uniform(x0 + half_x, x1 - half_x)
        cz = np.random.uniform(z0 + half_z, z1 - half_z)
        mesh_try.apply_translation([cx, 0, cz])

        aabb = (mesh_try.bounds[0, 0], mesh_try.bounds[0, 2],
                mesh_try.bounds[1, 0], mesh_try.bounds[1, 2])
        if any(aabb_overlap_2d(aabb, e, buffer) for e in placed_aabbs):
            continue

        center = mesh_try.bounds.mean(axis=0).astype(np.float32)
        bbox = (center[0], center[1], center[2],
                natural_size[0], natural_size[1], natural_size[2],
                yaw, CLASS_TO_IDX[class_name])
        return mesh_try, bbox, aabb
    return None


def sample_room_background(room_x, room_y, room_z, n_floor, n_walls, n_ceiling):
    floor = np.column_stack([
        np.random.uniform(0, room_x, n_floor),
        np.zeros(n_floor),
        np.random.uniform(0, room_z, n_floor),
    ])
    nw = max(1, n_walls // 4)
    w1 = np.column_stack([np.zeros(nw), np.random.uniform(0, room_y, nw), np.random.uniform(0, room_z, nw)])
    w2 = np.column_stack([np.full(nw, room_x), np.random.uniform(0, room_y, nw), np.random.uniform(0, room_z, nw)])
    w3 = np.column_stack([np.random.uniform(0, room_x, nw), np.random.uniform(0, room_y, nw), np.zeros(nw)])
    w4 = np.column_stack([np.random.uniform(0, room_x, nw), np.random.uniform(0, room_y, nw), np.full(nw, room_z)])
    ceiling = np.column_stack([
        np.random.uniform(0, room_x, n_ceiling),
        np.full(n_ceiling, room_y),
        np.random.uniform(0, room_z, n_ceiling),
    ])
    return np.concatenate([floor, w1, w2, w3, w4, ceiling], axis=0)


# ============================================
# Point sampling: area-weighted with per-object floor
# ============================================
def sample_object_points(placed_meshes, budget):
    """Distribute `budget` points across objects proportional to sqrt(area),
    never below MIN_PTS_PER_OBJ per object."""
    areas = np.array([max(float(m.area), 1e-6) for m in placed_meshes])
    w = np.sqrt(areas)
    n_alloc = np.maximum(MIN_PTS_PER_OBJ,
                         (budget * w / w.sum()).astype(int))
    if n_alloc.sum() > budget:
        n_alloc = np.maximum(MIN_PTS_PER_OBJ,
                             (n_alloc * budget / n_alloc.sum()).astype(int))
    pts_list, assign = [], []
    for i, (m, n) in enumerate(zip(placed_meshes, n_alloc)):
        try:
            pts, _ = trimesh.sample.sample_surface(m, int(n))
            pts_list.append(pts.astype(np.float32))
            assign.extend([i] * len(pts))
        except Exception:
            continue
    if not pts_list:
        return None, None
    return np.concatenate(pts_list, axis=0), np.array(assign, dtype=np.int32)


# ============================================
# Scene builders
# ============================================
def _finalize_scene(placed_meshes, placed_bboxes, room_dims):
    room_x, room_y, room_z = room_dims
    obj_budget = int(N_POINTS * OBJ_POINT_SHARE)
    object_pts, object_assignment = sample_object_points(placed_meshes, obj_budget)
    if object_pts is None:
        return None

    n_bg = max(0, N_POINTS - len(object_pts))
    bg_pts = sample_room_background(room_x, room_y, room_z,
                                    n_floor=int(n_bg * 0.5),
                                    n_walls=int(n_bg * 0.4),
                                    n_ceiling=int(n_bg * 0.1)).astype(np.float32)

    all_pts = np.concatenate([object_pts, bg_pts], axis=0)
    rgb = np.zeros_like(all_pts)
    pc_full = np.concatenate([all_pts, rgb], axis=1).astype(np.float32)

    is_fg_orig = np.concatenate([
        np.ones(len(object_pts), dtype=bool),
        np.zeros(len(bg_pts), dtype=bool),
    ])
    assignment_orig = np.full(len(all_pts), -1, dtype=np.int32)
    assignment_orig[:len(object_pts)] = object_assignment

    N = len(pc_full)
    if N >= N_POINTS:
        idx = np.random.choice(N, N_POINTS, replace=False)
    else:
        extra = np.random.choice(N, N_POINTS - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
    pc_full = pc_full[idx]
    is_fg = is_fg_orig[idx]
    assignment = assignment_orig[idx]

    centers = np.array([b[:3] for b in placed_bboxes], dtype=np.float32)
    votes = np.zeros((len(pc_full), 10), dtype=np.float32)
    votes[:, 0] = is_fg.astype(np.float32)
    if is_fg.sum() > 0:
        fg_assign = np.clip(assignment[is_fg], 0, len(centers) - 1)
        offsets = centers[fg_assign] - pc_full[is_fg, :3]
        votes[is_fg, 1:4] = offsets
        votes[is_fg, 4:7] = offsets
        votes[is_fg, 7:10] = offsets

    return {
        'pc': pc_full,
        'bbox': np.array(placed_bboxes, dtype=np.float32),
        'votes': votes,
        '_room_dims': room_dims,
    }


def generate_room_scene(mesh_registry, n_objects_range=(8, 14)):
    """ROOM scene: floor layout, class-balanced, small-class boosted."""
    room_x = np.random.uniform(4.0, 6.0)
    room_y = np.random.uniform(2.5, 3.0)
    room_z = np.random.uniform(4.0, 6.0)

    n_objects = np.random.randint(*n_objects_range)
    available = [c for c, p in mesh_registry.items() if len(p) > 0]
    small_avail = [c for c in available if c in SMALL_CLASSES]

    placed_meshes, placed_aabbs, placed_bboxes = [], [], []
    for k in range(n_objects):
        # every other slot is forced to a small class (they need the exposure)
        if small_avail and k % 2 == 1:
            class_name = pick_class(small_avail)
        else:
            class_name = pick_class(available)
        mesh_path = np.random.choice(mesh_registry[class_name])
        try:
            base = load_normalized_mesh(mesh_path, class_name)
        except Exception:
            continue
        res = try_place(base, class_name, (0, 0, room_x, room_z), 0.0, placed_aabbs)
        if res is None:
            continue
        mesh_try, bbox, aabb = res
        placed_meshes.append(mesh_try)
        placed_bboxes.append(bbox)
        placed_aabbs.append(aabb)
        GLOBAL_CLASS_COUNTS[class_name] += 1

    if not placed_meshes:
        return None
    return _finalize_scene(placed_meshes, placed_bboxes, (room_x, room_y, room_z))


def generate_tabletop_scene(mesh_registry, n_items_range=(4, 8)):
    """TABLETOP scene: a table/desk with small military items on its surface,
    plus 1-3 large objects on the floor for context."""
    room_x = np.random.uniform(3.5, 5.0)
    room_y = np.random.uniform(2.5, 3.0)
    room_z = np.random.uniform(3.5, 5.0)

    placed_meshes, placed_aabbs, placed_bboxes = [], [], []

    # 1) the table
    surf_class = np.random.choice(TABLE_SURFACES)
    if not mesh_registry.get(surf_class):
        return None
    table_path = np.random.choice(mesh_registry[surf_class])
    try:
        table = load_normalized_mesh(table_path, surf_class)
    except Exception:
        return None
    res = try_place(table, surf_class, (0.5, 0.5, room_x - 0.5, room_z - 0.5),
                    0.0, placed_aabbs)
    if res is None:
        return None
    table_mesh, table_bbox, table_aabb = res
    placed_meshes.append(table_mesh)
    placed_bboxes.append(table_bbox)
    placed_aabbs.append(table_aabb)
    GLOBAL_CLASS_COUNTS[surf_class] += 1

    # tabletop region: table AABB inset by 10 cm, at the table's top height
    top_y = float(table_mesh.bounds[1, 1])
    tx0, tz0, tx1, tz1 = table_aabb
    region = (tx0 + 0.10, tz0 + 0.10, tx1 - 0.10, tz1 - 0.10)

    # 2) small items ON the table (separate collision set from the floor)
    tabletop_aabbs = []
    avail_items = [c for c in TABLE_ITEMS if mesh_registry.get(c)]
    n_items = np.random.randint(*n_items_range)
    for _ in range(n_items):
        class_name = pick_class(avail_items)
        mesh_path = np.random.choice(mesh_registry[class_name])
        try:
            base = load_normalized_mesh(mesh_path, class_name)
        except Exception:
            continue
        res = try_place(base, class_name, region, top_y, tabletop_aabbs, buffer=0.02)
        if res is None:
            continue
        mesh_try, bbox, aabb = res
        placed_meshes.append(mesh_try)
        placed_bboxes.append(bbox)
        tabletop_aabbs.append(aabb)
        GLOBAL_CLASS_COUNTS[class_name] += 1

    if len(placed_meshes) < 3:   # need the table + at least 2 items
        return None

    # 3) 1-3 large objects on the floor for context
    large = [c for c, p in mesh_registry.items()
             if p and CLASS_REAL_SIZE[c] >= 0.8 and c not in TABLE_SURFACES]
    for _ in range(np.random.randint(1, 4)):
        class_name = pick_class(large)
        mesh_path = np.random.choice(mesh_registry[class_name])
        try:
            base = load_normalized_mesh(mesh_path, class_name)
        except Exception:
            continue
        res = try_place(base, class_name, (0, 0, room_x, room_z), 0.0, placed_aabbs)
        if res is None:
            continue
        mesh_try, bbox, aabb = res
        placed_meshes.append(mesh_try)
        placed_bboxes.append(bbox)
        placed_aabbs.append(aabb)
        GLOBAL_CLASS_COUNTS[class_name] += 1

    return _finalize_scene(placed_meshes, placed_bboxes, (room_x, room_y, room_z))


def generate_scene(mesh_registry):
    if np.random.rand() < TABLETOP_FRACTION:
        return generate_tabletop_scene(mesh_registry)
    return generate_room_scene(mesh_registry)


# ============================================
# Saving + main loop
# ============================================
def save_scene(scene, scene_idx, out_dir):
    sid = f'{scene_idx:06d}'
    np.savez_compressed(out_dir / f'{sid}_pc.npz', pc=scene['pc'])
    np.save(out_dir / f'{sid}_bbox.npy', scene['bbox'])
    np.savez_compressed(out_dir / f'{sid}_votes.npz', votes=scene['votes'])


def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def fill_split(split_name, out_dir, n_target, mesh_registry, gc_every=50):
    print(f'\n=== {split_name.upper()} ({n_target} scenes) ===')
    out_dir.mkdir(parents=True, exist_ok=True)
    class_counter = Counter()
    failed = 0
    t_start = time.time()
    last_print = t_start

    for i in range(n_target):
        scene_file = out_dir / f'{i:06d}_pc.npz'
        if scene_file.exists():
            bbox_file = out_dir / f'{i:06d}_bbox.npy'
            if bbox_file.exists():
                try:
                    for box in np.load(bbox_file):
                        cname = CLASS_NAMES[int(box[7])]
                        class_counter[cname] += 1
                        GLOBAL_CLASS_COUNTS[cname] += 1
                except Exception:
                    pass
            continue

        attempts, scene = 0, None
        while scene is None and attempts < 5:
            scene = generate_scene(mesh_registry)
            attempts += 1
        if scene is None:
            failed += 1
            continue

        save_scene(scene, i, out_dir)
        for box in scene['bbox']:
            class_counter[CLASS_NAMES[int(box[7])]] += 1

        del scene
        if (i + 1) % gc_every == 0:
            gc.collect()

        now = time.time()
        if (i + 1) % 100 == 0 or now - last_print > 10:
            elapsed = now - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (n_target - i - 1) / rate / 60 if rate > 0 else 0
            print(f'  {i + 1:5d}/{n_target}  {rate:.1f} scenes/s  '
                  f'mem={memory_mb():.0f} MB  ETA {eta_min:.1f} min', flush=True)
            last_print = now

    print(f'  done: {n_target} scenes in {(time.time() - t_start) / 60:.1f} min '
          f'({failed} failed)')
    return class_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/Users/dosvatsky/3D Object Detection/data/mesh_dataset_v1')
    parser.add_argument('--out-root', default='/Users/dosvatsky/3D Object Detection/data/synthetic_v2_60class')
    parser.add_argument('--n-train', type=int, default=8000)
    parser.add_argument('--n-val', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print('Building mesh registry (flat layout)...')
    registry = build_mesh_registry(data_root)
    total = 0
    for c in CLASS_NAMES:
        n = len(registry[c])
        total += n
        if n == 0:
            print(f'  WARNING: no meshes for {c}')
    print(f'  total: {total} meshes across {NUM_CLASSES} classes')
    print(f'  tabletop-eligible items: {len(TABLE_ITEMS)}')

    counter_train = fill_split('train', out_root / 'train', args.n_train, registry)
    counter_val = fill_split('val', out_root / 'val', args.n_val, registry)

    combined = counter_train + counter_val
    total_scenes = args.n_train + args.n_val
    print('\n' + '=' * 60)
    print('DATASET STATISTICS')
    print('=' * 60)
    print(f'Total scenes: {total_scenes}   Total objects: {sum(combined.values())}')
    print(f'\n{"Class":22s} {"count":>7s}  {"avg/scene":>10s}')
    print('-' * 50)
    for cls in CLASS_NAMES:
        cnt = combined[cls]
        print(f'{cls:22s} {cnt:>7d}  {cnt / total_scenes:>9.2f}')

    with open(out_root / 'classes.txt', 'w') as f:
        for i, c in enumerate(CLASS_NAMES):
            f.write(f'{i}\t{c}\n')
    with open(out_root / 'train_idx.txt', 'w') as f:
        f.writelines(f'{i:06d}\n' for i in range(args.n_train))
    with open(out_root / 'val_idx.txt', 'w') as f:
        f.writelines(f'{i:06d}\n' for i in range(args.n_val))
    print(f'\nWrote index files to {out_root}')
    print('Done.')


if __name__ == '__main__':
    main()
