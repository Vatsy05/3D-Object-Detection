#!/usr/bin/env python3
"""
Phase 8.2 - Synthetic scene generator (memory-safe standalone script)

Run from terminal:
    cd "/Users/dosvatsky/3D Object Detection"
    python scripts/generate_synthetic_dataset.py --n-train 5000 --n-val 1000

Resumes automatically: if scene N already exists on disk, it is skipped.
Designed to use bounded memory (~2-3 GB) regardless of dataset size.
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

# ============================================
# Class registry (must match notebook Cell 1)
# ============================================
FURNITURE_CLASSES = [
    'bed', 'table', 'sofa', 'chair', 'toilet',
    'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub',
]
MILITARY_CLASSES = [
    'ammo_box', 'binoculars', 'combat_knife', 'flashlight', 'gas_mask',
    'hand_grenade', 'helmet', 'magazine', 'military_radio', 'pistol',
    'rifle', 'rocket_launcher', 'shotgun', 'sniper_rifle',
    'tactical_backpack', 'tactical_vest', 'wire_cutter',
]
CLASS_NAMES = FURNITURE_CLASSES + MILITARY_CLASSES
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

CLASS_REAL_SIZE = {
    'bed': 2.0, 'table': 1.5, 'sofa': 2.0, 'chair': 0.55, 'toilet': 0.6,
    'desk': 1.4, 'dresser': 1.0, 'night_stand': 0.55, 'bookshelf': 0.85, 'bathtub': 1.6,
    'ammo_box': 0.35, 'binoculars': 0.22, 'combat_knife': 0.30, 'flashlight': 0.18,
    'gas_mask': 0.28, 'hand_grenade': 0.12, 'helmet': 0.28, 'magazine': 0.18,
    'military_radio': 0.30, 'pistol': 0.22, 'rifle': 0.95, 'rocket_launcher': 1.20,
    'shotgun': 0.95, 'sniper_rifle': 1.20, 'tactical_backpack': 0.55,
    'tactical_vest': 0.50, 'wire_cutter': 0.25,
}

FLAT_CLASSES = {
    'bed', 'sofa', 'table', 'desk', 'bathtub',
    'pistol', 'rifle', 'shotgun', 'sniper_rifle', 'combat_knife',
    'rocket_launcher', 'magazine', 'hand_grenade', 'wire_cutter',
    'ammo_box', 'binoculars',
}
TALL_CLASSES = {
    'chair', 'bookshelf', 'dresser', 'night_stand', 'toilet',
    'helmet', 'gas_mask', 'flashlight', 'military_radio',
    'tactical_backpack', 'tactical_vest',
}

# ============================================
# Bounded LRU mesh cache (prevents memory blowup)
# ============================================
MESH_CACHE_MAX = 40  # max cached meshes
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
# Mesh registry — scan disk once
# ============================================
def build_mesh_registry(data_root):
    """Scan disk for meshes per class. Returns dict {class_name: [paths...]}."""
    registry = {c: [] for c in CLASS_NAMES}
    for c in FURNITURE_CLASSES:
        folder = data_root / 'furniture' / c
        if folder.exists():
            registry[c] = sorted(folder.glob('*.off'))
    for c in MILITARY_CLASSES:
        folder = data_root / 'military' / c
        if folder.exists():
            registry[c] = sorted(folder.glob('*.glb'))
    return registry


# ============================================
# Mesh loading + orientation
# ============================================
def auto_orient_mesh(m, class_name):
    """Rotate so longest axis is horizontal for FLAT, vertical for TALL."""
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
    """Load mesh, decimate, orient to Y-up, scale, place on floor."""
    cache_key = (str(path), class_name)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    m = trimesh.load(str(path), force='mesh', skip_materials=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))

    # Strip textures, materials, attributes to save memory
    try:
        m.visual = trimesh.visual.ColorVisuals(mesh=m)
    except Exception:
        pass

    if len(m.vertices) > decimate_threshold:
        try:
            m = m.simplify_quadric_decimation(decimate_threshold // 2)
        except Exception:
            pass

    # ModelNet40 .off files are Z-up; rotate to Y-up
    if str(path).lower().endswith('.off'):
        R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        m.apply_transform(R)

    m = auto_orient_mesh(m, class_name)

    # Center XZ at origin
    m.apply_translation(-m.centroid)

    # Scale uniformly so longest extent = real-world size
    target = CLASS_REAL_SIZE[class_name]
    longest = float(m.extents.max())
    if longest > 1e-6:
        m.apply_scale(target / longest)

    # Place min Y at floor
    min_y = float(m.bounds[0, 1])
    m.apply_translation([0, -min_y, 0])

    cache_put(cache_key, m.copy())
    return m


# ============================================
# Scene generation
# ============================================
def aabb_overlap_2d(a, b, buffer=0.05):
    """2D XZ-plane AABB overlap check. a = (x_min, z_min, x_max, z_max)."""
    return not (a[2] + buffer < b[0] or b[2] + buffer < a[0] or
                a[3] + buffer < b[1] or b[3] + buffer < a[1])


def sample_room_background(room_x, room_y, room_z, n_floor=1500, n_walls=1500, n_ceiling=500):
    """Sample background points from floor/walls/ceiling."""
    floor = np.column_stack([
        np.random.uniform(0, room_x, n_floor),
        np.zeros(n_floor),
        np.random.uniform(0, room_z, n_floor),
    ])
    nw = n_walls // 4
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


def generate_scene(mesh_registry,
                   n_objects_range=(6, 12),
                   output_n_points=20000,
                   max_placement_attempts=20):
    """Generate a single synthetic scene."""
    room_x = np.random.uniform(4.0, 6.0)
    room_y = np.random.uniform(2.5, 3.0)
    room_z = np.random.uniform(4.0, 6.0)

    n_objects = np.random.randint(*n_objects_range)
    available_classes = [c for c, paths in mesh_registry.items() if len(paths) > 0]

    placed_meshes = []
    placed_aabbs_xz = []
    placed_bboxes = []
    placed_centers = []

    for _ in range(n_objects):
        class_name = np.random.choice(available_classes)
        mesh_path = np.random.choice(mesh_registry[class_name])
        try:
            base_mesh = load_normalized_mesh(mesh_path, class_name)
        except Exception:
            continue

        # Try to place with random yaw + position, avoid collisions
        placed = False
        for _ in range(max_placement_attempts):
            yaw = np.random.uniform(0, 2 * np.pi)
            mesh_try = base_mesh.copy()
            R = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
            mesh_try.apply_transform(R)
            min_y_after = float(mesh_try.bounds[0, 1])
            mesh_try.apply_translation([0, -min_y_after, 0])

            half_x = (mesh_try.bounds[1, 0] - mesh_try.bounds[0, 0]) / 2
            half_z = (mesh_try.bounds[1, 2] - mesh_try.bounds[0, 2]) / 2

            margin_x = max(0.1, half_x + 0.2)
            margin_z = max(0.1, half_z + 0.2)
            if margin_x * 2 >= room_x or margin_z * 2 >= room_z:
                break  # mesh too big for room

            cx = np.random.uniform(margin_x, room_x - margin_x)
            cz = np.random.uniform(margin_z, room_z - margin_z)

            mesh_try.apply_translation([cx, 0, cz])
            aabb = (mesh_try.bounds[0, 0], mesh_try.bounds[0, 2],
                    mesh_try.bounds[1, 0], mesh_try.bounds[1, 2])

            if any(aabb_overlap_2d(aabb, e) for e in placed_aabbs_xz):
                continue

            # accept
            center = mesh_try.bounds.mean(axis=0).astype(np.float32)
            size = mesh_try.extents.astype(np.float32)
            placed_meshes.append(mesh_try)
            placed_aabbs_xz.append(aabb)
            placed_bboxes.append((center[0], center[1], center[2],
                                  size[0], size[1], size[2],
                                  yaw, CLASS_TO_IDX[class_name]))
            placed_centers.append(center)
            placed = True
            break

    if len(placed_meshes) == 0:
        return None

    # Sample object points
    n_per_obj = output_n_points // 4 // max(1, len(placed_meshes))
    n_per_obj = max(200, n_per_obj)
    object_pts_list = []
    object_assignment = []
    for i, m in enumerate(placed_meshes):
        try:
            pts, _ = trimesh.sample.sample_surface(m, n_per_obj)
            object_pts_list.append(pts.astype(np.float32))
            object_assignment.extend([i] * len(pts))
        except Exception:
            continue

    if len(object_pts_list) == 0:
        return None

    object_pts = np.concatenate(object_pts_list, axis=0)
    object_assignment = np.array(object_assignment, dtype=np.int32)

    # Sample background
    n_bg = output_n_points - len(object_pts)
    if n_bg > 0:
        bg_pts = sample_room_background(room_x, room_y, room_z,
                                        n_floor=int(n_bg * 0.5),
                                        n_walls=int(n_bg * 0.4),
                                        n_ceiling=int(n_bg * 0.1))
        bg_pts = bg_pts.astype(np.float32)
    else:
        bg_pts = np.empty((0, 3), dtype=np.float32)

    all_pts = np.concatenate([object_pts, bg_pts], axis=0)
    rgb = np.zeros_like(all_pts)
    pc_full = np.concatenate([all_pts, rgb], axis=1).astype(np.float32)

    # is_fg/assignment for original points
    is_fg_orig = np.concatenate([
        np.ones(len(object_pts), dtype=bool),
        np.zeros(len(bg_pts), dtype=bool),
    ])
    # assignment: object index for fg points, -1 for bg
    assignment_orig = np.full(len(all_pts), -1, dtype=np.int32)
    assignment_orig[:len(object_pts)] = object_assignment

    # Resample to exactly output_n_points (subsample without replacement; pad with replacement)
    N = len(pc_full)
    if N >= output_n_points:
        idx = np.random.choice(N, output_n_points, replace=False)
    else:
        extra = np.random.choice(N, output_n_points - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])

    pc_full = pc_full[idx]
    is_fg = is_fg_orig[idx]
    assignment = assignment_orig[idx]

    # Build votes
    votes = np.zeros((len(pc_full), 10), dtype=np.float32)
    votes[:, 0] = is_fg.astype(np.float32)
    fg_mask = is_fg
    if fg_mask.sum() > 0:
        centers_arr = np.array(placed_centers, dtype=np.float32)
        fg_assign = assignment[fg_mask]
        # safety clip in case of edge cases
        fg_assign = np.clip(fg_assign, 0, len(placed_centers) - 1)
        offsets = centers_arr[fg_assign] - pc_full[fg_mask, :3]
        votes[fg_mask, 1:4] = offsets
        votes[fg_mask, 4:7] = offsets
        votes[fg_mask, 7:10] = offsets

    bboxes_arr = np.array(placed_bboxes, dtype=np.float32)

    scene = {
        'pc': pc_full,
        'bbox': bboxes_arr,
        'votes': votes,
        '_room_dims': (room_x, room_y, room_z),
    }
    return scene


def save_scene(scene, scene_idx, out_dir):
    """Save scene to VoteNet-compatible files."""
    sid = f'{scene_idx:06d}'
    np.savez_compressed(out_dir / f'{sid}_pc.npz', pc=scene['pc'])
    np.save(out_dir / f'{sid}_bbox.npy', scene['bbox'])
    np.savez_compressed(out_dir / f'{sid}_votes.npz', votes=scene['votes'])


# ============================================
# Main loop
# ============================================
def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def fill_split(split_name, out_dir, n_target, mesh_registry, gc_every=50):
    """Generate scenes for one split. Skips scenes that already exist."""
    print(f'\n=== {split_name.upper()} ({n_target} scenes) ===')
    out_dir.mkdir(parents=True, exist_ok=True)
    class_counter = Counter()
    failed = 0
    t_start = time.time()
    last_print = t_start

    for i in range(n_target):
        scene_file = out_dir / f'{i:06d}_pc.npz'
        if scene_file.exists():
            # already done — count classes from existing bbox file for stats
            bbox_file = out_dir / f'{i:06d}_bbox.npy'
            if bbox_file.exists():
                try:
                    bb = np.load(bbox_file)
                    for box in bb:
                        class_counter[CLASS_NAMES[int(box[7])]] += 1
                except Exception:
                    pass
            continue

        attempts = 0
        scene = None
        while scene is None and attempts < 5:
            scene = generate_scene(mesh_registry)
            attempts += 1
        if scene is None:
            failed += 1
            continue

        save_scene(scene, i, out_dir)
        for box in scene['bbox']:
            class_counter[CLASS_NAMES[int(box[7])]] += 1

        # explicit cleanup
        del scene
        if (i + 1) % gc_every == 0:
            gc.collect()

        # progress log every 10s or every 100 scenes
        now = time.time()
        if (i + 1) % 100 == 0 or now - last_print > 10:
            elapsed = now - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (n_target - i - 1) / rate / 60 if rate > 0 else 0
            print(f'  {i + 1:5d}/{n_target}  '
                  f'{rate:.1f} scenes/s  '
                  f'mem={memory_mb():.0f} MB  '
                  f'ETA {eta_min:.1f} min')
            last_print = now

    elapsed = time.time() - t_start
    print(f'  done: {n_target} scenes in {elapsed / 60:.1f} min ({failed} failed)')
    return class_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/Users/dosvatsky/3D Object Detection/data/mesh_dataset_v1')
    parser.add_argument('--out-root', default='/Users/dosvatsky/3D Object Detection/data/synthetic_v1')
    parser.add_argument('--n-train', type=int, default=5000)
    parser.add_argument('--n-val', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print('Building mesh registry...')
    registry = build_mesh_registry(data_root)
    total = 0
    for c in CLASS_NAMES:
        n = len(registry[c])
        total += n
        if n == 0:
            print(f'  WARNING: no meshes for {c}')
    print(f'  total: {total} meshes across {NUM_CLASSES} classes')

    print(f'Initial memory: {memory_mb():.0f} MB')
    print(f'Mesh cache cap: {MESH_CACHE_MAX} entries')

    counter_train = fill_split('train', out_root / 'train', args.n_train, registry)
    counter_val = fill_split('val', out_root / 'val', args.n_val, registry)

    # Combined stats
    combined = counter_train + counter_val
    total_objects = sum(combined.values())
    total_scenes = args.n_train + args.n_val

    print('\n' + '=' * 60)
    print('DATASET STATISTICS')
    print('=' * 60)
    print(f'Total scenes: {total_scenes}')
    print(f'Total objects: {total_objects}')
    print(f'\nClass distribution:')
    print(f'{"Class":18s} {"count":>7s}  {"avg/scene":>10s}')
    print('-' * 50)
    for cls in CLASS_NAMES:
        cnt = combined[cls]
        avg = cnt / total_scenes
        bar = '#' * int(avg * 10)
        print(f'{cls:18s} {cnt:>7d}  {avg:>9.2f}  {bar}')

    # split index files
    with open(out_root / 'classes.txt', 'w') as f:
        for i, c in enumerate(CLASS_NAMES):
            f.write(f'{i}\t{c}\n')
    with open(out_root / 'train_idx.txt', 'w') as f:
        f.writelines(f'{i:06d}\n' for i in range(args.n_train))
    with open(out_root / 'val_idx.txt', 'w') as f:
        f.writelines(f'{i:06d}\n' for i in range(args.n_val))

    print(f'\nWrote {out_root}/classes.txt')
    print(f'Wrote {out_root}/train_idx.txt')
    print(f'Wrote {out_root}/val_idx.txt')
    print(f'\nFinal memory: {memory_mb():.0f} MB')
    print('Done.')


if __name__ == '__main__':
    main()
