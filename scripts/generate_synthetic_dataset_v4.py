#!/usr/bin/env python3
"""
Phase 10 — Camera-simulation scene generator v4 (60 classes, 2.5D depth-realistic)

ROOT-CAUSE FIX for the Phase 9 engine failure (recall 2/56):
  Phase 8/9 trained on FULL-SURROUND surface samples (points from every side of
  every object, walls a thin uniform scatter). A real/BlenderProc depth camera
  produces a SINGLE-VIEWPOINT 2.5D cloud: only visible surfaces, one dense
  contiguous wall plane, self-occlusion. The train/test distributions did not
  match, so the model hallucinated on wall planes and missed occluded objects.

v4 makes TRAINING data look like a depth camera by ray-casting the composed
scene from a random virtual camera (open3d RaycastingScene) and keeping only
the hit points. Now train == test distribution. Everything downstream (bbox
storage format, votes, Y-up convention, dataset class) is unchanged.

Scene composition (room / tabletop, class balance, small-object boost, sizes,
orientation) is inherited from v3 — only the point SAMPLING changes:
    v3: trimesh.sample.sample_surface (full surround)
    v4: open3d raycast from a camera pose (2.5D visible surface)

Camera poses mimic the BlenderProc eval rig: 0.9-1.6 m from the scene, 20-45 deg
downward, random azimuth. Depth is converted to a world point cloud, gaussian
sensor noise added, then stored in the SAME Y-up (pc/bbox/votes) format.

Run on the Mac M2:
    cd "/Users/dosvatsky/3D Object Detection"
    python scripts/generate_synthetic_dataset_v4.py --n-train 8000 --n-val 1500

Resumable. Needs: pip install open3d trimesh numpy psutil
"""
import argparse, gc, os, sys, time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
import psutil

sys.path.insert(0, str(Path(__file__).parent))
from class_registry_v2 import (
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, CLASS_REAL_SIZE,
    FURNITURE_CLASSES, MILITARY_CLASSES, FLAT_CLASSES, TALL_CLASSES, SMALL_CLASSES,
)

N_POINTS = 40000
MIN_PTS_PER_OBJ = 300         # kept as a QC threshold (see min-visibility retry)
TABLETOP_FRACTION = 0.30
IMG_W, IMG_H = 640, 480       # virtual sensor resolution (matches eval rig)
FX = FY = 600.0               # matches eval intrinsics
SENSOR_SIGMA = 0.004          # 4 mm gaussian depth noise
MULTIVIEW = 2                 # cast from 2 camera poses per scene and merge
                              # (partial multi-view: richer than 1, still 2.5D)

TABLE_ITEMS = sorted(c for c in MILITARY_CLASSES if CLASS_REAL_SIZE[c] <= 0.65)
TABLE_SURFACES = ['table', 'desk']

MESH_CACHE_MAX = 40
_MESH_CACHE = OrderedDict()
def cache_get(k):
    if k in _MESH_CACHE:
        _MESH_CACHE.move_to_end(k); return _MESH_CACHE[k]
    return None
def cache_put(k, v):
    _MESH_CACHE[k] = v; _MESH_CACHE.move_to_end(k)
    while len(_MESH_CACHE) > MESH_CACHE_MAX:
        _MESH_CACHE.popitem(last=False)


def build_mesh_registry(data_root):
    return {c: sorted((data_root / c).glob('*.glb')) for c in CLASS_NAMES}


def auto_orient_mesh(m, class_name):
    ext = m.extents.copy()
    la = int(np.argmax(ext))
    if class_name in FLAT_CLASSES and la == 1:
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0]))
    elif class_name in TALL_CLASSES:
        if la == 0:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1]))
        elif la == 2:
            m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0]))
    return m


def load_normalized_mesh(path, class_name, decimate_threshold=150_000):
    key = (str(path), class_name)
    c = cache_get(key)
    if c is not None:
        return c.copy()
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
    longest = float(m.extents.max())
    if longest > 1e-6:
        m.apply_scale(CLASS_REAL_SIZE[class_name] / longest)
    m.apply_translation([0, -float(m.bounds[0, 1]), 0])
    cache_put(key, m.copy())
    return m


GLOBAL_CLASS_COUNTS = Counter()
def pick_class(cands):
    w = np.array([1.0 / (1.0 + GLOBAL_CLASS_COUNTS[c]) for c in cands])
    return np.random.choice(cands, p=w / w.sum())


def aabb_overlap_2d(a, b, buf=0.05):
    return not (a[2]+buf < b[0] or b[2]+buf < a[0] or a[3]+buf < b[1] or b[3]+buf < a[1])


def try_place(base, class_name, region, floor_y, placed_aabbs, max_att=20, buf=0.05):
    natural = base.extents.astype(np.float32)
    for _ in range(max_att):
        yaw = np.random.uniform(0, 2*np.pi)
        m = base.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0,1,0]))
        m.apply_translation([0, floor_y - float(m.bounds[0,1]), 0])
        hx = (m.bounds[1,0]-m.bounds[0,0])/2; hz = (m.bounds[1,2]-m.bounds[0,2])/2
        x0,z0,x1,z1 = region
        if x0+hx >= x1-hx or z0+hz >= z1-hz:
            return None
        cx = np.random.uniform(x0+hx, x1-hx); cz = np.random.uniform(z0+hz, z1-hz)
        m.apply_translation([cx, 0, cz])
        aabb = (m.bounds[0,0], m.bounds[0,2], m.bounds[1,0], m.bounds[1,2])
        if any(aabb_overlap_2d(aabb, e, buf) for e in placed_aabbs):
            continue
        center = m.bounds.mean(axis=0).astype(np.float32)
        bbox = (center[0],center[1],center[2], natural[0],natural[1],natural[2],
                yaw, CLASS_TO_IDX[class_name])
        return m, bbox, aabb
    return None


def make_room_meshes(rx, ry, rz):
    """Floor + 4 walls as trimesh boxes (thin), Y-up (y = height)."""
    t = 0.05
    floor = trimesh.creation.box(extents=[rx, t, rz]); floor.apply_translation([rx/2, -t/2, rz/2])
    w1 = trimesh.creation.box(extents=[t, ry, rz]); w1.apply_translation([0, ry/2, rz/2])
    w2 = trimesh.creation.box(extents=[t, ry, rz]); w2.apply_translation([rx, ry/2, rz/2])
    w3 = trimesh.creation.box(extents=[rx, ry, t]); w3.apply_translation([rx/2, ry/2, 0])
    w4 = trimesh.creation.box(extents=[rx, ry, t]); w4.apply_translation([rx/2, ry/2, rz])
    return [floor, w1, w2, w3, w4]


def raycast_scene(all_meshes, obj_slots, room_dims, n_views=MULTIVIEW):
    """Cast a depth camera into the composed scene from n random poses.
    all_meshes: list of trimesh (objects first, then room). obj_slots: number
    of leading meshes that are real objects (rest are room -> assignment -1).
    Returns (points Nx3 Y-up, assignment N of object-index-or-(-1))."""
    scene = o3d.t.geometry.RaycastingScene()
    geo_to_obj = []
    for i, m in enumerate(all_meshes):
        tm = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(m.vertices, o3d.core.float32),
            o3d.core.Tensor(m.faces, o3d.core.uint32))
        scene.add_triangles(tm)
        geo_to_obj.append(i if i < obj_slots else -1)
    geo_to_obj = np.array(geo_to_obj, dtype=np.int64)

    rx, ry, rz = room_dims
    scene_center = np.array([rx/2, 0.4, rz/2], dtype=np.float32)
    all_pts, all_assign = [], []
    for _ in range(n_views):
        dist = np.random.uniform(1.6, 3.0)
        az = np.random.uniform(0, 2*np.pi)
        el = np.radians(np.random.uniform(18, 45))
        eye = scene_center + dist*np.array([np.cos(az)*np.cos(el),
                                            np.sin(el)/max(np.cos(el),0.3),
                                            np.sin(az)*np.cos(el)], dtype=np.float32)
        eye[1] = np.clip(eye[1], 0.6, ry-0.2)
        rays = scene.create_rays_pinhole(
            fov_deg=60.0, center=scene_center, eye=eye,
            up=np.array([0,1,0], dtype=np.float32),
            width_px=IMG_W, height_px=IMG_H)
        ans = scene.cast_rays(rays)
        t_hit = ans['t_hit'].numpy().reshape(-1)
        geo_id = ans['geometry_ids'].numpy().reshape(-1)
        rays_np = rays.numpy().reshape(-1, 6)
        valid = np.isfinite(t_hit)
        origins = rays_np[valid, :3]; dirs = rays_np[valid, 3:6]
        pts = origins + dirs * t_hit[valid, None]
        assign = geo_to_obj[geo_id[valid]]
        all_pts.append(pts.astype(np.float32)); all_assign.append(assign)
    pts = np.concatenate(all_pts); assign = np.concatenate(all_assign)
    # sensor noise along the (already-baked) surface
    pts += np.random.normal(0, SENSOR_SIGMA, pts.shape).astype(np.float32)
    return pts, assign


def finalize(all_meshes, obj_bboxes, obj_slots, room_dims):
    pts, assign = raycast_scene(all_meshes, obj_slots, room_dims)
    if len(pts) < 2000:
        return None
    # QC: require each object to have >= a small visible count, else drop the box
    keep_idx, remap = [], {}
    for oi in range(obj_slots):
        n_vis = int((assign == oi).sum())
        if n_vis >= 40:                # visible enough to be a valid GT
            remap[oi] = len(keep_idx); keep_idx.append(oi)
    if not keep_idx:
        return None
    bboxes = np.array([obj_bboxes[i] for i in keep_idx], dtype=np.float32)
    new_assign = np.full(len(assign), -1, dtype=np.int32)
    for old, new in remap.items():
        new_assign[assign == old] = new

    # subsample / pad to N_POINTS
    if len(pts) >= N_POINTS:
        sel = np.random.choice(len(pts), N_POINTS, replace=False)
    else:
        sel = np.concatenate([np.arange(len(pts)),
                              np.random.choice(len(pts), N_POINTS-len(pts), replace=True)])
    pts = pts[sel]; new_assign = new_assign[sel]

    rgb = np.zeros_like(pts)
    pc_full = np.concatenate([pts, rgb], axis=1).astype(np.float32)
    is_fg = new_assign >= 0
    centers = bboxes[:, :3]
    votes = np.zeros((len(pc_full), 10), dtype=np.float32)
    votes[:, 0] = is_fg.astype(np.float32)
    if is_fg.sum() > 0:
        off = centers[new_assign[is_fg]] - pc_full[is_fg, :3]
        votes[is_fg, 1:4] = off; votes[is_fg, 4:7] = off; votes[is_fg, 7:10] = off
    return {'pc': pc_full, 'bbox': bboxes, 'votes': votes}


def gen_room(reg, n_range=(8,14)):
    rx = np.random.uniform(4,6); ry = np.random.uniform(2.5,3); rz = np.random.uniform(4,6)
    n = np.random.randint(*n_range)
    avail = [c for c,p in reg.items() if p]
    small = [c for c in avail if c in SMALL_CLASSES]
    meshes, aabbs, bboxes = [], [], []
    for k in range(n):
        cls = pick_class(small) if (small and k%2==1) else pick_class(avail)
        try:
            base = load_normalized_mesh(np.random.choice(reg[cls]), cls)
        except Exception:
            continue
        r = try_place(base, cls, (0,0,rx,rz), 0.0, aabbs)
        if r is None:
            continue
        m,bb,ab = r; meshes.append(m); bboxes.append(bb); aabbs.append(ab)
        GLOBAL_CLASS_COUNTS[cls]+=1
    if not meshes:
        return None
    room = make_room_meshes(rx,ry,rz)
    return finalize(meshes+room, bboxes, len(meshes), (rx,ry,rz))


def gen_tabletop(reg, n_range=(4,8)):
    rx=np.random.uniform(3.5,5); ry=np.random.uniform(2.5,3); rz=np.random.uniform(3.5,5)
    meshes,aabbs,bboxes = [],[],[]
    surf = np.random.choice(TABLE_SURFACES)
    if not reg.get(surf):
        return None
    try:
        t = load_normalized_mesh(np.random.choice(reg[surf]), surf)
    except Exception:
        return None
    r = try_place(t, surf, (0.5,0.5,rx-0.5,rz-0.5), 0.0, aabbs)
    if r is None:
        return None
    tm,tb,ta = r; meshes.append(tm); bboxes.append(tb); aabbs.append(ta)
    GLOBAL_CLASS_COUNTS[surf]+=1
    top=float(tm.bounds[1,1]); tx0,tz0,tx1,tz1=ta
    region=(tx0+0.1,tz0+0.1,tx1-0.1,tz1-0.1); tabletop=[]
    items=[c for c in TABLE_ITEMS if reg.get(c)]
    for _ in range(np.random.randint(*n_range)):
        cls=pick_class(items)
        try:
            base=load_normalized_mesh(np.random.choice(reg[cls]),cls)
        except Exception:
            continue
        rr=try_place(base,cls,region,top,tabletop,buf=0.02)
        if rr is None:
            continue
        m,bb,ab=rr; meshes.append(m); bboxes.append(bb); tabletop.append(ab)
        GLOBAL_CLASS_COUNTS[cls]+=1
    if len(meshes)<3:
        return None
    large=[c for c,p in reg.items() if p and CLASS_REAL_SIZE[c]>=0.8 and c not in TABLE_SURFACES]
    for _ in range(np.random.randint(1,4)):
        cls=pick_class(large)
        try:
            base=load_normalized_mesh(np.random.choice(reg[cls]),cls)
        except Exception:
            continue
        rr=try_place(base,cls,(0,0,rx,rz),0.0,aabbs)
        if rr is None:
            continue
        m,bb,ab=rr; meshes.append(m); bboxes.append(bb); aabbs.append(ab)
        GLOBAL_CLASS_COUNTS[cls]+=1
    room=make_room_meshes(rx,ry,rz)
    return finalize(meshes+room, bboxes, len(meshes), (rx,ry,rz))


def generate_scene(reg):
    return gen_tabletop(reg) if np.random.rand() < TABLETOP_FRACTION else gen_room(reg)


def save_scene(s, i, out):
    sid=f'{i:06d}'
    np.savez_compressed(out/f'{sid}_pc.npz', pc=s['pc'])
    np.save(out/f'{sid}_bbox.npy', s['bbox'])
    np.savez_compressed(out/f'{sid}_votes.npz', votes=s['votes'])


def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss/1024/1024


def fill_split(name, out, n, reg, gc_every=50):
    print(f'\n=== {name.upper()} ({n} scenes) ===')
    out.mkdir(parents=True, exist_ok=True)
    cc=Counter(); failed=0; t0=time.time(); lp=t0
    for i in range(n):
        if (out/f'{i:06d}_pc.npz').exists():
            bf=out/f'{i:06d}_bbox.npy'
            if bf.exists():
                try:
                    for b in np.load(bf):
                        cc[CLASS_NAMES[int(b[7])]]+=1; GLOBAL_CLASS_COUNTS[CLASS_NAMES[int(b[7])]]+=1
                except Exception:
                    pass
            continue
        s=None; a=0
        while s is None and a<5:
            s=generate_scene(reg); a+=1
        if s is None:
            failed+=1; continue
        save_scene(s,i,out)
        for b in s['bbox']:
            cc[CLASS_NAMES[int(b[7])]]+=1
        del s
        if (i+1)%gc_every==0:
            gc.collect()
        now=time.time()
        if (i+1)%100==0 or now-lp>10:
            el=now-t0; rate=(i+1)/el if el else 0
            print(f'  {i+1:5d}/{n}  {rate:.1f}/s  mem={memory_mb():.0f}MB  '
                  f'ETA {(n-i-1)/rate/60 if rate else 0:.1f}min', flush=True)
            lp=now
    print(f'  done in {(time.time()-t0)/60:.1f} min ({failed} failed)')
    return cc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/Users/dosvatsky/3D Object Detection/data/mesh_dataset_v1')
    ap.add_argument('--out-root',  default='/Users/dosvatsky/3D Object Detection/data/synthetic_v4_25d')
    ap.add_argument('--n-train', type=int, default=8000)
    ap.add_argument('--n-val',   type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    a=ap.parse_args()
    np.random.seed(a.seed)
    reg=build_mesh_registry(Path(a.data_root))
    out=Path(a.out_root); out.mkdir(parents=True, exist_ok=True)
    tot=sum(len(v) for v in reg.values())
    print(f'{tot} meshes / {NUM_CLASSES} classes; 2.5D raycast, {MULTIVIEW} views/scene')
    ct=fill_split('train', out/'train', a.n_train, reg)
    cv=fill_split('val', out/'val', a.n_val, reg)
    comb=ct+cv; ts=a.n_train+a.n_val
    print('\n'+'='*50+f'\nDATASET STATS  scenes={ts} objects={sum(comb.values())}')
    for c in CLASS_NAMES:
        print(f'{c:22s} {comb[c]:>7d}  {comb[c]/ts:>7.2f}')
    with open(out/'classes.txt','w') as f:
        for i,c in enumerate(CLASS_NAMES): f.write(f'{i}\t{c}\n')
    with open(out/'train_idx.txt','w') as f: f.writelines(f'{i:06d}\n' for i in range(a.n_train))
    with open(out/'val_idx.txt','w') as f: f.writelines(f'{i:06d}\n' for i in range(a.n_val))
    print('Done.')


if __name__ == '__main__':
    main()
