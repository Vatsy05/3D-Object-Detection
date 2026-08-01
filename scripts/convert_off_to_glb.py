#!/usr/bin/env python3
"""One-time conversion: ModelNet40 .off -> .glb (for Habitat-Sim compatibility).

Run:
    cd "/Users/dosvatsky/3D Object Detection"
    python scripts/convert_off_to_glb.py
"""

from pathlib import Path
import trimesh

ROOT = Path('/Users/dosvatsky/3D Object Detection/data/mesh_dataset_v1/furniture')

n_converted = 0
n_skipped = 0
n_failed = 0

for off in sorted(ROOT.rglob('*.off')):
    glb = off.with_suffix('.glb')
    if glb.exists():
        n_skipped += 1
        continue
    try:
        m = trimesh.load(str(off), force='mesh')
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        m.export(str(glb))
        n_converted += 1
        if n_converted % 20 == 0:
            print(f'  converted {n_converted} so far...')
    except Exception as e:
        n_failed += 1
        print(f'  FAILED {off.relative_to(ROOT.parent)}: {e}')

print()
print(f'Converted:                       {n_converted}')
print(f'Skipped (already had .glb):      {n_skipped}')
print(f'Failed:                          {n_failed}')

# Verify
all_glb = list(ROOT.rglob('*.glb'))
print(f'\nTotal furniture .glb files now:  {len(all_glb)}')
for cls_dir in sorted(ROOT.iterdir()):
    if cls_dir.is_dir():
        count = len(list(cls_dir.glob('*.glb')))
        print(f'  {cls_dir.name:14s}  {count} .glb files')
