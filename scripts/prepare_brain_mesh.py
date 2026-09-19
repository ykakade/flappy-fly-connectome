"""Simplify the official MaleCNS brain shell for the browser.

Usage: python scripts/prepare_brain_mesh.py /path/to/brain-shell.ngmesh
Pinned source URL and license are in dist/assets/BRAIN-SOURCE.md.
"""
import json
from pathlib import Path
import struct
import sys
import numpy as np

raw = Path(sys.argv[1]).read_bytes()
count = struct.unpack('<I', raw[:4])[0]
vertices = np.frombuffer(raw, dtype='<f4', count=count * 3, offset=4).reshape(-1, 3) / 8
faces = np.frombuffer(raw, dtype='<u4', offset=4 + count * 12).reshape(-1, 3)
# Convert nanometers to the same 8 nm voxel coordinates as somaLocation.
_, inverse = np.unique(np.round(vertices / 1600).astype(np.int32), axis=0, return_inverse=True)
counts = np.bincount(inverse)
vertices = np.column_stack([np.bincount(inverse, weights=vertices[:, i]) / counts for i in range(3)])
faces = inverse[faces]
faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])]
_, indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
faces = faces[np.sort(indices)]
Path('dist/assets/brain-shell.json').write_text(json.dumps(dict(
    source='MaleCNS brain-shell-v2.2', generation='1758224654323697', units='8nm voxels',
    vertices=np.round(vertices, 1).tolist(), faces=faces.tolist(),
), separators=(',', ':')))
print(f'{len(vertices)} vertices, {len(faces)} faces')
