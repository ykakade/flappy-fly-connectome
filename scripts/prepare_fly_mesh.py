"""Convert the Apache-2.0 flybody OBJ/XML assets to a compact browser mesh.

Usage: python scripts/prepare_fly_mesh.py /path/to/flybody/fruitfly/assets
Source revision and attribution: dist/assets/flybody/README.md.
"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

source = Path(sys.argv[1])
out = Path('dist/assets/flybody')
out.mkdir(parents=True, exist_ok=True)
xml = ET.parse(source / 'fruitfly.xml').getroot()
chunks = []
mesh_index = {}
byte_offset = 0

def add_array(values, dtype):
    global byte_offset
    array = np.asarray(values, dtype=dtype).ravel()
    descriptor = dict(offset=byte_offset, count=len(array))
    data = array.tobytes()
    chunks.append(data)
    byte_offset += len(data)
    return descriptor

for mesh in xml.findall('asset/mesh'):
    vertices, faces = [], []
    for line in (source / mesh.attrib['file']).read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            indices = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])
    v = np.asarray(vertices)
    # Vertex clustering preserves the full surface, removing subpixel detail.
    # Original OBJ units are millimeters; MJCF scales them by 0.1.
    cell = .014 if 'black' in mesh.attrib['name'] else .008
    _, inverse = np.unique(np.round(v / cell).astype(np.int32), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    clustered = np.column_stack([np.bincount(inverse, weights=v[:, i]) / count for i in range(3)])
    f = inverse[np.asarray(faces)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 0] != f[:, 2]) & (f[:, 1] != f[:, 2])]
    _, unique = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(unique)]
    mesh_index[mesh.attrib['name']] = dict(positions=add_array(clustered * .1, '<f4'), indices=add_array(f, '<u4'))

def vector(element, name, fallback):
    return [float(x) for x in element.get(name, fallback).split()]

def transform(element):
    return dict(position=vector(element, 'pos', '0 0 0'), quaternion=vector(element, 'quat', '1 0 0 0'))

def body(element):
    return dict(name=element.get('name'), **transform(element), meshes=[dict(mesh=g.get('mesh'), material=g.get('material', 'body'), **transform(g)) for g in element.findall('geom') if g.get('mesh')], children=[body(b) for b in element.findall('body')])

model = dict(source='TuragaLab/flybody', revision='d015e9bfe441bd90ae431bac24c55cb74bdbce26', meshes=mesh_index,
             materials={m.get('name'): vector(m, 'rgba', '1 1 1 1') for m in xml.findall('asset/material')},
             body=body(xml.find('worldbody/body')))
(out / 'fly.json').write_text(json.dumps(model, separators=(',', ':')))
(out / 'fly.bin').write_bytes(b''.join(chunks))
print(f'{len(mesh_index)} parts; {byte_offset / 1e6:.2f} MB')
