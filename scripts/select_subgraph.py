"""Deterministic maximum-strength connected expansion from a visual→descending edge."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--max-nodes", type=int, default=96)
    p.add_argument("--max-edges", type=int, default=4096)
    p.add_argument("--output", type=Path, default=Path("data/graph.json"))
    a = p.parse_args()
    nodes = {int(n["bodyId"]): n for n in json.loads((a.cache / "candidate-nodes.json").read_text())}
    edges = np.load(a.cache / "candidates.npz")["edges"]
    visual = {i for i, n in nodes.items() if n["superclass"] == "visual_projection"}
    descending = {i for i, n in nodes.items() if n["superclass"] == "descending_neuron"}
    direct = edges[np.isin(edges[:, 0], list(visual)) & np.isin(edges[:, 1], list(descending))]
    if not len(direct):
        raise RuntimeError("No real visual-to-descending edge; refusing to fabricate a graph")
    anchor = sorted(direct.tolist(), key=lambda e: (-e[2], e[0], e[1]))[0]
    adj = {}
    for u, v, w in edges.tolist():
        adj.setdefault(u, []).append((-w, u, v))
        adj.setdefault(v, []).append((-w, u, v))
    selected = set(anchor[:2])
    tree = {tuple(anchor[:2])}
    heap = []
    for n in selected:
        for e in adj[n]:
            heapq.heappush(heap, e)
    while len(selected) < a.max_nodes and heap:
        nw, u, v = heapq.heappop(heap)
        if (u in selected) == (v in selected):
            continue
        new = v if u in selected else u
        selected.add(new)
        tree.add((u, v))
        for e in adj[new]:
            heapq.heappush(heap, e)
    induced = edges[np.isin(edges[:, 0], list(selected)) & np.isin(edges[:, 1], list(selected))]
    ordered = sorted(induced.tolist(), key=lambda e: ((e[0], e[1]) not in tree, -e[2], e[0], e[1]))[: a.max_edges]
    if a.max_edges < len(tree):
        raise ValueError("Edge cap cannot preserve spanning tree")
    ordered.sort(key=lambda e: (e[0], e[1]))
    ids = sorted(selected)
    index = {n: i for i, n in enumerate(ids)}
    strength = {i: 0 for i in ids}
    for u, v, w in ordered:
        strength[u] += w
    inputs = sorted(selected & visual, key=lambda i: (-strength[i], i))[:16]
    reachable = set(inputs)
    for _ in range(4):
        reachable |= {v for u, v, w in ordered if u in reachable}
    outputs = sorted((selected & descending & reachable) - set(inputs))[:16]
    if not inputs or not outputs:
        raise RuntimeError("Selected graph has no visual-to-descending route in four passes")
    weights = np.array([e[2] for e in ordered], float)
    dst = np.array([index[e[1]] for e in ordered])
    norm = np.bincount(dst, weights=weights, minlength=len(ids))
    artifact = dict(
        dataset="MaleCNS v1.0",
        license="CC-BY-4.0",
        attribution="FlyEM (HHMI Janelia), Cambridge, MRC LMB, Google Research",
        source="https://male-cns.janelia.org/download/",
        nodes=[nodes[i] for i in ids],
        edges=[
            dict(
                source=str(u),
                target=str(v),
                count=w,
                source_index=index[u],
                target_index=index[v],
                normalized_weight=float(w / norm[index[v]]),
            )
            for u, v, w in ordered
        ],
        inputs=[index[i] for i in inputs],
        outputs=[index[i] for i in outputs],
        selection=dict(
            method=__doc__,
            anchor_ids=anchor[:2],
            max_nodes=a.max_nodes,
            max_edges=a.max_edges,
            message_passes=4,
            retained_nodes=len(ids),
            retained_edges=len(ordered),
            retained_synapse_count=int(weights.sum()),
            excluded_reason="CPU/browser budget; only traced visual_projection, cb_intrinsic and descending_neuron candidates, non-self edges with >=5 synapses; greedily retain connected strongest neighborhood. This is not a full visual-to-motor circuit.",
            **json.loads((a.cache / "preprocessing.json").read_text()),
        ),
    )
    # IDs are strings in browser artifacts to preserve identity independently of JS numeric limits.
    for node in artifact["nodes"]:
        node["bodyId"] = str(node["bodyId"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(
        "Retained",
        len(ids),
        "nodes",
        len(ordered),
        "edges; input",
        len(inputs),
        "output",
        len(outputs),
        "SHA256",
        hashlib.sha256(a.output.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
