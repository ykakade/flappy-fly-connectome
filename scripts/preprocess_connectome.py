"""Stream the real Feather graph into a traced visual/central/descending candidate graph."""

import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--min-weight", type=int, default=5)
    a = p.parse_args()
    nodes = feather.read_table(a.cache / "body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    assert {"bodyId", "superclass", "type", "status"} <= set(nodes.columns)
    keep = nodes[
        (nodes.status == "Traced") & nodes.superclass.isin(["visual_projection", "cb_intrinsic", "descending_neuron"])
    ]
    ids = keep.bodyId.to_numpy()
    chunks = []
    total = 0
    with pa.memory_map(str(a.cache / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"), "r") as f:
        r = pa.ipc.open_file(f)
        assert r.schema.names == ["body_pre", "body_post", "weight"], r.schema
        for i in range(r.num_record_batches):
            batch = r.get_batch(i)
            pre, post, weight = [batch.column(j).to_numpy() for j in range(3)]
            total += len(pre)
            mask = (weight >= a.min_weight) & (pre != post) & np.isin(pre, ids) & np.isin(post, ids)
            if mask.any():
                chunks.append(np.stack([pre[mask], post[mask], weight[mask]], axis=1))
    edges = np.concatenate(chunks)
    np.savez_compressed(a.cache / "candidates.npz", edges=edges)
    # Official annotations preserved including nulls, no invented signs or transmitter labels.
    (a.cache / "candidate-nodes.json").write_text(keep.to_json(orient="records"))
    (a.cache / "preprocessing.json").write_text(
        json.dumps(
            dict(
                raw_edges=total,
                candidate_edges=len(edges),
                candidate_nodes=len(keep),
                min_weight=a.min_weight,
                allowed_superclasses=["visual_projection", "cb_intrinsic", "descending_neuron"],
                status="Traced",
            ),
            indent=2,
        )
    )
    print("Candidate graph:", len(keep), "nodes,", len(edges), "edges from", total, "rows")


if __name__ == "__main__":
    main()
