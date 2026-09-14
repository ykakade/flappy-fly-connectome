# MaleCNS data provenance and attribution

The **MaleCNS v1.0** structural connectome was acquired, reconstructed and annotated by **FlyEM (HHMI Janelia), the University of Cambridge, the MRC Laboratory of Molecular Biology, and Google Research**. It is not a validated dynamical model of a brain. No biological electrophysiology, learned plasticity or consciousness is supplied by the structural wiring alone.

- Official project: https://www.janelia.org/project-team/flyem/male-cns-connectome
- Official download instructions: https://male-cns.janelia.org/download/
- Publication: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, Cell (2026), https://doi.org/10.1016/j.cell.2026.08.015
- Google Research announcement: https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
- License: **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/

The official project and download pages link this license. It permits redistribution and adaptation with credit, license link, and indication of changes, without implying endorsement. `graph.json` and `dist/graph.json` are small modified subsets under CC BY 4.0, separately from the MIT original code. Changes include node/edge filtering, selection, index mapping and incoming-count normalization. We retain official annotations and directed counts. No transmitter labels or excitatory/inhibitory classifications are inferred. Optional official neurotransmitter predictions were not downloaded or used.

## Verified files

`dataset.json` records exact URLs, immutable Google Storage generation URLs, byte sizes, complete inspected Arrow schemas, row counts, publisher MD5 and CRC32C headers, and locally calculated SHA256. Download inspection occurred 2026-09-13. The v1.0 release date shown by Janelia is 2026-06-08; the Google announcement is 2026-09-03.

Both files are **Feather v2 / Apache Arrow IPC files**, at:
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

| Filename | Exact bytes | Publisher MD5 |
|---|---:|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `50a7718770c57220f160ba4f431ab89e` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `f30e9dcca25cfd021bf1e7b3d975599e` |

The download script pins generations, verifies size and MD5, computes SHA256, and fails closed if bytes differ. CRC32C is recorded from official headers; it is not separately recomputed. The website does not publish SHA256; those values are our local hashes, not claimed publisher signatures.

The annotations contain 211,577 segment records including glia, orphans and unimportant segments; these are not all neurons. `bodyId: int64` is the original segment identifier. Other used fields are `status: string`, `superclass: string`, `type: string`. The actual complete schema is recorded, not guessed. The connectivity schema is exactly `body_pre: int64`, `body_post: int64`, `weight: int64`. It contains 151,856,684 segment-pair rows, including fragments; this row count must not be confused with the announcement's approximately 125 million synaptic connections among neurons.

Raw files are deliberately ignored and are **not committed**. The local cache is about 1.1 GB plus preprocessing products. The browser downloads only the derived graph and exported checkpoints. Original IDs are strings in the browser artifact; `source_index` and `target_index` index the stored node list. Edge IDs are original source/target pairs because the official pair table provides no individual edge ID column.

## Reproduce selection

```sh
python scripts/download_malecns.py
python scripts/preprocess_connectome.py
python scripts/select_subgraph.py
```

1. Keep annotations with `status == Traced` and `superclass` in `visual_projection`, `cb_intrinsic`, `descending_neuron`.
2. Stream actual connection batches; keep non-self pairs between these nodes with `weight >= 5`. This yields 42,675 candidate nodes and 2,039,742 candidate connections.
3. Choose the strongest visual-projection → descending-neuron connection; resolve ties by ascending original source ID, then target ID.
4. Expand a weakly connected neighborhood from its endpoints, always adding the strongest incident edge to one new node, with the same deterministic tie order. Stop at 96 nodes.
5. Keep induced edges; if exceeding 4,096, preserve the expansion tree first, then highest counts. The delivered graph has 602 edges, so no induced edge is removed by this cap.
6. Select up to 16 visual input nodes by retained outgoing count, then ID. Select up to 16 descending outputs reachable in at most four directed passes, by ID. The artifact has 11 inputs and 16 outputs. These are engineered selection rules, not a claim of natural game sensors or flight muscles.
7. Preserve all annotation columns and raw edge counts. Divide each edge count by the total retained incoming count of its destination to form its numerical weight.

Exact anchor IDs, retained synapse count, parameters, exclusion rationale and counts are stored in `graph.json.selection`. Exclusions keep CPU/browser computation small; this is neither the entire connectome nor a complete validated visual-to-motor pathway. Selection is based only on structure and annotations, not game reward or test results.
