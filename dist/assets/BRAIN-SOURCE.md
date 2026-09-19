# Brain surface

The surface is the official [MaleCNS brain shell](https://storage.googleapis.com/flyem-male-cns/rois/brain-shell-v2.2/mesh/brain-shell.ngmesh?generation=1758224654323697), object generation `1758224654323697`.

Source: FlyEM at HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as described in the [MaleCNS data release](https://male-cns.janelia.org/download/).

The mesh is simplified by vertex clustering. Nanometers are divided by eight so it shares the neuron annotations’ coordinate system. The surface is context only. The 96 colored nodes are the model’s selected neurons, and lines show its connections, not traced nerve paths.

Rebuild with `python scripts/prepare_brain_mesh.py /path/to/brain-shell.ngmesh`.
