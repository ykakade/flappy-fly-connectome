# Fruit-fly body

Anatomical body meshes from [TuragaLab/flybody](https://github.com/TuragaLab/flybody), revision `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.

Copyright Google DeepMind and HHMI Janelia Research Campus. Licensed under [Apache 2.0](LICENSE).

The meshes were simplified by vertex clustering and exported with their original body hierarchy and materials. The browser changes the pose and animates a front leg to press a button. These gestures illustrate game decisions; they are not a simulation of the fly’s muscles.

To rebuild, download the source repository and run:

```sh
python scripts/prepare_fly_mesh.py /path/to/flybody/fruitfly/assets
```
