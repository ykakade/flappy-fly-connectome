# Checkpoints

Each architecture (`graph`, `shuffled`, `mlp`) includes `untrained.pt`, `early.pt` (16,384 decisions), `mid.pt` (65,536 decisions), `best.pt`, and `last.pt`. Names refer to the actual recorded decision count; early and mid belong to the initial 131,072-decision pilot, followed by a longer continuation. Best is selected by the mean score on validation seeds 9001–9008, with first-best tie breaking. Test seeds never select a checkpoint.

Each `.pt` stores weights, fixed topology, Adam state, Torch/NumPy/Python RNG states, obstacle-seed generator state, all environment states, configuration, dataset graph SHA256, step count, wall time and hardware. Exact continuation is tested against uninterrupted training. Resume rejects incompatible graph, architecture or rollout configuration. Load only trusted PyTorch checkpoints; browser artifacts use plain JSON and cannot execute pickle code.

`dist/checkpoints.json` contains inference-only weights (no Python server required), plus actual step counts and measured evaluation means. It includes post-training interventions. MLP activations are explicitly labeled hidden units, never biological neurons.

`progress.json` is the complete local metric record, including failed early updates. The Training Journey tab is the equivalent local metric viewer: serve `dist` and view validation curves for all architectures. `dist/training.json` is an exported copy; rerun evaluation to refresh it after further training.
