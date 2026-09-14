# Flappy Fly

**Can a controller built from real fruit-fly wiring learn a Flappy Bird–style game?** In this small state-input experiment, yes: the selected MaleCNS-connectome-derived controller improved from **0.00 to 16.88 gates** on 32 unseen seeds. The shuffled graph and MLP scored higher. **These results do not establish an advantage for biological wiring.**

[Browser demo](https://flappy-fly-connectome.pox.chatgpt.site) · [GitHub](https://github.com/ykakade/flappy-fly-connectome) · [Official dataset](https://male-cns.janelia.org/download/)

**MaleCNS v1.0 · 96 original neurons · 602 directed connections · 69,237 summed synapse counts.** A deterministic algorithm selects a connected neighborhood of the official visual-projection → descending-neuron edge **10002 → 10059**, preserving original IDs, annotations and edge counts.

> This is a **connectome-derived controller experiment**, not a real fly playing a game or a biologically validated brain simulation. MaleCNS supplies structural wiring. Tanh dynamics, game-state adapters, PPO and the animated fly body are engineering additions. Inputs are structured state, **not pixels**. No consciousness, biological plasticity or full-brain emulation is claimed.

## Run the saved browser demo

```sh
git clone https://github.com/ykakade/flappy-fly-connectome.git
cd flappy-fly-connectome
npm run build
npm run dev
```

Open http://localhost:4173. This path needs Node 20+ and Python 3, with **no npm runtime dependencies**. Inference runs entirely in JavaScript; the hosted demo requires no local server or raw connectome download.

- **Play:** space/up arrow, mouse click or tap to flap; restart button or R to reset. Best score is device-local and includes all modes.
- **Watch AI:** online inference from a saved checkpoint; actions sampled from its flap probability.
- **Human vs. Fly:** solid human and outlined AI avatars use independent environments with identical obstacle seeds. Both scores are shown.
- **Training Journey:** untrained, early, mid and validation-selected best checkpoints on the same seed; validation curves for all architectures.
- **Ablation Lab:** selected intervention versus original controller, same obstacle and action-sampling seed, with scores, score difference and probability traces.

All 96 retained neurons are displayed in the activity panel. Hover for original IDs and annotations. Each color is the actual final-pass tanh activation, raw −1…+1, with **no aggregation**. Positions are schematic, not anatomy. MLP values are labeled hidden units. Human mode generates no neural activity; final colors after collision are the final computed values, not a recorded movie.

## Measured results

Each architecture received **2,097,152 policy decisions**, seed **42**, under identical PPO settings. Best checkpoints were selected only on validation seeds **9001–9008**, with first-best tie breaking. Selected checkpoints consequently have different step counts. Test evaluation uses the same **32 unseen seeds, 10001–10032**, a **3,600-frame / 60-second cap**, and common seeded action-sampling uniforms. The score ceiling is 22 gates.

| Controller | Mean | Median | SD | Survival (s) | Selected decisions |
|---|---:|---:|---:|---:|---:|
| untrained | 0.00 | 0.0 | 0.00 | 2.06 | 0 |
| early | 0.00 | 0.0 | 0.00 | 2.14 | 16,384 |
| mid | 0.09 | 0.0 | 0.29 | 3.54 | 65,536 |
| best | 16.88 | 22.0 | 6.60 | 47.48 | 573,440 |
| shuffled | 18.44 | 22.0 | 6.44 | 51.20 | 786,432 |
| mlp | 18.16 | 22.0 | 7.00 | 50.42 | 1,146,880 |
| lesion | 0.00 | 0.0 | 0.00 | 1.72 | 573,440 |
| random | 0.00 | 0.0 | 0.00 | 1.17 | 573,440 |
| bypass | 0.19 | 0.0 | 0.39 | 4.49 | 573,440 |

Full training time on Apple M1 CPU (one thread per process; runs overlapped): graph **276.8 s**, shuffled **273.7 s**, mlp **45.9 s**.

The original and shuffled graphs have **383 trainable parameters**; the MLP has **380**, 0.8% fewer. Shuffling preserves exact node/edge counts and in/out-degree sequences. Total decisions, environment, PPO settings and checkpoint selection are controlled; wall-clock compute and effective expressivity are not matched.

Interventions are **post-training disruptions without retraining**: output lesion zeros all incoming edges to selected readouts; random weights multiply fixed normalized strengths by seeded standard-normal draws (seed 991, not biological signs); bypass sends mean input-adapter tanh activity directly to every output readout slot. They test this implementation's dependence on its graph, not superiority over a separately optimized bypass architecture.

Full per-episode scores, median, population SD (`ddof=0`), survival, seeds, checkpoint paths/hashes, step counts, parameter counts and hardware are in [metrics/evaluation.json](metrics/evaluation.json). Complete training histories, including early zero-score failures, are preserved in `checkpoints/*/progress.json`. Additional run metadata is in [metrics/training-runs.json](metrics/training-runs.json).

## Reproduce from official data

The release used Python 3.13 on Apple M1, macOS 26.2, Torch CPU with one thread per process. Reserve about 1.1 GB of download/cache space plus several GB of RAM for preprocessing. Raw files are ignored by Git. Exact versions are recorded in `requirements-release.txt`; portable installation can instead use `pip install -e '.[dev]'`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-release.txt
pip install --no-deps -e .

# Pin official object generations, verify publisher MD5/size, record SHA256 and actual schemas.
python scripts/download_malecns.py
python scripts/preprocess_connectome.py
python scripts/select_subgraph.py

# Headless smoke training and continuation.
python -m flappy_fly.train --steps 4096 --out checkpoints/smoke
python -m flappy_fly.train --steps 8192 --out checkpoints/smoke --resume checkpoints/smoke/last.pt

# Equal release budgets, separate architectures.
python -m flappy_fly.train --kind graph --seed 42 --steps 2097152 --out checkpoints/graph
python -m flappy_fly.train --kind shuffled --seed 42 --steps 2097152 --out checkpoints/shuffled
python -m flappy_fly.train --kind mlp --seed 42 --steps 2097152 --out checkpoints/mlp

# Held-out evaluation and browser export.
python -m flappy_fly.evaluate --episodes 32 --seed-start 10001
pytest -q
ruff check .
npm test
npm run build
npm run dev
```

The actual release ran a **131,072-decision pilot for each architecture**, then continued each `last.pt` to 2,097,152. The exact continuation commands were:

```sh
python -m flappy_fly.train --kind graph --steps 2097152 --out checkpoints/graph --resume checkpoints/graph/last.pt
python -m flappy_fly.train --kind shuffled --steps 2097152 --out checkpoints/shuffled --resume checkpoints/shuffled/last.pt
python -m flappy_fly.train --kind mlp --steps 2097152 --out checkpoints/mlp --resume checkpoints/mlp/last.pt
```

For a longer run, use `--steps 8388608` with `--resume checkpoints/graph/last.pt`. Steps mean **total target policy decisions**, rounded up to complete rollout batches (default 4,096). Episode length (`--max-frames`), environments, rollout, epochs, seed, output and graph are configurable. Checkpoints save Adam, every RNG and environment state; resume rejects incompatible settings. Interrupted/uninterrupted training is tested for exact agreement within the same software/hardware environment.

**Local metric viewer:** serve `dist` and open Training Journey. `progress.json` records every update; validation is every fourth update. Rerun evaluation to refresh `dist/training.json` after additional training. This is the equivalent local viewer; TensorBoard is not required.

## Data and model

[Data documentation](data/README.md) records the verified official files, Feather v2 schemas, immutable generation URLs, publisher checksums, complete selection algorithm, attribution and CC BY 4.0 terms. `data/dataset.json` stores machine-readable provenance. No column names were inferred from examples.

Selection keeps `Traced` visual-projection, central-intrinsic and descending neurons, removes self-connections and counts below five, then greedily expands the strongest incident edge from a real visual→descending anchor. It stops at 96 nodes, preserving connectedness. The 4,096-edge cap does not remove any of this subset's 602 induced edges. The subset is selected from structure alone, independent of game reward. Most neurons are excluded for CPU/browser scale, not biological irrelevance.

The six observations are normalized vertical position, vertical velocity, horizontal distance, gap top, gap bottom and previous action. A trained 6→11 affine adapter injects state into **11 officially annotated visual-projection nodes**. Initialize `h = tanh(injection)`, then perform four sparse directed passes:

```text
message[target] += h[source] × normalized_synapse_count[source,target]
h = tanh(injection + trained_node_bias + message × trained_node_gain)
P(flap) = sigmoid(trained_readout(h[16 selected descending nodes]))
```

PyTorch indexed accumulation and JavaScript edge loops avoid a dense N×N adjacency. Incoming strengths are normalized by the total retained incoming synapse count of each target. **Topology and normalized counts remain fixed during ordinary training.** Adapters, per-node gains/biases, policy and critic are trained. Gains are numerical model parameters, not inferred excitatory/inhibitory types. Official transmitter predictions were not downloaded or used. The critic reads all nodes during training but never selects browser actions. Activities reset and are recomputed each decision; no persistent biological dynamical state is claimed.

PPO: gamma 0.99, GAE lambda 0.95, clip 0.2, Adam 0.0003, four epochs, minibatches 512, value coefficient 0.5, entropy 0.005, gradient clipping 0.5. Initial policy bias −2 is shared engineered exploration initialization. There are no hardcoded successful action sequences or single fixed training levels.

**Reward per physics frame:** +0.01 survival, +1 gate passed, −1 collision, −0.001 flap. Terminal frames also receive the survival term. Two actions are requested once per four 60-Hz physics frames. The time cap is a terminal finite-horizon condition. Training obstacle seeds are sampled with replacement from 1…8000; validation/test ranges are disjoint. The avatar's artwork can extend outside its normalized collision bounds.

The reference physics is `src/flappy_fly/environment.py`, mirrored in `dist/environment.js`. Recorded-action tests check every state exactly, including RNG, collisions, passing and subsequent gates. Browser probabilities and activations agree with Torch within 2e-6; **all 288 delivered evaluation episodes reproduce their scores and survival frames in JavaScript**.

## Repository and CI

```text
data/                Provenance and small original-ID graph
scripts/             Official download, preprocessing, selection, build validation
src/flappy_fly/       Headless physics, sparse controller, PPO, evaluation
dist/                Static demo and browser inference artifacts
configs/             Smoke/release parameter records
checkpoints/         Full restorable models and progress metrics
metrics/             Measured results and verification record
tests/               Physics parity, graph integrity, restoration, resume, browser inference
.github/workflows/   Python/lint/frontend CI and optional GitHub Pages workflow
```

GitHub Pages can also host `dist/`: select GitHub Actions as the Pages source and run the manual **Publish GitHub Pages (optional)** workflow. The primary release uses Sites static hosting. Only trusted `.pt` checkpoints should be loaded; public browser assets use plain JSON.

## Limitations

- One 96-node structural subset, one training seed per architecture. No pixel vision, full-brain dynamics, biological plasticity or consciousness.
- An easy engineered game with a 60-second ceiling. Median scores saturate; longer/harder episodes and multiple training seeds/subgraphs are needed for stronger conclusions.
- Finite training seed range. Consecutive test seeds have correlated first LCG draws, though later obstacles diversify. Different physics, noisy observations and raw images are untested.
- Parameter count does not match compute or expressivity. Lesions/bypass are post-training, not separately optimized baselines. Shuffled and MLP scores exceed the original graph here.
- Exact resume is local to a software/hardware environment. Different Torch versions or architectures may change learning trajectories; browser Float64 inference over Float32 weights can theoretically diverge at marginal future thresholds.
- Original fly artwork is AI-generated. Wing motion is a cosmetic bob, not a biomechanical simulation. No third-party game artwork is copied.

## License and citations

Original code: [MIT](LICENSE). MaleCNS and modified graph artifacts: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit **FlyEM (HHMI Janelia), University of Cambridge, MRC Laboratory of Molecular Biology and Google Research**. Selection, indexing and normalization are disclosed modifications; no endorsement is implied.

Official publication: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, **Cell (2026)**, [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015). See [Janelia](https://www.janelia.org/project-team/flyem/male-cns-connectome), the [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/), [PPO](https://arxiv.org/abs/1707.06347) and [CITATION.cff](CITATION.cff).
