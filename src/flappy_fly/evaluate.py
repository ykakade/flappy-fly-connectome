"""Evaluate exported checkpoints on common unseen seeds and emit browser inference artifacts."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .model import load_graph, make_model
from .train import episode


def restore(path, graph, graph_sha):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["graph_sha256"] != graph_sha:
        raise ValueError("Checkpoint graph checksum mismatch")
    model = make_model(graph, ck["kind"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=10001)
    p.add_argument("--max-frames", type=int, default=3600)
    a = p.parse_args()
    if a.seed_start < 10001:
        raise ValueError("Held-out seed range must begin at 10001 or above")
    torch.set_num_threads(1)
    graph = load_graph()
    sha = hashlib.sha256(Path("data/graph.json").read_bytes()).hexdigest()
    specs = [
        ("untrained", "checkpoints/graph/untrained.pt", "none"),
        ("early", "checkpoints/graph/early.pt", "none"),
        ("mid", "checkpoints/graph/mid.pt", "none"),
        ("best", "checkpoints/graph/best.pt", "none"),
        ("shuffled", "checkpoints/shuffled/best.pt", "none"),
        ("mlp", "checkpoints/mlp/best.pt", "none"),
        ("lesion", "checkpoints/graph/best.pt", "lesion"),
        ("random", "checkpoints/graph/best.pt", "random"),
        ("bypass", "checkpoints/graph/best.pt", "bypass"),
    ]
    results = []
    exports = {}
    for name, path, intervention in specs:
        model, ck = restore(path, graph, sha)
        model.intervention = intervention
        episodes = [episode(model, seed, a.max_frames) for seed in range(a.seed_start, a.seed_start + a.episodes)]
        for e in episodes:
            e.pop("trajectory")
        scores = [e["score"] for e in episodes]
        rec = dict(
            name=name,
            kind=ck["kind"],
            intervention=intervention,
            mean=float(np.mean(scores)),
            median=float(np.median(scores)),
            std=float(np.std(scores)),
            survival_seconds=float(np.mean([e["survival_seconds"] for e in episodes])),
            evaluation_episodes=a.episodes,
            evaluation_seeds=[e["seed"] for e in episodes],
            training_steps=ck["steps"],
            training_seed=ck["seed"],
            parameters=ck["parameters"],
            checkpoint=path,
            checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            wall_seconds=ck["wall_seconds"],
            hardware=ck["hardware"],
            episodes=episodes,
        )
        results.append(rec)
        exports[name] = dict(
            kind=ck["kind"],
            intervention=intervention,
            weights={k: v.tolist() for k, v in model.state_dict().items()},
            training_steps=ck["steps"],
            parameters=ck["parameters"],
            evaluation_mean=rec["mean"],
            checkpoint=path,
        )
        if intervention == "random":
            generator = torch.Generator().manual_seed(991)
            exports[name]["random_edge_weights"] = (
                torch.randn(len(model.edge_weight), generator=generator) * model.edge_weight
            ).tolist()
        print(name, rec["mean"], rec["survival_seconds"], flush=True)
    report = dict(
        graph_sha256=sha,
        dataset="MaleCNS v1.0",
        max_frames=a.max_frames,
        decision_repeat=4,
        evaluation_policy="Bernoulli sample using LCG(seed XOR 0xABCDEF); same random uniforms across controllers; no test-seed checkpoint selection",
        std_definition="Population standard deviation (ddof=0)",
        validation_seeds=list(range(9001, 9009)),
        training_seeds="Obstacle seeds drawn with replacement from 1..8000",
        results=results,
    )
    Path("metrics").mkdir(exist_ok=True)
    runs = []
    for kind in ["graph", "shuffled", "mlp"]:
        last = torch.load(f"checkpoints/{kind}/last.pt", weights_only=False, map_location="cpu")
        runs.append(
            dict(
                kind=kind,
                training_decisions=last["steps"],
                wall_seconds=last["wall_seconds"],
                hardware=last["hardware"],
                seed=last["seed"],
                parameters=last["parameters"],
                checkpoint=f"checkpoints/{kind}/last.pt",
            )
        )
    Path("metrics/training-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    Path("metrics/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("dist/checkpoints.json").write_text(json.dumps(exports, separators=(",", ":")) + "\n")
    shutil.copy("data/graph.json", "dist/graph.json")
    shutil.copy("metrics/evaluation.json", "dist/evaluation.json")
    shutil.copy("data/dataset.json", "dist/dataset.json")
    journey = {k: json.loads(Path(f"checkpoints/{k}/progress.json").read_text()) for k in ["graph", "shuffled", "mlp"]}
    Path("dist/training.json").write_text(json.dumps(journey) + "\n")


if __name__ == "__main__":
    main()
