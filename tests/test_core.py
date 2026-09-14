import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from flappy_fly.environment import Environment, RNG
from flappy_fly.model import load_graph, make_model, topology
from flappy_fly.train import episode

ROOT = Path(__file__).resolve().parents[1]


def node(payload):
    return json.loads(
        subprocess.check_output(["node", "tests/parity.mjs"], input=json.dumps(payload), text=True, cwd=ROOT)
    )


def test_real_graph_integrity():
    graph = load_graph(ROOT / "data/graph.json")
    ids = {n["bodyId"] for n in graph["nodes"]}
    assert len(ids) == 96 and len(graph["edges"]) == 602
    assert all(n["status"] == "Traced" for n in graph["nodes"])
    assert all(graph["nodes"][i]["superclass"] == "visual_projection" for i in graph["inputs"])
    assert all(graph["nodes"][i]["superclass"] == "descending_neuron" for i in graph["outputs"])
    pairs = set()
    for edge in graph["edges"]:
        assert edge["source"] in ids and edge["target"] in ids
        assert edge["count"] >= 5
        assert graph["nodes"][edge["source_index"]]["bodyId"] == edge["source"]
        assert graph["nodes"][edge["target_index"]]["bodyId"] == edge["target"]
        pairs.add((edge["source"], edge["target"]))
    assert len(pairs) == 602
    reached = {next(iter(ids))}
    for _ in range(len(ids)):
        reached |= {v for u, v in pairs if u in reached} | {u for u, v in pairs if v in reached}
    assert reached == ids
    metadata = json.loads((ROOT / "data/dataset.json").read_text())
    assert metadata["version"] == "v1.0"
    assert metadata["files"][1]["md5"] == "f30e9dcca25cfd021bf1e7b3d975599e"


@pytest.mark.parametrize("seed", [1, 42, 10001, 4294967295])
def test_physics_parity(seed):
    rng = RNG(123456)
    actions = [int(rng.random() < 0.13) for _ in range(1000)]
    env = Environment(seed)
    expected = []
    for a in actions:
        env.step(a)
        expected.append(env.state())
    actual = node(dict(mode="environment", seed=seed, max_frames=3600, actions=actions))
    assert actual == expected


def test_physics_passing_and_pipe_generation():
    env = Environment(997, 3600)
    actions = []
    expected = []
    # Test-only feedback policy exercises score, scrolling and RNG after multiple gates.
    for _ in range(900):
        p = next(p for p in env.pipes if p["x"] + 0.09 >= 0.198)
        action = int(env.y > p["center"] + 0.03 and env.vy > 0)
        actions.append(action)
        env.step(action)
        expected.append(env.state())
        if env.done:
            break
    assert env.score >= 1
    actual = node(dict(mode="environment", seed=997, max_frames=3600, actions=actions))
    assert actual == expected


def test_rng_and_state_restore():
    env = Environment(42)
    for a in [0, 0, 1, 0, 0]:
        env.step(a)
    restored = Environment.restore(env.state())
    for a in [1, 0, 0, 0] * 20:
        assert restored.step(a) == env.step(a)
        assert restored.state() == env.state()


def test_shuffled_control_is_degree_and_parameter_matched():
    g = load_graph(ROOT / "data/graph.json")
    s, d, w = topology(g)
    ss, dd, ww = topology(g, "shuffled")
    assert len(s) == len(ss)
    assert set(zip(s, d)) != set(zip(ss, dd))
    assert len(set(zip(ss, dd))) == len(ss)
    assert np.array_equal(np.bincount(s), np.bincount(ss))
    assert np.array_equal(np.bincount(d), np.bincount(dd))
    original = make_model(g)
    shuffled = make_model(g, "shuffled")
    mlp = make_model(g, "mlp")

    def count(m):
        return sum(p.numel() for p in m.parameters())

    assert count(original) == count(shuffled)
    assert abs(count(mlp) - count(original)) / count(original) < 0.02


def test_checkpoint_restore_and_deterministic_evaluation(tmp_path):
    g = load_graph(ROOT / "data/graph.json")
    torch.manual_seed(42)
    m = make_model(g)
    path = tmp_path / "smoke.pt"
    torch.save(m.state_dict(), path)
    other = make_model(g)
    other.load_state_dict(torch.load(path, weights_only=True))
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0]])
    assert torch.equal(m(x)[0], other(x)[0])
    assert episode(m, 10001) == episode(other, 10001)


@pytest.mark.skipif(
    not (ROOT / "dist/checkpoints.json").exists(), reason="Export checkpoints before full integration tests"
)
def test_all_browser_controllers_match_torch():
    g = load_graph(ROOT / "data/graph.json")
    artifacts = json.loads((ROOT / "dist/checkpoints.json").read_text())
    observations = np.random.default_rng(124).uniform(-1, 1, (40, 6)).tolist()
    for name, artifact in artifacts.items():
        ck = torch.load(ROOT / artifact["checkpoint"], weights_only=False, map_location="cpu")
        m = make_model(g, artifact["kind"])
        m.load_state_dict(ck["model"])
        m.intervention = artifact["intervention"]
        with torch.no_grad():
            logits, _, h = m(torch.tensor(observations, dtype=torch.float32), True)
        actual = node(dict(mode="controller", artifact=artifact, observations=observations))
        np.testing.assert_allclose(
            [r["probability"] for r in actual], logits.sigmoid().numpy(), atol=2e-6, rtol=2e-6, err_msg=name
        )
        np.testing.assert_allclose([r["activity"] for r in actual], h.numpy(), atol=2e-6, rtol=2e-6, err_msg=name)


def test_checkpoint_graph_hash():
    path = ROOT / "checkpoints/graph/best.pt"
    if not path.exists():
        pytest.skip("Train smoke checkpoint first")
    ck = torch.load(path, weights_only=False, map_location="cpu")
    assert ck["graph_sha256"] == hashlib.sha256((ROOT / "data/graph.json").read_bytes()).hexdigest()


def test_resume_matches_uninterrupted_training(tmp_path):
    import sys

    def run(out, steps, resume=None):
        cmd = [
            sys.executable,
            "-m",
            "flappy_fly.train",
            "--steps",
            str(steps),
            "--envs",
            "4",
            "--rollout",
            "8",
            "--out",
            str(out),
        ]
        if resume:
            cmd += ["--resume", str(resume)]
        subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)

    full = tmp_path / "full"
    split = tmp_path / "split"
    run(full, 256)
    run(split, 128)
    run(split, 256, split / "last.pt")
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(split / "last.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert a["environments"] == b["environments"]
