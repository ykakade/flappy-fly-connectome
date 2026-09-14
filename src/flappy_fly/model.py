"""Fixed sparse connectome topology with engineered adapters; no biological sign claims."""

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def load_graph(path="data/graph.json"):
    return json.loads(Path(path).read_text())


def topology(graph, kind="graph", seed=77):
    src = np.array([e["source_index"] for e in graph["edges"]])
    dst = np.array([e["target_index"] for e in graph["edges"]])
    counts = np.array([e["count"] for e in graph["edges"]], dtype=float)
    rng = np.random.default_rng(seed)
    if kind == "shuffled":
        # Directed double-edge swaps preserve in/out-degree, node/edge counts; no duplicates/self edges.
        pairs = set(zip(src.tolist(), dst.tolist()))
        for _ in range(len(src) * 20):
            i, j = rng.integers(len(src), size=2)
            u, v, x, y = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
            if u == x or v == y or u == y or x == v or (u, y) in pairs or (x, v) in pairs:
                continue
            pairs.remove((u, v))
            pairs.remove((x, y))
            pairs.add((u, y))
            pairs.add((x, v))
            dst[i], dst[j] = y, v
    totals = np.bincount(dst, weights=counts, minlength=len(graph["nodes"]))
    weight = counts / np.maximum(totals[dst], 1)
    return src, dst, weight


class Controller(nn.Module):
    def __init__(self, graph, kind="graph", seed=77):
        super().__init__()
        self.kind = kind
        self.n = len(graph["nodes"])
        self.passes = 4
        self.register_buffer("inputs", torch.tensor(graph["inputs"], dtype=torch.long))
        self.register_buffer("outputs", torch.tensor(graph["outputs"], dtype=torch.long))
        src, dst, weight = topology(graph, kind, seed)
        self.register_buffer("src", torch.tensor(src))
        self.register_buffer("dst", torch.tensor(dst))
        self.register_buffer("edge_weight", torch.tensor(weight, dtype=torch.float32))
        self.input = nn.Linear(6, len(graph["inputs"]))
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.gain = nn.Parameter(torch.ones(self.n))
        self.policy = nn.Linear(len(graph["outputs"]), 1)
        self.value = nn.Linear(self.n, 1)
        nn.init.constant_(self.policy.bias, -2.0)
        nn.init.normal_(self.policy.weight, 0, 0.05)
        self.intervention = "none"

    def forward(self, x, return_activity=False):
        injection = torch.zeros((len(x), self.n), device=x.device)
        injection[:, self.inputs] = self.input(x)
        h = torch.tanh(injection)
        weight = self.edge_weight
        if self.intervention == "lesion":
            weight = weight * (~torch.isin(self.dst, self.outputs)).float()
        elif self.intervention == "random":
            g = torch.Generator().manual_seed(991)
            weight = torch.randn(len(weight), generator=g, device="cpu").to(x.device) * weight
        for _ in range(self.passes):
            msg = torch.zeros_like(h).index_add(1, self.dst, h[:, self.src] * weight)
            h = torch.tanh(injection + self.bias + msg * self.gain)
        if self.intervention == "bypass":
            # Post-training intervention; average input-adapter activity feeds every readout node.
            out = torch.tanh(self.input(x)).mean(dim=1, keepdim=True).expand(-1, len(self.outputs))
        else:
            out = h[:, self.outputs]
        logits = self.policy(out).squeeze(-1)
        value = self.value(h).squeeze(-1)
        return (logits, value, h) if return_activity else (logits, value)


class MLP(nn.Module):
    def __init__(self, graph, kind="mlp", seed=77):
        super().__init__()
        budget = 7 * len(graph["inputs"]) + 3 * len(graph["nodes"]) + len(graph["outputs"]) + 2
        self.hidden = nn.Linear(6, round((budget - 2) / 9))
        self.policy = nn.Linear(self.hidden.out_features, 1)
        self.value = nn.Linear(self.hidden.out_features, 1)
        self.kind = "mlp"
        nn.init.constant_(self.policy.bias, -2.0)
        nn.init.normal_(self.policy.weight, 0, 0.05)

    def forward(self, x, return_activity=False):
        h = torch.tanh(self.hidden(x))
        logits, value = self.policy(h).squeeze(-1), self.value(h).squeeze(-1)
        return (logits, value, h) if return_activity else (logits, value)


def make_model(graph, kind="graph", seed=77):
    return MLP(graph) if kind == "mlp" else Controller(graph, kind, seed)
