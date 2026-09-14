"""CPU PPO; full RNG/optimizer/environment checkpointing for exact continuation."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Bernoulli
from .environment import Environment, RNG
from .model import load_graph, make_model


def episode(model, seed, max_frames=3600, trajectory=False):
    env = Environment(seed, max_frames)
    rng = RNG(seed ^ 0xABCDEF)
    trace = []
    with torch.no_grad():
        while not env.done:
            logits, _, h = model(torch.tensor([env.observation()], dtype=torch.float32), True)
            p = float(logits.sigmoid()[0])
            action = int(rng.random() < p)
            if trajectory:
                trace.append(dict(frame=env.frame, probability=p, action=action, activity=h[0].tolist()))
            env.step(action)
    return dict(seed=seed, score=env.score, survival_seconds=env.frame / 60, frames=env.frame, trajectory=trace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["graph", "shuffled", "mlp"], default="graph")
    p.add_argument("--steps", type=int, default=131072, help="Total target policy decisions (4 physics frames each)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph", default="data/graph.json")
    p.add_argument("--out", type=Path, default=Path("checkpoints/graph"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-frames", type=int, default=3600)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    graph = load_graph(a.graph)
    graph_sha = hashlib.sha256(Path(a.graph).read_bytes()).hexdigest()
    model = make_model(graph, a.kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    seed_rng = np.random.default_rng(a.seed)

    def new_env():
        # Training range never overlaps held-out validation (9001+) or test (10001+) seeds.
        return Environment(int(seed_rng.integers(1, 8001)), a.max_frames)

    envs = [new_env() for _ in range(a.envs)]
    step = 0
    elapsed_before = 0.0
    best = -float("inf")
    history = []
    if a.resume:
        ck = torch.load(a.resume, weights_only=False, map_location="cpu")
        for field, value in [("graph_sha256", graph_sha), ("kind", a.kind)]:
            if ck[field] != value:
                raise ValueError("Resume mismatch: " + field)
        for field in ["envs", "rollout", "epochs", "max_frames", "seed"]:
            if ck["config"][field] != getattr(a, field):
                raise ValueError("Resume configuration mismatch: " + field)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        seed_rng.bit_generator.state = ck["seed_rng"]
        envs = [Environment.restore(s) for s in ck["environments"]]
        step = ck["steps"]
        elapsed_before = ck["wall_seconds"]
        best = ck["best_validation"]
        history = ck["history"]
    a.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    hardware = dict(
        platform=platform.platform(),
        processor=platform.processor(),
        machine=platform.machine(),
        device="CPU",
        torch_threads=1,
        torch_version=torch.__version__,
    )

    def save(name):
        ck = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            seed_rng=seed_rng.bit_generator.state,
            environments=[e.state() for e in envs],
            steps=step,
            wall_seconds=elapsed_before + time.perf_counter() - start,
            kind=a.kind,
            seed=a.seed,
            graph_sha256=graph_sha,
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            hardware=hardware,
            parameters=sum(x.numel() for x in model.parameters()),
            best_validation=best,
            history=history,
        )
        torch.save(ck, a.out / (name + ".pt"))

    if step == 0:
        save("untrained")
    completed = []
    while step < a.steps:
        observations = []
        actions = []
        logps = []
        rewards = []
        dones = []
        values = []
        for _ in range(a.rollout):
            obs = torch.tensor([e.observation() for e in envs], dtype=torch.float32)
            with torch.no_grad():
                logits, val = model(obs)
                distribution = Bernoulli(logits=logits)
                action = distribution.sample()
                lp = distribution.log_prob(action)
            rew = []
            done = []
            for i, e in enumerate(envs):
                _, r, d = e.step(int(action[i]))
                rew.append(r)
                done.append(d)
                if d:
                    completed.append(e.score)
                    envs[i] = new_env()
            observations.append(obs)
            actions.append(action)
            logps.append(lp)
            values.append(val)
            rewards.append(torch.tensor(rew))
            dones.append(torch.tensor(done, dtype=torch.float32))
        with torch.no_grad():
            _, next_value = model(torch.tensor([e.observation() for e in envs], dtype=torch.float32))
        vals = torch.stack(values)
        rs = torch.stack(rewards)
        ds = torch.stack(dones)
        adv = torch.zeros_like(rs)
        gae = torch.zeros(a.envs)
        # Episode duration is part of the finite-horizon task: max-frame endings are terminal.
        for t in reversed(range(a.rollout)):
            nv = next_value if t == a.rollout - 1 else vals[t + 1]
            delta = rs[t] + 0.99 * nv * (1 - ds[t]) - vals[t]
            gae = delta + 0.99 * 0.95 * (1 - ds[t]) * gae
            adv[t] = gae
        returns = (adv + vals).flatten()
        advantages = adv.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs = torch.cat(observations)
        act = torch.cat(actions)
        old_lp = torch.cat(logps)
        batch = len(act)
        for _ in range(a.epochs):
            for indices in torch.randperm(batch).split(512):
                logits, v = model(obs[indices])
                dist = Bernoulli(logits=logits)
                ratio = (dist.log_prob(act[indices]) - old_lp[indices]).exp()
                surrogate = torch.minimum(ratio * advantages[indices], ratio.clamp(0.8, 1.2) * advantages[indices])
                loss = -surrogate.mean() + 0.5 * (v - returns[indices]).square().mean() - 0.005 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        step += batch
        record = dict(
            steps=step,
            mean_recent_train_score=float(np.mean(completed[-100:])) if completed else 0.0,
            loss=float(loss.detach()),
            wall_seconds=elapsed_before + time.perf_counter() - start,
        )
        if step % (batch * 4) == 0 or step >= a.steps:
            validation = [episode(model, s, a.max_frames)["score"] for s in range(9001, 9009)]
            record["validation_mean"] = float(np.mean(validation))
            if record["validation_mean"] > best:
                best = record["validation_mean"]
                history.append(record)
                save("best")
                history.pop()
            print(json.dumps(record), flush=True)
        history.append(record)
        if step == batch * 4:
            save("early")
        if step == batch * 16:
            save("mid")
        save("last")
        (a.out / "progress.json").write_text(json.dumps(history, indent=2))
    print("Finished", a.kind, step, "decisions", flush=True)


if __name__ == "__main__":
    main()
