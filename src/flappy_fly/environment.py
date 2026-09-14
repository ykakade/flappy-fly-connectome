"""Deterministic reference physics. Mirrored in dist/environment.js; parity tested."""


class RNG:
    def __init__(self, seed):
        self.state = int(seed) & 0xFFFFFFFF

    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296


class Environment:
    FPS = 60
    REPEAT = 4

    def __init__(self, seed=1, max_frames=3600):
        self.rng = RNG(seed)
        self.seed = seed
        self.max_frames = max_frames
        self.y, self.vy, self.previous = 0.5, 0.0, 0
        self.frame, self.score, self.done = 0, 0, False
        self.pipes = []
        for i in range(3):
            self.add_pipe(1.10 + 0.55 * i)

    def add_pipe(self, x):
        self.pipes.append(dict(x=x, center=0.28 + 0.44 * self.rng.random(), passed=False))

    def observation(self):
        p = next(p for p in self.pipes if p["x"] + 0.09 >= 0.22 - 0.022)
        return [
            2 * self.y - 1,
            self.vy / 0.02,
            p["x"] - 0.22,
            2 * (p["center"] - 0.16) - 1,
            2 * (p["center"] + 0.16) - 1,
            float(self.previous),
        ]

    def tick(self, action):
        if self.done:
            return 0.0
        if action:
            self.vy = -0.012
        self.previous = int(bool(action))
        self.vy = min(self.vy + 0.0006, 0.025)
        self.y += self.vy
        self.frame += 1
        reward = 0.01 - 0.001 * bool(action)
        for p in self.pipes:
            p["x"] -= 0.0035
            if not p["passed"] and p["x"] + 0.09 < 0.22 - 0.022:
                p["passed"] = True
                self.score += 1
                reward += 1.0
            if p["x"] < 0.22 + 0.022 and p["x"] + 0.09 > 0.22 - 0.022:
                if self.y - 0.022 < p["center"] - 0.16 or self.y + 0.022 > p["center"] + 0.16:
                    self.done = True
        if self.y < 0.022 or self.y > 1 - 0.022:
            self.done = True
        if self.done:
            reward -= 1.0
        if self.frame >= self.max_frames:
            self.done = True
        if self.pipes[0]["x"] < -0.1:
            self.pipes.pop(0)
            self.add_pipe(self.pipes[-1]["x"] + 0.55)
        return reward

    def step(self, action):
        reward = 0.0
        for i in range(self.REPEAT):
            reward += self.tick(action if i == 0 else 0)
            if self.done:
                break
        self.previous = int(action)
        return self.observation(), reward, self.done

    def state(self):
        return dict(
            seed=self.seed,
            max_frames=self.max_frames,
            rng=self.rng.state,
            y=self.y,
            vy=self.vy,
            previous=self.previous,
            frame=self.frame,
            score=self.score,
            done=self.done,
            pipes=[p.copy() for p in self.pipes],
        )

    @classmethod
    def restore(cls, state):
        env = cls(state["seed"], state["max_frames"])
        for k, v in state.items():
            if k != "rng":
                setattr(env, k, v)
        env.rng.state = state["rng"]
        return env
