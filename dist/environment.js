// Reference: src/flappy_fly/environment.py. Fixed 60 Hz, action once per four frames.
export class RNG {
  constructor(seed) {
    this.state = seed >>> 0;
  }
  random() {
    this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0;
    return this.state / 4294967296;
  }
}
export class Environment {
  constructor(seed = 1, maxFrames = 3600) {
    this.rng = new RNG(seed);
    this.seed = seed;
    this.max_frames = maxFrames;
    this.y = 0.5;
    this.vy = 0;
    this.previous = 0;
    this.frame = 0;
    this.score = 0;
    this.done = false;
    this.pipes = [];
    for (let i = 0; i < 3; i++) this.addPipe(1.1 + 0.55 * i);
  }
  addPipe(x) {
    this.pipes.push({
      x,
      center: 0.28 + 0.44 * this.rng.random(),
      passed: false,
    });
  }
  observation() {
    const p = this.pipes.find((p) => p.x + 0.09 >= 0.22 - 0.022);
    return [
      2 * this.y - 1,
      this.vy / 0.02,
      p.x - 0.22,
      2 * (p.center - 0.16) - 1,
      2 * (p.center + 0.16) - 1,
      this.previous,
    ];
  }
  tick(action) {
    if (this.done) return 0;
    if (action) this.vy = -0.012;
    this.previous = +!!action;
    this.vy = Math.min(this.vy + 0.0006, 0.025);
    this.y += this.vy;
    this.frame++;
    let reward = 0.01 - 0.001 * !!action;
    for (const p of this.pipes) {
      p.x -= 0.0035;
      if (!p.passed && p.x + 0.09 < 0.22 - 0.022) {
        p.passed = true;
        this.score++;
        reward++;
      }
      if (
        p.x < 0.22 + 0.022 &&
        p.x + 0.09 > 0.22 - 0.022 &&
        (this.y - 0.022 < p.center - 0.16 || this.y + 0.022 > p.center + 0.16)
      )
        this.done = true;
    }
    if (this.y < 0.022 || this.y > 1 - 0.022) this.done = true;
    if (this.done) reward--;
    if (this.frame >= this.max_frames) this.done = true;
    if (this.pipes[0].x < -0.1) {
      this.pipes.shift();
      this.addPipe(this.pipes.at(-1).x + 0.55);
    }
    return reward;
  }
  step(action) {
    let reward = 0;
    for (let i = 0; i < 4; i++) {
      reward += this.tick(i === 0 ? action : 0);
      if (this.done) break;
    }
    this.previous = action;
    return [this.observation(), reward, this.done];
  }
  state() {
    return {
      seed: this.seed,
      max_frames: this.max_frames,
      rng: this.rng.state,
      y: this.y,
      vy: this.vy,
      previous: this.previous,
      frame: this.frame,
      score: this.score,
      done: this.done,
      pipes: this.pipes.map((p) => ({ ...p })),
    };
  }
}
