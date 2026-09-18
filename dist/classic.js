import { RNG } from "./environment.js";

// Pixel coordinates at 288 x 512, with the floor at y = 400.
// The experiment environment stays separate so saved results remain reproducible.
export const CLASSIC = Object.freeze({
  width: 288, height: 512, floor: 400, birdX: 72,
  birdWidth: 34, birdHeight: 24, hitWidth: 24, hitHeight: 20,
  gravity: 0.25, flap: -4.5, maxFall: 5,
  speed: 2, pipeWidth: 52, gap: 100, spacing: 160,
});

export class ClassicEnvironment {
  constructor(seed = 1) {
    this.seed = seed;
    this.rng = new RNG(seed);
    this.y = 200 / CLASSIC.floor;
    this.vy = 0;
    this.frame = 0;
    this.score = 0;
    this.done = false;
    this.max_frames = Infinity;
    this.pipes = [];
    for (let i = 0; i < 3; i++) this.addPipe(360 + i * CLASSIC.spacing);
  }
  addPipe(x) {
    this.pipes.push({
      x: x / CLASSIC.width,
      center: (100 + Math.floor(this.rng.random() * 180)) / CLASSIC.floor,
      passed: false,
    });
  }
  tick(action) {
    if (this.done) return;
    this.vy = action ? CLASSIC.flap / CLASSIC.floor
      : Math.min(this.vy + CLASSIC.gravity / CLASSIC.floor, CLASSIC.maxFall / CLASSIC.floor);
    this.y += this.vy;
    // Touching the top stops upward travel; only pipes and ground end a run.
    if (this.y < CLASSIC.hitHeight / 2 / CLASSIC.floor) {
      this.y = CLASSIC.hitHeight / 2 / CLASSIC.floor;
      this.vy = 0;
    }
    this.frame++;
    const x = CLASSIC.birdX, y = this.y * CLASSIC.floor;
    for (const pipe of this.pipes) {
      pipe.x -= CLASSIC.speed / CLASSIC.width;
      const left = pipe.x * CLASSIC.width, center = pipe.center * CLASSIC.floor;
      if (x + CLASSIC.hitWidth / 2 > left && x - CLASSIC.hitWidth / 2 < left + CLASSIC.pipeWidth &&
        (y - CLASSIC.hitHeight / 2 < center - CLASSIC.gap / 2 ||
         y + CLASSIC.hitHeight / 2 > center + CLASSIC.gap / 2)) this.done = true;
    }
    if (y + CLASSIC.hitHeight / 2 >= CLASSIC.floor) this.done = true;
    if (!this.done) for (const pipe of this.pipes) {
      if (!pipe.passed && pipe.x * CLASSIC.width + CLASSIC.pipeWidth / 2 < x) {
        pipe.passed = true;
        this.score++;
      }
    }
    if (this.pipes[0].x * CLASSIC.width < -CLASSIC.pipeWidth) {
      this.pipes.shift();
      this.addPipe(this.pipes.at(-1).x * CLASSIC.width + CLASSIC.spacing);
    }
  }
}
