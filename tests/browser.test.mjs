import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { Environment, RNG } from "../dist/environment.js";
import { Controller } from "../dist/controller.js";
test("fixed seed gives identical full episodes", () => {
  const a = new Environment(101),
    b = new Environment(101),
    r = new RNG(191);
  for (let i = 0; i < 1000; i++) {
    let action = +(r.random() < 0.12);
    assert.deepEqual(a.step(action), b.step(action));
    assert.deepEqual(a.state(), b.state());
  }
});
test("exported controllers run finite live inference", () => {
  const checkpoints = JSON.parse(fs.readFileSync("dist/checkpoints.json"));
  for (const artifact of Object.values(checkpoints)) {
    const controller = new Controller(artifact),
      env = new Environment(10001);
    for (let i = 0; i < 10; i++) {
      const r = controller.forward(env.observation());
      assert.ok(
        Number.isFinite(r.probability) &&
          r.probability >= 0 &&
          r.probability <= 1,
      );
      assert.ok(r.activity.every(Number.isFinite));
      env.step(+(r.probability > 0.5));
    }
  }
});
test("all exported evaluation scores reproduce in JS", () => {
  const cp = JSON.parse(fs.readFileSync("dist/checkpoints.json")),
    report = JSON.parse(fs.readFileSync("dist/evaluation.json"));
  for (const row of report.results) {
    const m = new Controller(cp[row.name]);
    for (const expected of row.episodes) {
      const env = new Environment(expected.seed, report.max_frames),
        r = new RNG(expected.seed ^ 0xabcdef);
      while (!env.done) {
        const p = m.forward(env.observation()).probability;
        env.step(+(r.random() < p));
      }
      assert.equal(
        env.score,
        expected.score,
        row.name + " score " + expected.seed,
      );
      assert.equal(
        env.frame,
        expected.frames,
        row.name + " duration " + expected.seed,
      );
    }
  }
});

const { ClassicEnvironment, CLASSIC } = await import("../dist/classic.js");
test("classic controls flap immediately and a fall ends at the ground", () => {
  const env = new ClassicEnvironment(10001);
  env.tick(1);
  assert.ok(env.y < .5);
  assert.equal(env.vy * CLASSIC.floor, CLASSIC.flap);
  while (!env.done && env.frame < 1000) env.tick(0);
  assert.ok(env.done);
  assert.equal(env.score, 0);
  assert.ok(env.y * CLASSIC.floor + CLASSIC.hitHeight / 2 >= CLASSIC.floor);
  const frame = env.frame;
  env.tick(1);
  assert.equal(env.frame, frame);
});
test("classic scores once per pipe and collisions do not award points", () => {
  const env = new ClassicEnvironment(2);
  env.pipes[0] = { x: (CLASSIC.birdX - CLASSIC.pipeWidth / 2 + 1) / CLASSIC.width, center: .5, passed: false };
  env.tick(0);
  assert.equal(env.score, 1);
  env.tick(0);
  assert.equal(env.score, 1);
  env.pipes[1] = { x: (CLASSIC.birdX - CLASSIC.pipeWidth / 2 + 1) / CLASSIC.width, center: .9, passed: false };
  env.tick(0);
  assert.ok(env.done);
  assert.equal(env.score, 1);
});
test("classic stays deterministic, recycles pipes, and has no one-minute limit", () => {
  const a = new ClassicEnvironment(42), b = new ClassicEnvironment(42);
  assert.deepEqual(a.pipes, b.pipes);
  a.frame = 3600;
  a.pipes[0].x = -1;
  a.tick(0);
  assert.equal(a.done, false);
  assert.equal(a.pipes.length, 3);
  assert.ok(Math.abs((a.pipes[2].x - a.pipes[1].x) * CLASSIC.width - CLASSIC.spacing) < 1e-10);
});
test("displaying experiment decisions one frame at a time preserves saved results", () => {
  const cp = JSON.parse(fs.readFileSync("dist/checkpoints.json")), report = JSON.parse(fs.readFileSync("dist/evaluation.json"));
  for (const row of report.results) {
    const model = new Controller(cp[row.name]);
    for (const expected of row.episodes) {
      const env = new Environment(expected.seed, report.max_frames), rng = new RNG(expected.seed ^ 0xabcdef);
      while (!env.done) {
        const action = +(rng.random() < model.forward(env.observation()).probability);
        for (let i = 0; i < 4 && !env.done; i++) env.tick(i === 0 ? action : 0);
        env.previous = action;
      }
      assert.equal(env.score, expected.score, row.name + " score " + expected.seed);
      assert.equal(env.frame, expected.frames, row.name + " duration " + expected.seed);
    }
  }
});
