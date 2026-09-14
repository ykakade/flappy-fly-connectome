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
