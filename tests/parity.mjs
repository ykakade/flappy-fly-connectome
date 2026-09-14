import fs from "node:fs";
import { Environment } from "../dist/environment.js";
import { Controller } from "../dist/controller.js";
const req = JSON.parse(fs.readFileSync(0, "utf8"));
if (req.mode === "environment") {
  const env = new Environment(req.seed, req.max_frames);
  const states = [];
  for (const action of req.actions) {
    env.step(action);
    states.push(env.state());
  }
  console.log(JSON.stringify(states));
} else {
  const model = new Controller(req.artifact);
  console.log(JSON.stringify(req.observations.map((x) => model.forward(x))));
}
