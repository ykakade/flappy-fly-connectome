import fs from "node:fs";
import { execFileSync } from "node:child_process";
for (const file of [
  "index.html",
  "style.css",
  "app.js",
  "environment.js",
  "controller.js",
  "classic.js",
  "assets/README.md",
  "assets/LICENSE",
  "graph.json",
  "checkpoints.json",
  "evaluation.json",
  "training.json",
  "dataset.json",
]) {
  if (!fs.existsSync("dist/" + file))
    throw new Error("Missing production asset: " + file);
}
for (const file of ["app.js", "environment.js", "classic.js", "controller.js"])
  execFileSync(process.execPath, ["--check", "dist/" + file], {
    stdio: "inherit",
  });
const html = fs.readFileSync("dist/index.html", "utf8");
for (const m of html.matchAll(/(?:src|href)="([^"#:]+)"/g)) {
  if (!/^https?:|^data:/.test(m[1]) && !fs.existsSync("dist/" + m[1]))
    throw new Error("Broken asset " + m[1]);
}
const graph = JSON.parse(fs.readFileSync("dist/graph.json"));
if (
  !graph.edges.length ||
  !graph.nodes.every((n) => typeof n.bodyId === "string")
)
  throw new Error("Invalid graph");
console.log(
  `Static demo verified: ${graph.nodes.length} real nodes, ${graph.edges.length} edges.`,
);

for (const name of ["background-day", "base", "pipe-green", "message", "gameover",
  ...["yellow", "blue"].flatMap(color => ["up", "mid", "down"].map(wing => `${color}bird-${wing}flap`)),
  ...Array.from({length: 10}, (_, i) => String(i))]) {
  if (!fs.existsSync(`dist/assets/${name}.png`)) throw new Error(`Missing sprite: ${name}`);
}
for (const name of ["wing", "hit", "point", "die"]) {
  if (!fs.existsSync(`dist/assets/${name}.wav`)) throw new Error(`Missing sound: ${name}`);
}
