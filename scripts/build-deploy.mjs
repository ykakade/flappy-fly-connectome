import fs from "node:fs";
import path from "node:path";

const output = ".deploy";
fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(output, { recursive: true });
fs.cpSync("selector", output, { recursive: true });
fs.cpSync("dist", path.join(output, "flappy-fly"), { recursive: true });

for (const file of [
  "index.html",
  "style.css",
  "theme.js",
  "fonts/figtree.woff2",
  "fonts/cabin.woff2",
  "fonts/yash-marginalia.ttf",
  "flappy-fly/index.html",
  "flappy-fly/app.js",
]) {
  if (!fs.existsSync(path.join(output, file))) {
    throw new Error(`Missing deployment asset: ${file}`);
  }
}

console.log("Deployment bundle ready at .deploy/.");
