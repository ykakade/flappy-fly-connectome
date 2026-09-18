import { Environment, RNG } from "./environment.js";
import { ClassicEnvironment, CLASSIC } from "./classic.js";
import { Controller } from "./controller.js";
const $ = (id) => document.getElementById(id),
  canvas = $("game"), ctx = canvas.getContext("2d"), nc = $("neural").getContext("2d");
const sprites = {};
const spriteNames = ["background-day", "base", "pipe-green", ...["yellow", "blue"].flatMap(color => ["up", "mid", "down"].map(wing => `${color}bird-${wing}flap`)), ...Array.from({length: 10}, (_, i) => String(i))];
const spriteReady = Promise.all(spriteNames.map(name => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = resolve;
  img.onerror = () => reject(new Error("Could not load " + name));
  img.src = `assets/${name}.png`;
  sprites[name] = img;
})));
const sounds = Object.fromEntries(["wing", "point", "hit", "die"].map(name => [name, new Audio(`assets/${name}.wav`)]));
let soundOn = true;
function sound(name) {
  if (!soundOn) return;
  const audio = sounds[name];
  audio.currentTime = 0;
  audio.play().catch(() => {});
}
let graph, checkpoints, report, training,
  mode = "play", running = false, env = new ClassicEnvironment(10001), other = null,
  controller = null, reference = null, rng = null, otherRng = null, pending = false,
  activity = [], probability = null, trace = [], otherTrace = [], last = 0,
  accumulator = 0, frameTime = 0, decisionFrame = 0, lastAction = 0, otherAction = 0,
  deathAt = 0, deathY = 0, deathVelocity = 0, assetsLoaded = false;
let best = 0;
function bestKey() { return mode === "play" ? "flappy-fly-classic-best-v1" : "flappy-fly-experiment-best-v1"; }
function loadBest() {
  best = 0;
  try { best = Math.max(0, Number(localStorage.getItem(bestKey())) || 0); } catch {}
  $("best").textContent = best;
}
const labels = { play: "Your turn", watch: "Watch the computer", versus: "You vs. computer", journey: "Training", lab: "Change the wiring" };
const checkpointNames = { best: "Trained fly wiring", early: "Early training", mid: "Mid-training", untrained: "Before training", shuffled: "Shuffled wiring", mlp: "Standard network", lesion: "Output connections cut", random: "Random strengths", bypass: "Wiring skipped" };
function seed() {
  const value = Number($("seed").value);
  return Number.isInteger(value) && value >= 1 && value <= 4294967295 ? value : 10001;
}
function selectedKey() {
  return mode === "lab" ? ($("intervention").value === "none" ? "best" : $("intervention").value) : $("checkpoint").value;
}
function refreshController() {
  if (!checkpoints) return;
  const a = checkpoints[selectedKey()];
  controller = new Controller(a);
  reference = new Controller(checkpoints.best);
  $("steps").textContent = a.training_steps.toLocaleString();
  $("eval").textContent = a.evaluation_mean.toFixed(2) + " pipes";
  $("edges").textContent = a.kind === "mlp" ? "N/A" : graph.edges.length.toLocaleString();
  $("node-count").textContent = a.kind === "mlp" ? a.weights["hidden.bias"].length + " units" : graph.nodes.length + " neurons";
  $("activity-note").textContent = a.kind === "mlp" ? "Each dot is a unit in the network. Color shows its value from -1 to +1." : "Each dot represents a neuron in the model. Hover for its ID. The layout is a diagram, not a brain scan.";
}
function clearActivity() {
  activity = []; probability = null;
  $("prob").textContent = "N/A";
  $("prob-fill").style.width = "0";
  $("activity-status").textContent = "Idle";
}
function reset() {
  running = false; pending = false; accumulator = 0; last = 0; decisionFrame = 0; deathAt = 0;
  env = mode === "play" ? new ClassicEnvironment(seed()) : new Environment(seed(), report?.max_frames || 3600);
  other = ["versus", "lab"].includes(mode) ? new Environment(seed(), report?.max_frames || 3600) : null;
  rng = new RNG(seed() ^ 0xabcdef); otherRng = new RNG(seed() ^ 0xabcdef);
  trace = []; otherTrace = [];
  clearActivity(); refreshController(); loadBest();
  $("score").textContent = "0";
  $("overlay").hidden = false;
  $("overlay").classList.remove("game-ended");
  $("overlay").classList.toggle("experiment", mode !== "play");
  $("overlay").querySelector("h2").classList.toggle("sr-only", mode === "play");
  $("overlay").querySelector("h2").textContent = mode === "play" ? "Get ready" : labels[mode];
  $("ready-image").hidden = mode !== "play";
  $("gameover-image").hidden = true;
  $("end-card").hidden = true;
  $("overlay-copy").hidden = mode === "play";
  $("overlay-copy").textContent = mode === "versus" ? "You are yellow. The computer is blue." : mode === "lab" ? "Yellow: changed wiring. Blue: original wiring." : "Choose a controller, then press Start.";
  $("start").textContent = mode === "play" ? "Play" : "Start";
  $("run-status").textContent = ["play", "versus"].includes(mode) ? "Space, click, or tap to flap" : "Ready";
  $("inference-label").textContent = mode === "play" ? "Human control" : checkpointNames[selectedKey()];
  $("activity-copy").textContent = "Watch the dots change as the computer decides when to flap.";
  $("comparison-results").textContent = "";
  drawChart();
}
function start() {
  if (!assetsLoaded) return;
  if (mode !== "play" && !checkpoints) {
    $("run-status").textContent = "The controller could not load. Try reloading.";
    return;
  }
  reset(); running = true;
  pending = ["play", "versus"].includes(mode);
  $("overlay").hidden = true;
  canvas.focus({preventScroll: true});
  $("run-status").textContent = "Flying";
  $("activity-status").textContent = mode === "play" ? "Idle" : "Live";
}
function changeMode(next) {
  if (!(next in labels)) throw new Error("Unknown mode");
  mode = next;
  document.body.dataset.mode = mode;
  document.querySelectorAll("[data-mode]").forEach(b => {
    if (b.tagName !== "BUTTON") return;
    const active = b.dataset.mode === mode;
    b.setAttribute("aria-selected", String(active)); b.tabIndex = active ? 0 : -1;
  });
  $("workspace").setAttribute("aria-labelledby", "tab-" + mode);
  document.querySelector(".instrument").hidden = mode === "play";
  $("mode-note").hidden = mode === "play";
  $("mode-label").textContent = labels[mode];
  $("score-label").textContent = mode === "lab" ? "Changed wiring" : mode === "versus" ? "You" : "Score";
  $("intervention-label").hidden = mode !== "lab";
  $("checkpoint").disabled = mode === "play" || mode === "lab";
  $("comparison").hidden = !["versus", "lab", "journey"].includes(mode);
  $("comparison-title").textContent = mode === "journey" ? "Scores during training" : mode === "lab" ? "What changed?" : "Your scores";
  $("comparison-description").textContent = mode === "journey" ? "Choose a training stage to watch. This chart shows how each controller improved." : mode === "lab" ? "Both controllers get the same pipes and random choices. Only the wiring changes." : "Yellow is you. Blue is the computer. You both get the same pipes.";
  reset();
}
function recordInference(model, state, random, records) {
  const r = model.forward(state.observation());
  const action = +(random.random() < r.probability);
  records.push({frame: state.frame, probability: r.probability, action});
  return {...r, action};
}
function finish() {
  $("overlay").hidden = false;
  $("overlay").classList.add("game-ended");
  $("overlay").classList.remove("experiment");
  $("overlay").querySelector("h2").classList.add("sr-only");
  $("overlay").querySelector("h2").textContent = "Game over";
  $("ready-image").hidden = true;
  $("gameover-image").hidden = false;
  $("end-card").hidden = false;
  $("end-score").textContent = env.score;
  $("end-best").textContent = best;
  $("overlay-copy").hidden = true;
  $("start").textContent = "Play again";
  $("run-status").textContent = env.frame >= env.max_frames ? "Time up" : "Game over";
  $("activity-status").textContent = mode === "play" ? "Idle" : "Finished";
}
function step() {
  if (!running) return;
  const before = env.score, wasDone = env.done;
  if (mode === "play") {
    env.tick(+pending);
    if (pending) sound("wing");
    pending = false;
  } else {
    // Spread each four-frame experiment decision over four display frames.
    // Restore previous action at the boundary exactly as Environment.step does.
    if (!env.done) {
      let action = 0;
      if (decisionFrame === 0) {
        if (mode === "versus") { action = +pending; pending = false; }
        else {
          const r = recordInference(controller, env, rng, trace);
          action = r.action; activity = r.activity; probability = r.probability;
        }
        lastAction = action;
        if (action) sound("wing");
      }
      env.tick(action);
      if (decisionFrame === 3 || env.done) env.previous = lastAction;
    }
    if (other && !other.done) {
      let action = 0;
      if (decisionFrame === 0) {
        const r = recordInference(mode === "lab" ? reference : controller, other, otherRng, otherTrace);
        action = r.action; otherAction = action;
        if (mode === "versus") { activity = r.activity; probability = r.probability; }
      }
      other.tick(action);
      if (decisionFrame === 3 || other.done) other.previous = otherAction;
    }
    decisionFrame = (decisionFrame + 1) % 4;
  }
  if (env.score > before) sound("point");
  if (env.done && !wasDone && env.frame < env.max_frames) sound("hit");
  $("score").textContent = env.score;
  if (env.score > best) {
    best = env.score; $("best").textContent = best;
    try { localStorage.setItem(bestKey(), String(best)); } catch {}
  }
  if (probability !== null) {
    $("prob").textContent = (probability * 100).toFixed(1) + "%";
    $("prob-fill").style.width = probability * 100 + "%";
  }
  if (other) {
    $("comparison-results").textContent = mode === "lab" ? `Changed wiring: ${env.score}. Original: ${other.score}. Difference: ${env.score - other.score}.` : `You: ${env.score}. Computer: ${other.score}.`;
    $("run-status").textContent = mode === "lab" ? `Changed ${env.score} / Original ${other.score}` : `You ${env.score} / Computer ${other.score}`;
    if (decisionFrame === 0) drawChart();
  }
  if (env.done && (!other || other.done)) {
    running = false;
    if (mode === "play") {
      deathAt = frameTime; deathY = env.y * 400; deathVelocity = Math.max(0, env.vy * 400);
      sound("die");
    } else finish();
  }
}
function sprite(name, x, y, w, h) {
  const img = sprites[name];
  if (img?.naturalWidth) ctx.drawImage(img, Math.round(x), Math.round(y), w ?? img.width, h ?? img.height);
}
function drawBird(x, y, velocity, blue = false, scale = 1) {
  ctx.save(); ctx.translate(Math.round(x), Math.round(y));
  ctx.rotate(Math.max(-.45, Math.min(Math.PI / 2, velocity > 1 ? (velocity - 1) * .22 : -.35)));
  const wing = !env.done && !matchMedia("(prefers-reduced-motion: reduce)").matches ? ["up", "mid", "down", "mid"][Math.floor(frameTime / 100) % 4] : "mid";
  sprite(`${blue ? "blue" : "yellow"}bird-${wing}flap`, -17 * scale, -12 * scale, 34 * scale, 24 * scale);
  ctx.restore();
}
function drawScore(score) {
  const digits = String(score).split("");
  const width = digits.reduce((sum, digit) => sum + (sprites[digit]?.width || 24) + 1, -1);
  let x = (288 - width) / 2;
  for (const digit of digits) { sprite(digit, x, 32); x += (sprites[digit]?.width || 24) + 1; }
}
function draw() {
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#4ec0ca"; ctx.fillRect(0, 0, 288, 512);
  sprite("background-day", 0, 0);
  const display = other && env.done && !other.done ? other : env;
  const classic = mode === "play";
  const pipeWidth = classic ? CLASSIC.pipeWidth : .09 * 288;
  const halfGap = classic ? CLASSIC.gap / 2 : .16 * 400;
  if (env.frame > 0) for (const pipe of display.pipes) {
    const x = Math.round(pipe.x * 288), top = Math.round(pipe.center * 400 - halfGap), bottom = Math.round(pipe.center * 400 + halfGap);
    ctx.save(); ctx.translate(x, top); ctx.scale(1, -1);
    sprite("pipe-green", 0, 0, pipeWidth, 320); ctx.restore();
    sprite("pipe-green", x, bottom, pipeWidth, 320);
  }
  if (running || env.frame > 0 || mode !== "play") {
    const x = classic ? CLASSIC.birdX : .22 * 288;
    // Experiment sprites fit their original collision box.
    const birdScale = classic ? 1 : .48;
    drawBird(x, classic && deathAt ? deathY : env.y * 400, classic && deathAt ? 10 : env.vy * 400, false, birdScale);
    if (other) drawBird(x, other.y * 400, other.vy * 400, true, birdScale);
  }
  const offset = env.frame === 0 ? Math.floor(frameTime / (1000 / 60) * 2) % 48 : Math.floor(env.frame * (classic ? 2 : .0035 * 288)) % 48;
  sprite("base", -offset, 400);
  if (running) drawScore(env.score);
  if (deathAt && frameTime - deathAt < 100 && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    ctx.fillStyle = "rgba(255,255,255,.65)"; ctx.fillRect(0, 0, 288, 512);
  }
  if (mode !== "play") drawNeural();
}
let positions = [];
function graphPositions() {
  if (!graph) return;
  const ins = new Set(graph.inputs),
    outs = new Set(graph.outputs);
  const groups = [
    graph.inputs,
    graph.nodes.map((_, i) => i).filter((i) => !ins.has(i) && !outs.has(i)),
    graph.outputs,
  ];
  positions = Array(graph.nodes.length);
  groups.forEach((nodes, g) =>
    nodes.forEach((node, j) => {
      const cols = g === 1 ? 5 : 2,
        rows = Math.ceil(nodes.length / cols);
      positions[node] = {
        x: 30 + g * 180 + (j % cols) * ((g === 1 ? 155 : 35) / cols),
        y: 20 + Math.floor(j / cols) * (210 / Math.max(rows - 1, 1)),
      };
    }),
  );
}
function activityColor(value) {
  if (value >= 0)
    return `rgb(${45 + Math.round(value * 152)},${68 + Math.round(value * 175)},${83 + Math.round(value * 35)})`;
  return `rgb(${45 + Math.round(-value * 68)},${68 + Math.round(-value * 73)},${83 + Math.round(-value * 134)})`;
}
function drawNeural() {
  nc.clearRect(0, 0, 480, 250);
  if (!graph || !positions.length) return;
  if (controller?.kind === "mlp" && mode !== "play") {
    activity.forEach((v, i) => {
      nc.fillStyle = activityColor(v);
      nc.beginPath();
      nc.arc(
        25 + (i % 10) * 44,
        25 + Math.floor(i / 10) * 43,
        5,
        0,
        Math.PI * 2,
      );
      nc.fill();
    });
    return;
  }
  const w = controller?.w;
  const edgeList = w
    ? w.src.map((s, e) => [s, w.dst[e]])
    : graph.edges.map((e) => [e.source_index, e.target_index]);
  nc.lineWidth = 0.65;
  for (const [s, t] of edgeList) {
    if (controller?.intervention === "lesion" && graph.outputs.includes(t))
      continue;
    const a = positions[s],
      b = positions[t];
    nc.strokeStyle = activity.length
      ? "rgba(168,202,170,.14)"
      : "rgba(100,137,160,.14)";
    nc.beginPath();
    nc.moveTo(a.x, a.y);
    nc.lineTo(b.x, b.y);
    nc.stroke();
  }
  positions.forEach((p, i) => {
    nc.fillStyle = activity.length ? activityColor(activity[i]) : "#486175";
    nc.beginPath();
    nc.arc(
      p.x,
      p.y,
      graph.inputs.includes(i) || graph.outputs.includes(i) ? 4 : 3,
      0,
      Math.PI * 2,
    );
    nc.fill();
  });
}
$("neural").addEventListener("pointermove", (e) => {
  if (controller?.kind === "mlp") {
    e.currentTarget.title =
      "Actual MLP hidden-unit activations; not biological neurons";
    return;
  }
  const r = e.currentTarget.getBoundingClientRect(),
    x = ((e.clientX - r.left) * 480) / r.width,
    y = ((e.clientY - r.top) * 250) / r.height;
  let nearest = positions
    .map((p, i) => ({ i, d: Math.hypot(x - p.x, y - p.y) }))
    .sort((a, b) => a.d - b.d)[0];
  if (nearest && nearest.d < 15) {
    const n = graph.nodes[nearest.i];
    e.currentTarget.title = `ID ${n.bodyId} · ${n.type || "untyped"} · ${n.superclass} · activity ${activity[nearest.i]?.toFixed(4) ?? "idle"}`;
  } else e.currentTarget.title = "All retained neurons; schematic layout";
});
function drawChart() {
  const c = $("prob-chart"),
    cx = c.getContext("2d");
  cx.clearRect(0, 0, c.width, c.height);
  if ($("comparison").hidden) return;
  let series, maxX, maxY;
  if (mode === "journey") {
    series = Object.entries(training || {}).map(([k, records]) => ({
      name: k,
      points: records
        .filter((r) => "validation_mean" in r)
        .map((r) => [r.steps, r.validation_mean]),
    }));
    maxX = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[0])));
    maxY = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[1])));
  } else {
    series = [
      {
        name: "Changed wiring",
        points: trace.map((p) => [p.frame / 60, p.probability]),
      },
      {
        name: mode === "lab" ? "Original" : "Computer",
        points: otherTrace.map((p) => [p.frame / 60, p.probability]),
      },
    ];
    maxX = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p[0])));
    maxY = 1;
  }
  cx.strokeStyle = "#dbe3e9";
  cx.beginPath();
  cx.moveTo(45, 15);
  cx.lineTo(45, 150);
  cx.lineTo(985, 150);
  cx.stroke();
  cx.font = "12px monospace";
  cx.fillStyle = "#5c6b7b";
  cx.fillText(mode === "journey" ? "Training score" : "Chance of a flap", 45, 13);
  cx.fillText(
    mode === "journey"
      ? `${maxX.toLocaleString()} decisions`
      : `${maxX.toFixed(1)} seconds`,
    800,
    175,
  );
  series.forEach((s, j) => {
    cx.strokeStyle = ["#679e38", "#547ac0", "#c48432"][j];
    cx.lineWidth = 1.5;
    cx.beginPath();
    s.points.forEach(([x, y], i) => {
      const px = 45 + (x / maxX) * 935,
        py = 150 - (y / maxY) * 125;
      if (i === 0) cx.moveTo(px, py);
      else cx.lineTo(px, py);
    });
    cx.stroke();
    cx.fillStyle = cx.strokeStyle;
    cx.fillText(s.name, 190 + j * 190, 13);
  });
}
function loop(t) {
  frameTime = t;
  if (!last) last = t;
  const delta = Math.min(t - last, 100);
  last = t;
  if (!document.hidden) {
    if (running) {
      accumulator += delta;
      while (accumulator >= 1000 / 60 && running) { step(); accumulator -= 1000 / 60; }
    } else if (deathAt && $("overlay").hidden) {
      deathVelocity = Math.min(deathVelocity + .5 * delta / (1000 / 60), 10);
      deathY = Math.min(388, deathY + deathVelocity * delta / (1000 / 60));
      if (deathY >= 388 && t - deathAt > 500) finish();
    }
    draw();
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
$("start").addEventListener("click", start);
$("reset").addEventListener("click", reset);
$("sound").addEventListener("click", () => {
  soundOn = !soundOn;
  $("sound").textContent = soundOn ? "Sound on" : "Sound off";
  $("sound").setAttribute("aria-pressed", String(soundOn));
  if (!soundOn) Object.values(sounds).forEach(audio => audio.pause());
});
$("seed").addEventListener("change", () => { $("seed").value = seed(); reset(); });
$("checkpoint").addEventListener("change", reset);
$("intervention").addEventListener("change", reset);
document.querySelectorAll("button[data-mode]").forEach(b => {
  b.addEventListener("click", () => changeMode(b.dataset.mode));
  b.addEventListener("keydown", e => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const buttons = [...document.querySelectorAll("button[data-mode]")].filter(button => !button.disabled), i = buttons.indexOf(b);
    const next = e.key === "Home" ? 0 : e.key === "End" ? buttons.length - 1 : (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    changeMode(buttons[next].dataset.mode); buttons[next].focus();
  });
});
function flap() {
  if (!["play", "versus"].includes(mode)) return;
  if (!running) {
    if (deathAt && $("overlay").hidden) return;
    start(); return;
  }
  if (!env.done) pending = true;
}
canvas.addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  e.preventDefault(); flap();
});
document.addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "BUTTON", "A", "SUMMARY"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space" || e.code === "ArrowUp") { e.preventDefault(); if (!e.repeat) flap(); }
  if (e.code === "KeyR" && !e.repeat) reset();
});
document.addEventListener("visibilitychange", () => { last = 0; accumulator = 0; });
async function load() {
  try {
    const data = await Promise.all(["graph.json", "checkpoints.json", "evaluation.json", "training.json"].map(async p => {
      const r = await fetch(p); if (!r.ok) throw new Error(p + " " + r.status); return r.json();
    }));
    [graph, checkpoints, report, training] = data;
    graphPositions(); refreshController();
    const show = ["untrained", "best", "shuffled", "mlp", "lesion", "random", "bypass"];
    $("metrics-body").replaceChildren(...show.map(name => {
      const r = report.results.find(x => x.name === name), tr = document.createElement("tr");
      [checkpointNames[name], r.mean.toFixed(2), r.median.toFixed(1), r.std.toFixed(2), r.survival_seconds.toFixed(1) + " s"].forEach(v => {
        const td = document.createElement("td"); td.textContent = v; tr.append(td);
      }); return tr;
    }));
    const un = report.results.find(r => r.name === "untrained"), trained = report.results.find(r => r.name === "best");
    $("result-summary").textContent = `The fly-wiring controller went from ${un.mean.toFixed(2)} to ${trained.mean.toFixed(2)} pipes on average after training. Shuffled wiring and a standard network scored higher. Fly wiring did not give it an advantage in this test.`;
    $("eval-details").textContent = `${trained.evaluation_episodes} pipe layouts that were not used for training. Each run lasted up to ${report.max_frames / 60} seconds. “Spread” is the standard deviation. These scores use the experiment rules.`;
    $("graph-explanation").textContent = `The model uses ${graph.nodes.length} neurons and ${graph.edges.length} connections from MaleCNS. Training adjusts how it uses those connections. This is a small model built from wiring data, not a simulation of a living fly.`;
    drawChart();
  } catch {
    $("activity-copy").textContent = "The controllers could not load. Reload to try again.";
    $("result-summary").textContent = "Scores could not load. You can still play.";
    $("metrics-body").textContent = "";
    document.querySelectorAll('button[data-mode]:not([data-mode="play"])').forEach(b => b.disabled = true);
  }
}
reset();
spriteReady.then(() => { assetsLoaded = true; }).catch(() => {
  $("run-status").textContent = "Game art could not load. Please reload.";
  $("start").disabled = true;
});
load();
window.flappyFly = {
  getState: () => ({ mode, running, seed: seed(), score: env.score, otherScore: other?.score,
    frame: env.frame, y: env.y, vy: env.vy, done: env.done, pipes: env.pipes.map(p => ({...p})),
    probability, activity: activity.slice(), controller: selectedKey(), ready: assetsLoaded && !!checkpoints }),
  setMode: changeMode, start, reset,
};

if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  for (const tool of [
    {
      name: "read_flappy_fly_state",
      description: "Read the current live flight state and controller.",
      inputSchema: {
        type: "object",
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true },
      execute: () => window.flappyFly.getState(),
    },
    {
      name: "configure_flappy_fly",
      description:
        "Select experiment mode and obstacle seed, resetting the visible flight without starting it.",
      inputSchema: {
        type: "object",
        properties: {
          mode: { type: "string", enum: Object.keys(labels) },
          seed: { type: "integer", minimum: 1, maximum: 4294967295 },
        },
        required: ["mode", "seed"],
        additionalProperties: false,
      },
      execute: (input) => {
        if (
          !input ||
          !(input.mode in labels) ||
          !Number.isInteger(input.seed) ||
          input.seed < 1 ||
          input.seed > 4294967295
        )
          throw new Error("Invalid mode or seed");
        $("seed").value = input.seed;
        changeMode(input.mode);
        return window.flappyFly.getState();
      },
    },
  ]) {
    try {
      Promise.resolve(
        document.modelContext.registerTool(tool, { signal: lifecycle.signal }),
      ).catch(() => {});
    } catch {}
  }
  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
}
