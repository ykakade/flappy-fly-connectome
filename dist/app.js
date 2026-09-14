import { Environment, RNG } from "./environment.js";
import { Controller } from "./controller.js";
const $ = (id) => document.getElementById(id),
  canvas = $("game"),
  ctx = canvas.getContext("2d"),
  nc = $("neural").getContext("2d");
const fly = new Image();
fly.src = "fly.png";
let graph,
  checkpoints,
  report,
  training,
  mode = "play",
  running = false,
  env = new Environment(10001),
  other = null,
  controller = null,
  reference = null,
  rng = null,
  otherRng = null,
  pending = false,
  activity = [],
  probability = null,
  trace = [],
  otherTrace = [],
  last = 0,
  accumulator = 0,
  frameTime = 0;
let best = 0;
try {
  best = Number(localStorage.getItem("flappy-fly-best-v1")) || 0;
} catch {}
$("best").textContent = String(best).padStart(2, "0");
const labels = {
  play: "HUMAN FLIGHT",
  watch: "LIVE AI FLIGHT",
  versus: "HUMAN / AI COMPARISON",
  journey: "TRAINING JOURNEY",
  lab: "CONNECTION INTERVENTION",
};
const checkpointNames = {
  best: "Best · MaleCNS",
  early: "Early · MaleCNS",
  mid: "Mid · MaleCNS",
  untrained: "Untrained · MaleCNS",
  shuffled: "Trained shuffled graph",
  mlp: "MLP baseline",
  lesion: "Output-edge lesion",
  random: "Random graph weights",
  bypass: "Adapter bypass",
};
function seed() {
  const value = Number($("seed").value);
  return Number.isInteger(value) && value >= 1 && value <= 4294967295
    ? value
    : 10001;
}
function selectedKey() {
  return mode === "lab"
    ? $("intervention").value === "none"
      ? "best"
      : $("intervention").value
    : $("checkpoint").value;
}
function refreshController() {
  if (!checkpoints) return;
  const key = selectedKey(),
    a = checkpoints[key];
  controller = new Controller(a);
  reference = new Controller(checkpoints.best);
  $("steps").textContent = a.training_steps.toLocaleString();
  $("eval").textContent = a.evaluation_mean.toFixed(2) + " gates";
  $("edges").textContent =
    a.kind === "mlp" ? "— (MLP)" : graph.edges.length.toLocaleString();
  $("node-count").textContent =
    a.kind === "mlp"
      ? a.weights["hidden.bias"].length + " hidden units"
      : graph.nodes.length + " neurons";
  $("activity-note").textContent =
    a.kind === "mlp"
      ? "Each dot is a hidden unit, not a biological neuron. Raw tanh activity: −1 to +1. Live inference."
      : `All ${graph.nodes.length} retained neuron IDs are displayed; hover to inspect. No aggregation. Raw tanh activity: −1 to +1. Positions are schematic, not anatomy.`;
}
function clearActivity() {
  activity = [];
  probability = null;
  $("prob").textContent = "—";
  $("prob-fill").style.width = "0";
  $("activity-status").textContent = "IDLE";
}
function reset() {
  running = false;
  pending = false;
  accumulator = 0;
  env = new Environment(seed(), report?.max_frames || 3600);
  other = ["versus", "lab"].includes(mode)
    ? new Environment(seed(), report?.max_frames || 3600)
    : null;
  rng = new RNG(seed() ^ 0xabcdef);
  otherRng = new RNG(seed() ^ 0xabcdef);
  trace = [];
  otherTrace = [];
  clearActivity();
  refreshController();
  $("score").textContent = "00";
  $("overlay").hidden = false;
  $("overlay").querySelector(".eyebrow").textContent =
    mode === "play"
      ? "YOUR TURN AT THE CONTROLS"
      : "SAME WORLD. A DIFFERENT CONTROLLER.";
  $("overlay").querySelector("h2").innerHTML =
    mode === "play"
      ? "A small flap.<br>A big experiment."
      : mode === "versus"
        ? "You and the fly.<br>One flight path."
        : mode === "lab"
          ? "Change a pathway.<br>Watch what changes."
          : "Real wiring.<br>Live decisions.";
  $("overlay").querySelector("p").textContent =
    mode === "play"
      ? "Guide your fruit fly through the gates."
      : mode === "lab"
        ? "Solid fly: intervention. Outline: original graph."
        : mode === "versus"
          ? "You are the solid fly. AI is the orange outline."
          : "Every flap is computed from the selected checkpoint.";
  $("start").innerHTML =
    (mode === "play"
      ? "Start flying"
      : mode === "versus"
        ? "Start comparison"
        : "Run checkpoint") + " <span>↗</span>";
  $("overlay").querySelector(".hint").hidden = !["play", "versus"].includes(
    mode,
  );
  $("run-status").textContent = "Ready · seed " + seed();
  $("inference-label").textContent =
    mode === "play"
      ? "Human control"
      : "Live inference · " + checkpointNames[selectedKey()];
  $("activity-copy").textContent =
    mode === "play"
      ? "Controller activity appears during AI flight. Human mode does not generate neural signals."
      : "Each color comes from the controller used in this flight. Biological wiring; engineered dynamics.";
  $("comparison-results").textContent = "";
  drawChart();
}
function start() {
  if (mode !== "play" && !checkpoints) {
    $("run-status").textContent = "Controller unavailable — please reload.";
    return;
  }
  reset();
  running = true;
  $("overlay").hidden = true;
  canvas.focus();
  $("run-status").textContent = "Flight in progress";
  $("activity-status").textContent = mode === "play" ? "HUMAN" : "LIVE";
}
function changeMode(next) {
  if (!(next in labels)) throw new Error("Unknown mode");
  mode = next;
  document.querySelectorAll("[data-mode]").forEach((b) => {
    const active = b.dataset.mode === mode;
    b.setAttribute("aria-selected", String(active));
    b.tabIndex = active ? 0 : -1;
  });
  $("mode-label").textContent = labels[mode];
  $("score-label").textContent =
    mode === "lab" ? "INTERVENTION" : mode === "versus" ? "YOU" : "SCORE";
  $("intervention-label").hidden = mode !== "lab";
  $("checkpoint").disabled = mode === "play" || mode === "lab";
  $("comparison").hidden = !["versus", "lab", "journey"].includes(mode);
  $("comparison-title").textContent =
    mode === "journey"
      ? "The training record."
      : mode === "lab"
        ? "Same seed. Change only the intervention."
        : "Same seed. Two flights.";
  $("comparison-description").textContent =
    mode === "journey"
      ? "Select an untrained, early, mid, or validation-selected best checkpoint. The obstacle seed remains fixed. The plot shows held-out validation scores over training."
      : mode === "lab"
        ? "Compare the intervention (green) against the original best controller (blue), using identical obstacle and action-sampling seeds. A shuffled controller was trained separately with the same budget."
        : "Space, click, or tap controls your fly. The AI uses the identical obstacle seed. Scores are shown independently.";
  reset();
}
function recordInference(model, state, random, records) {
  const r = model.forward(state.observation());
  const action = +(random.random() < r.probability);
  records.push({ frame: state.frame, probability: r.probability, action });
  return { ...r, action };
}
function step() {
  if (!running) return;
  if (!env.done) {
    let action = 0;
    if (["play", "versus"].includes(mode)) {
      action = +pending;
      pending = false;
    } else {
      const r = recordInference(controller, env, rng, trace);
      action = r.action;
      activity = r.activity;
      probability = r.probability;
    }
    env.step(action);
  }
  if (other && !other.done) {
    const r = recordInference(
      mode === "lab" ? reference : controller,
      other,
      otherRng,
      otherTrace,
    );
    other.step(r.action);
    if (mode === "versus") {
      activity = r.activity;
      probability = r.probability;
    }
  }
  $("score").textContent = String(env.score).padStart(2, "0");
  if (env.score > best) {
    best = env.score;
    $("best").textContent = String(best).padStart(2, "0");
    try {
      localStorage.setItem("flappy-fly-best-v1", String(best));
    } catch {}
  }
  if (probability !== null) {
    $("prob").textContent = (probability * 100).toFixed(1) + "%";
    $("prob-fill").style.width = probability * 100 + "%";
  }
  if (other) {
    $("comparison-results").textContent =
      mode === "lab"
        ? `Intervention: ${env.score} gates · Original: ${other.score} gates · Score change: ${env.score - other.score >= 0 ? "+" : ""}${env.score - other.score}`
        : `You: ${env.score} gates · AI: ${other.score} gates`;
    $("run-status").textContent =
      mode === "lab"
        ? `Intervention ${env.score} / Original ${other.score}`
        : `You ${env.score} / AI ${other.score}`;
    drawChart();
  }
  if (env.done && (!other || other.done)) {
    running = false;
    $("activity-status").textContent = mode === "play" ? "HUMAN" : "FINAL";
    $("overlay").hidden = false;
    $("overlay").querySelector(".eyebrow").textContent =
      env.frame >= env.max_frames ? "EPISODE LIMIT REACHED" : "FLIGHT COMPLETE";
    $("overlay").querySelector("h2").textContent =
      env.score + " gates cleared.";
    $("overlay").querySelector("p").textContent =
      `${(env.frame / 60).toFixed(1)} seconds · Seed ${seed()}${other ? " · Comparison: " + other.score + " gates" : ""}`;
    $("start").innerHTML = "Replay same seed <span>↻</span>";
    $("run-status").textContent =
      "Finished · " + (env.frame / 60).toFixed(1) + " s";
  }
}
function drawFly(x, y, rotation, ghost = false, scale = 1) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale, scale);
  ctx.rotate(Math.max(-0.35, Math.min(0.65, rotation)));
  if (ghost) {
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = "#d47a2c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(0, 0, 23, 15, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  const flutter =
    running && !matchMedia("(prefers-reduced-motion: reduce)").matches
      ? Math.sin(frameTime * 0.05) * 2
      : 0;
  if (fly.complete) ctx.drawImage(fly, -40, -40 + flutter, 80, 80);
  ctx.restore();
}
function draw() {
  const W = 960,
    H = 540;
  ctx.fillStyle = "#deedf5";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#c5dce9";
  ctx.lineWidth = 0.6;
  const offset = (env.frame * 0.0035 * W) % 40;
  for (let x = -offset; x < W; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 0; y < H; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  const display = other && env.done && !other.done ? other : env;
  for (const p of display.pipes) {
    const x = p.x * W,
      w = 0.09 * W,
      top = (p.center - 0.16) * H,
      bot = (p.center + 0.16) * H;
    ctx.fillStyle = "#aac5d4";
    ctx.fillRect(x, 0, w, top);
    ctx.fillRect(x, bot, w, H - bot);
    ctx.fillStyle = "#799daf";
    ctx.fillRect(x - 5, top - 10, w + 10, 10);
    ctx.fillRect(x - 5, bot, w + 10, 10);
    ctx.strokeStyle = "#8aaec0";
    for (let y = 8; y < top - 14; y += 13) {
      ctx.beginPath();
      ctx.moveTo(x + 8, y);
      ctx.lineTo(x + 22, y);
      ctx.stroke();
    }
    ctx.fillStyle = "#587d92";
    ctx.font = "11px monospace";
    ctx.fillText("GATE", x + 13, top - 24);
  }
  ctx.strokeStyle = "#92b6cd";
  ctx.setLineDash([4, 7]);
  ctx.beginPath();
  ctx.moveTo(0.22 * W, 0);
  ctx.lineTo(0.22 * W, H);
  ctx.stroke();
  ctx.setLineDash([]);
  if (!running && env.frame === 0) {
    drawFly(
      canvas.clientWidth < 600 ? 800 : 750,
      canvas.clientWidth < 600 ? 415 : 285,
      0,
      false,
      canvas.clientWidth < 600 ? 1.8 : 2.4,
    );
  } else {
    drawFly(0.22 * W, env.y * H, env.vy * 25);
    if (other) drawFly(0.22 * W, other.y * H, other.vy * 25, true);
  }
  drawNeural();
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
        name: "Intervention",
        points: trace.map((p) => [p.frame / 60, p.probability]),
      },
      {
        name: mode === "lab" ? "Original" : "AI",
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
  cx.fillText(mode === "journey" ? "Validation gates" : "P(flap)", 45, 13);
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
  const delta = Math.min(t - last, 150);
  last = t;
  if (running) {
    accumulator += delta;
    while (accumulator >= 1000 / 15 && running) {
      step();
      accumulator -= 1000 / 15;
    }
  }
  draw();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
$("start").addEventListener("click", start);
$("reset").addEventListener("click", reset);
$("seed").addEventListener("change", () => {
  $("seed").value = seed();
  reset();
});
$("checkpoint").addEventListener("change", reset);
$("intervention").addEventListener("change", reset);
document.querySelectorAll("[data-mode]").forEach((b) => {
  b.addEventListener("click", () => changeMode(b.dataset.mode));
  b.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const buttons = [...document.querySelectorAll("[data-mode]")],
      i = buttons.indexOf(b),
      next =
        e.key === "Home"
          ? 0
          : e.key === "End"
            ? buttons.length - 1
            : (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) %
              buttons.length;
    changeMode(buttons[next].dataset.mode);
    buttons[next].focus();
  });
});
function flap() {
  if (!["play", "versus"].includes(mode)) return;
  if (!running) {
    if (env.frame === 0) start();
    else return;
  }
  pending = true;
}
canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  flap();
});
document.addEventListener("keydown", (e) => {
  if (
    (e.code === "Space" || e.code === "ArrowUp") &&
    !["INPUT", "SELECT", "BUTTON", "A"].includes(document.activeElement.tagName)
  ) {
    e.preventDefault();
    flap();
  }
  if (
    e.code === "KeyR" &&
    !["INPUT", "SELECT"].includes(document.activeElement.tagName)
  )
    reset();
});
document.addEventListener("visibilitychange", () => {
  last = 0;
  accumulator = 0;
});
async function load() {
  try {
    [graph, checkpoints, report, training] = await Promise.all(
      [
        "graph.json",
        "checkpoints.json",
        "evaluation.json",
        "training.json",
      ].map(async (p) => {
        const r = await fetch(p);
        if (!r.ok) throw new Error(p + " " + r.status);
        return r.json();
      }),
    );
    graphPositions();
    const show = [
      "untrained",
      "best",
      "shuffled",
      "mlp",
      "lesion",
      "random",
      "bypass",
    ];
    $("metrics-body").replaceChildren(
      ...show.map((name) => {
        const r = report.results.find((x) => x.name === name),
          tr = document.createElement("tr");
        [
          checkpointNames[name],
          r.mean.toFixed(2),
          r.median.toFixed(1),
          r.std.toFixed(2),
          r.survival_seconds.toFixed(1) + " s",
        ].forEach((v) => {
          const td = document.createElement("td");
          td.textContent = v;
          tr.append(td);
        });
        return tr;
      }),
    );
    const un = report.results.find((r) => r.name === "untrained"),
      trained = report.results.find((r) => r.name === "best");
    $("result-summary").textContent =
      `The selected connectome controller averaged ${trained.mean.toFixed(2)} gates, compared with ${un.mean.toFixed(2)} before training. ${trained.mean > un.mean ? "It improved in this run." : "This run did not establish useful learning."} One training seed and a small subgraph cannot establish a biological advantage.`;
    $("eval-details").textContent =
      `${trained.evaluation_episodes} common unseen seeds (${trained.evaluation_seeds[0]}–${trained.evaluation_seeds.at(-1)}). Population SD. Episode cap ${report.max_frames / 60}s. Seeded stochastic actions. Best selected on separate validation seeds.`;
    $("graph-explanation").textContent =
      `A deterministic strongest-edge expansion retains ${graph.nodes.length} nodes and ${graph.edges.length} real connections around a visual-to-descending edge. An engineered adapter injects six game-state values into ${graph.inputs.length} visual nodes. Four sparse passes feed ${graph.outputs.length} descending readout nodes. PPO trains adapters, node gains and biases; measured normalized edge weights and topology stay fixed. The controller does not see pixels.`;
    reset();
  } catch (e) {
    $("activity-copy").textContent =
      "AI artifacts could not load. Human flight remains available.";
    $("run-status").textContent = "AI unavailable: " + e.message;
    document
      .querySelectorAll('[data-mode]:not([data-mode="play"])')
      .forEach((b) => (b.disabled = true));
  }
}
load();
// Public readback uses the same live state as the visible interface and automated QA.
window.flappyFly = {
  getState: () => ({
    mode,
    running,
    seed: seed(),
    score: env.score,
    otherScore: other?.score,
    frame: env.frame,
    probability,
    activity: activity.slice(),
    controller: selectedKey(),
    ready: !!checkpoints,
  }),
  setMode: changeMode,
  start,
  reset,
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
