// Float64 browser inference; weights are exported Float32 PyTorch values. Tested to 2e-6.
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const linear = (x, w, b) => w.map((row, i) => dot(row, x) + b[i]);
export class Controller {
  constructor(artifact) {
    this.artifact = artifact;
    this.w = artifact.weights;
    this.kind = artifact.kind;
    this.intervention = artifact.intervention || "none";
  }
  forward(obs) {
    const w = this.w;
    if (this.kind === "mlp") {
      const h = linear(obs, w["hidden.weight"], w["hidden.bias"]).map(
        Math.tanh,
      );
      return {
        probability:
          1 /
          (1 + Math.exp(-linear(h, w["policy.weight"], w["policy.bias"])[0])),
        activity: h,
      };
    }
    const n = w.bias.length,
      inject = Array(n).fill(0),
      input = linear(obs, w["input.weight"], w["input.bias"]);
    w.inputs.forEach((node, i) => (inject[node] = input[i]));
    let h = inject.map(Math.tanh);
    let weights = w.edge_weight;
    // Random intervention weights are exported by evaluation to reproduce the seeded Torch draws.
    if (this.intervention === "random" && this.artifact.random_edge_weights)
      weights = this.artifact.random_edge_weights;
    const outputs = new Set(w.outputs);
    for (let pass = 0; pass < 4; pass++) {
      const msg = Array(n).fill(0);
      for (let e = 0; e < w.src.length; e++) {
        if (this.intervention === "lesion" && outputs.has(w.dst[e])) continue;
        msg[w.dst[e]] += h[w.src[e]] * weights[e];
      }
      h = h.map((_, i) =>
        Math.tanh(inject[i] + w.bias[i] + msg[i] * w.gain[i]),
      );
    }
    const out =
      this.intervention === "bypass"
        ? w.outputs.map(
            () =>
              input.map(Math.tanh).reduce((a, b) => a + b, 0) / input.length,
          )
        : w.outputs.map((i) => h[i]);
    return {
      probability:
        1 /
        (1 + Math.exp(-linear(out, w["policy.weight"], w["policy.bias"])[0])),
      activity: h,
    };
  }
}
