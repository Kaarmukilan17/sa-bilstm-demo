import * as ort from "./assets/ort/ort.wasm.min.mjs";

const MAX_LEN = 300;
const BATCH = 32;
const WARMUP = 3;
const TIMED = 20;

// Measured by us in Python on an Intel Core i5-1135G7 (mean latency per batch of 32, ms).
const REFERENCE = [
  { name: "PyTorch (CPU)", ms: 160.3, thr: 200 },
  { name: "ONNX Runtime (CPU)", ms: 116.7, thr: 274 },
  { name: "OpenVINO FP32 (CPU)", ms: 111.8, thr: 286 },
  { name: "OpenVINO FP16 (CPU)", ms: 102.6, thr: 312, best: true },
];

const $ = (id) => document.getElementById(id);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

let session = null;
let vocab = null;
let samples = [];
let picked = [];
let yours = null;

/* ---------- tokenizer: must match the training notebook ---------- */
// Training: re.findall(r"\b\w+\b", text.lower()), first 300 tokens,
// unknown -> 1, padding (0) appended at the END.
function encode(text) {
  const tokens = String(text).toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  const ids = new Array(MAX_LEN).fill(0);
  const n = Math.min(tokens.length, MAX_LEN);
  for (let i = 0; i < n; i++) {
    const id = vocab[tokens[i]];
    ids[i] = id === undefined ? 1 : id;
  }
  return ids;
}

function makeTensor(rows) {
  const data = new BigInt64Array(rows.length * MAX_LEN);
  rows.forEach((ids, r) => ids.forEach((v, c) => (data[r * MAX_LEN + c] = BigInt(v))));
  return new ort.Tensor("int64", data, [rows.length, MAX_LEN]);
}

async function predict(tensor) {
  const out = await session.run({ input_ids: tensor });
  return Array.from(out.logits.data, sigmoid);
}

/* ---------- setup ---------- */
async function init() {
  const base = new URL("./assets/", import.meta.url);
  ort.env.wasm.wasmPaths = new URL("ort/", base).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(navigator.hardwareConcurrency || 2, 4)
    : 1;

  try {
    const [vres, mres] = await Promise.all([
      fetch(new URL("vocabulary.json", base)),
      fetch(new URL("sa_bilstm.onnx", base)),
    ]);
    if (!vres.ok || !mres.ok) throw new Error("model files not found");
    vocab = await vres.json();
    session = await ort.InferenceSession.create(await mres.arrayBuffer(), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    $("modelstate").textContent = "Could not load the model: " + e.message;
    return;
  }
  $("modelstate").textContent = "Model ready (runs locally in your browser).";
  $("classify").disabled = false;

  try {
    const r = await fetch(new URL("demo_samples.json", base));
    if (!r.ok) throw new Error();
    samples = (await r.json()).filter((s) => s && typeof s.text === "string");
  } catch {
    samples = [];
  }
  if (samples.length >= BATCH) {
    $("runbatch").disabled = false;
    $("shuffle").disabled = false;
    drawArticles();
    $("batchstate").textContent = `${samples.length} test articles available.`;
  } else {
    $("batchstate").textContent =
      "Add assets/demo_samples.json (see README) to enable the batch demo.";
  }
  renderChart();
}

/* ---------- single article ---------- */
$("classify").addEventListener("click", async () => {
  const text = $("article").value.trim();
  if (!text) return;
  const [p] = await predict(makeTensor([encode(text)]));
  const box = $("single");
  const fake = p >= 0.5;
  box.className = "verdict " + (fake ? "fake" : "real");
  $("single-verdict").textContent = fake ? "Likely FAKE" : "Likely GENUINE";
  $("single-detail").textContent = `P(fake) = ${p.toFixed(3)}  |  P(genuine) = ${(1 - p).toFixed(3)}`;
  const meter = box.querySelector(".meter");
  meter.hidden = false;
  $("single-meter").style.width = (p * 100).toFixed(1) + "%";
});
$("clear").addEventListener("click", () => {
  $("article").value = "";
  $("single").className = "verdict empty";
  $("single-verdict").textContent = "Waiting for input";
  $("single-detail").textContent = "The model outputs a fake-probability. 0.50 or above is called fake.";
  $("single").querySelector(".meter").hidden = true;
});

/* ---------- batch of 32 ---------- */
function drawArticles() {
  const idx = samples.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  picked = idx.slice(0, BATCH).map((i) => samples[i]);
}
$("shuffle").addEventListener("click", () => {
  drawArticles();
  $("batchstate").textContent = "New articles drawn. Press Run.";
});

const tick = () => new Promise((r) => setTimeout(r, 0));
const pct = (arr, q) => {
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
};

$("runbatch").addEventListener("click", async () => {
  const btns = ["runbatch", "shuffle", "classify"].map($);
  btns.forEach((b) => (b.disabled = true));
  const prog = $("prog");
  prog.hidden = false;
  prog.max = WARMUP + TIMED;
  prog.value = 0;

  const tensor = makeTensor(picked.map((s) => encode(s.text))); // not timed
  let probs = [];
  const times = [];
  for (let i = 0; i < WARMUP + TIMED; i++) {
    $("batchstate").textContent = i < WARMUP ? "Warming up..." : `Timing run ${i - WARMUP + 1} of ${TIMED}...`;
    await tick();
    const t0 = performance.now();
    probs = await predict(tensor);
    const dt = performance.now() - t0;
    if (i >= WARMUP) times.push(dt);
    prog.value = i + 1;
  }
  prog.hidden = true;
  $("batchstate").textContent = "Done.";

  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const thr = (BATCH * 1000) / mean;
  const correct = picked.filter((s, i) => (probs[i] >= 0.5 ? 1 : 0) === Number(s.label)).length;

  $("s-mean").textContent = mean.toFixed(1);
  $("s-med").textContent = pct(times, 0.5).toFixed(1);
  $("s-p95").textContent = pct(times, 0.95).toFixed(1);
  $("s-thr").textContent = thr.toFixed(0);
  $("s-acc").textContent = `${correct}/${BATCH}`;
  $("s-acc-d").textContent = `${((correct / BATCH) * 100).toFixed(1)}% correct`;
  $("stats").hidden = false;

  const threads = ort.env.wasm.numThreads;
  const dev = $("device");
  dev.hidden = false;
  dev.textContent =
    `This device: ${navigator.hardwareConcurrency || "unknown"} logical cores reported, ` +
    `${threads} WebAssembly thread${threads > 1 ? "s" : ""} used` +
    (self.crossOriginIsolated ? "" : " (multi-threading is off on this host)") +
    `. ${WARMUP} warm-up and ${TIMED} timed runs of the same batch; tokenising is not timed.`;

  fillTable(probs);
  yours = { name: "Your browser (ONNX Runtime Web)", ms: mean, thr, you: true };
  renderChart();

  ["runbatch", "shuffle", "classify"].map($).forEach((b) => (b.disabled = false));
});

function fillTable(probs) {
  const body = $("results").querySelector("tbody");
  body.textContent = "";
  picked.forEach((s, i) => {
    const truth = Number(s.label) === 1;
    const pred = probs[i] >= 0.5;
    const tr = document.createElement("tr");
    if (truth !== pred) tr.className = "wrong";
    const cells = [
      String(i + 1),
      s.text.replace(/\s+/g, " ").slice(0, 150) + (s.text.length > 150 ? "..." : ""),
      truth ? "Fake" : "Genuine",
      pred ? "Fake" : "Genuine",
      probs[i].toFixed(3),
      truth === pred ? "correct" : "wrong",
    ];
    cells.forEach((c, k) => {
      const td = document.createElement("td");
      td.textContent = c;
      if (k === 2 || k === 3) td.className = c === "Fake" ? "tag-fake" : "tag-real";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  $("tablewrap").hidden = false;
}

/* ---------- comparison chart ---------- */
function renderChart() {
  const rows = yours ? [...REFERENCE, yours] : REFERENCE;
  const max = Math.max(...rows.map((r) => r.ms));
  const el = $("chart");
  el.textContent = "";
  const head = document.createElement("div");
  head.className = "bar-head";
  head.innerHTML = "<span>Runtime</span><span>Latency per batch of 32 (ms)</span><span>Latency | throughput</span>";
  el.appendChild(head);
  rows.forEach((r) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (r.best ? " best" : "") + (r.you ? " you" : "");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = r.name;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = ((r.ms / max) * 100).toFixed(1) + "%";
    track.appendChild(fill);
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = `${r.ms.toFixed(1)} ms | ${Math.round(r.thr)} articles/s`;
    row.append(name, track, val);
    el.appendChild(row);
  });
  $("yourrow-note").hidden = !yours;
}

renderChart();
init();
