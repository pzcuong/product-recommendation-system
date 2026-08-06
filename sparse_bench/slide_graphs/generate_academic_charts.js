#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "output", "charts", "cearfn_academic");
fs.mkdirSync(OUT, { recursive: true });

const W = 2400, H = 1200;
const C = {
  blue: "#2B6CB0",
  blue2: "#67A3D9",
  gray1: "#9E9E9E",
  gray2: "#BDBDBD",
  gray3: "#D0D0D0",
  orange: "#E76F51",
  green: "#0B8F55",
  red: "#C73E3E",
  purple: "#7A64B7",
  ink: "#111111",
  grid: "#D9D9D9",
  white: "#FFFFFF",
};

const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const t = (x, y, s, size = 38, o = {}) =>
  `<text x="${x}" y="${y}" font-family="${o.family || "Georgia, 'Times New Roman', serif"}" font-size="${size}" font-weight="${o.weight || 400}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || "start"}"${o.rotate ? ` transform="rotate(${o.rotate} ${x} ${y})"` : ""}>${esc(s)}</text>`;
const r = (x, y, w, h, fill, stroke = "none", sw = 0) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
const l = (x1, y1, x2, y2, stroke = C.ink, sw = 3, dash = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const c = (x, y, rad, fill, stroke = C.white, sw = 3) =>
  `<circle cx="${x}" cy="${y}" r="${rad}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
const fmt = (v, d = 3) => Number(v).toFixed(d);
const scale = (v, lo, hi, a, b) => a + ((v - lo) / (hi - lo)) * (b - a);
const hexRgb = (hex) => [
  parseInt(hex.slice(1, 3), 16),
  parseInt(hex.slice(3, 5), 16),
  parseInt(hex.slice(5, 7), 16),
];
const blend = (from, to, amount) => {
  const a = hexRgb(from), b = hexRgb(to);
  return `#${a.map((v, i) => Math.round(v + (b[i] - v) * amount)
    .toString(16).padStart(2, "0")).join("")}`;
};

function frame(title, body, source) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  ${r(0, 0, W, H, C.white)}
  ${t(W / 2, 82, title, 60, { weight: 700, anchor: "middle" })}
  ${body.join("")}
  ${t(W - 80, H - 28, source, 22, { anchor: "end", fill: "#555555", family: "Arial, Helvetica, sans-serif" })}
  </svg>`;
}

async function save(name, title, body, source) {
  const svg = frame(title, body, source);
  fs.writeFileSync(path.join(OUT, `${name}.svg`), svg);
  await sharp(Buffer.from(svg)).png().toFile(path.join(OUT, `${name}.png`));
}

function axes(body, cfg) {
  const { x0, x1, y0, y1, ymin = 0, ymax, ticks = 5, ylabel = "" } = cfg;
  const tickStep = (ymax - ymin) / ticks;
  const tickDigits = tickStep < .02 ? 3 : tickStep < .1 ? 2 : 1;
  for (let i = 0; i <= ticks; i++) {
    const v = ymin + ((ymax - ymin) * i) / ticks;
    const y = scale(v, ymin, ymax, y0, y1);
    body.push(l(x0, y, x1, y, i === 0 ? C.ink : C.grid, i === 0 ? 4 : 2));
    body.push(t(x0 - 25, y + 12, v.toFixed(tickDigits), 31, { anchor: "end" }));
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  body.push(l(x0, y1, x1, y1, C.ink, 4));
  if (ylabel) body.push(t(60, (y0 + y1) / 2, ylabel, 48, { anchor: "middle", rotate: -90 }));
}

function legend(body, entries, x, y, cols = entries.length) {
  const colW = 500;
  entries.forEach(([label, color], i) => {
    const row = Math.floor(i / cols), col = i % cols;
    const xx = x + col * colW, yy = y + row * 54;
    body.push(r(xx, yy - 28, 64, 28, color));
    body.push(t(xx + 82, yy - 3, label, 31));
  });
}

async function fusionAblation() {
  const body = [];
  const data = [
    ["Video Games", [.11901, .12571, .14685]],
    ["Baby Products", [.03879, .04873, .05346]],
    ["Diginetica", [.49565, .44852, .51681]],
  ];
  const x0 = 190, x1 = 2320, y0 = 1060, y1 = 170, ymax = .56;
  axes(body, { x0, x1, y0, y1, ymax, ticks: 5, ylabel: "Recall@20" });
  legend(body, [["Memory-only (β=0)", C.gray1], ["Neural-only (β=1)", C.gray2], ["Fused (CEARF-N)", C.blue]], 260, 230, 1);
  const centers = [590, 1250, 1910], barW = 165;
  data.forEach(([domain, vals], di) => {
    vals.forEach((v, si) => {
      const x = centers[di] + (si - 1) * (barW + 4);
      const y = scale(v, 0, ymax, y0, y1);
      const color = [C.gray1, C.gray2, C.blue][si];
      body.push(r(x - barW / 2, y, barW, y0 - y, color));
      body.push(t(x, y - 18, fmt(v), 34, { anchor: "middle" }));
    });
    body.push(t(centers[di], 1115, domain, 39, { anchor: "middle" }));
  });
  await save("01-fusion-ablation-recall20", "Fusion ablation: memory-only vs neural-only vs fused", body, "Current locked three-seed means • Supplementary evidence audit");
}

async function cutoffAblation() {
  const body = [];
  const datasets = {
    "Video Games": { Memory: [.06932, .08821, .11901], Neural: [.06158, .08471, .12571], Fused: [.07803, .10368, .14685] },
    "Baby Products": { Memory: [.02288, .02900, .03879], Neural: [.02337, .03214, .04873], Fused: [.02767, .03668, .05346] },
    Diginetica: { Memory: [.29174, .37334, .49565], Neural: [.26398, .33539, .44852], Fused: [.30738, .39210, .51681] },
  };
  const panelW = 700, gaps = 65, left = 170;
  Object.entries(datasets).forEach(([domain, series], pi) => {
    const px = left + pi * (panelW + gaps);
    const y0 = 1010, y1 = 235;
    const ymax = Math.max(...Object.values(series).flat()) * 1.14;
    axes(body, { x0: px, x1: px + panelW, y0, y1, ymax, ticks: 4, ylabel: pi === 0 ? "Recall" : "" });
    body.push(t(px + panelW / 2, 172, domain, 42, { weight: 700, anchor: "middle" }));
    const centers = [px + 150, px + 350, px + 550], bw = 54;
    ["R@6", "R@10", "R@20"].forEach((lab, i) => body.push(t(centers[i], 1060, lab, 30, { anchor: "middle" })));
    ["Memory", "Neural", "Fused"].forEach((name, si) => {
      const color = [C.gray1, C.gray2, C.blue][si];
      series[name].forEach((v, ki) => {
        const x = centers[ki] + (si - 1) * (bw + 3);
        const y = scale(v, 0, ymax, y0, y1);
        body.push(r(x - bw / 2, y, bw, y0 - y, color));
        body.push(t(x, y - 10, fmt(v), 22, { anchor: "middle" }));
      });
    });
  });
  legend(body, [["Memory-only", C.gray1], ["Neural-only", C.gray2], ["Fused CEARF-N", C.blue]], 470, 1132, 3);
  await save("02-fusion-ablation-all-cutoffs", "Fusion remains strongest across Recall cutoffs", body, "Current locked three-seed means • Supplementary evidence audit");
}

async function baselines() {
  const body = [];
  const methods = ["V-SKNN", "STAN", "Transition", "GRU4Rec", "NARM", "SR-GNN", "SIGMA", "CEARF-N"];
  const vals = {
    "Video Games": [.11936, .12115, .04223, .10368, .13705, .12267, .08644, .14685],
    "Baby Products": [.05038, .04944, .00735, .04501, .02990, .05208, .02473, .05346],
    Diginetica: [.50506, .51502, .40102, .41785, .53406, .49267, .37219, .51681],
  };
  const colors = [C.gray1, C.gray2, C.gray3, "#7F8C8D", C.orange, "#8FA6B8", "#B7A7C8", C.blue];
  const x0 = 180, x1 = 2320, y0 = 1010, y1 = 275, ymax = .58;
  axes(body, { x0, x1, y0, y1, ymax, ticks: 5, ylabel: "Recall@20" });
  const centers = [560, 1250, 1940], bw = 54;
  Object.entries(vals).forEach(([domain, data], di) => {
    data.forEach((v, mi) => {
      const x = centers[di] + (mi - 3.5) * (bw + 3);
      const y = scale(v, 0, ymax, y0, y1);
      body.push(r(x - bw / 2, y, bw, y0 - y, colors[mi]));
      if (methods[mi] === "CEARF-N" || (domain === "Diginetica" && methods[mi] === "NARM")) {
        body.push(t(x, y - 14, fmt(v), 25, { anchor: "middle", weight: 700 }));
      }
    });
    body.push(t(centers[di], 1065, domain, 38, { anchor: "middle" }));
  });
  legend(body, methods.map((m, i) => [m, colors[i]]), 175, 170, 4);
  await save("03-baseline-comparison-recall20", "Recall@20 comparison against reported baselines", body, "Main paper Table 1 • Amazon CEARF-N uses TF–IDF/SVD; comparator rows are ID-only");
}

async function externalGate() {
  const body = [];
  const data = [
    ["Video Games", .14685, .15342, .00658],
    ["Baby Products", .05364, .05882, .00518],
    ["Diginetica", .53406, .53790, .00384],
  ];
  const x0 = 190, x1 = 2320, y0 = 1050, y1 = 245, ymax = .59;
  axes(body, { x0, x1, y0, y1, ymax, ticks: 5, ylabel: "Recall@20" });
  legend(body, [["Validation-selected singleton", C.gray1], ["External expert gate", C.blue]], 260, 180, 1);
  const centers = [620, 1300, 1980], bw = 210;
  data.forEach(([domain, singleton, gate, delta], i) => {
    [[singleton, C.gray1, -1], [gate, C.blue, 1]].forEach(([v, color, off]) => {
      const x = centers[i] + off * (bw / 2 + 5);
      const y = scale(v, 0, ymax, y0, y1);
      body.push(r(x - bw / 2, y, bw, y0 - y, color));
      body.push(t(x, y - 18, fmt(v), 34, { anchor: "middle" }));
    });
    body.push(t(centers[i], 1105, domain, 38, { anchor: "middle" }));
    body.push(t(centers[i] + 110, scale(gate, 0, ymax, y0, y1) - 70, `Δ +${fmt(delta)}`, 31, { anchor: "middle", fill: C.green, weight: 700 }));
  });
  await save("04-external-gate-recall20", "External expert gate vs validation-selected singleton", body, "hard_gate_singleton_audit.json • All 95% CIs exclude zero");
}

async function metadata() {
  const body = [];
  const domains = {
    "Video Games": { CEARF: [.13208, .14685, .14794], NARM: [.13705, .15387, .15392] },
    "Baby Products": { CEARF: [.05055, .05346, .05761], NARM: [.02990, .06628, .06493] },
  };
  const cond = ["ID-only", "TF–IDF/SVD", "MiniLM"];
  Object.entries(domains).forEach(([domain, series], pi) => {
    const px = 220 + pi * 1130, panelW = 850, y0 = 1000, y1 = 240;
    const ymax = pi === 0 ? .17 : .075;
    axes(body, { x0: px, x1: px + panelW, y0, y1, ymax, ticks: 5, ylabel: pi === 0 ? "Recall@20" : "" });
    body.push(t(px + panelW / 2, 175, domain, 43, { weight: 700, anchor: "middle" }));
    const centers = [px + 180, px + 425, px + 670], bw = 92;
    cond.forEach((lab, i) => body.push(t(centers[i], 1055, lab, 30, { anchor: "middle" })));
    [["CEARF", C.blue, -1], ["NARM", C.orange, 1]].forEach(([name, color, off]) => {
      series[name].forEach((v, i) => {
        const x = centers[i] + off * 49;
        const y = scale(v, 0, ymax, y0, y1);
        body.push(r(x - bw / 2, y, bw, y0 - y, color));
        body.push(t(x, y - 12, fmt(v), 27, { anchor: "middle" }));
      });
    });
  });
  legend(body, [["CEARF-N", C.blue], ["NARM", C.orange]], 750, 1120, 2);
  await save("05-metadata-matched-fairness", "Metadata-matched fairness: CEARF-N vs NARM", body, "Main paper Table 4 • Three-seed means");
}

async function transferMargins() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "validation_test_transfer_audit.json")));
  const body = [];
  const rows = [
    ["Adaptive router", audit.stages.adaptive_router.summary],
    ["External expert gate", audit.stages.external_expert_gate.summary],
  ];
  const x0 = 270, x1 = 2280, y0 = 960, y1 = 245, ymin = -.0025, ymax = .006;
  for (let i = -2; i <= 6; i++) {
    const v = i / 1000;
    const y = scale(v, ymin, ymax, y0, y1);
    body.push(l(x0, y, x1, y, v === 0 ? C.ink : C.grid, v === 0 ? 4 : 2));
    body.push(t(x0 - 25, y + 10, v.toFixed(3), 30, { anchor: "end" }));
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  const centers = [800, 1700], bw = 240;
  rows.forEach(([name, d], i) => {
    const values = [d.mean_validation_admission_margin, d.mean_test_retention_margin];
    values.forEach((v, j) => {
      const x = centers[i] + (j === 0 ? -130 : 130);
      const yZero = scale(0, ymin, ymax, y0, y1);
      const yVal = scale(v, ymin, ymax, y0, y1);
      const color = j === 0 ? C.gray1 : v < 0 ? C.red : C.blue;
      body.push(r(x - bw / 2, Math.min(yVal, yZero), bw, Math.abs(yZero - yVal), color));
      body.push(t(x, v >= 0 ? yVal - 18 : yVal + 45, (v >= 0 ? "" : "−") + Math.abs(v).toFixed(5), 34, { anchor: "middle" }));
    });
    body.push(t(centers[i], 1040, name, 39, { anchor: "middle" }));
    body.push(t(centers[i], 1090, `${d.retained_nonnegative}/${d.admitted_decisions} retained`, 31, { anchor: "middle", fill: i === 0 ? C.red : C.green, weight: 700 }));
  });
  legend(body, [["Validation admission margin A", C.gray1], ["Test retention margin R", C.blue], ["Negative test retention", C.red]], 400, 170, 3);
  body.push(t(80, 600, "Utility margin", 47, { anchor: "middle", rotate: -90 }));
  await save("06b-validation-test-margins", "Validation-to-test utility margins by gate type", body, "validation_test_transfer_audit.json • U = 0.5·R@6 + 0.5·R@20 • Backup for RQ2");
}

function modalCount(values) {
  const counts = new Map();
  values.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  return Math.max(...counts.values());
}

async function admissionReliability() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "validation_test_transfer_audit.json")));
  const hardGate = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "hard_gate_singleton_audit.json")));
  const body = [];
  const stageRows = [
    ["Adaptive router", audit.stages.adaptive_router, C.orange],
    ["External expert gate", audit.stages.external_expert_gate, C.blue],
  ];

  // Panel A: all four quantities named by RQ2 on a common decision-count scale.
  const ax0 = 170, ax1 = 1250, ay0 = 990, ay1 = 280, amax = 9;
  for (let i = 0; i <= 9; i += 1) {
    const y = scale(i, 0, amax, ay0, ay1);
    body.push(l(ax0, y, ax1, y, i === 0 ? C.ink : C.grid, i === 0 ? 4 : 2));
    if (i % 3 === 0) body.push(t(ax0 - 22, y + 10, String(i), 29, { anchor: "end" }));
  }
  body.push(l(ax0, ay0, ax0, ay1, C.ink, 4));
  body.push(t((ax0 + ax1) / 2, 205, "Decision outcomes", 42, { weight: 700, anchor: "middle" }));
  const categories = [
    ["Eligible", "eligible_decisions"],
    ["Admitted", "admitted_decisions"],
    ["Retained", "retained_nonnegative"],
    ["Reversed", "reversals"],
  ];
  const centers = [330, 610, 890, 1170], bw = 92;
  categories.forEach(([label, key], ci) => {
    stageRows.forEach(([, stage, color], si) => {
      const value = stage.summary[key];
      const x = centers[ci] + (si === 0 ? -52 : 52);
      const y = scale(value, 0, amax, ay0, ay1);
      body.push(r(x - bw / 2, y, bw, ay0 - y, color));
      body.push(t(x, y - 14, String(value), 34, { anchor: "middle", weight: 700 }));
    });
    body.push(t(centers[ci], 1045, label, 31, { anchor: "middle" }));
  });
  legend(body, [["Adaptive router", C.orange], ["External expert gate", C.blue]], 330, 1120, 2);
  body.push(t(58, 635, "Number of decisions", 43, { anchor: "middle", rotate: -90 }));

  // Panel B: stability is defined, rather than implied by aggregate margins.
  const bx0 = 1400, bx1 = 2300;
  body.push(t((bx0 + bx1) / 2, 205, "Cross-seed stability", 42, { weight: 700, anchor: "middle" }));
  const domainOrder = [
    ["Video_Games", "Video"],
    ["Baby_Products", "Baby"],
    ["Diginetica_HID", "Diginetica"],
  ];
  const xCenters = [1580, 1845, 2140];
  domainOrder.forEach(([, label], i) => body.push(t(xCenters[i], 290, label, 30, { anchor: "middle", weight: 700 })));
  const stabilityRows = [
    ["Router admit/reject", audit.stages.adaptive_router.records, "admitted"],
    ["Router exact family", audit.stages.adaptive_router.records, "selected"],
    ["External admit/reject", audit.stages.external_expert_gate.records, "admitted"],
    ["External exact subset", audit.stages.external_expert_gate.records, "selected"],
  ];
  const yCenters = [440, 590, 790, 940];
  stabilityRows.forEach(([label, records, key], ri) => {
    body.push(t(1375, yCenters[ri] + 10, label, 27, { anchor: "end" }));
    domainOrder.forEach(([domain], di) => {
      const values = records.filter((record) => record.domain === domain).map((record) => String(record[key]));
      const stable = modalCount(values);
      const color = stable === 3 ? C.green : C.orange;
      body.push(c(xCenters[di], yCenters[ri], 46, color, C.white, 4));
      body.push(t(xCenters[di], yCenters[ri] + 11, `${stable}/3`, 29, { anchor: "middle", fill: C.white, weight: 700 }));
    });
  });
  body.push(l(bx0, 690, bx1, 690, C.grid, 2));
  const domainKeys = ["Video_Games", "Baby_Products", "Diginetica_HID"];
  const prettyExpert = { cearf: "C", stan: "S", vsknn: "V", narm: "N" };
  const membership = domainKeys.map((domain) => {
    const selected = hardGate.domains[domain].runs.map((run) =>
      new Set(run.selected.replace(/^rrf_/, "").split("_")));
    const intersection = [...selected[0]].filter((x) => selected.every((set) => set.has(x)));
    const union = new Set(selected.flatMap((set) => [...set]));
    const values = hardGate.domains[domain].runs.map(
      (run) => run["selected_test_recall@20"]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = Math.sqrt(values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / (values.length - 1));
    return {
      core: intersection.map((x) => prettyExpert[x]).join("+"),
      jaccard: intersection.length / union.size,
      sd,
    };
  });
  membership.forEach((d, i) => {
    body.push(t(xCenters[i], 1002, `core ${d.core}`, 26, { anchor: "middle", weight: 700, fill: C.blue }));
    body.push(t(xCenters[i], 1038, `J=${d.jaccard.toFixed(3)} · SD=${d.sd.toFixed(5)}`, 22, { anchor: "middle", fill: "#444444", family: "Arial, Helvetica, sans-serif" }));
  });
  body.push(t((bx0 + bx1) / 2, 1090, "Exact membership can change while final Recall@20 remains stable", 24, { anchor: "middle", fill: "#555555", family: "Arial, Helvetica, sans-serif" }));
  legend(body, [["3/3 stable", C.green], ["2/3 stable", C.orange]], 1540, 1140, 2);

  await save(
    "06-validation-test-transfer",
    "RQ2: Are validation admission decisions reliable?",
    body,
    "validation_test_transfer_audit.json • Retained + reversed = admitted • Test never triggers reselection"
  );
}

async function routerSeeds() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "validation_test_transfer_audit.json")));
  const records = audit.stages.adaptive_router.records.filter((x) => x.admitted);
  const body = [];
  const x0 = 220, x1 = 2320, y0 = 960, y1 = 240, ymin = -.007, ymax = .005;
  for (let i = -6; i <= 4; i += 2) {
    const v = i / 1000;
    const y = scale(v, ymin, ymax, y0, y1);
    body.push(l(x0, y, x1, y, v === 0 ? C.ink : C.grid, v === 0 ? 4 : 2));
    body.push(t(x0 - 25, y + 10, v.toFixed(3), 29, { anchor: "end" }));
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  const centers = records.map((_, i) => 430 + i * 410), bw = 130;
  records.forEach((rec, i) => {
    const lab = `${rec.domain === "Video_Games" ? "Video" : rec.domain === "Baby_Products" ? "Baby" : "Digi"}-${rec.seed}`;
    [rec.validation_admission_margin, rec.test_retention_margin].forEach((v, j) => {
      const x = centers[i] + (j === 0 ? -70 : 70);
      const zero = scale(0, ymin, ymax, y0, y1), yy = scale(v, ymin, ymax, y0, y1);
      const color = j === 0 ? C.gray1 : v < 0 ? C.red : C.blue;
      body.push(r(x - bw / 2, Math.min(zero, yy), bw, Math.abs(zero - yy), color));
      body.push(t(x, v >= 0 ? yy - 12 : yy + 38, (v >= 0 ? "" : "−") + Math.abs(v).toFixed(4), 25, { anchor: "middle" }));
    });
    body.push(t(centers[i], 1020, lab, 30, { anchor: "middle" }));
  });
  legend(body, [["Validation A", C.gray1], ["Test R ≥ 0", C.blue], ["Test reversal", C.red]], 520, 170, 3);
  body.push(t(75, 600, "Utility margin", 47, { anchor: "middle", rotate: -90 }));
  await save("07-router-transfer-per-seed", "Adaptive-router admission and test retention by seed", body, "cearfn_v2_nested_results.json • Five positively admitted router decisions");
}

async function inference() {
  const runtime = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "cearfn_inference_benchmark.json")));
  const body = [];
  const domains = [
    ["Video Games", runtime.domains.Video_Games],
    ["Baby Products", runtime.domains.Baby_Products],
    ["Diginetica", runtime.domains.Diginetica_HID],
  ];
  const panels = [
    { x0: 220, x1: 1120, max: 450, key: "queries_per_second", title: "Throughput (queries/s)", color: C.blue, digits: 1 },
    { x0: 1350, x1: 2250, max: 4.6, key: "amortized_milliseconds_per_query", title: "Amortized time (ms/query)", color: C.gray1, digits: 3 },
  ];
  panels.forEach((p) => {
    body.push(t((p.x0 + p.x1) / 2, 190, p.title, 42, { weight: 700, anchor: "middle" }));
    const y0 = 980, y1 = 270;
    for (let i = 0; i <= 5; i++) {
      const v = p.max * i / 5;
      const y = scale(v, 0, p.max, y0, y1);
      body.push(l(p.x0, y, p.x1, y, i === 0 ? C.ink : C.grid, i === 0 ? 4 : 2));
      body.push(t(p.x0 - 20, y + 10, v.toFixed(p.max < 10 ? 1 : 0), 28, { anchor: "end" }));
    }
    body.push(l(p.x0, y0, p.x0, y1, C.ink, 4));
    const centers = [430, 680, 930].map((v) => v + (p.x0 - 220));
    domains.forEach(([name, d], i) => {
      const v = d[p.key], y = scale(v, 0, p.max, y0, y1);
      body.push(r(centers[i] - 90, y, 180, y0 - y, p.color));
      body.push(t(centers[i], y - 16, v.toFixed(p.digits), 33, { anchor: "middle" }));
      body.push(t(centers[i], 1035, name.replace(" Products", ""), 29, { anchor: "middle" }));
    });
  });
  body.push(t(W / 2, 1115, "Warm full-catalogue path; excludes loading, training, index construction, and metric computation", 30, { anchor: "middle" }));
  await save("08-inference-performance", "CEARF-N inference performance", body, "cearfn_inference_benchmark.json • Apple M2 Pro, 32 GB; CPU CEARF + Metal PASGR");
}

async function rentalComparison() {
  const body = [];
  const data = [
    ["MostPop", .229175, C.gray3],
    ["ItemKNN", .230739, C.gray2],
    ["MGCOT-MPS", .238170, C.orange],
    ["DT-RRF", .268752, "#8492A6"],
    ["SKNN", .269456, C.gray1],
    ["MGCOT–CEARF", .280798, "#2A9D8F"],
    ["CEARF", .284709, C.blue],
  ];
  const x0 = 190, x1 = 2320, y0 = 1000, y1 = 200, ymax = .32;
  axes(body, { x0, x1, y0, y1, ymax, ticks: 4, ylabel: "Recall@6" });
  const centers = data.map((_, i) => 355 + i * 300);
  const barW = 205;
  data.forEach(([name, value, color], i) => {
    const y = scale(value, 0, ymax, y0, y1);
    body.push(r(centers[i] - barW / 2, y, barW, y0 - y, color));
    body.push(t(centers[i], y - 18, value.toFixed(5), 31, {
      anchor: "middle",
      weight: name === "CEARF" ? 700 : 400,
    }));
    body.push(t(centers[i], 1055, name, 29, {
      anchor: "middle",
      weight: name === "CEARF" ? 700 : 400,
    }));
  });
  const fusion = data[5][1], memory = data[6][1];
  body.push(t(2260, 160,
              `Validation: fusion +.00817; test: −${(memory - fusion).toFixed(5)} vs CEARF`, 29,
              { anchor: "end", fill: C.red, weight: 700 }));
  await save(
    "09-rental-masked-loo-recall6",
    "Rental masked leave-one-out: controlled method comparison",
    body,
    "RENTAL_SOTA_AUDIT.md • 2,557 test queries • train-only adjacency • nested-selected configurations"
  );
}

async function evidenceRescueDamage() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "evidence_necessity_audit.json")));
  const body = [];
  const names = ["Video Games", "Baby Products", "Diginetica"];
  const rows = names.map((name) => {
    const s = audit.domains[name].overall.summary;
    return [name, s.rescue_rate.mean, s.damage_rate.mean, s.net_rescue_rate.mean];
  });
  const x0 = 230, x1 = 2300, y0 = 900, y1 = 260, ymin = -.03, ymax = .055;
  for (let i = -3; i <= 5; i++) {
    const value = i / 100;
    const y = scale(value, ymin, ymax, y0, y1);
    body.push(l(x0, y, x1, y, value === 0 ? C.ink : C.grid, value === 0 ? 4 : 2));
    body.push(t(x0 - 24, y + 11, value.toFixed(2), 30, { anchor: "end" }));
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  const zero = scale(0, ymin, ymax, y0, y1);
  const centers = [600, 1260, 1920], bw = 210;
  rows.forEach(([name, rescue, damage, net], i) => {
    [[rescue, C.blue, -115], [-damage, C.red, 115]].forEach(([value, color, offset]) => {
      const x = centers[i] + offset;
      const y = scale(value, ymin, ymax, y0, y1);
      body.push(r(x - bw / 2, Math.min(y, zero), bw, Math.abs(zero - y), color));
      body.push(t(x, value >= 0 ? y - 16 : y + 42, `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(4)}`, 32, { anchor: "middle" }));
    });
    body.push(t(centers[i], 975, name, 38, { anchor: "middle" }));
    body.push(t(centers[i], 1030, `Net +${net.toFixed(5)}`, 32, { anchor: "middle", fill: C.green, weight: 700 }));
  });
  legend(body, [["Rescue: memory miss → fusion hit", C.blue], ["Damage: memory hit → fusion miss", C.red]], 650, 165, 1);
  body.push(t(80, 600, "Share of test queries", 45, { anchor: "middle", rotate: -90 }));
  await save("10-evidence-rescue-damage", "Where fusion adds value beyond memory", body, "evidence_necessity_audit.json • Locked test ranks • Rescue − damage = ΔRecall@20");
}

async function popularityNetRescue() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "evidence_necessity_audit.json")));
  const body = [];
  const names = ["Video Games", "Baby Products", "Diginetica"];
  const strata = ["head", "torso", "tail"];
  const colors = [C.gray1, C.blue2, C.blue];
  const x0 = 220, x1 = 2310, y0 = 980, y1 = 255, ymin = -.005, ymax = .04;
  for (let i = 0; i <= 9; i++) {
    const value = ymin + i * .005;
    const y = scale(value, ymin, ymax, y0, y1);
    body.push(l(x0, y, x1, y, Math.abs(value) < 1e-9 ? C.ink : C.grid, Math.abs(value) < 1e-9 ? 4 : 2));
    if (i % 2 === 1 || Math.abs(value) < 1e-9) {
      body.push(t(x0 - 24, y + 10, value.toFixed(3), 29, { anchor: "end" }));
    }
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  const zero = scale(0, ymin, ymax, y0, y1);
  const centers = [590, 1260, 1930], bw = 135;
  names.forEach((name, i) => {
    strata.forEach((stratum, j) => {
      const value = audit.domains[name].popularity_strata[stratum].summary.net_rescue_rate.mean;
      const x = centers[i] + (j - 1) * (bw + 8);
      const y = scale(value, ymin, ymax, y0, y1);
      const fill = value < 0 ? C.red : colors[j];
      body.push(r(x - bw / 2, Math.min(y, zero), bw, Math.abs(zero - y), fill));
      body.push(t(x, value >= 0 ? y - 14 : y + 38, `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(4)}`, 27, { anchor: "middle" }));
    });
    body.push(t(centers[i], 1040, name, 37, { anchor: "middle" }));
  });
  legend(body, [["Head", colors[0]], ["Torso", colors[1]], ["Tail", colors[2]], ["Negative", C.red]], 390, 170, 4);
  body.push(t(80, 610, "Net rescue rate", 45, { anchor: "middle", rotate: -90 }));
  await save("11-net-rescue-by-popularity", "Neural residual is not uniformly a long-tail effect", body, "evidence_necessity_audit.json • Target popularity from training-frequency 20/60/20 strata");
}

async function regimeDiagnostics() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "evidence_necessity_audit.json")));
  const body = [];
  const domains = [
    ["Video Games", "Video", C.blue],
    ["Baby Products", "Baby", C.orange],
    ["Diginetica", "Diginetica", C.green],
  ];
  const panels = [
    {
      x: 160, y: 150, title: "Session length (router buckets)",
      section: "router_context_length_strata",
      categories: [["short", "L≤2"], ["mid", "3–7"], ["long", ">7"]],
    },
    {
      x: 1290, y: 150, title: "Last observed item popularity",
      section: "router_last_item_popularity_strata",
      categories: [["head", "Head"], ["tail", "Tail"]],
    },
    {
      x: 160, y: 620, title: "Cross-memory agreement",
      section: "memory_agreement_strata",
      categories: [["low", "Low"], ["high", "High"]],
    },
    {
      x: 1290, y: 620, title: "Transition branching (validation-frozen)",
      section: "transition_branching_strata",
      categories: [["low", "Low"], ["mid", "Mid"], ["high", "High"]],
    },
  ];
  const panelW = 950, panelH = 390, ymax = .045;
  panels.forEach((panel, pi) => {
    const x0 = panel.x + 80, x1 = panel.x + panelW;
    const y1 = panel.y + 85, y0 = panel.y + panelH;
    body.push(t((x0 + x1) / 2, panel.y + 38, panel.title, 34, { weight: 700, anchor: "middle" }));
    [0, .02, .04].forEach((value) => {
      const yy = scale(value, 0, ymax, y0, y1);
      body.push(l(x0, yy, x1, yy, value === 0 ? C.ink : C.grid, value === 0 ? 3 : 2));
      body.push(t(x0 - 16, yy + 9, value.toFixed(2), 24, { anchor: "end" }));
    });
    body.push(l(x0, y0, x0, y1, C.ink, 3));
    const span = (x1 - x0) / panel.categories.length;
    panel.categories.forEach(([key, label], ci) => {
      const center = x0 + span * (ci + .5);
      domains.forEach(([domain, , color], di) => {
        const block = audit.domains[domain][panel.section][key];
        const count = block.summary.n_queries;
        const value = block.summary.net_rescue_rate.mean;
        const bw = Math.min(78, span / 4);
        const xx = center + (di - 1) * (bw + 7);
        if (count === 0) {
          body.push(t(xx, y0 - 12, "—", 30, { anchor: "middle", fill: color, weight: 700 }));
        } else {
          const yy = scale(value, 0, ymax, y0, y1);
          body.push(r(xx - bw / 2, yy, bw, y0 - yy, color));
        }
      });
      body.push(t(center, y0 + 38, label, 26, { anchor: "middle" }));
    });
    if (pi % 2 === 0) body.push(t(panel.x + 18, (y0 + y1) / 2, "Net ΔRecall@20", 27, { anchor: "middle", rotate: -90 }));
  });
  legend(body, domains.map(([, label, color]) => [label, color]), 560, 1145, 3);
  body.push(t(W / 2, 1092, "Low memory agreement is the only direction that repeats across all three domains; branching is domain-dependent", 28, { anchor: "middle", weight: 700 }));
  await save(
    "12-rq1-regime-diagnostics",
    "RQ1: When does neural evidence add value beyond memory?",
    body,
    "evidence_necessity_audit.json • Locked test audit • Branch thresholds frozen on validation • ‘—’ = empty stratum"
  );
}

async function rescueMechanisms() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "evidence_necessity_audit.json")));
  const body = [];
  const names = ["Video Games", "Baby Products", "Diginetica"];
  const mechanisms = [
    ["outside_all_memory_top120", "New candidate: outside all memory top-120", C.blue],
    ["only_in_unselected_memory_component_top120", "Found only by an unselected memory", C.purple],
    ["promoted_from_selected_memory_rank21_120", "Promoted from selected-memory rank 21–120", C.blue2],
  ];
  const x0 = 220, x1 = 2320, y0 = 900, y1 = 250, ymin = -.03, ymax = .055;
  for (let i = -3; i <= 5; i += 1) {
    const value = i / 100;
    const yy = scale(value, ymin, ymax, y0, y1);
    body.push(l(x0, yy, x1, yy, value === 0 ? C.ink : C.grid, value === 0 ? 4 : 2));
    body.push(t(x0 - 24, yy + 10, value.toFixed(2), 29, { anchor: "end" }));
  }
  body.push(l(x0, y0, x0, y1, C.ink, 4));
  const zero = scale(0, ymin, ymax, y0, y1);
  const centers = [590, 1260, 1930], bw = 240;
  names.forEach((name, i) => {
    const summary = audit.domains[name].overall.summary;
    let cumulative = 0;
    mechanisms.forEach(([key, , color]) => {
      const value = summary.rescue_mechanisms[key].query_rate.mean;
      const yBottom = scale(cumulative, ymin, ymax, y0, y1);
      cumulative += value;
      const yTop = scale(cumulative, ymin, ymax, y0, y1);
      body.push(r(centers[i] - bw - 15, yTop, bw, yBottom - yTop, color));
    });
    const outsideShare = summary.rescue_mechanisms.outside_all_memory_top120.share_of_rescues.mean;
    body.push(t(centers[i] - bw / 2 - 15, scale(cumulative, ymin, ymax, y0, y1) - 16, `Rescue ${cumulative.toFixed(4)}`, 29, { anchor: "middle", weight: 700 }));
    body.push(t(centers[i] - bw / 2 - 15, scale(cumulative, ymin, ymax, y0, y1) - 51, `${(outsideShare * 100).toFixed(1)}% truly new`, 25, { anchor: "middle", fill: C.blue, weight: 700 }));
    const damage = summary.damage_rate.mean;
    const yd = scale(-damage, ymin, ymax, y0, y1);
    body.push(r(centers[i] + 15, zero, bw, yd - zero, C.red));
    body.push(t(centers[i] + bw / 2 + 15, yd + 38, `−${damage.toFixed(4)}`, 29, { anchor: "middle" }));
    body.push(t(centers[i], 975, name, 37, { anchor: "middle" }));
    body.push(t(centers[i], 1022, `Net +${summary.net_rescue_rate.mean.toFixed(5)}`, 29, { anchor: "middle", fill: C.green, weight: 700 }));
  });
  const shortLegend = [
    ["New: outside all memory top-120", C.blue, 250, 150],
    ["Unselected-memory retrieval", C.purple, 1280, 150],
    ["Promotion from rank 21–120", C.blue2, 250, 205],
    ["Damage", C.red, 1280, 205],
  ];
  shortLegend.forEach(([label, color, xx, yy]) => {
    body.push(r(xx, yy - 28, 64, 28, color));
    body.push(t(xx + 82, yy - 3, label, 29));
  });
  body.push(t(75, 610, "Share of test queries", 43, { anchor: "middle", rotate: -90 }));
  await save(
    "13-rq1-rq3-rescue-mechanisms",
    "RQ1/RQ3: Is neural gain discovery or reranking?",
    body,
    "evidence_necessity_audit.json • Disjoint rescue partition • Three-seed mean query rates"
  );
}

async function shortcutDiagnostics() {
  const shortcut = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "shortcut_diagnostics_audit.json")));
  const evidence = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "evidence_necessity_audit.json")));
  const body = [];
  const domains = [
    ["Video Games", "Video", C.blue],
    ["Baby Products", "Baby", C.orange],
    ["Diginetica", "Diginetica", C.green],
  ];

  const lx0 = 150, lx1 = 1450, ly0 = 960, ly1 = 250, ymax = .58;
  [0, .1, .2, .3, .4, .5].forEach((value) => {
    const yy = scale(value, 0, ymax, ly0, ly1);
    body.push(l(lx0, yy, lx1, yy, value === 0 ? C.ink : C.grid, value === 0 ? 4 : 2));
    body.push(t(lx0 - 18, yy + 9, value.toFixed(1), 27, { anchor: "end" }));
  });
  body.push(l(lx0, ly0, lx0, ly1, C.ink, 4));
  body.push(t((lx0 + lx1) / 2, 185, "Component target coverage / Recall@20", 38, { anchor: "middle", weight: 700 }));
  const components = [
    ["transition", "Transition"],
    ["similar_session", "Session"],
    ["selected_memory", "Selected memory"],
    ["fusion", "Fusion"],
  ];
  const groupSpan = (lx1 - lx0) / components.length;
  components.forEach(([key, label], ci) => {
    const center = lx0 + groupSpan * (ci + .5);
    domains.forEach(([domain, , color], di) => {
      const value = key === "fusion"
        ? evidence.domains[domain].overall.summary["fused_recall@20"].mean
        : shortcut.domains[domain]["component_recall@20"][key];
      const bw = 72, xx = center + (di - 1) * 78;
      const yy = scale(value, 0, ymax, ly0, ly1);
      body.push(r(xx - bw / 2, yy, bw, ly0 - yy, color));
    });
    body.push(t(center, 1012, label, 27, { anchor: "middle" }));
  });

  const rx0 = 1640, rx1 = 2310, ry0 = 950, ry1 = 300, ymin = -.0045, rymax = .001;
  body.push(t((rx0 + rx1) / 2, 185, "No-transition memory intervention", 38, { anchor: "middle", weight: 700 }));
  [-.004, -.003, -.002, -.001, 0, .001].forEach((value) => {
    const yy = scale(value, ymin, rymax, ry0, ry1);
    body.push(l(rx0, yy, rx1, yy, value === 0 ? C.ink : C.grid, value === 0 ? 4 : 2));
    body.push(t(rx0 - 18, yy + 9, value.toFixed(3), 26, { anchor: "end" }));
  });
  body.push(l(rx0, ry0, rx0, ry1, C.ink, 4));
  const rcenters = [1755, 1975, 2195];
  domains.forEach(([domain, label, color], i) => {
    const block = shortcut.domains[domain].transition_removal;
    const value = block.delta_no_transition_minus_locked;
    const zero = scale(0, ymin, rymax, ry0, ry1);
    const yy = scale(value, ymin, rymax, ry0, ry1);
    body.push(r(rcenters[i] - 70, Math.min(zero, yy), 140, Math.abs(zero - yy), color));
    body.push(t(rcenters[i], value < 0 ? yy + 38 : zero - 14, value.toFixed(5), 25, { anchor: "middle", weight: 700 }));
    body.push(t(rcenters[i], 1005, label, 25, { anchor: "middle" }));
    body.push(t(rcenters[i], 1043, `T active ${(100 * block.active_share).toFixed(1)}%`, 22, { anchor: "middle", fill: "#555555", family: "Arial, Helvetica, sans-serif" }));
  });
  legend(body, domains.map(([, label, color]) => [label, color]), 520, 1115, 3);
  body.push(t(1975, 1072, "Memory path only; PASGR/router are not refitted", 23, { anchor: "middle", fill: C.red, weight: 700 }));
  await save(
    "14-rq3-shortcut-diagnostics",
    "RQ3: Complementary evidence or benchmark shortcut?",
    body,
    "shortcut_diagnostics_audit.json • Amazon locked memory is session-only • Union coverage is not an achievable ranker"
  );
}

async function teacherPopularityAttribution() {
  const audit = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "teacher_interaction_audit.json")));
  const body = [];
  const domains = ["Video Games", "Baby Products"];
  const teachers = ["None", "TF-IDF/SVD", "MiniLM"];
  const teacherLabels = ["None", "TF–IDF", "MiniLM"];
  const strata = ["overall", "head", "torso", "tail"];
  const stratumLabels = ["Overall", "Head", "Torso", "Tail"];
  const maxAbs = .03;
  domains.forEach((domain, di) => {
    const px = 170 + di * 1150;
    body.push(t(px + 480, 195, domain, 42, { anchor: "middle", weight: 700 }));
    const cellW = 190, cellH = 155, startX = px + 255, startY = 310;
    stratumLabels.forEach((label, i) => body.push(t(startX + i * cellW + cellW / 2, 275, label, 29, { anchor: "middle", weight: 700 })));
    teachers.forEach((teacher, ti) => {
      body.push(t(startX - 25, startY + ti * cellH + 91, teacherLabels[ti], 30, { anchor: "end", weight: 700 }));
      strata.forEach((stratum, si) => {
        const block = stratum === "overall"
          ? audit.domains[domain].overall
          : audit.domains[domain].popularity_strata[stratum];
        const value = block.effects.architecture_gap_cearfn_minus_narm[teacher];
        const intensity = Math.min(Math.abs(value) / maxAbs, 1);
        const color = value >= 0
          ? blend(C.white, C.blue, .18 + .72 * intensity)
          : blend(C.white, C.red, .18 + .72 * intensity);
        const xx = startX + si * cellW, yy = startY + ti * cellH;
        body.push(r(xx, yy, cellW - 6, cellH - 6, color, C.white, 3));
        body.push(t(xx + (cellW - 6) / 2, yy + 91, `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(5)}`, 30, { anchor: "middle", fill: intensity > .58 ? C.white : C.ink, weight: 700 }));
      });
    });
    const eff = audit.domains[domain].overall.effects.interaction_difference_in_differences;
    body.push(t(px + 635, 845, `Teacher interaction (DiD): TF ${eff["TF-IDF/SVD"].toFixed(5)} · Mini ${eff.MiniLM.toFixed(5)}`, 27, { anchor: "middle", weight: 700 }));
  });
  legend(body, [["Positive: CEARF-N leads", C.blue], ["Negative: NARM leads", C.red]], 650, 955, 2);
  body.push(t(W / 2, 1040, "Matched-teacher aggregate reversals are head-driven; mean torso/tail gaps remain positive", 32, { anchor: "middle", weight: 700 }));
  body.push(t(W / 2, 1092, "Descriptive system-level interaction—not a causal factorial decomposition", 27, { anchor: "middle", fill: C.red }));
  await save(
    "15-rq4-teacher-popularity",
    "RQ4: Does the architecture conclusion survive matched information budgets?",
    body,
    "teacher_interaction_audit.json • Cell = Recall@20(CEARF-N) − Recall@20(NARM) • Three-seed means"
  );
}

async function contactSheet() {
  const names = fs.readdirSync(OUT)
    .filter((name) => /^\d\d-.*\.png$/.test(name))
    .sort();
  const tileW = 600, tileH = 300, cols = 3;
  const composites = [];
  for (let i = 0; i < names.length; i++) {
    const input = await sharp(path.join(OUT, names[i]))
      .resize(tileW, tileH, { fit: "contain", background: C.white })
      .png()
      .toBuffer();
    composites.push({
      input,
      left: (i % cols) * tileW,
      top: Math.floor(i / cols) * tileH,
    });
  }
  await sharp({
    create: {
      width: cols * tileW,
      height: Math.ceil(names.length / cols) * tileH,
      channels: 3,
      background: C.white,
    },
  }).composite(composites).png().toFile(path.join(OUT, "contact-sheet.png"));
}

async function main() {
  await fusionAblation();
  await cutoffAblation();
  await baselines();
  await externalGate();
  await metadata();
  await admissionReliability();
  await transferMargins();
  await routerSeeds();
  await inference();
  await rentalComparison();
  await evidenceRescueDamage();
  await popularityNetRescue();
  await regimeDiagnostics();
  await rescueMechanisms();
  await shortcutDiagnostics();
  await teacherPopularityAttribution();
  await contactSheet();
  const names = fs.readdirSync(OUT).filter((x) => x.endsWith(".png") && x !== "contact-sheet.png").sort();
  fs.writeFileSync(path.join(OUT, "README.txt"), [
    "CEARF-N standalone academic charts",
    "",
    "Each chart is supplied as PNG (2400×1200) and editable SVG.",
    "All values are the current locked values used in the paper and supplementary material.",
    "",
    ...names,
    "",
  ].join("\n"));
  process.stdout.write(`${OUT}\n${names.length} charts generated\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
