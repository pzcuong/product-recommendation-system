#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "output", "slides", "cearfn_graphs");
fs.mkdirSync(OUT, { recursive: true });

const W = 1920;
const H = 1080;
const C = {
  ink: "#14213D",
  muted: "#526174",
  grid: "#D9E0E8",
  pale: "#F3F6FA",
  cearf: "#2864DC",
  fusion: "#00A39A",
  memory: "#F2A900",
  neural: "#7B61C8",
  narm: "#E56B3E",
  stan: "#64748B",
  vsknn: "#94A3B8",
  danger: "#D64545",
  success: "#008A72",
  white: "#FFFFFF",
};

const esc = (s) =>
  String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const text = (x, y, value, size = 28, opts = {}) => {
  const {
    fill = C.ink,
    weight = 400,
    anchor = "start",
    family = "Arial, Helvetica, sans-serif",
    opacity = 1,
    italic = false,
  } = opts;
  return `<text x="${x}" y="${y}" fill="${fill}" opacity="${opacity}" font-family="${family}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}"${italic ? ' font-style="italic"' : ""}>${esc(value)}</text>`;
};

const line = (x1, y1, x2, y2, stroke = C.grid, width = 2, dash = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;

const rect = (x, y, w, h, fill, rx = 0, stroke = "none", sw = 0, opacity = 1) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" opacity="${opacity}" stroke="${stroke}" stroke-width="${sw}"/>`;

const circle = (cx, cy, r, fill, stroke = C.white, sw = 3) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;

const pathEl = (d, stroke, width = 4, fill = "none", dash = "") =>
  `<path d="${d}" stroke="${stroke}" stroke-width="${width}" fill="${fill}" stroke-linecap="round" stroke-linejoin="round"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;

const titleBlock = (title, subtitle) => [
  text(100, 88, title, 48, { weight: 700 }),
  text(100, 132, subtitle, 25, { fill: C.muted }),
  line(100, 164, 1820, 164, C.grid, 2),
];

const footer = (source) => [
  line(100, 1020, 1820, 1020, C.grid, 1),
  text(100, 1055, source, 19, { fill: C.muted }),
  text(1820, 1055, "CEARF-N • ADMA 2026", 19, { fill: C.muted, anchor: "end" }),
];

const svg = (body) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${rect(0, 0, W, H, C.white)}${body.join("")}</svg>`;

async function save(name, body, alt, source) {
  const content = svg([...titleBlock(alt.title, alt.subtitle), ...body, ...footer(source)]);
  const svgPath = path.join(OUT, `${name}.svg`);
  const pngPath = path.join(OUT, `${name}.png`);
  fs.writeFileSync(svgPath, content);
  await sharp(Buffer.from(content)).png().toFile(pngPath);
  return {
    id: name,
    title: alt.title,
    subtitle: alt.subtitle,
    svg: path.basename(svgPath),
    png: path.basename(pngPath),
    alt_text: alt.description,
  };
}

function scale(v, lo, hi, px0, px1) {
  return px0 + ((v - lo) / (hi - lo)) * (px1 - px0);
}

function fmt(v, digits = 5) {
  return Number(v).toFixed(digits).replace(/^0/, "");
}

async function architecture() {
  const y = 420;
  const body = [];
  const boxes = [
    { x: 100, w: 250, title: "Session prefix", sub: "q = (i₁,…,iL)", fill: C.pale },
    { x: 440, w: 310, title: "CEARF memory", sub: "transition • session • popularity", fill: "#FFF5D9" },
    { x: 840, w: 310, title: "PASGR residual", sub: "semantic neural rank", fill: "#EEE9FB" },
    { x: 1240, w: 290, title: "Rank fusion", sub: "RRF + validation-gated router", fill: "#DFF6F2" },
    { x: 1620, w: 200, title: "Top-K", sub: "test once", fill: "#E8F0FF" },
  ];
  boxes.forEach((b) => {
    body.push(rect(b.x, y, b.w, 180, b.fill, 22, C.grid, 2));
    body.push(text(b.x + b.w / 2, y + 72, b.title, 31, { weight: 700, anchor: "middle" }));
    body.push(text(b.x + b.w / 2, y + 120, b.sub, 22, { fill: C.muted, anchor: "middle" }));
  });
  for (let i = 0; i < boxes.length - 1; i++) {
    const x1 = boxes[i].x + boxes[i].w + 12;
    const x2 = boxes[i + 1].x - 18;
    body.push(line(x1, y + 90, x2, y + 90, C.ink, 4));
    body.push(`<polygon points="${x2},${y + 90} ${x2 - 18},${y + 78} ${x2 - 18},${y + 102}" fill="${C.ink}"/>`);
  }
  body.push(rect(410, 710, 1160, 135, C.pale, 20, C.grid, 2));
  body.push(text(990, 760, "Locked validation contract", 32, { weight: 700, anchor: "middle" }));
  body.push(text(990, 806, "Every optional stage competes against an absent or pure-endpoint state before test access", 25, { anchor: "middle", fill: C.muted }));
  body.push(line(595, 710, 595, 620, C.memory, 5));
  body.push(line(995, 710, 995, 620, C.neural, 5));
  body.push(line(1385, 710, 1385, 620, C.fusion, 5));
  return save(
    "01-method-and-admission-flow",
    body,
    {
      title: "CEARF-N: evidence enters only after validation admission",
      subtitle: "Memory and neural ranks remain independently rejectable; test is accessed once after lock.",
      description: "Pipeline from session prefix through CEARF memory and PASGR neural residual to rank fusion and Top-K output, with a locked validation contract under the optional stages.",
    },
    "Source: main paper, Fig. 1 and Sec. 3"
  );
}

const mainPerf = {
  Video: { "V-SKNN": .11936, STAN: .12115, Transition: .04223, GRU4Rec: .10368, NARM: .13705, "SR-GNN": .12267, SIGMA: .08644, "CEARF-N": .14685 },
  Baby: { "V-SKNN": .05038, STAN: .04944, Transition: .00735, GRU4Rec: .04501, NARM: .02990, "SR-GNN": .05208, SIGMA: .02473, "CEARF-N": .05346 },
  Diginetica: { "V-SKNN": .50506, STAN: .51502, Transition: .40102, GRU4Rec: .41785, NARM: .53406, "SR-GNN": .49267, SIGMA: .37219, "CEARF-N": .51681 },
};

async function mainPerformance() {
  const body = [];
  const panels = [
    { key: "Video", x: 100, color: C.cearf },
    { key: "Baby", x: 695, color: C.cearf },
    { key: "Diginetica", x: 1290, color: C.narm },
  ];
  panels.forEach((p) => {
    const entries = Object.entries(mainPerf[p.key]).sort((a, b) => b[1] - a[1]);
    const max = entries[0][1] * 1.08;
    body.push(text(p.x + 250, 230, p.key, 32, { weight: 700, anchor: "middle" }));
    entries.forEach(([name, value], i) => {
      const y = 285 + i * 78;
      const barX = p.x + 145;
      const width = (value / max) * 355;
      const highlighted = name === "CEARF-N" || (p.key === "Diginetica" && name === "NARM");
      const fill = name === "CEARF-N" ? C.cearf : name === "NARM" && p.key === "Diginetica" ? C.narm : "#C9D2DD";
      body.push(text(barX - 15, y + 24, name, 22, { anchor: "end", weight: highlighted ? 700 : 400 }));
      body.push(rect(barX, y, 355, 34, C.pale, 8));
      body.push(rect(barX, y, width, 34, fill, 8));
      body.push(text(barX + Math.max(width + 10, 62), y + 25, fmt(value), 21, { weight: highlighted ? 700 : 400 }));
    });
  });
  body.push(rect(100, 925, 1720, 52, "#EDF4FF", 12));
  body.push(text(960, 960, "CEARF-N leads Video and Baby; ID-only NARM remains strongest on Diginetica.", 27, { anchor: "middle", weight: 700 }));
  return save(
    "02-recall20-against-baselines",
    body,
    {
      title: "Recall@20 against eight reported systems",
      subtitle: "Full-catalogue evaluation; deterministic baselines have no seed variance.",
      description: "Three ranked horizontal bar panels comparing Recall at 20 on Video, Baby, and Diginetica. CEARF-N leads Video and Baby while NARM leads Diginetica.",
    },
    "Source: main paper, Table 1 • Amazon CEARF-N uses TF–IDF/SVD; comparators are ID-only"
  );
}

const endpoints = {
  Video: { Memory: [.06932, .08821, .11901], Neural: [.06158, .08471, .12571], Fusion: [.07803, .10368, .14685] },
  Baby: { Memory: [.02288, .02900, .03879], Neural: [.02337, .03214, .04873], Fusion: [.02767, .03668, .05346] },
  Diginetica: { Memory: [.29174, .37334, .49565], Neural: [.26398, .33539, .44852], Fusion: [.30738, .39210, .51681] },
};

async function endpointFusion() {
  const body = [];
  const labels = ["R@6", "R@10", "R@20"];
  const colors = { Memory: C.memory, Neural: C.neural, Fusion: C.fusion };
  Object.entries(endpoints).forEach(([domain, series], panelIndex) => {
    const x0 = 120 + panelIndex * 600;
    const y0 = 850;
    const top = 270;
    const vals = Object.values(series).flat();
    const max = Math.max(...vals) * 1.12;
    body.push(text(x0 + 230, 225, domain, 31, { weight: 700, anchor: "middle" }));
    for (let g = 0; g <= 4; g++) {
      const v = (max * g) / 4;
      const y = scale(v, 0, max, y0, top);
      body.push(line(x0 + 65, y, x0 + 500, y, C.grid, 1));
      body.push(text(x0 + 55, y + 7, v.toFixed(2), 18, { fill: C.muted, anchor: "end" }));
    }
    labels.forEach((lab, i) => body.push(text(x0 + 145 + i * 145, 892, lab, 21, { anchor: "middle" })));
    Object.entries(series).forEach(([name, values]) => {
      const points = values.map((v, i) => [x0 + 145 + i * 145, scale(v, 0, max, y0, top)]);
      body.push(pathEl(`M ${points.map(([x, y]) => `${x} ${y}`).join(" L ")}`, colors[name], name === "Fusion" ? 6 : 4));
      points.forEach(([x, y], i) => {
        body.push(circle(x, y, name === "Fusion" ? 10 : 8, colors[name]));
        if (name === "Fusion") body.push(text(x, y - 18, fmt(values[i]), 18, { anchor: "middle", weight: 700 }));
      });
    });
  });
  const legend = [["Memory only", C.memory], ["Neural only", C.neural], ["Selected fusion", C.fusion]];
  legend.forEach(([lab, color], i) => {
    const x = 540 + i * 320;
    body.push(line(x, 958, x + 50, 958, color, 6));
    body.push(circle(x + 25, 958, 8, color));
    body.push(text(x + 66, 966, lab, 22));
  });
  return save(
    "03-fusion-vs-direct-endpoints",
    body,
    {
      title: "Selected fusion exceeds both direct endpoints",
      subtitle: "The internally controlled claim holds at every cutoff on all three domains.",
      description: "Small multiple line charts comparing memory only, neural only, and selected fusion at Recall 6, 10, and 20 across three domains. Fusion is highest at each cutoff.",
    },
    "Source: main paper, Table 2"
  );
}

const gateDelta = [
  { name: "Video", d: .00658, lo: .00558, hi: .00757, gate: .15342, single: .14685 },
  { name: "Baby", d: .00518, lo: .00467, hi: .00569, gate: .05882, single: .05364 },
  { name: "Diginetica", d: .00384, lo: .00212, hi: .00556, gate: .53790, single: .53406 },
];

async function externalGate() {
  const body = [];
  const x0 = 430, x1 = 1640, lo = 0, hi = .008;
  for (let i = 0; i <= 8; i++) {
    const v = i / 1000;
    const x = scale(v, lo, hi, x0, x1);
    body.push(line(x, 265, x, 820, C.grid, 1));
    body.push(text(x, 860, v.toFixed(3), 20, { fill: C.muted, anchor: "middle" }));
  }
  gateDelta.forEach((d, i) => {
    const y = 360 + i * 190;
    body.push(text(350, y + 10, d.name, 29, { weight: 700, anchor: "end" }));
    body.push(line(scale(d.lo, lo, hi, x0, x1), y, scale(d.hi, lo, hi, x0, x1), y, C.ink, 5));
    body.push(line(scale(d.lo, lo, hi, x0, x1), y - 18, scale(d.lo, lo, hi, x0, x1), y + 18, C.ink, 4));
    body.push(line(scale(d.hi, lo, hi, x0, x1), y - 18, scale(d.hi, lo, hi, x0, x1), y + 18, C.ink, 4));
    body.push(circle(scale(d.d, lo, hi, x0, x1), y, 15, C.fusion));
    body.push(text(scale(d.d, lo, hi, x0, x1), y - 34, `+${fmt(d.d)}`, 24, { anchor: "middle", weight: 700 }));
    body.push(text(350, y + 48, `${fmt(d.single)} → ${fmt(d.gate)}`, 20, { fill: C.muted, anchor: "end" }));
  });
  body.push(text(1035, 915, "Δ Recall@20 with query-clustered 95% CI", 25, { anchor: "middle", weight: 700 }));
  body.push(rect(500, 935, 1070, 55, "#E9F7F4", 12));
  body.push(text(1035, 972, "All three intervals exclude zero; no external-gate admission reverses on test.", 26, { anchor: "middle", weight: 700 }));
  return save(
    "04-external-expert-gate",
    body,
    {
      title: "External expert admission adds test-time value",
      subtitle: "All 15 non-empty subsets of CEARF-N, STAN, V-SKNN, and NARM were validation-eligible.",
      description: "Forest plot of Recall at 20 gains and 95 percent confidence intervals for the external expert gate on Video, Baby, and Diginetica. All gains are positive.",
    },
    "Source: hard_gate_singleton_audit.json • 20,000 query-clustered bootstrap repetitions"
  );
}

const teacher = {
  Video: { "CEARF-N": [.13208, .14685, .14794], NARM: [.13705, .15387, .15392] },
  Baby: { "CEARF-N": [.05055, .05346, .05761], NARM: [.02990, .06628, .06493] },
};

async function metadataFairness() {
  const body = [];
  const labs = ["ID-only", "TF–IDF/SVD", "MiniLM"];
  Object.entries(teacher).forEach(([domain, series], panel) => {
    const x0 = 160 + panel * 880;
    const y0 = 845, top = 285;
    const vals = Object.values(series).flat();
    const lo = Math.min(...vals) * .85;
    const hi = Math.max(...vals) * 1.08;
    body.push(text(x0 + 340, 230, domain, 34, { weight: 700, anchor: "middle" }));
    for (let g = 0; g <= 4; g++) {
      const v = lo + ((hi - lo) * g) / 4;
      const y = scale(v, lo, hi, y0, top);
      body.push(line(x0 + 60, y, x0 + 680, y, C.grid, 1));
      body.push(text(x0 + 48, y + 7, v.toFixed(3), 19, { anchor: "end", fill: C.muted }));
    }
    labs.forEach((lab, i) => body.push(text(x0 + 155 + i * 230, 892, lab, 21, { anchor: "middle" })));
    [["CEARF-N", C.cearf], ["NARM", C.narm]].forEach(([name, color]) => {
      const pts = series[name].map((v, i) => [x0 + 155 + i * 230, scale(v, lo, hi, y0, top)]);
      body.push(pathEl(`M ${pts.map(([x, y]) => `${x} ${y}`).join(" L ")}`, color, 6));
      pts.forEach(([x, y], i) => {
        body.push(circle(x, y, 11, color));
        body.push(text(x, y - 20, fmt(series[name][i]), 20, { anchor: "middle", weight: 700 }));
      });
    });
  });
  body.push(line(670, 957, 720, 957, C.cearf, 6));
  body.push(text(738, 965, "CEARF-N", 23));
  body.push(line(940, 957, 990, 957, C.narm, 6));
  body.push(text(1008, 965, "NARM", 23));
  return save(
    "05-metadata-matched-fairness",
    body,
    {
      title: "Matched semantic information reverses the Amazon ordering",
      subtitle: "Architecture claims change once CEARF-N and NARM receive the same teacher.",
      description: "Two line-chart panels show CEARF-N and NARM Recall at 20 under ID-only, TF-IDF SVD, and MiniLM initialization. NARM leads both metadata-matched conditions.",
    },
    "Source: main paper, Table 4 • Three-seed means"
  );
}

async function transferSummary(transfer) {
  const body = [];
  const stages = [
    { name: "Adaptive router", data: transfer.stages.adaptive_router.summary, color: C.danger },
    { name: "External expert gate", data: transfer.stages.external_expert_gate.summary, color: C.success },
  ];
  stages.forEach((s, i) => {
    const y = 350 + i * 330;
    body.push(text(150, y - 55, s.name, 32, { weight: 700 }));
    body.push(text(150, y - 15, `${s.data.retained_nonnegative}/${s.data.admitted_decisions} admitted decisions retain non-negative test utility`, 22, { fill: C.muted }));
    const xA = 800, xR = 1510;
    const maxAbs = .006;
    const baseY = y + 80;
    const yA = baseY - (s.data.mean_validation_admission_margin / maxAbs) * 120;
    const yR = baseY - (s.data.mean_test_retention_margin / maxAbs) * 120;
    body.push(line(650, baseY, 1660, baseY, C.grid, 2));
    body.push(pathEl(`M ${xA} ${yA} L ${xR} ${yR}`, s.color, 7));
    body.push(circle(xA, yA, 15, C.cearf));
    body.push(circle(xR, yR, 15, s.color));
    body.push(text(xA, yA - 28, fmt(s.data.mean_validation_admission_margin), 25, { anchor: "middle", weight: 700 }));
    body.push(text(xR, yR - 28, fmt(s.data.mean_test_retention_margin), 25, { anchor: "middle", weight: 700 }));
    body.push(text(xA, baseY + 46, "Validation A", 23, { anchor: "middle" }));
    body.push(text(xR, baseY + 46, "Test R", 23, { anchor: "middle" }));
    const keepW = 360 * s.data.retention_rate;
    body.push(rect(230, y + 35, 360, 44, "#F0D5D5", 10));
    body.push(rect(230, y + 35, keepW, 44, s.color, 10));
    body.push(text(410, y + 67, `${Math.round(s.data.retention_rate * 100)}% retained`, 23, { anchor: "middle", weight: 700, fill: C.white }));
  });
  body.push(text(1155, 930, "Mean admission margin", 22, { anchor: "middle", fill: C.muted }));
  return save(
    "06-validation-to-test-transfer",
    body,
    {
      title: "Validation-to-test transfer is stage-dependent",
      subtitle: "Gating controls test access; it does not guarantee that a small validation gain will generalize.",
      description: "Two slope diagrams compare mean validation admission margin to mean test retention margin. Adaptive routing falls below zero with 40 percent retention, while external expert gating remains positive with 100 percent retention.",
    },
    "Source: validation_test_transfer_audit.json • U = 0.5·R@6 + 0.5·R@20"
  );
}

async function routerScatter(transfer) {
  const body = [];
  const records = transfer.stages.adaptive_router.records.filter((r) => r.admitted);
  const x0 = 320, x1 = 1700, y0 = 870, y1 = 255;
  const xmin = 0, xmax = .0045, ymin = -.007, ymax = .0015;
  const zeroY = scale(0, ymin, ymax, y0, y1);
  body.push(rect(x0, zeroY, x1 - x0, y0 - zeroY, "#FBEAEA", 0, "none", 0, .8));
  body.push(rect(x0, y1, x1 - x0, zeroY - y1, "#E8F7F2", 0, "none", 0, .8));
  for (let i = 0; i <= 4; i++) {
    const v = i * .001;
    const x = scale(v, xmin, xmax, x0, x1);
    body.push(line(x, y1, x, y0, C.grid, 1));
    body.push(text(x, 910, v.toFixed(3), 20, { anchor: "middle", fill: C.muted }));
  }
  for (let i = -7; i <= 1; i += 2) {
    const v = i / 1000;
    const y = scale(v, ymin, ymax, y0, y1);
    body.push(line(x0, y, x1, y, v === 0 ? C.ink : C.grid, v === 0 ? 3 : 1));
    body.push(text(x0 - 20, y + 7, v.toFixed(3), 20, { anchor: "end", fill: C.muted }));
  }
  records.forEach((r) => {
    const x = scale(r.validation_admission_margin, xmin, xmax, x0, x1);
    const y = scale(r.test_retention_margin, ymin, ymax, y0, y1);
    const domain = r.domain === "Video_Games" ? "V" : r.domain === "Baby_Products" ? "B" : "D";
    const color = r.reversal ? C.danger : C.success;
    body.push(circle(x, y, 18, color));
    body.push(text(x + 24, y - 14, `${domain}${r.seed}`, 23, { weight: 700 }));
  });
  body.push(text(1010, 955, "Validation admission margin A", 25, { anchor: "middle", weight: 700 }));
  body.push(`<text x="105" y="560" fill="${C.ink}" font-family="Arial, Helvetica, sans-serif" font-size="25" font-weight="700" text-anchor="middle" transform="rotate(-90 105 560)">Test retention margin R</text>`);
  body.push(text(1660, zeroY - 28, "retained", 23, { fill: C.success, anchor: "end", weight: 700 }));
  body.push(text(1660, zeroY + 42, "reversed", 23, { fill: C.danger, anchor: "end", weight: 700 }));
  body.push(text(320, 225, "Labels: V = Video, B = Baby, D = Diginetica; number = seed", 21, { fill: C.muted }));
  return save(
    "07-router-transfer-by-seed",
    body,
    {
      title: "Small router-selection gains are fragile",
      subtitle: "Three of five positively admitted adaptive routers reverse against the regime null on test.",
      description: "Scatter plot of validation admission margin versus test retention margin for five admitted router decisions. Video seed 42 and Baby seeds 42 and 456 are below zero and reverse.",
    },
    "Source: cearfn_v2_nested_results.json; recomputed by audit_validation_test_transfer.py"
  );
}

async function efficiency(runtime) {
  const body = [];
  const domains = [
    ["Video", runtime.domains.Video_Games],
    ["Baby", runtime.domains.Baby_Products],
    ["Diginetica", runtime.domains.Diginetica_HID],
  ];
  const comps = [
    ["memory", C.memory],
    ["neural", C.neural],
    ["router_feature", C.cearf],
    ["router_beta", "#62B6CB"],
    ["fusion", C.fusion],
  ];
  domains.forEach(([name, d], i) => {
    const y = 320 + i * 210;
    body.push(text(250, y + 38, name, 29, { weight: 700, anchor: "end" }));
    let x = 320;
    comps.forEach(([key, color]) => {
      const w = 950 * d.component_share[key];
      body.push(rect(x, y, w, 62, color, 0));
      if (w > 80) body.push(text(x + w / 2, y + 40, `${Math.round(d.component_share[key] * 100)}%`, 22, { anchor: "middle", weight: 700, fill: key === "memory" ? C.ink : C.white }));
      x += w;
    });
    body.push(text(1350, y + 24, `${d.queries_per_second.toFixed(1)} q/s`, 27, { weight: 700 }));
    body.push(text(1350, y + 58, `${d.amortized_milliseconds_per_query.toFixed(3)} ms/query`, 22, { fill: C.muted }));
  });
  comps.forEach(([key, color], i) => {
    const label = { memory: "Memory", neural: "Neural", router_feature: "Router features", router_beta: "β assignment", fusion: "RRF fusion" }[key];
    const x = 250 + i * 300;
    body.push(rect(x, 925, 28, 28, color, 5));
    body.push(text(x + 40, 948, label, 20));
  });
  body.push(text(790, 850, "Share of warm end-to-end inference time", 25, { anchor: "middle", weight: 700 }));
  return save(
    "08-inference-throughput-and-cost",
    body,
    {
      title: "Warm full-catalogue inference: 241–394 queries/s",
      subtitle: "Memory retrieval dominates runtime; routing and fusion together remain a small share.",
      description: "Stacked bars show component shares of inference time for Video, Baby, and Diginetica, with queries per second and amortized milliseconds per query.",
    },
    "Source: cearfn_inference_benchmark.json • Apple M2 Pro, 32 GB; CPU memory + Metal PASGR"
  );
}

async function main() {
  const transfer = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "validation_test_transfer_audit.json")));
  const runtime = JSON.parse(fs.readFileSync(path.join(ROOT, "sparse_bench", "cearfn_inference_benchmark.json")));
  const manifest = [];
  manifest.push(await architecture());
  manifest.push(await mainPerformance());
  manifest.push(await endpointFusion());
  manifest.push(await externalGate());
  manifest.push(await metadataFairness());
  manifest.push(await transferSummary(transfer));
  manifest.push(await routerScatter(transfer));
  manifest.push(await efficiency(runtime));
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify({ generated_at: new Date().toISOString(), resolution: `${W}x${H}`, charts: manifest }, null, 2) + "\n");
  const readme = `# CEARF-N slide graphs

All charts are available as editable SVG and 1920×1080 PNG.

Recommended presentation order:

1. Method and admission flow
2. Recall@20 against baselines
3. Fusion versus direct endpoints
4. External expert gate
5. Metadata-matched fairness
6. Validation-to-test transfer
7. Router transfer by seed
8. Inference throughput and component cost

Scientific reading:

- The defensible internal claim is that selected fusion exceeds both direct endpoints.
- The external gate transfers on all nine matched-seed decisions.
- Metadata matching reverses the Amazon model ordering in favour of NARM.
- Validation admission is not a generalization guarantee: adaptive routing reverses in three of five admitted decisions.

Regenerate with:

\`NODE_PATH=/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules /Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node sparse_bench/slide_graphs/generate_slide_graphs.js\`
`;
  fs.writeFileSync(path.join(OUT, "README.md"), readme);
  process.stdout.write(`${OUT}\n${manifest.length} charts generated\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
