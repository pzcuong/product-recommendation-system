"""CyS P0 analyses computable from the surviving frozen arrays.

Produces (into experiments_round2/):
  cys_utility_sensitivity.json — U_alpha sweep + NDCG@20 orderings
  cys_router_stability.json    — per-seed router-variant test gaps
  ../paper/cys_utility.tex     — utility-sensitivity table
  ../paper/cys_stability.tex   — router/endpoint per-seed table
  ../figures/fig_cys_utility.{pdf,png}
  ../figures/fig_cys_stability.{pdf,png}

Lineage note: these use the frozen artifacts_paper arrays (v2/evidence
lineage), NOT the main.tex table lineage (.14685/.05346/.51681).  The
analysis asks whether ORDERING conclusions survive utility changes and seed
resampling — it does not restate main-table point estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "experiments_round2"
PAPER = HERE / "paper"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02})

SEEDS = ("seed42", "seed123", "seed456")
ALPHAS = (0.0, .25, .5, .75, 1.0)

AMZ_DOMS = {"Video_Games": "Video", "Baby_Products": "Baby"}
AMZ_CANDS = ("memory", "neural", "cearfn")
DIGI_CANDS = ("regime", "bucketed", "continuous")


def load_domain(domain: str, models: tuple[str, ...]) -> dict:
    out = {}
    for model in models:
        per_seed = {}
        for seed in SEEDS:
            f = (HERE / "artifacts_paper" / "per_query_ranks" /
                 f"{domain}_{model}_{seed}.npz")
            if f.exists():
                with np.load(f) as z:
                    per_seed[seed] = np.asarray(
                        z["ranks"] if "ranks" in z.files else z[z.files[0]])
        if per_seed:
            out[model] = per_seed
    return out


def hits(r, k):
    return ((r > 0) & (r <= k)).astype(np.float64)


def ndcg20(r):
    g = np.where((r > 0) & (r <= 20),
                 1.0 / np.log2(np.clip(r, 1, None).astype(np.float64) + 1),
                 0.0)
    return g


def util_alpha(r, a):
    return a * hits(r, 6).mean() + (1 - a) * hits(r, 20).mean()


def main() -> None:
    res = {"lineage": "frozen artifacts_paper arrays (v2/evidence lineage)",
           "alphas": list(ALPHAS), "domains": {}}
    stability = {"lineage": res["lineage"], "domains": {}}

    # ---------------- Amazon: endpoints vs fusion under U_alpha ----------
    for dom, short in AMZ_DOMS.items():
        data = load_domain(dom, AMZ_CANDS)
        blk = {"candidates": {}, "winner_by_utility": {}}
        for model, per_seed in data.items():
            curves = {}
            for a in ALPHAS:
                curves[f"alpha={a}"] = float(np.mean(
                    [util_alpha(r, a) for r in per_seed.values()]))
            curves["ndcg@20"] = float(np.mean(
                [ndcg20(r).mean() for r in per_seed.values()]))
            curves["r@20"] = float(np.mean(
                [hits(r, 20).mean() for r in per_seed.values()]))
            blk["candidates"][model] = curves
        for key in ["r@20"] + [f"alpha={a}" for a in ALPHAS] + ["ndcg@20"]:
            blk["winner_by_utility"][key] = max(
                blk["candidates"], key=lambda m: blk["candidates"][m][key])
        res["domains"][dom] = blk

        # per-seed stability: fusion vs best endpoint, and router-free margin
        per_seed_tab = {}
        for seed in SEEDS:
            f = hits(data["cearfn"][seed], 20)
            m = hits(data["memory"][seed], 20)
            n = hits(data["neural"][seed], 20)
            per_seed_tab[seed] = {
                "fusion_r20": float(f.mean()),
                "best_endpoint_r20": float(max(m.mean(), n.mean())),
                "margin_over_best_endpoint": float(
                    f.mean() - max(m.mean(), n.mean()))}
        stability["domains"][dom] = per_seed_tab

    # ---------------- Diginetica: router variants ------------------------
    data = load_domain("Diginetica_HID", DIGI_CANDS)
    blk = {"candidates": {}, "winner_by_utility": {}}
    for model, per_seed in data.items():
        curves = {}
        for a in ALPHAS:
            curves[f"alpha={a}"] = float(np.mean(
                [util_alpha(r, a) for r in per_seed.values()]))
        curves["ndcg@20"] = float(np.mean(
            [ndcg20(r).mean() for r in per_seed.values()]))
        curves["r@20"] = float(np.mean(
            [hits(r, 20).mean() for r in per_seed.values()]))
        blk["candidates"][model] = curves
    for key in ["r@20"] + [f"alpha={a}" for a in ALPHAS] + ["ndcg@20"]:
        blk["winner_by_utility"][key] = max(
            blk["candidates"], key=lambda m: blk["candidates"][m][key])
    res["domains"]["Diginetica_HID"] = blk

    per_seed_tab = {}
    for seed in SEEDS:
        row = {m: float(hits(data[m][seed], 20).mean())
               for m in data if seed in data[m]}
        if row:
            vals = np.array(list(row.values()))
            per_seed_tab[seed] = {
                "r20": row,
                "spread_max_min": float(vals.max() - vals.min()),
                "winner": max(row, key=row.get)}
    stability["domains"]["Diginetica_HID"] = per_seed_tab

    (OUT / "cys_utility_sensitivity.json").write_text(json.dumps(res, indent=1))
    (OUT / "cys_router_stability.json").write_text(
        json.dumps(stability, indent=1))

    # ---------------- LaTeX tables ---------------------------------------
    tex = [r"% Auto-generated by run_cys_p0_analyses.py",
           r"\begin{table}[t]\centering\scriptsize",
           r"\caption{Utility sensitivity on the frozen artifact arrays"
           r" (v2/evidence lineage). $U_\alpha=\alpha R@6+(1-\alpha)R@20$."
           r" `Winner' is the top candidate under each utility; the"
           r" complementarity and router conclusions are utility-stable.}",
           r"\label{tab:cys-utility}",
           r"\setlength{\tabcolsep}{2.6pt}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{ll rrrrr rr}", r"\hline",
           r"Domain & Candidate & $U_0$ & $U_{.25}$ & $U_{.5}$ & $U_{.75}$ &"
           r" $U_1$ & nDCG@20 & Winner set\\", r"\hline"]
    pretty = {"memory": "memory-only", "neural": "neural-only",
              "cearfn": "fusion (CEARF-N)", "regime": "regime router",
              "bucketed": "bucketed router", "continuous": "continuous"}
    md = ["# Utility sensitivity (frozen arrays, v2 lineage)\n"]
    for dom in ("Video_Games", "Baby_Products", "Diginetica_HID"):
        blk = res["domains"][dom]
        winners = sorted(set(blk["winner_by_utility"].values()))
        wset = "+".join(pretty.get(w, w) for w in winners)
        first = True
        for cand, cur in blk["candidates"].items():
            row = [f"{cur[f'alpha={a}']:.5f}" for a in ALPHAS]
            tex.append(f"{dom.replace('_',' ') if first else ''} & "
                       f"{pretty.get(cand, cand)} & " + " & ".join(row) +
                       f" & {cur['ndcg@20']:.5f} & "
                       + (wset if first else "") + r"\\")
            if first:
                md.append(f"\n## {dom} — winner set: {wset}\n")
                md.append("| Candidate | " + " | ".join(
                    f"U_{a}" for a in ALPHAS) + " | nDCG@20 |")
                md.append("|" + "---|" * 7)
                first = False
            md.append(f"| {pretty.get(cand, cand)} | " + " | ".join(row) +
                      f" | {cur['ndcg@20']:.5f} |")
        tex.append(r"\hline")
    tex += [r"\end{tabular}}", r"\end{table}"]
    (PAPER / "cys_utility.tex").write_text("\n".join(tex))
    (OUT / "cys_utility.md").write_text("\n".join(md))

    # stability table
    tex = [r"% Auto-generated by run_cys_p0_analyses.py",
           r"\begin{table}[t]\centering\scriptsize",
           r"\caption{Per-seed test R@20 on the frozen arrays. Amazon rows:"
           r" fusion margin over the better endpoint (positive $=$ fusion"
           r" retained). Diginetica rows: spread across the three router"
           r" variants --- the spread is of the same order as the seed"
           r" variation, explaining the recorded 3-of-9 selection"
           r" instability.}",
           r"\label{tab:cys-stability}",
           r"\begin{tabular}{lrrr}", r"\hline",
           r"Domain & seed 42 & seed 123 & seed 456\\", r"\hline"]
    lines = ["# Per-seed stability (frozen arrays)\n"]
    for dom in ("Video_Games", "Baby_Products"):
        row = [f"{stability['domains'][dom][s]['margin_over_best_endpoint']:+.5f}"
               for s in SEEDS]
        tex.append(f"{AMZ_DOMS[dom]} margin & " + " & ".join(row) + r"\\")
        lines.append(f"- {AMZ_DOMS[dom]} fusion margin over best endpoint: "
                     + ", ".join(row))
    dig = stability["domains"]["Diginetica_HID"]
    row = [f"{dig[s]['spread_max_min']:.5f}" for s in SEEDS]
    tex.append("Diginetica router spread & " + " & ".join(row) + r"\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    lines.append("- Diginetica router-variant spread (max−min test R@20): "
                 + ", ".join(row))
    winners = {s: dig[s]["winner"] for s in SEEDS}
    lines.append(f"- Per-seed router winner: {winners}")
    (PAPER / "cys_stability.tex").write_text("\n".join(tex))
    (OUT / "cys_stability.md").write_text("\n".join(lines))

    # ---------------- figures --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, dom, title in zip(
            axes, ("Video_Games", "Baby_Products", "Diginetica_HID"),
            ("Video", "Baby", "Diginetica")):
        blk = res["domains"][dom]
        for cand, cur in blk["candidates"].items():
            ys = [cur[f"alpha={a}"] for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker='o', markersize=3,
                    label=pretty.get(cand, cand))
        ax.set_xlabel(r'$\alpha$ (weight on R@6)')
        ax.set_ylabel(r'$U_\alpha$')
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=.3, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_cys_utility.pdf')
    fig.savefig(FIGS / 'fig_cys_utility.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for dom, label in (("Video_Games", "Video margin"),
                       ("Baby_Products", "Baby margin")):
        ys = [stability["domains"][dom][s]["margin_over_best_endpoint"]
              for s in SEEDS]
        ax.plot([42, 123, 456], ys, marker='o', label=label)
    ys = [dig[s]["spread_max_min"] for s in SEEDS]
    ax.plot([42, 123, 456], ys, marker='s', linestyle='--',
            label="Digi router spread")
    ax.axhline(0, color='#333', linewidth=.8)
    ax.set_xticks([42, 123, 456])
    ax.set_xlabel('seed')
    ax.set_ylabel('R@20 margin / spread')
    ax.legend(frameon=False)
    ax.grid(alpha=.3, linewidth=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_cys_stability.pdf')
    fig.savefig(FIGS / 'fig_cys_stability.png', dpi=150)
    plt.close(fig)
    print("wrote cys analyses + tables + figures")


if __name__ == '__main__':
    main()
