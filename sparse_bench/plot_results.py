#!/usr/bin/env python3
"""Generate legacy exploratory CEARF-N figures.

This script contains pre-lock values (including Diginetica .49028/.48265) and
is not a source for the ADMA submission.  Canonical tables are defined in
CANONICAL_EXPERIMENT_LINEAGE.md and use the locked nested .51681/.53406
lineage.  The explicit warning prevents exploratory plots from being mistaken
for submission evidence.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Color palette (colorblind-friendly)
COLORS = {
    'CEARF-N': '#2166ac',
    'NARM': '#b2182b',
    'SR-GNN': '#d6604d',
    'GRU4Rec': '#4dac26',
    'SASRec': '#f4a582',
    'SIGMA': '#8c510a',
    'memory': '#999999',
    'neural': '#bababa',
    'regime': '#2166ac',
    'bucketed': '#4393c3',
    'continuous': '#92c5de',
}


def load_data():
    """Load all results from JSON files."""
    data = {}

    # v1 evidence (memory/neural/fused for ablation)
    ev = json.load(open(HERE / "cearfn_evidence_results.json"))
    for ds in ('Video_Games', 'Baby_Products'):
        runs = ev[ds]['runs']
        data[ds] = {
            'memory': [r['CEARF']['recall@20'] for r in runs],
            'neural': [r['PASGR']['recall@20'] for r in runs],
            'fused_v1': [r['CEARF-N']['recall@20'] for r in runs],
        }

    # v2 results (from nometa + earlier data)
    nm = json.load(open(HERE / "cearfn_v2_nometa_results.json"))
    for ds in ('Video_Games', 'Baby_Products'):
        runs = nm[ds]['runs']
        data[ds]['no_meta'] = [r['regime']['recall@20'] for r in runs]

    # Diginetica
    digi_nm = json.load(open(HERE / "cearfn_v2_nometa_results.json"))
    if 'Diginetica_HID' in digi_nm:
        runs = digi_nm['Diginetica_HID']['runs']
        data['Diginetica_HID'] = {
            'memory': [0.48674] * 3,
            'neural': [0.20273] * 3,
            'fused_v1': [0.49028] * 3,
            'no_meta': [r['regime']['recall@20'] for r in runs],
        }

    return data


def fig1_main_results():
    """Figure 1: Main results — Recall@20 grouped bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

    # Data (3-seed means)
    datasets = ['Video\nGames', 'Baby\nProducts', 'Diginetica']
    methods = ['GRU4Rec', 'SASRec', 'NARM', 'SR-GNN', 'SIGMA', 'CEARF-N v2']
    values = {
        'Video\nGames':    [0.10368, 0.03794, 0.13705, 0.12267, 0.08644, 0.14701],
        'Baby\nProducts':  [0.04501, 0.03163, 0.02990, 0.05208, 0.02473, 0.05597],
        'Diginetica':      [0.41461, 0.26911, 0.48265, 0.43220, 0.37336, 0.49028],
    }

    for ax, ds in zip(axes, datasets):
        vals = values[ds]
        colors = [COLORS.get(m, '#333') for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Recall@20' if ds == 'Video\nGames' else '')
        ax.set_title(ds, fontweight='bold')

        # Highlight winner
        best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('#000')
        bars[best_idx].set_linewidth(1.5)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()
    path = FIG_DIR / 'fig1_main_results.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig1_main_results.png')
    plt.close()
    print(f'Saved {path}')


def fig2_fusion_ablation():
    """Figure 2: Fusion ablation — memory-only vs neural-only vs fused."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    datasets = ['Video Games', 'Baby Products', 'Diginetica']
    memory =  [0.11901, 0.03879, 0.48674]
    neural =  [0.12605, 0.04492, 0.20273]
    fused =   [0.14577, 0.05370, 0.49028]

    x = np.arange(len(datasets))
    w = 0.25

    bars1 = ax.bar(x - w, memory, w, label='Memory-only (β=0)', color=COLORS['memory'], edgecolor='white')
    bars2 = ax.bar(x, neural, w, label='Neural-only (β=1)', color=COLORS['neural'], edgecolor='white')
    bars3 = ax.bar(x + w, fused, w, label='Fused (CEARF-N)', color=COLORS['CEARF-N'], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Recall@20')
    ax.set_title('Fusion ablation: memory-only vs neural-only vs fused', fontweight='bold')
    ax.legend(frameon=False, loc='upper left')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    # Add arrows showing fusion gain
    for i, (m, n, f) in enumerate(zip(memory, neural, fused)):
        ax.annotate('', xy=(x[i]+w, f), xytext=(x[i]+w, max(m, n)),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    plt.tight_layout()
    path = FIG_DIR / 'fig2_fusion_ablation.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig2_fusion_ablation.png')
    plt.close()
    print(f'Saved {path}')


def fig3_paired_ci():
    """Figure 3: Paired bootstrap CI forest plot."""
    fig, ax = plt.subplots(figsize=(7, 4))

    comparisons = [
        ('VG vs NARM', 0.00997, 0.00797, 0.01196),
        ('VG vs SR-GNN', 0.02434, 0.02230, 0.02640),
        ('Baby vs NARM', 0.02607, 0.02479, 0.02736),
        ('Baby vs SR-GNN', 0.00389, 0.00264, 0.00514),
        ('Digi vs NARM', 0.00762, 0.00426, 0.01097),
    ]

    y = np.arange(len(comparisons))
    labels = [c[0] for c in comparisons]
    means = [c[1] for c in comparisons]
    ci_lo = [c[2] for c in comparisons]
    ci_hi = [c[3] for c in comparisons]

    ax.errorbar(means, y, xerr=[[m-lo for m, lo in zip(means, ci_lo)],
                                 [hi-m for hi, m in zip(ci_hi, means)]],
                fmt='o', color=COLORS['CEARF-N'], markersize=6, capsize=4, linewidth=1.5)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Δ Recall@20 (CEARF-N v2 − baseline)')
    ax.set_title('Paired bootstrap CI (95%)', fontweight='bold')

    # Add p-value annotations
    pvalues = ['7.3e-65', '<1e-100', '<1e-100', '5.1e-26', '1.3e-14']
    for i, p in enumerate(pvalues):
        ax.text(ci_hi[i] + 0.0005, i, f'p={p}', va='center', fontsize=7, color='gray')

    plt.tight_layout()
    path = FIG_DIR / 'fig3_paired_ci.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig3_paired_ci.png')
    plt.close()
    print(f'Saved {path}')


def fig4_no_metadata():
    """Figure 4: No-metadata comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # VG
    ax = axes[0]
    vals = [0.14701, 0.13208, 0.13705]
    labels = ['CEARF-N\n(full)', 'CEARF-N\n(no-meta)', 'NARM\n(best)']
    colors = [COLORS['CEARF-N'], '#92c5de', COLORS['NARM']]
    bars = ax.bar(range(3), vals, color=colors, edgecolor='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Recall@20')
    ax.set_title('Video Games', fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 0.17)

    # Baby
    ax = axes[1]
    vals = [0.05597, 0.05055, 0.05208]
    labels = ['CEARF-N\n(full)', 'CEARF-N\n(no-meta)', 'SR-GNN\n(best)']
    colors = [COLORS['CEARF-N'], '#92c5de', COLORS['SR-GNN']]
    bars = ax.bar(range(3), vals, color=colors, edgecolor='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Recall@20')
    ax.set_title('Baby Products', fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 0.065)

    plt.tight_layout()
    path = FIG_DIR / 'fig4_no_metadata.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig4_no_metadata.png')
    plt.close()
    print(f'Saved {path}')


def fig5_gru4rec_stability():
    """Figure 5: GRU4Rec seed-dependent instability on Diginetica."""
    fig, ax = plt.subplots(figsize=(6, 3))

    seeds = [42, 123, 456, 137, 234, 345, 567]
    r20 = [0.41722, 0.41201, 0.01359, 0.09494, 0.00192, 0.00192, 0.00192]
    converged = [r > 0.3 for r in r20]
    colors = [COLORS['GRU4Rec'] if c else '#d9d9d9' for c in converged]

    bars = ax.bar(range(len(seeds)), r20, color=colors, edgecolor='white')
    ax.axhline(y=0.48265, color=COLORS['NARM'], linestyle='--', linewidth=1, label='NARM (.483)')
    ax.axhline(y=0.49028, color=COLORS['CEARF-N'], linestyle='--', linewidth=1, label='CEARF-N (.490)')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed={s}' for s in seeds], rotation=45, ha='right')
    ax.set_ylabel('Recall@20')
    ax.set_title('GRU4Rec on Diginetica: seed-dependent instability', fontweight='bold')
    ax.legend(frameon=False, loc='upper right')

    # Add convergence label
    ax.text(0.5, 0.95, '2/7 converge', transform=ax.transAxes, fontsize=9,
            va='top', ha='center', color=COLORS['GRU4Rec'], fontweight='bold')

    plt.tight_layout()
    path = FIG_DIR / 'fig5_gru4rec_stability.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig5_gru4rec_stability.png')
    plt.close()
    print(f'Saved {path}')


def fig6_gating_config():
    """Figure 6: Validation-gated PASGR configuration."""
    fig, ax = plt.subplots(figsize=(5, 3))

    components = ['graph_weight', 'prototype_transport', 'contrastive_weight', 'inbatch_weight']
    vg_vals = [0.35, 0, 0.15, 0.10]
    baby_vals = [0.0, 0, 0.15, 0.0]
    vg_labels = ['0.35', 'OFF', '0.15', '0.10']
    baby_labels = ['0.0', 'OFF', '0.15', '0.0']

    x = np.arange(len(components))
    w = 0.35

    bars1 = ax.bar(x - w/2, [1 if v > 0 else 0 for v in vg_vals], w,
                   label='Video Games', color=COLORS['CEARF-N'], edgecolor='white')
    bars2 = ax.bar(x + w/2, [1 if v > 0 else 0 for v in baby_vals], w,
                   label='Baby Products', color=COLORS['bucketed'], edgecolor='white')

    # Add labels
    for bar, label in zip(bars1, vg_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontsize=8)
    for bar, label in zip(bars2, baby_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Graph\ntransport', 'Prototype\ntransport',
                         'Contrastive\nloss', 'In-batch\nloss'])
    ax.set_ylabel('Selected (1) / Rejected (0)')
    ax.set_title('Validation-gated PASGR components', fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.4)

    plt.tight_layout()
    path = FIG_DIR / 'fig6_gating_config.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig6_gating_config.png')
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    fig1_main_results()
    fig2_fusion_ablation()
    fig3_paired_ci()
    fig4_no_metadata()
    fig5_gru4rec_stability()
    fig6_gating_config()
    print(f'\nAll figures saved to {FIG_DIR}')
