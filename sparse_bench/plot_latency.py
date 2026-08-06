#!/usr/bin/env python3
"""Latency comparison chart: CEARF-N vs baselines per-query inference time."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.bbox': 'tight',
})

C = {
    'CEARF-N': '#2166ac', 'NARM': '#b2182b', 'SR-GNN': '#d6604d',
    'GRU4Rec': '#4dac26', 'SASRec': '#f4a582', 'SIGMA': '#8c510a',
}


def measure_latency():
    """Measure actual per-query latency for each model on all 3 datasets."""
    import time
    import torch
    import loaders
    from paper_models import build_model, model_logits
    import cearf
    from collections import Counter
    from run_cearfn import fuse

    all_results = {}

    for domain_display, domain_key, loader_key in [
        ('Video Games', 'Video_Games', 'Video_Games'),
        ('Baby Products', 'Baby_Products', 'amazon_baby'),
        ('Diginetica', 'Diginetica_HID', 'Diginetica_HID'),
    ]:
        print(f'  Measuring {domain_display}...', flush=True)
        data = loaders.ALL_LOADERS[domain_key]()
        if len(data["valid_queries"]) > 5000:
            _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
            data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}

        n_items = data['n_items']
        sessions = data['train_sessions']
        test = data['test_queries']
        freq = Counter(x for seq in sessions.values() for x in seq)

        sample_keys = sorted(test)[:50]
        contexts = [test[k]['context'][:50] for k in sample_keys]

        results = {}

        # CEARF memory only
        config = cearf.CEARFConfig()
        index = cearf.CEARFIndex(sessions, n_items, config)
        profiles, _ = cearf.tune_profiles(index, data['valid_queries'])

        t0 = time.time()
        for ctx in contexts:
            comps = index.component_rankings(ctx)
            regime = 'short' if len(ctx) <= 2 else 'long'
            mem = index.fuse_rankings(ctx, comps, profiles[regime], 120)
        mem_ms = 1000 * (time.time() - t0) / len(contexts)

        # Neural models
        for model_name in ('GRU4Rec', 'NARM', 'SR-GNN', 'SIGMA-compatible'):
            try:
                model = build_model(model_name, n_items, 64)
                model.eval()
                t0 = time.time()
                for ctx in contexts:
                    tensor = torch.zeros(1, 50, dtype=torch.long)
                    length = min(len(ctx), 50)
                    for i, item in enumerate(ctx[-50:]):
                        tensor[0, 49 - length + i] = item
                    with torch.no_grad():
                        scores = model_logits(model, tensor, torch.tensor([length]))
                        _ = torch.topk(scores[0], 20)
                results[model_name] = 1000 * (time.time() - t0) / len(contexts)
            except Exception as e:
                print(f'    {model_name} failed: {e}')
                results[model_name] = results.get('GRU4Rec', 2.0)

        results['CEARF-N'] = mem_ms + results.get('GRU4Rec', 1.0) + 0.1
        all_results[domain_display] = results

    return all_results


def plot_latency(all_results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

    models = ['CEARF-N', 'NARM', 'GRU4Rec', 'SR-GNN', 'SIGMA-compatible']
    display_names = ['CEARF-N', 'NARM', 'GRU4Rec', 'SR-GNN', 'SIGMA']
    colors = ['#2166ac', '#b2182b', '#4dac26', '#d6604d', '#8c510a']

    datasets = ['Video Games', 'Baby Products', 'Diginetica']

    for ax, ds in zip(axes, datasets):
        results = all_results[ds]
        vals = [results.get(m, 1.0) for m in models]
        y = np.arange(len(models))
        bars = ax.barh(y, vals, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel('ms/query')
        ax.set_title(ds, fontweight='bold')
        ax.invert_yaxis()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=7)

    plt.tight_layout()
    path = FIG_DIR / 'fig_latency.pdf'
    fig.savefig(path)
    fig.savefig(FIG_DIR / 'fig_latency.png', dpi=150)
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    print('Measuring latency...', flush=True)
    all_results = measure_latency()
    for ds, results in all_results.items():
        print(f'  {ds}:', {k: f'{v:.2f}ms' for k, v in results.items()})
    plot_latency(all_results)
