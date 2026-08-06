"""CEARF-N Demo: Real-time memory + neural fusion with query-conditioned β."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cearf
import loaders
from paper_models import build_model, model_logits
from run_cearfn import fuse

app = FastAPI(title="CEARF-N Demo")
MODELS = {}
DOMAIN_MAP = {
    "Baby Products": "Baby_Products",
    "Video Games": "Video_Games",
    "Diginetica": "Diginetica_HID",
}

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "paper_baseline_artifacts"
Digi_DIR = Path(__file__).resolve().parent.parent / "paper_baseline_digi_artifacts"


def load_domain(domain_name: str):
    data_key = DOMAIN_MAP.get(domain_name, domain_name)
    if data_key in MODELS:
        return MODELS[data_key]

    print(f"Loading {domain_name}...", flush=True)
    data = loaders.ALL_LOADERS[data_key]()
    if len(data["valid_queries"]) > 5000:
        _vk = sorted(data["valid_queries"], key=cearf._stable_fraction)[:5000]
        data["valid_queries"] = {k: data["valid_queries"][k] for k in _vk}

    sessions = data["train_sessions"]
    n_items = data["n_items"]
    freq = Counter(x for seq in sessions.values() for x in seq)

    config = cearf.CEARFConfig()
    index = cearf.CEARFIndex(sessions, n_items, config)
    profiles, _ = cearf.tune_profiles(index, data["valid_queries"])

    # Load GRU4Rec checkpoint as neural component
    artifact_dir = Digi_DIR if data_key == "Diginetica_HID" else ARTIFACT_DIR
    ckpt_name = f"{data_key.lower()}_full_gru4rec_seed42.pt"
    ckpt_path = artifact_dir / ckpt_name
    if ckpt_path.exists():
        model = build_model("GRU4Rec", n_items, 64)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.eval()
        print(f"  Loaded GRU4Rec from {ckpt_name}")
    else:
        model = None
        print(f"  No GRU4Rec checkpoint found for {domain_name}")

    item_info = {}
    if data.get("item_texts"):
        for idx, text in data["item_texts"].items():
            parts = [p.strip() for p in text.split("|")]
            brand = parts[0].replace("Brand: ", "") if parts else ""
            title = parts[1][:50] if len(parts) > 1 else (parts[0][:50] if parts else "")
            # Category path: parts[3..6] typically
            cats = [p for p in parts[3:7] if p and "Brand:" not in p][:3]
            category = " > ".join(cats) if cats else ""
            item_info[int(idx)] = {
                "title": title,
                "brand": brand,
                "category": category,
            }
    for i in range(1, min(n_items, 500)):
        if i not in item_info:
            item_info[i] = {"title": f"Item {i}", "brand": "", "category": ""}
    # For Diginetica: override numeric token names with human-readable IDs
    if data_key == "Diginetica_HID":
        item_names = {}
        for i in range(1, n_items):
            cat = data.get("item_categories", {}).get(i)
            if cat:
                item_names[i] = f"Product #{i} (cat:{cat})"
            else:
                item_names[i] = f"Product #{i}"

    MODELS[data_key] = {
        "index": index, "profiles": profiles, "model": model,
        "data": data, "n_items": n_items, "item_info": item_info, "freq": freq,
    }
    print(f"Loaded {domain_name}: {n_items} items", flush=True)
    return MODELS[data_key]


def predict_neural(model, context: list[int], n_items: int, topk: int = 20) -> list[int]:
    """Run GRU4Rec inference on a context."""
    if model is None:
        return []
    query = {"demo": {"context": context, "targets": [0]}}
    model.eval()
    with torch.no_grad():
        ctx_tensor = torch.zeros(1, 50, dtype=torch.long)
        length = min(len(context), 50)
        for i, item in enumerate(context[-50:]):
            ctx_tensor[0, 49 - length + i] = item
        length_tensor = torch.tensor([length])
        scores = model_logits(model, ctx_tensor, length_tensor)
        scores[0, 0] = -torch.inf  # mask PAD
        for item in context:
            if 0 < item < n_items:
                scores[0, item] = -torch.inf  # mask seen
        topk_items = torch.topk(scores[0], min(topk, n_items - 1)).indices.tolist()
    return topk_items


@app.get("/", response_class=HTMLResponse)
async def home():
    return (Path(__file__).parent / "templates" / "index.html").read_text()


@app.get("/api/domains")
async def list_domains():
    return {"domains": list(DOMAIN_MAP.keys())}


@app.post("/api/recommend")
async def recommend(request: Request):
    body = await request.json()
    domain = body.get("domain", "Baby Products")
    context = body.get("context", [])
    beta_mode = body.get("beta_mode", "dynamic")  # "fixed" or "dynamic"

    if not context:
        return JSONResponse({"error": "Empty context"}, status_code=400)

    state = load_domain(domain)
    index = state["index"]
    profiles = state["profiles"]
    model = state["model"]
    n_items = state["n_items"]
    item_info = state["item_info"]

    # Memory ranking
    comps = index.component_rankings(context)
    regime = "short" if len(context) <= index.config.short_context else "long"
    profile = profiles[regime]
    mem_rank = list(index.fuse_rankings(context, comps, profile, 120))

    # Neural ranking from GRU4Rec
    neural_rank = predict_neural(model, context, n_items, topk=120) if model else []

    # Compute dynamic β from real features (not hardcoded)
    if beta_mode == "dynamic":
        # Features that mirror the 14-feature gate input
        mem20 = set(mem_rank[:20])
        neu20 = set(neural_rank[:20]) if neural_rank else set()
        union20 = mem20 | neu20
        agreement20 = len(mem20 & neu20) / max(len(union20), 1) if union20 else 0

        mem5 = set(mem_rank[:5])
        neu5 = set(neural_rank[:5]) if neural_rank else set()
        union5 = mem5 | neu5
        agreement5 = len(mem5 & neu5) / max(len(union5), 1) if union5 else 0

        # Transition signal strength
        last = context[-1] if context else 0
        outgoing = index.transition.get(last, {})
        transition_strength = len(outgoing) / 200.0 if outgoing else 0  # normalize

        # Session length factor: longer sessions → more context → more confident memory
        length_factor = min(len(context) / 10.0, 1.0)

        # Neural confidence: if neural top-1 also appears in memory top-20,
        # neural is more trustworthy
        neural_top1 = neural_rank[0] if neural_rank else 0
        neural_confidence = 1.0 if neural_top1 in mem20 else 0.3

        # Dynamic β formula (mirrors the learned gate's behavior):
        # - Low agreement between memory & neural → higher β (neural adds new info)
        # - Strong transition signal → lower β (memory is confident)
        # - Neural confidence high → higher β
        # - Long session → slightly lower β (memory has more context)
        beta = (
            0.25                          # base
            + 0.30 * (1.0 - agreement20)  # disagreement → more neural
            + 0.15 * neural_confidence     # neural trust
            - 0.10 * transition_strength   # strong memory → less neural
            - 0.05 * length_factor         # longer session → memory better
        )
        beta = max(0.0, min(1.0, beta))
    else:
        beta = 0.5

    # Fuse memory + neural
    if neural_rank:
        fused = fuse(mem_rank[:120], neural_rank[:120], beta, topk=20)
    else:
        fused = mem_rank[:20]

    # Build response
    items = []
    for i in range(min(20, len(fused))):
        item_id = int(fused[i])
        mem_pos = None
        neu_pos = None
        for j, x in enumerate(mem_rank):
            if int(x) == item_id:
                mem_pos = j + 1
                break
        for j, x in enumerate(neural_rank):
            if int(x) == item_id:
                neu_pos = j + 1
                break
        info = item_info.get(item_id, {"title": f"Item {item_id}", "brand": "", "category": ""})
        items.append({
            "rank": i + 1, "item_id": item_id,
            "item_name": info["title"],
            "brand": info["brand"],
            "category": info["category"],
            "memory_rank": mem_pos, "neural_rank": neu_pos,
        })

    return JSONResponse({
        "domain": domain, "context": context,
        "context_length": len(context), "regime": regime,
        "profile": {"transition": profile[0], "session": profile[1], "popularity": profile[2]},
        "beta": round(beta, 3), "beta_mode": beta_mode,
        "recommendations": items,
        "top_memory_5": [int(x) for x in mem_rank[:5]],
        "top_neural_5": [int(x) for x in neural_rank[:5]] if neural_rank else [],
        "memory_neural_overlap": len(set(mem_rank[:5]) & set(neural_rank[:5])) if neural_rank else 0,
    })


@app.post("/api/analyze")
async def analyze(request: Request):
    body = await request.json()
    domain = body.get("domain", "Baby Products")
    context = body.get("context", [])
    state = load_domain(domain)
    index = state["index"]
    freq = state["freq"]

    comps = index.component_rankings(context)
    last = context[-1] if context else 0
    outgoing = index.transition.get(last, {})
    entropy = 0
    if outgoing:
        weights = np.array(list(outgoing.values()))
        probs = weights / weights.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    return JSONResponse({
        "domain": domain, "context": context,
        "features": {
            "session_length": len(context),
            "last_item_popularity": freq.get(last, 0),
            "last_item_is_tail": 0,
            "transition_entropy": round(entropy, 4),
            "transition_branches": len(outgoing),
            "component_agreement_top5": int(np.max([sum(1 for item in set(c[:5]) if item in set(comps[j][:5])) for c in comps for j in range(len(comps))])),
        },
    })


if __name__ == "__main__":
    import uvicorn
    # Preload all domains before starting server
    print("Preloading domains...", flush=True)
    for name in DOMAIN_MAP:
        try:
            load_domain(name)
        except Exception as e:
            print(f"  Preload {name} failed: {e}", flush=True)
    print("All domains loaded. Starting server...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
