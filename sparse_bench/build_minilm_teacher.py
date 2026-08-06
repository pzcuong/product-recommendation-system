#!/usr/bin/env python3
"""Build checksummed MiniLM item-teacher matrices for Amazon domains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import loaders


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def text_fingerprint(texts: dict[int, str], n_items: int,
                     model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode())
    digest.update(str(n_items).encode())
    for item in sorted(texts):
        digest.update(str(int(item)).encode())
        digest.update(b"\0")
        digest.update(str(texts[item]).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*",
                        default=["Video_Games", "Baby_Products"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "semantic_teacher_artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    dimension = int(model.get_sentence_embedding_dimension())
    for domain in args.domains:
        data = loaders.ALL_LOADERS[domain]()
        n_items = int(data["n_items"])
        texts = {
            int(item): str(text)
            for item, text in data.get("item_texts", {}).items()
            if 0 < int(item) < n_items and str(text).strip()
        }
        fingerprint = text_fingerprint(texts, n_items, args.model)
        matrix_path = args.output_dir / f"{domain.lower()}_minilm.npy"
        manifest_path = args.output_dir / f"{domain.lower()}_minilm.json"
        if matrix_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("fingerprint") == fingerprint
                    and manifest.get("model") == args.model):
                print(f"[MINILM] {domain} cache hit: {matrix_path}", flush=True)
                continue

        ids = sorted(texts)
        matrix = np.zeros((n_items, dimension), dtype=np.float32)
        if ids:
            encoded = model.encode(
                [texts[item] for item in ids],
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            matrix[ids] = np.asarray(encoded, dtype=np.float32)
        np.save(matrix_path, matrix)
        matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        manifest = {
            "domain": domain,
            "model": args.model,
            "dimension": dimension,
            "n_items": n_items,
            "items_with_text": len(ids),
            "coverage": len(ids) / max(n_items - 1, 1),
            "fingerprint": fingerprint,
            "matrix_sha256": matrix_sha256,
            "matrix": str(matrix_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
