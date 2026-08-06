"""Record external-dataset and SSM-backend availability for ADMA artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .mamba4rec_reference import backend_status, sigma_provenance


def main():
    status = backend_status()
    payload = {
        "dataset": "RetailRocket",
        "loader": "sparse_bench.loaders.load_retailrocket",
        "dataset_path": "archive/crossdomain_data/events.csv",
        "backend": {
            "official_mamba4rec": status.__dict__ | {
                "ready": status.official_mamba4rec_ready,
            },
            "sigma_reference": sigma_provenance(),
        },
        "interpretation": (
            "Official Mamba4Rec is a required external baseline when run on CUDA; "
            "the current MPS machine can run only the documented SIGMA-compatible "
            "pure-PyTorch reference."
        ),
    }
    out = Path("sparse_bench/scaling_study/artifacts/external_backend_status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

