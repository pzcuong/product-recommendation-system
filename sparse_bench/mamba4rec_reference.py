"""Reference-compatible Mamba4Rec/SIGMA backends for the ADMA study.

Mamba4Rec's official implementation depends on ``mamba-ssm`` and
``causal-conv1d`` (CUDA).  This module never silently substitutes the local
Python recurrence: callers must inspect ``backend_status`` and record whether
the official backend or the documented SIGMA fallback was used.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendStatus:
    mamba_ssm: bool
    causal_conv1d: bool
    cuda: bool

    @property
    def official_mamba4rec_ready(self) -> bool:
        return self.mamba_ssm and self.causal_conv1d and self.cuda


def backend_status() -> BackendStatus:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return BackendStatus(
        mamba_ssm=importlib.util.find_spec("mamba_ssm") is not None,
        causal_conv1d=importlib.util.find_spec("causal_conv1d") is not None,
        cuda=cuda,
    )


class OfficialMamba4RecUnavailable(RuntimeError):
    pass


def build_official_mamba(d_model: int, d_state: int = 16,
                         d_conv: int = 4, expand: int = 2):
    """Construct the official Mamba block, failing loudly when unavailable."""
    status = backend_status()
    if not status.official_mamba4rec_ready:
        raise OfficialMamba4RecUnavailable(
            "Official Mamba4Rec backend requires CUDA + mamba-ssm + "
            f"causal-conv1d; status={status}"
        )
    from mamba_ssm import Mamba
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def sigma_provenance() -> dict:
    """Metadata for the local SIGMA-compatible implementation."""
    return {
        "name": "SIGMA-compatible-pytorch",
        "paper": "Liu et al., SIGMA, AAAI 2025",
        "official_cuda_kernel": False,
        "implementation": "sparse_bench.sigma_model",
        "differences": [
            "pure-PyTorch selective scan instead of mamba_ssm",
            "same full-softmax/session-head protocol as the local benchmark",
        ],
    }

