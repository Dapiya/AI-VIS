"""Torch device helpers for AI-VIS.

This module centralizes device selection (CUDA/MPS/CPU) and cache management so
the rest of the codebase can stay backend-agnostic.
"""

from __future__ import annotations

from typing import Union

import torch

TorchDeviceLike = Union[str, torch.device]


def is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    return bool(mps_backend.is_available())


def get_torch_device(*, device: TorchDeviceLike | None = None, gpu_id: str | None = None) -> torch.device:
    """Return the best inference device available.

    Priority order:
      1) Explicit `device` argument
      2) CUDA (if available)
      3) MPS (Apple Silicon; if available)
      4) CPU

    `gpu_id` is treated as a disable flag when set to "-1" (or None).
    """
    if device is not None:
        return torch.device(device)

    if gpu_id is None or str(gpu_id) == "-1":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if is_mps_available():
        return torch.device("mps")

    return torch.device("cpu")


def empty_torch_cache(device: TorchDeviceLike | None) -> None:
    """Best-effort cache clear for the active accelerator backend."""
    if device is None:
        return

    dev = torch.device(device)
    if dev.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if dev.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            try:
                mps.empty_cache()
            except Exception:
                pass
