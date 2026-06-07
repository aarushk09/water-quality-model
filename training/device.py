"""Resolve training device: CUDA (NVIDIA) > MPS (Apple GPU) > CPU."""

from __future__ import annotations

from typing import Optional, Union

import torch


def resolve_device(preference: Optional[str] = "auto") -> torch.device:
    """
    Pick the best available accelerator.

    preference:
        auto — cuda if available, else mps, else cpu
        cuda, mps, cpu — force a backend (raises if unavailable)
    """
    pref = (preference or "auto").lower().strip()

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Install a CUDA-enabled PyTorch build "
                "and NVIDIA drivers, or set training.device to 'mps' or 'cpu'."
            )
        return torch.device("cuda")

    if pref == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError(
                "MPS requested but not available (requires Apple Silicon + recent macOS/PyTorch)."
            )
        return torch.device("mps")

    if pref == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device preference '{preference}'. Use auto, cuda, mps, or cpu."
    )


def device_label(device: torch.device) -> str:
    """Human-readable device name for logs."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name})"
    if device.type == "mps":
        return "mps (Apple GPU / Metal)"
    return "cpu"
