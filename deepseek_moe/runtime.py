"""Device and precision helpers, so the same code runs on CPU, Apple GPUs (MPS) and CUDA."""

from __future__ import annotations

import contextlib
import platform

import torch


def pick_device(name: str = "auto") -> str:
    """'auto' picks CUDA, then Apple-Silicon MPS, then CPU. Anything else is passed through.

    MPS is skipped on Intel Macs: their AMD GPUs benchmarked slower than the CPU for this model.
    """
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available() and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def resolve_precision(precision: str, device: str) -> str:
    """'auto' -> bf16 on CUDA GPUs that support it, fp32 everywhere else (CPU stays exact)."""
    if precision != "auto":
        return precision
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def autocast(device: str, precision: str):
    """Mixed-precision context for forward passes; a no-op for fp32."""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def _autocast_enabled(device_type: str) -> bool:
    try:  # torch >= 2.4
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # torch 2.2 (Intel Mac)
        if device_type == "cuda":
            return torch.is_autocast_enabled()
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return False


def full_precision(x: torch.Tensor):
    """Temporarily disable autocast (e.g. for the MoE router, kept in fp32 as in DeepSeek-V3)."""
    if _autocast_enabled(x.device.type):
        return torch.autocast(device_type=x.device.type, enabled=False)
    return contextlib.nullcontext()


def setup_backends(device: str) -> None:
    """Fast-but-safe matmul settings for CUDA (TF32 for any fp32 matmuls on Ampere+)."""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
