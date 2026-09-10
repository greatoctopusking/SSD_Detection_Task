"""设备选择与后端配置：CUDA 优先、自动回退 CPU，可选 AMP/TF32。"""
from __future__ import annotations

import torch


def resolve_device(prefer: str = "auto") -> torch.device:
    """prefer: auto | cuda | cpu。cuda 不可用时回退 CPU 并提示。"""
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        if not torch.cuda.is_available():
            print("[device] 配置要求 cuda，但当前不可用（检查是否为 CUDA 版 torch / 驱动是否正常），已回退 CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_backends(device: torch.device) -> None:
    """固定输入尺寸下的加速开关（CUDA）。"""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True      # 自动挑选最快卷积算法（输入尺寸固定时收益明显）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.benchmark = False


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"cuda:{idx} ({name}, {total:.1f} GB)"
    return "cpu"


def amp_supported(device: torch.device) -> bool:
    return device.type == "cuda"
