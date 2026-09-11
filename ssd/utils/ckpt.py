"""检查点（ckpt）保存与加载。

统一格式（train.py 保存）：
    {
        "epoch": int, "global_step": int, "best_value": float|None,
        "model": state_dict, "optimizer": state_dict|None, "scaler": state_dict|None,
        "history": dict, "config": dict,
    }
兼容读取：旧的"裸 state_dict"（如早期 train.py 保存的 ssd-XX.pth），
以及 DataParallel 保存的 "module." 前缀权重。
"""
from __future__ import annotations

import os

import torch


def strip_module_prefix(state_dict: dict) -> dict:
    """去掉 DataParallel/DDP 的 "module." 前缀。"""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    path: str,
    net: torch.nn.Module,
    optimizer=None,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    best_value=None,
    best_map=None,
    best_loss=None,
    history: dict | None = None,
    config: dict | None = None,
) -> str:
    """保存完整训练状态（原子写：先写 .tmp 再替换，避免中断损坏 ckpt）。

    best_value 为当前 best_metric 对应的值；best_map / best_loss 分别记录两条最优曲线，
    两者尺度不同（mAP 与 loss），必须分开保存，避免比较时混用。"""
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_value": best_value,
        "best_map": best_map,
        "best_loss": best_loss,
        "model": strip_module_prefix(net.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": history or {},
        "config": config or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """读取 ckpt，统一返回 dict 形式（裸 state_dict 会被包装）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        ck["model"] = strip_module_prefix(ck["model"])
        return ck
    # 裸 state_dict（旧格式）
    return {
        "epoch": 0, "global_step": 0, "best_value": None, "best_map": None, "best_loss": None,
        "model": strip_module_prefix(ck if isinstance(ck, dict) else {}),
        "optimizer": None, "scaler": None, "history": {}, "config": {},
    }


def load_model_state(net: torch.nn.Module, path: str, map_location="cpu", strict: bool = True) -> dict:
    """只把权重加载进 net（eval / infer 用），返回 ckpt 元信息。"""
    ck = load_checkpoint(path, map_location=map_location)
    net.load_state_dict(ck["model"], strict=strict)
    return ck


def find_latest_checkpoint(ckpt_root: str, prefer: str = "best") -> str | None:
    """在 ckpt_root 下找最近一次 run 目录中的 checkpoint（prefer: best|current）。

    找不到时间戳 run 目录时，回退兼容早期的 checkpoints/ssd-XX.pth。"""
    if not os.path.isdir(ckpt_root):
        return None
    runs = sorted(
        d for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d)) and len(d) == 15 and d[8] == "_"
    )
    for run in reversed(runs):
        for name in (f"{prefer}_checkpoint.pth", "current_checkpoint.pth", "best_checkpoint.pth"):
            p = os.path.join(ckpt_root, run, name)
            if os.path.isfile(p):
                return p
    legacy = sorted(f for f in os.listdir(ckpt_root) if f.endswith(".pth"))
    return os.path.join(ckpt_root, legacy[-1]) if legacy else None
