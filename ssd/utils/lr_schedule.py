"""学习率调度 —— 对应手册 get_lr：warmup（线性上升）+ cosine 退火。

    lr_init --warmup_steps--> lr_max --cosine--> lr_end
其中 lr_end 默认 = lr_init * lr_end_ratio（手册用 lr_end = 0.001*0.05 = 0.00005）。

说明：
    - warmup_epochs 支持小数（如 0.5）；
    - 若 warmup 步数 >= 总步数（例如 `--epochs 1` 的冒烟训练），自动压缩为总步数的 10%（至少 1 步）并提示。
"""
from __future__ import annotations

import math

import numpy as np


def build_lr_schedule(
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 2,
    lr_init: float = 0.001,
    lr_max: float = 0.05,
    lr_end: float | None = None,
    lr_end_ratio: float = 0.05,
) -> np.ndarray:
    """返回长度 = total_steps 的 float32 学习率序列（逐步对应一个训练 step）。"""
    total_steps = int(total_epochs) * int(steps_per_epoch)
    if total_steps <= 0:
        raise ValueError(
            f"total_steps 必须 > 0（total_epochs={total_epochs}, steps_per_epoch={steps_per_epoch}）"
        )

    warmup_steps = int(round(float(warmup_epochs) * steps_per_epoch))
    if warmup_steps >= total_steps:
        clamped = max(1, int(total_steps * 0.1))
        print(
            f"[lr_schedule] warmup 步数({warmup_steps}) >= 总步数({total_steps})，"
            f"自动压缩为 {clamped} 步（总步数的 10%）"
        )
        warmup_steps = clamped
    if lr_end is None:
        lr_end = lr_init * lr_end_ratio

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            progress = (i - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_end + (lr_max - lr_end) * (1.0 + math.cos(math.pi * progress)) / 2.0
        lr_each_step.append(max(lr, 0.0))
    return np.asarray(lr_each_step, dtype=np.float32)
