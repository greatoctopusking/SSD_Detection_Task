"""学习率调度 —— 对应手册 get_lr：warmup（线性上升）+ cosine 退火。

    lr_init --warmup_steps--> lr_max --cosine--> lr_end
其中 lr_end 默认 = lr_init * lr_end_ratio（手册用 lr_end = 0.001*0.05）。
"""
from __future__ import annotations

import math
import numpy as np


def build_lr_schedule(
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 2,
    lr_init: float = 0.001,
    lr_max: float = 0.05,
    lr_end: float | None = None,
    lr_end_ratio: float = 0.05,
) -> np.ndarray:
    """返回长度 = total_steps 的 float32 学习率序列。"""
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    if warmup_steps >= total_steps:
        raise ValueError("warmup_epochs 不能大于 total_epochs")
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
