"""CPU 测速：量测 1 个真实训练 step（forward+loss+backward+optimizer）的耗时，
据此反推合理的子集大小 x epoch 数，避免盲跑几小时才发现太慢。

用法（模型已实现，可直接运行）：
    python scripts/benchmark_cpu.py --batch 4 --steps 3
输出：平均 s/step、s/图像，以及 train=500/1000/2000 张 x 3/5/10 epoch 的预估墙钟时间。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ssd.config import load_config, resolve_config_paths  # noqa: E402
from ssd.losses import SSDLoss  # noqa: E402
from ssd.model.anchor import generate_default_boxes  # noqa: E402
from ssd.model.SSD300 import SSD300  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    m = cfg["model"]
    t = cfg["train"]
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    default_boxes, default_boxes_tlbr = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )
    print(f"[benchmark] anchors: {default_boxes.shape}")

    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False)
    net.train()
    criterion = SSDLoss(num_classes=m["num_classes"])
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=1e-4, momentum=t["momentum"], weight_decay=t["weight_decay"])

    B = args.batch
    img = torch.randn(B, 3, m["input_size"][0], m["input_size"][1])
    loc = torch.randn(B, 8732, 4) * 0.05
    label = torch.zeros(B, 8732, dtype=torch.long)
    pos_idx = torch.randint(0, 8732, (B, 20))
    label.scatter_(1, pos_idx, torch.randint(1, m["num_classes"], (B, 20)))
    num = (label > 0).sum(dim=1, keepdim=True).float().clamp(min=1)

    def step():
        opt.zero_grad()
        pl, pc = net(img)
        loss = criterion(pl, pc, loc, label, num)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    per_img = avg / B
    print(f"[benchmark] 平均 {avg:.2f} s/step (batch={B})  ~ {per_img:.2f} s/图像")
    print("[benchmark] 预估墙钟（仅训练，不含数据/评估开销）:")
    for n_img, n_epochs in ((500, 5), (500, 10), (1000, 5), (2000, 5)):
        total_s = n_img / B * avg * n_epochs
        print(f"   train {n_img:5d} 张 x {n_epochs:2d} epoch  ~ {total_s / 3600:6.1f} 小时"
              f"  ({total_s / 60:6.0f} 分钟)")


if __name__ == "__main__":
    main()
