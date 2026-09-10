"""训练/推理测速（CUDA / CPU 通用）：量测 1 个真实训练 step 的耗时，据此定 batch/epoch 规模。

用法：
    cd D:\\GithubRepositories\\SSD_Detection_Task
    conda activate ssd

    python scripts/benchmark.py --batch 16 --steps 5          # 用 yaml 的 device
    python scripts/benchmark.py --batch 4  --steps 3 --device cpu --no-amp
输出：平均 s/step、s/图像、显存峰值，以及若干 epoch 数的预估墙钟。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ssd.config import load_config, resolve_config_paths  # noqa: E402
from ssd.losses import build_loss  # noqa: E402
from ssd.model.anchor import generate_default_boxes  # noqa: E402
from ssd.model.SSD300 import SSD300  # noqa: E402
from ssd.utils.device import configure_backends, describe_device, resolve_device  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None, help="auto|cuda|cpu")
    ap.add_argument("--batch", type=int, default=None, help="默认取 yaml 的 train.batch_size")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    m, t = cfg["model"], cfg["train"]
    B = int(args.batch or t["batch_size"])

    device = resolve_device(args.device or cfg.get("device", "auto"))
    configure_backends(device)
    amp_enabled = bool(t.get("amp", True)) and device.type == "cuda" and not args.no_amp
    print(f"[benchmark] device = {describe_device(device)} | batch = {B} | AMP = {amp_enabled}")
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    default_boxes, _ = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"], boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"], img_size=m["input_size"][0],
        scales_min=m["scales_min"], scales_max=m["scales_max"],
    )
    print(f"[benchmark] anchors: {default_boxes.shape}")

    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False).to(device)
    net.train()
    criterion = build_loss(m["num_classes"], cfg.get("loss"))
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=1e-4, momentum=float(t["momentum"]), weight_decay=float(t["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    img = torch.randn(B, 3, m["input_size"][0], m["input_size"][1], device=device)
    loc = torch.randn(B, 8732, 4, device=device) * 0.05
    label = torch.zeros(B, 8732, dtype=torch.long, device=device)
    pos_idx = torch.randint(0, 8732, (B, 20), device=device)
    label.scatter_(1, pos_idx, torch.randint(1, m["num_classes"], (B, 20), device=device))
    num = (label > 0).sum(dim=1, keepdim=True).float().clamp(min=1)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pl, pc = net(img)
            loss = criterion(pl, pc, loc, label, num)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(t.get("grad_clip", 10.0)))
        scaler.step(opt)
        scaler.update()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    per_img = avg / B
    print(f"[benchmark] 平均 {avg:.3f} s/step (batch={B})  ~ {per_img:.4f} s/图像")
    if device.type == "cuda":
        print(f"[benchmark] 显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("[benchmark] 预估墙钟（仅训练 step，不含数据加载/评估；实际约 +20~40% 数据开销）:")
    for n_img, n_epochs in ((5000, 5), (5000, 30), (118000, 1), (118000, 30), (118000, 60)):
        total_s = n_img / B * avg * n_epochs
        print(f"   {n_img:6d} 张 x {n_epochs:2d} epoch  ~ {total_s / 3600:7.2f} 小时"
              f"  ({total_s / 60:7.1f} 分钟)")


if __name__ == "__main__":
    main()
