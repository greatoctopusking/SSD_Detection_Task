"""训练入口 —— CPU 小规模训练（第 06 步运行）。

用法：
    conda activate ssd
    python train.py                                   # 用 configs/ssd300_coco.yaml
    python train.py --epochs 8 --batch-size 4         # 临时覆盖超参

流程：读配置 → 生成 8732 锚点 → 构建 train DataLoader → SSD300(+预训练骨干)
     → SSDLoss(focal+BCE) → warmup+cosine 学习率 + Momentum → 训练并记录/保存 ckpt。

依赖：第 04 步实现 ssd/model/{anchor,backbone,ssd}.py 后即可运行。
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from ssd.config import ensure_dirs, load_config, resolve_config_paths, resolve_path
from ssd.data.coco import build_dataloader
from ssd.losses import SSDLoss
from ssd.model.anchor import generate_default_boxes      # 第 04 步实现
from ssd.model.ssd import SSD300                          # 第 04 步实现
from ssd.utils.lr_schedule import build_lr_schedule


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="yaml 路径（默认 configs/ssd300_coco.yaml）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 train.epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 train.batch_size")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    ensure_dirs(cfg)
    set_seed(args.seed)

    m = cfg["model"]
    t = cfg["train"]
    p = cfg["paths"]
    if args.epochs:
        t["epochs"] = args.epochs
    if args.batch_size:
        t["batch_size"] = args.batch_size

    # ---------- 锚点（只算一次，全流程共享） ----------
    default_boxes, default_boxes_tlbr = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )
    print(f"[train] 锚点数: {default_boxes.shape}")

    # ---------- 数据 ----------
    train_loader, train_ds = build_dataloader(cfg, "train", default_boxes, default_boxes_tlbr)
    steps_per_epoch = len(train_loader)
    print(f"[train] 训练图 {len(train_ds)} 张 / batch {t['batch_size']} / "
          f"每 epoch {steps_per_epoch} step / 共 {t['epochs']} epoch")

    # ---------- 模型 / 损失 / 优化 ----------
    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=bool(m["pretrained_backbone"]))
    criterion = SSDLoss(num_classes=m["num_classes"])
    lr_arr = build_lr_schedule(
        total_epochs=t["epochs"],
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=t["warmup_epochs"],
        lr_init=t["lr_init"],
        lr_max=t["lr_max"],
        lr_end_ratio=t["lr_end_ratio"],
    )
    opt = torch.optim.SGD(
        [p for p in net.parameters() if p.requires_grad],
        lr=float(lr_arr[0]), momentum=t["momentum"], weight_decay=t["weight_decay"],
    )

    log_path = os.path.join(resolve_path(p["log_dir"]), "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    global_step = 0
    wall_start = time.time()

    def log_line(msg: str):
        msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_line(f"== 开始训练: {t['epochs']} epochs, batch={t['batch_size']}, "
             f"lr_max={t['lr_max']}, seed={args.seed} ==")
    net.train()

    for epoch in range(1, t["epochs"] + 1):
        ep_loss, ep_steps, ep_t0 = 0.0, 0, time.time()
        for img, gt_loc, gt_label, num_match in train_loader:
            opt.zero_grad()
            pred_loc, pred_cls = net(img)
            loss = criterion(pred_loc, pred_cls, gt_loc, gt_label, num_match)
            loss.backward()
            opt.step()

            if global_step < len(lr_arr):
                opt.param_groups[0]["lr"] = float(lr_arr[global_step])
            global_step += 1
            ep_loss += float(loss.item())
            ep_steps += 1

            if ep_steps % 20 == 0 or ep_steps == steps_per_epoch:
                log_line(f"epoch {epoch}/{t['epochs']} step {ep_steps}/{steps_per_epoch} "
                         f"loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}")

        avg_loss = ep_loss / max(ep_steps, 1)
        ep_time = time.time() - ep_t0
        log_line(f"epoch {epoch}/{t['epochs']} 完成 | avg loss {avg_loss:.4f} | "
                 f"{ep_time:.0f}s ({ep_time / max(ep_steps, 1):.2f} s/step)")

        # 定期保存
        if epoch % t["save_ckpt_epochs"] == 0 or epoch == t["epochs"]:
            ckpt = os.path.join(resolve_path(p["ckpt_dir"]), f"ssd-{epoch:02d}.pth")
            torch.save(net.state_dict(), ckpt)
            log_line(f"已保存 {ckpt}")

    total = time.time() - wall_start
    log_line(f"== 训练结束，总耗时 {total / 60:.1f} 分钟 ==")


if __name__ == "__main__":
    main()
