"""评估入口 —— 加载 ckpt 在 val 子集上算 mAP/AP/AR（第 07 步运行）。

用法：
    python eval.py --ckpt checkpoints/ssd-05.pth
    python eval.py --ckpt checkpoints/ssd-05.pth --min-score 0.05
输出：COCOeval 12 项指标（控制台）+ mAP 摘要 + outputs/predictions.json
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ssd.config import load_config, resolve_config_paths, resolve_path
from ssd.data.coco import build_dataloader
from ssd.model.anchor import generate_default_boxes      # 第 04 步实现
from ssd.model.ssd import SSD300                          # 第 04 步实现
from ssd.utils.coco_eval import COCOMetrics
from ssd.utils.postprocess import decode_boxes


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None, help="权重路径（默认 configs 的 eval.ckpt_path）")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--nms-threshold", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    m = cfg["model"]
    ecfg = cfg["eval"]
    ckpt_path = args.ckpt or resolve_path(ecfg["ckpt_path"])
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"ckpt 不存在: {ckpt_path}")

    min_score = args.min_score if args.min_score is not None else ecfg["min_score"]
    nms_thr = args.nms_threshold if args.nms_threshold is not None else ecfg["nms_threshold"]

    # ---------- 模型 + 权重 ----------
    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False)
    state = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    print(f"[eval] 已加载 {ckpt_path}")

    default_boxes, default_boxes_tlbr = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )

    # ---------- 数据 ----------
    val_loader, val_ds = build_dataloader(cfg, "val")
    print(f"[eval] 验证图 {len(val_ds)} 张")

    metrics = COCOMetrics(
        anno_json=cfg["data"]["anno_json"],
        min_score=min_score,
        nms_threshold=nms_thr,
        max_boxes=ecfg["max_boxes"],
    )

    t0 = time.time()
    with torch.no_grad():
        for img_ids, img, shapes in val_loader:
            pred_loc, pred_cls = net(img)
            scores = torch.sigmoid(pred_cls)
            for i in range(img.shape[0]):
                loc_np = pred_loc[i].cpu().numpy()
                sc_np = scores[i].cpu().numpy()
                boxes_pct = decode_boxes(loc_np, default_boxes)
                metrics.update(
                    int(img_ids[i]), boxes_pct, sc_np,
                    shapes[i].cpu().numpy(),
                )

    pred_json = os.path.join(resolve_path(cfg["paths"]["output_dir"]), "predictions.json")
    mAP = metrics.get_metrics(save_path=pred_json)
    print(f"\n[eval] mAP = {mAP:.4f}  （耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
