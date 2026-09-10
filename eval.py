"""评估入口 —— 在验证集上计算 mAP / AP / AR（CUDA/CPU 通用）。

用法：
    cd D:\\GithubRepositories\\SSD_Detection_Task
    conda activate ssd

    # 默认用 yaml 里 eval.ckpt_path
    python eval.py --ckpt checkpoints/20260101_120000/best_checkpoint.pth
    python eval.py --ckpt checkpoints/20260101_120000/current_checkpoint.pth --min-score 0.05
    python eval.py --ckpt <ckpt> --device cpu          # 强制 CPU

输出：控制台 COCOeval 12 项指标 + mAP 摘要；预测结果写入 outputs/predictions.json
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from ssd.config import ensure_dirs, load_config, resolve_config_paths, resolve_path, validate_config
from ssd.data.coco import build_dataloader
from ssd.model.anchor import generate_default_boxes
from ssd.model.SSD300 import SSD300
from ssd.utils.ckpt import find_latest_checkpoint, load_model_state
from ssd.utils.device import configure_backends, describe_device, resolve_device
from ssd.utils.evaluator import evaluate_map


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None, help="权重路径（默认 configs 的 eval.ckpt_path）")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--nms-threshold", type=float, default=None)
    ap.add_argument("--device", default=None, help="auto|cuda|cpu（默认取 yaml 的 device）")
    ap.add_argument("--sigmoid", action="store_true", help="用 sigmoid 概率（focal 训练时）")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    ensure_dirs(cfg)
    for w in validate_config(cfg):
        print(f"[eval][警告] {w}")

    m, ecfg = cfg["model"], cfg["eval"]
    ckpt_path = args.ckpt or resolve_path(ecfg["ckpt_path"])
    if not os.path.isfile(ckpt_path):
        auto = find_latest_checkpoint(resolve_path(cfg["paths"]["ckpt_root"]), prefer="best")
        if auto is None:
            raise SystemExit(f"ckpt 不存在: {ckpt_path}（checkpoints/ 下也没有可用的 run）")
        print(f"[eval] 配置的 ckpt 不存在，自动使用最近一次 run: {auto}")
        ckpt_path = auto
    min_score = args.min_score if args.min_score is not None else ecfg["min_score"]
    nms_thr = args.nms_threshold if args.nms_threshold is not None else ecfg["nms_threshold"]

    device = resolve_device(args.device or cfg.get("device", "auto"))
    configure_backends(device)
    print(f"[eval] device = {describe_device(device)}")

    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False).to(device)
    meta = load_model_state(net, ckpt_path, map_location=device)
    print(f"[eval] 已加载 {ckpt_path}（epoch={meta.get('epoch')} best={meta.get('best_value')}）")
    net.eval()

    default_boxes, _ = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )

    val_loader, val_ds = build_dataloader(cfg, "val")
    print(f"[eval] 验证图 {len(val_ds)} 张, min_score={min_score}, nms={nms_thr}")

    # 概率方式与训练损失配套：multibox → softmax；focal → sigmoid（可用 --sigmoid 强制）
    focal_trained = str((cfg.get("loss") or {}).get("type", "multibox")).lower() == "focal"
    use_sigmoid = bool(args.sigmoid or focal_trained)
    print(f"[eval] 概率方式: {'sigmoid（focal 训练）' if use_sigmoid else 'softmax（multibox 训练）'}")

    t0 = time.time()
    pred_json = os.path.join(resolve_path(cfg["paths"]["output_dir"]), "predictions.json")
    mAP = evaluate_map(
        net, val_loader, default_boxes, cfg["data"]["val"]["anno_json"], device,
        min_score=min_score, nms_threshold=nms_thr, max_boxes=ecfg["max_boxes"],
        save_path=pred_json, use_sigmoid=use_sigmoid,
    )
    print(f"\n[eval] mAP = {mAP:.4f}  （耗时 {time.time() - t0:.0f}s，预测已存 {pred_json}）")


if __name__ == "__main__":
    main()
