"""单图推理 + 画框（CUDA/CPU 通用）。支持任意图片路径（不必属于数据集）。

用法：
    cd D:\\GithubRepositories\\SSD_Detection_Task
    conda activate ssd

    python infer.py --ckpt checkpoints/20260101_120000/best_checkpoint.pth --image data/val2017/000000000139.jpg
    python infer.py --ckpt <ckpt> --image <你的图> --min-score 0.3 --device cpu

输出：outputs/infer_<文件名>.jpg（画好框的原图）+ 控制台检测列表
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import torch

from ssd.config import ensure_dirs, load_config, resolve_config_paths, resolve_path
from ssd.data import transforms as T
from ssd.data.coco import load_coco_meta
from ssd.model.anchor import generate_default_boxes
from ssd.model.SSD300 import SSD300
from ssd.utils.ckpt import find_latest_checkpoint, load_model_state
from ssd.utils.device import configure_backends, describe_device, resolve_device
from ssd.utils.postprocess import decode_boxes, filter_predictions
from ssd.utils.viz import draw_boxes


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None, help="权重路径（默认 configs 的 eval.ckpt_path）")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--out", default=None, help="输出图片路径（默认 outputs/infer_<文件名>.jpg）")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu")
    ap.add_argument("--sigmoid", action="store_true", help="用 sigmoid 概率（focal 训练时）")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    ensure_dirs(cfg)
    m, ecfg = cfg["model"], cfg["eval"]

    ckpt_path = args.ckpt or resolve_path(ecfg["ckpt_path"])
    if not os.path.isfile(ckpt_path):
        auto = find_latest_checkpoint(resolve_path(cfg["paths"]["ckpt_root"]), prefer="best")
        if auto is None:
            raise SystemExit(f"ckpt 不存在: {ckpt_path}（checkpoints/ 下也没有可用的 run）")
        print(f"[infer] 配置的 ckpt 不存在，自动使用最近一次 run: {auto}")
        ckpt_path = auto
    if not os.path.isfile(args.image):
        raise SystemExit(f"图片不存在: {args.image}")
    min_score = args.min_score if args.min_score is not None else ecfg["min_score"]

    device = resolve_device(args.device or cfg.get("device", "auto"))
    configure_backends(device)
    print(f"[infer] device = {describe_device(device)}")

    names = load_coco_meta(cfg["data"]["val"]["anno_json"])["names"]  # 索引即类别 label

    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False).to(device)
    load_model_state(net, ckpt_path, map_location=device)
    net.eval()

    default_boxes, _ = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"无法读取图片: {args.image}")
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img300, _ = T.preprocess_val(rgb, image_size=m["input_size"][0])
    x = torch.from_numpy(T.to_model_input(img300)).float().unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        pred_loc, pred_cls = net(x)
    # 概率方式与训练损失配套：multibox → softmax；focal → sigmoid（可用 --sigmoid 强制）
    focal_trained = str((cfg.get("loss") or {}).get("type", "multibox")).lower() == "focal"
    use_sigmoid = bool(args.sigmoid or focal_trained)
    probs = torch.sigmoid(pred_cls[0].float()) if use_sigmoid else torch.softmax(pred_cls[0].float(), dim=-1)
    scores = probs.cpu().numpy()
    boxes_pct = decode_boxes(pred_loc[0].float().cpu().numpy(), default_boxes)

    dets = filter_predictions(
        boxes_pct, scores, image_shape=(h, w),
        min_score=min_score,
        nms_threshold=ecfg["nms_threshold"],
        max_boxes=ecfg["max_boxes"],
    )
    print(f"[infer] {args.image} 检测到 {len(dets)} 个目标"
          f"（min_score={min_score}, {1000 * (time.time() - t0):.0f} ms）:")
    if dets:
        boxes_px = [d["box_px"] for d in dets]
        labels = [names[d["category"]] for d in dets]
        scores_l = [d["score"] for d in dets]
        for d, name in zip(dets, labels):
            y1, x1, y2, x2 = [round(v) for v in d["box_px"]]
            print(f"   {name:16s} score={d['score']:.3f}  box=[{x1},{y1},{x2},{y2}]")
        out_path = args.out or os.path.join(
            resolve_path(cfg["paths"]["output_dir"]),
            f"infer_{os.path.splitext(os.path.basename(args.image))[0]}.jpg",
        )
        draw_boxes(bgr.copy(), boxes_px, labels, scores_l, out_path=out_path)
        print(f"[infer] 结果已保存: {out_path}")
    else:
        print("[infer] 未检测到任何目标，可调低 --min-score 再试")


if __name__ == "__main__":
    main()
