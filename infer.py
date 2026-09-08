"""单图推理 + 画框（第 08 步运行）。支持任意图片路径（不必属于 json 子集）。

用法：
    python infer.py --ckpt checkpoints/ssd-05.pth --image data/images/000000000139.jpg
    python infer.py --ckpt checkpoints/ssd-05.pth --image <你的测试图> --min-score 0.3
输出：outputs/infer_<文件名>.jpg（画好框的原图）+ 控制台检测列表
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from ssd.config import load_config, resolve_config_paths, resolve_path
from ssd.data import transforms as T
from ssd.data.coco import load_coco_meta
from ssd.model.anchor import generate_default_boxes      # 第 04 步实现
from ssd.model.ssd import SSD300                          # 第 04 步实现
from ssd.utils.postprocess import decode_boxes, filter_predictions
from ssd.utils.viz import draw_boxes


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None, help="权重路径（默认 configs 的 eval.ckpt_path）")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--min-score", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    m = cfg["model"]
    ecfg = cfg["eval"]
    ckpt_path = args.ckpt or resolve_path(ecfg["ckpt_path"])
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"ckpt 不存在: {ckpt_path}")
    if not os.path.isfile(args.image):
        raise SystemExit(f"图片不存在: {args.image}")
    min_score = args.min_score if args.min_score is not None else ecfg["min_score"]

    meta = load_coco_meta(cfg["data"]["anno_json"])
    names = meta["names"]  # 索引即类别 label（0=背景）

    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=False)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
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
    x = torch.from_numpy(T.to_model_input(img300)).float().unsqueeze(0)

    with torch.no_grad():
        pred_loc, pred_cls = net(x)
    scores = torch.sigmoid(pred_cls[0]).numpy()
    boxes_pct = decode_boxes(pred_loc[0].numpy(), default_boxes)

    dets = filter_predictions(
        boxes_pct, scores, image_shape=(h, w),
        min_score=min_score,
        nms_threshold=ecfg["nms_threshold"],
        max_boxes=ecfg["max_boxes"],
    )
    print(f"[infer] {args.image} 检测到 {len(dets)} 个目标（min_score={min_score}）:")
    out_img = bgr.copy()
    if dets:
        boxes_px = [d["box_px"] for d in dets]
        labels = [f"{names[d['category']]}" for d in dets]
        scores_l = [d["score"] for d in dets]
        for d, name in zip(dets, labels):
            y1, x1, y2, x2 = [round(v) for v in d["box_px"]]
            print(f"   {name:16s} score={d['score']:.3f}  box=[{x1},{y1},{x2},{y2}]")
        out_name = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(resolve_path(cfg["paths"]["output_dir"]), f"infer_{out_name}.jpg")
        draw_boxes(out_img, boxes_px, labels, scores_l, out_path=out_path)
        print(f"[infer] 结果已保存: {out_path}")
    else:
        print("[infer] 未检测到任何目标，可调低 --min-score 再试")


if __name__ == "__main__":
    main()
