"""可视化：GT 抽查与推理结果画框（cv2 实现，保存到 outputs/）。"""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..data.coco import load_coco_meta

# 预置一组 BGR 颜色，按索引轮换
_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 200),
]


def _color_for(label_idx: int):
    return _PALETTE[label_idx % len(_PALETTE)]


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_px,               # 每项 (y1, x1, y2, x2) 像素
    labels,                 # 每项 str（类别名/显示文本）
    scores=None,            # 可选每项 float
    out_path: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """在 BGR 图上画框，返回副本；out_path 给出则写盘（自动建父目录）。"""
    img = image_bgr.copy()
    for i, (box, label) in enumerate(zip(boxes_px, labels)):
        y1, x1, y2, x2 = [int(round(float(v))) for v in box]
        color = _color_for(i)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = str(label)
        if scores is not None:
            text += f" {scores[i]:.2f}"
        # 文字背景条
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - th - baseline if y1 - th - baseline > 0 else y1 + baseline
        cv2.rectangle(img, (x1, ty), (x1 + tw, ty + th + baseline), color, -1)
        cv2.putText(img, text, (x1, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, img)
    return img


def draw_gt_sample(
    anno_json: str,
    image_dir: str,
    image_id: int,
    out_path: str,
) -> str | None:
    """把某张图的 GT 框画出来（数据抽查用）。返回输出路径；图片缺失返回 None。"""
    meta = load_coco_meta(anno_json)
    info = meta["images"].get(image_id)
    if info is None:
        return None
    path = os.path.join(image_dir, info["file_name"])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    boxes, labels, scores = [], [], []
    for a in meta["anns_by_image"].get(image_id, []):
        x, y, w, h = a["bbox"]
        boxes.append((y, x, y + h, x + w))
        labels.append(meta["cat_id_to_name"].get(a["category_id"], "?"))
        scores.append(None)
    draw_boxes(img, boxes, labels, out_path=out_path)
    return out_path
