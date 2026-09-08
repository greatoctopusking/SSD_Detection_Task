"""COCO 评估 —— 对应手册 COCOMetrics（pycocotools COCOeval 计算 mAP / AP / AR）。

用法：
    metrics = COCOMetrics(anno_json, ...)
    for 每张验证图:
        metrics.update(img_id, pred_boxes_percent(8732,4), sigmoid_scores(8732,81), image_shape)
    mAP = metrics.get_metrics(save_path='outputs/predictions.json')
"""
from __future__ import annotations

import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.coco import load_coco_meta
from .nms import nms


class COCOMetrics:
    def __init__(
        self,
        anno_json: str,
        min_score: float = 0.1,
        nms_threshold: float = 0.6,
        max_boxes: int = 100,
    ):
        meta = load_coco_meta(anno_json)
        self.num_classes = len(meta["cats"]) + 1                 # 80 + background
        self.label_to_cat_id = {i + 1: c["id"] for i, c in enumerate(meta["cats"])}
        self.min_score = min_score
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes

        self.coco_gt = COCO(anno_json)
        self.predictions: list[dict] = []
        self.img_ids: list[int] = []
        self.stats = None

    def update(
        self,
        img_id: int,
        pred_boxes: np.ndarray,   # (8732,4) 百分比 tlbr
        box_scores: np.ndarray,   # (8732, num_classes) sigmoid 后得分
        image_shape,              # (h, w)
    ):
        """单图预测送入评估器（手册 COCOMetrics.update 的 numpy 版）。"""
        h, w = float(image_shape[0]), float(image_shape[1])
        self.img_ids.append(int(img_id))

        for c in range(1, self.num_classes):
            class_scores = box_scores[:, c]
            keep = class_scores > self.min_score
            if not keep.any():
                continue
            class_boxes = pred_boxes[keep] * np.array([h, w, h, w], dtype=np.float32)
            sc = class_scores[keep]
            nms_idx = nms(class_boxes, sc, thres=self.nms_threshold, max_boxes=self.max_boxes)
            for i in nms_idx:
                loc = class_boxes[i]
                self.predictions.append(
                    {
                        "image_id": int(img_id),
                        "bbox": [float(loc[1]), float(loc[0]), float(loc[3] - loc[1]), float(loc[2] - loc[0])],
                        "score": float(sc[i]),
                        "category_id": int(self.label_to_cat_id[c]),
                    }
                )

    def get_metrics(self, save_path: str = "outputs/predictions.json") -> float:
        """跑 COCOeval，返回 mAP = stats[0]（IoU=0.50:0.95 的 AP）。"""
        if not self.predictions:
            print("[COCOMetrics] 无任何预测，mAP = 0.0")
            return 0.0

        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.predictions, f)

        coco_dt = self.coco_gt.loadRes(save_path)
        ev = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = self.img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        self.stats = ev.stats
        return float(ev.stats[0])
