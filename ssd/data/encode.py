"""锚点匹配与编码 —— 对应手册 ssd_bboxes_encode（纯 numpy，训练数据通路核心）。

输入约定：
    boxes           : (N,5) float32 像素坐标已按 0~1 归一化的 [y1, x1, y2, x2, label]，
                      label ∈ {1..80}（0 为背景，不参与匹配）；N 为本图 gt 数
    default_boxes   : (8732,4) float32 [cy, cx, h, w]，0~1 百分比（来自 ssd.model.anchor）
    default_boxes_tlbr: (8732,4) float32 [y1, x1, y2, x2]（同一批锚点的 tlbr 版）
    matching_threshold: 默认 0.5（IoU 阈值）

匹配原则（与手册/论文一致）：
    1) 每个 gt 与 IoU 最大的锚点强制匹配（该锚点负责该 gt）；
    2) 剩余锚点中与某 gt IoU > threshold 的也匹配该 gt；
    3) 每个锚点只匹配一个 gt；未匹配锚点 = 背景（label 0）。

输出：
    loc      : (8732,4) float32 —— 已编码偏移：
                   cy' = (cy_gt - cy_anchor) / (h_anchor * 0.1)
                   cx' = (cx_gt - cx_anchor) / (w_anchor * 0.1)
                   h'  = log(h_gt / h_anchor) / 0.2，w' 同理
    label    : (8732,) int64    —— 匹配到的类别 1..80，背景为 0
    num_match: (1,) int32       —— 正样本锚点数
"""
from __future__ import annotations

import numpy as np

VAR_XY = 0.1
VAR_WH = 0.2


def _split_tlbr(default_boxes_tlbr: np.ndarray):
    """拆出 1-D 的 y1,x1,y2,x2 与锚点面积。"""
    tlbr = np.asarray(default_boxes_tlbr, dtype=np.float32)
    y1, x1, y2, x2 = tlbr[:, 0], tlbr[:, 1], tlbr[:, 2], tlbr[:, 3]
    vol_anchors = (x2 - x1) * (y2 - y1)
    return y1, x1, y2, x2, vol_anchors


def match_and_encode(
    boxes: np.ndarray,
    default_boxes: np.ndarray,
    default_boxes_tlbr: np.ndarray,
    matching_threshold: float = 0.5,
):
    """将本图 gt boxes 匹配并编码到全部 8732 个锚点上。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f"boxes 应为 (N,5)，实际 {boxes.shape}")

    n_anchors = default_boxes_tlbr.shape[0]
    if default_boxes.shape != (n_anchors, 4):
        raise ValueError("default_boxes 与 default_boxes_tlbr 数量不一致")

    y1_a, x1_a, y2_a, x2_a, vol_a = _split_tlbr(default_boxes_tlbr)

    pre_scores = np.zeros(n_anchors, dtype=np.float32)
    t_label = np.zeros(n_anchors, dtype=np.int64)
    t_boxes = np.zeros((n_anchors, 4), dtype=np.float32)

    for bbox in boxes:
        label = int(bbox[4])
        if label <= 0:
            continue
        # 该 gt 与所有锚点的 IoU
        iy1 = np.maximum(y1_a, bbox[0])
        ix1 = np.maximum(x1_a, bbox[1])
        iy2 = np.minimum(y2_a, bbox[2])
        ix2 = np.minimum(x2_a, bbox[3])
        iw = np.maximum(ix2 - ix1, 0.0)
        ih = np.maximum(iy2 - iy1, 0.0)
        inter = iw * ih
        area_b = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        union = vol_a + area_b - inter
        scores = inter / np.maximum(union, 1e-8)

        idx = int(np.argmax(scores))
        scores[idx] = 2.0  # 强制该锚点必匹配
        mask = (scores > matching_threshold) & (scores > pre_scores)

        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)[0]

    # (cy, cx, h, w)，注意来自 tlbr：cy=(y1+y2)/2, cx=(x1+x2)/2, h=y2-y1, w=x2-x1
    loc = np.zeros((n_anchors, 4), dtype=np.float32)
    loc[:, 0] = (t_boxes[:, 0] + t_boxes[:, 2]) / 2.0   # cy
    loc[:, 1] = (t_boxes[:, 1] + t_boxes[:, 3]) / 2.0   # cx
    loc[:, 2] = t_boxes[:, 2] - t_boxes[:, 0]           # h
    loc[:, 3] = t_boxes[:, 3] - t_boxes[:, 1]           # w

    if index.size:
        loc_t = loc[index]
        db_t = default_boxes[index]
        loc_t[:, :2] = (loc_t[:, :2] - db_t[:, :2]) / (db_t[:, 2:] * VAR_XY)
        tmp = np.maximum(loc_t[:, 2:] / db_t[:, 2:], 1e-6)
        loc_t[:, 2:] = np.log(tmp) / VAR_WH
        loc[index] = loc_t

    num_match = np.array([index.size], dtype=np.int32)
    return loc.astype(np.float32), t_label.astype(np.int64), num_match
