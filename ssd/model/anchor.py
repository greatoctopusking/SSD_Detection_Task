"""锚点（PriorBox）生成 —— 对应手册 GeneratDefaultBoxes。★ 待与用户一起实现 ★

【接口契约 —— 后续所有模块都依赖它，请严格实现】：
    def generate_default_boxes(
        feature_map_sizes=(38,19,10,5,3,1),   # 6 个特征层尺寸
        boxes_per_point=(4,6,6,6,4,4),        # 每点锚点数
        fk_divisors=(8,16,32,64,100,300),     # img_size/该值 = 每层步长
        img_size=300,
        scales_min=0.1, scales_max=0.95,
    ) -> (default_boxes, default_boxes_tlbr)
        default_boxes     : np.ndarray (8732,4) float32，行序 [cy, cx, h, w]，取值 0~1（百分比）
        default_boxes_tlbr: np.ndarray (8732,4) float32，行序 [y1, x1, y2, x2]（供 IoU 用）
    （规则：scale 0.1~0.95 线性 + 末层追加 1.0；每层 [(s1,s1)] + 各长宽比 (w,h)(h,w) + (sqrt(s1*s2),sqrt(s1*s2))，
     总锚点数 = sum(f*h*w) = 38²·4 + 19²·6 + 10²·6 + 5²·6 + 3²·4 + 1²·4 = 8732；顺序与手册一致。）

供 data/encode.py、utils/postprocess.py、train.py、eval.py 使用；建议模块级缓存（只算一次）。
"""

import math
import numpy as np

import itertools as it

def convert2tlbr(dboxes):
    db_tlbr = []
    for cy, cx, h, w in dboxes:
        ly = cy - h / 2
        lx = cx - w / 2
        ry = cy + h / 2
        rx = cx + w / 2
        db_tlbr.append((ly, lx, ry, rx))
    return db_tlbr

def generate_default_boxes(
        feature_map_sizes=(38, 19, 10, 5, 3, 1),
        boxes_per_point=(4,6,6,6,4,4),
        fk_divisors=(8,16,32,64,100,300),
        img_size=300,
        scales_min=0.1,
        scales_max=0.95
):
    fk = img_size / np.array(fk_divisors, dtype=np.float64)

    n = len(feature_map_sizes)
    scale_ratio = (scales_max - scales_min) / (n - 1)
    scales = [scales_min + scale_ratio * i for i in range(n)]
    scales.append(1.0)

    anchor_wh_ratio = [(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)]

    default_boxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        s1 = scales[idx]
        s2 = scales[idx + 1]
        s3 = math.sqrt((s1 * s2))

        all_sizes = [(s1, s1)]
        for ratio in anchor_wh_ratio[idx]:
            w = s1 * math.sqrt(ratio)
            h = s1 / math.sqrt(ratio)
            all_sizes.extend([(w, h), (h, w)])
        all_sizes.append((s3, s3))
        assert len(all_sizes) == boxes_per_point[idx]

        for i, j in it.product(range(feature_size), repeat=2):
            cy = (i + 0.5) / fk[idx]
            cx = (j + 0.5) / fk[idx]
            for w, h in all_sizes:
                default_boxes.append((cy, cx, h, w))

    default_boxes_tlbr = convert2tlbr(default_boxes)
    default_boxes = np.asarray(default_boxes, dtype=np.float32)
    default_boxes_tlbr = np.asarray(default_boxes_tlbr, dtype=np.float32)
    return default_boxes, default_boxes_tlbr
