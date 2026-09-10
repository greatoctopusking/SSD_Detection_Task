"""数据增强与预处理 —— 迁移手册 preprocess_fn / _data_aug / random_sample_crop（纯 numpy+cv2）。

几何/颜色操作都在 RGB uint8 (H,W,3) 上进行；训练与验证共用归一化到模型输入的转换。
"""
from __future__ import annotations

import os

import numpy as np
import cv2

# DataLoader 多进程时，每个 worker 内部再开多线程会互相抢核；默认限制为 1 线程/进程
# （如需在单进程下加速，可设环境变量 CV2_NUM_THREADS）
cv2.setNumThreads(int(os.environ.get("CV2_NUM_THREADS", "1")))

# ImageNet 均值/方差（0~255 尺度，RGB 通道序）—— 与手册 Normalize(mean=[0.485*255,...]) 一致
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _rand(a: float = 0.0, b: float = 1.0) -> float:
    return np.random.rand() * (b - a) + a


def _intersect(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """box_a: (N,4) [y1,x1,y2,x2]；box_b: (4,)；返回 (N,) 交集面积。"""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_yx - min_yx, a_min=0.0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """IoU：box_a (N,4) tlbr vs box_b (4,) tlbr → (N,)。"""
    inter = _intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def random_sample_crop(image: np.ndarray, boxes: np.ndarray):
    """随机裁剪采样（手册算法）：
    整图 / IoU∈{0.1,0.3,0.5,0.7,0.9} 采样 / 纯随机，最多尝试 50 次。
    boxes: (N,5) [y1,x1,y2,x2,label] 像素坐标；返回裁剪后 image 与 boxes（坐标裁入裁剪框内）。"""
    height, width, _ = image.shape
    if boxes.shape[0] == 0:
        return image, boxes

    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
    if min_iou is None:
        return image, boxes

    for _ in range(50):
        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height
        if h / w < 0.5 or h / w > 2:      # 长宽比约束 0.5~2
            continue
        left = _rand() * (width - w)
        top = _rand() * (height - h)
        rect = np.array([int(top), int(left), int(top + h), int(left + w)], dtype=np.int64)

        overlap = jaccard_numpy(boxes[:, :4].astype(np.float32), rect.astype(np.float32))
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue
        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue

        image_t = image[rect[0]:rect[2], rect[1]:rect[3], :]
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        m1 = (rect[0] < centers[:, 0]) & (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) & (rect[3] > centers[:, 1])
        mask = m1 & m2 & drop_mask
        if not mask.any():
            continue

        boxes_t = boxes[mask, :].copy()
        boxes_t[:, 0:2] = np.maximum(boxes_t[:, 0:2], rect[:2]) - rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4]) - rect[:2]
        return image_t, boxes_t

    return image, boxes


def color_adjust_rgb(
    image_rgb: np.ndarray,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
) -> np.ndarray:
    """随机颜色抖动（手册 RandomColorAdjust(brightness/contrast/saturation=0.4) 的 cv2 实现）。"""
    img = image_rgb.astype(np.float32)

    b = _rand(1.0 - brightness, 1.0 + brightness)     # 亮度
    c = _rand(1.0 - contrast, 1.0 + contrast)         # 对比度
    s = _rand(1.0 - saturation, 1.0 + saturation)     # 饱和度

    # 亮度/饱和度在 HSV 上做
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * b, 0, 255)   # V = brightness
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)   # S = saturation
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # 对比度
    img = np.clip((img - 128.0) * c + 128.0, 0, 255)
    return img.astype(np.uint8)


def preprocess_train(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    image_size: int = 300,
    color_jitter: bool = True,
):
    """训练预处理：随机裁剪采样 → resize → 随机水平翻转 → (可选颜色抖动)。
    返回 (image_rgb(300,300,3) uint8, boxes_norm (M,5) float32 [y1,x1,y2,x2,label] 0~1)。
    boxes_norm 的行序/比例已与 300x300 输入空间对齐（可直接送 match_and_encode）。"""
    if boxes.shape[0] == 0:
        raise ValueError("训练样本需要至少 1 个 gt 框（make_subset 已过滤无标注图）")

    ih0, iw0, _ = image_rgb.shape
    boxes = np.asarray(boxes, dtype=np.float32)

    # 1) 随机裁剪（像素域）
    image, boxes = random_sample_crop(image_rgb, boxes)

    # 2) resize 到固定尺寸
    ih, iw, _ = image.shape
    image = cv2.resize(image, (image_size, image_size))

    # 3) 随机水平翻转（先翻图，坐标翻转留到归一化之后与手册一致）
    flip = _rand() < 0.5
    if flip:
        image = cv2.flip(image, 1)

    # 4) 可选颜色抖动（在最终尺寸上做，减少计算量）
    if color_jitter:
        image = color_adjust_rgb(image)

    # 5) 坐标归一化到 0~1（除以裁剪前原图尺寸；300x300 与锚点空间同比例）
    out = np.zeros_like(boxes)
    out[:, [0, 2]] = boxes[:, [0, 2]] / float(ih)
    out[:, [1, 3]] = boxes[:, [1, 3]] / float(iw)
    if flip:
        out[:, [1, 3]] = 1.0 - out[:, [3, 1]]
    out[:, 4] = boxes[:, 4]

    return image, out.astype(np.float32)


def preprocess_val(image_rgb: np.ndarray, image_size: int = 300):
    """验证/推理预处理：仅 resize。返回 (image(300,300,3) uint8, orig_shape (h, w))。"""
    h, w = image_rgb.shape[:2]
    image = cv2.resize(image_rgb, (image_size, image_size))
    return image, (h, w)


def to_model_input(image_rgb: np.ndarray, mean=MEAN, std=STD):
    """uint8 RGB (H,W,3) → float32 tensor (3,H,W)，按 ImageNet 均值/方差归一化（0~255 尺度）。"""
    img = image_rgb.astype(np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return np.ascontiguousarray(img)
