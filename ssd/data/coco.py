"""COCO 数据集 —— 对应手册 create_ssd_dataset（MindDataset → PyTorch Dataset/DataLoader）。

依赖：
    - 官方 instances json（data/annotations/instances_val2017.json）
    - 图片目录（data/images/，内含 file_name 对应的 .jpg）
    - 训练时还需锚点数组（来自 ssd.model.anchor，由调用方传入，避免循环依赖）

类别映射约定（train/eval/infer 全局一致）：
    类别按 json 中 categories 的 id 升序排，label 索引 = 排序后位置 + 1（1..80，0=背景）。
    meta['names']: 长度 81 的名称表（names[0]='__background__'）。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import transforms as T
from .encode import match_and_encode

MATCHING_THRESHOLD = 0.5  # 锚点匹配 IoU 阈值（手册值）


# ---------------------------------------------------------------- meta ----
def load_coco_meta(anno_json: str) -> dict:
    """解析标注 json，返回常用索引表。"""
    with open(anno_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cats = sorted(data["categories"], key=lambda c: c["id"])
    images = {im["id"]: im for im in data["images"]}

    anns_by_image: dict[int, list] = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0):          # 检测训练忽略 crowd 标注
            continue
        anns_by_image.setdefault(a["image_id"], []).append(a)

    return {
        "cats": cats,                                        # 按 id 升序
        "images": images,                                    # id -> info(file_name,w,h)
        "anns_by_image": anns_by_image,                      # image_id -> [ann,...]
        "label_of_cat": {c["id"]: i + 1 for i, c in enumerate(cats)},  # cat_id -> label(1..80)
        "cat_id_to_name": {c["id"]: c["name"] for c in cats},
        "names": ["__background__"] + [c["name"] for c in cats],       # 索引即 label
    }


def anns_to_boxes(anns: list, label_of_cat: dict) -> np.ndarray:
    """COCO ann -> (M,5) float32 [y1,x1,y2,x2,label]（像素坐标，y 轴向下）。"""
    if not anns:
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for a in anns:
        x, y, w, h = a["bbox"]
        rows.append([y, x, y + h, x + w, float(label_of_cat[a["category_id"]])])
    return np.asarray(rows, dtype=np.float32)


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------ dataset ----
class COCODataset(Dataset):
    """返回约定（训练）：(image(CHW float32), loc(8732,4), label(8732), num_match(1))
    返回约定（验证）：(img_id(int), image(CHW float32), image_shape(2,))"""

    def __init__(
        self,
        anno_json: str,
        image_dir: str,
        image_ids,
        is_training: bool = True,
        image_size: int = 300,
        default_boxes: np.ndarray | None = None,
        default_boxes_tlbr: np.ndarray | None = None,
        color_jitter: bool = True,
    ):
        if is_training and (default_boxes is None or default_boxes_tlbr is None):
            raise ValueError("训练数据集必须传入 default_boxes / default_boxes_tlbr")
        self.meta = load_coco_meta(anno_json)
        self.image_dir = image_dir
        self.is_training = is_training
        self.image_size = image_size
        self.default_boxes = default_boxes
        self.default_boxes_tlbr = default_boxes_tlbr
        self.color_jitter = color_jitter

        missing = [i for i in image_ids if i not in self.meta["images"]]
        if missing:
            raise ValueError(f"json 中找不到图片 id（前 5 个）：{missing[:5]}")
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        file_name = self.meta["images"][img_id]["file_name"]
        image = _read_rgb(os.path.join(self.image_dir, file_name))
        anns = self.meta["anns_by_image"].get(img_id, [])
        boxes = anns_to_boxes(anns, self.meta["label_of_cat"])

        if self.is_training:
            image, boxes_norm = T.preprocess_train(
                image, boxes, image_size=self.image_size, color_jitter=self.color_jitter
            )
            loc, label, num_match = match_and_encode(
                boxes_norm, self.default_boxes, self.default_boxes_tlbr, MATCHING_THRESHOLD
            )
            x = T.to_model_input(image)
            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(loc).float(),
                torch.from_numpy(label).long(),
                torch.from_numpy(num_match),
            )
        else:
            image, (h, w) = T.preprocess_val(image, image_size=self.image_size)
            x = T.to_model_input(image)
            return (
                img_id,
                torch.from_numpy(x).float(),
                torch.tensor([float(h), float(w)], dtype=torch.float32),
            )


# ------------------------------------------------------------- factory ----
def _read_ids(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return [int(i) for i in ids]


def build_dataloader(
    cfg: dict,
    split: str,
    default_boxes: np.ndarray | None = None,
    default_boxes_tlbr: np.ndarray | None = None,
):
    """按 cfg 构建 train/val 的 DataLoader。cfg 需为 resolve_config_paths() 后的版本。

    返回 (loader, dataset)；训练 split 自动过滤无标注图片（保证每个样本有 gt）。"""
    assert split in ("train", "val"), split
    data_cfg = cfg["data"]
    ids = _read_ids(
        data_cfg["subset"]["train_ids_json"] if split == "train" else data_cfg["subset"]["val_ids_json"]
    )
    is_training = split == "train"

    meta = load_coco_meta(data_cfg["anno_json"])
    if is_training:
        ids = [i for i in ids if meta["anns_by_image"].get(i)]
    else:
        ids = [i for i in ids if i in meta["images"]]

    ds = COCODataset(
        anno_json=data_cfg["anno_json"],
        image_dir=data_cfg["image_dir"],
        image_ids=ids,
        is_training=is_training,
        image_size=cfg["model"]["input_size"][0],
        default_boxes=default_boxes,
        default_boxes_tlbr=default_boxes_tlbr,
    )
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"] if is_training else cfg["eval"]["batch_size"]
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    return loader, ds
