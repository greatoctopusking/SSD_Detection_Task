"""COCO 数据集 —— 对应手册 create_ssd_dataset（MindDataset → PyTorch Dataset/DataLoader）。

数据组织（全量 COCO2017，路径在 yaml 里配置）：
    data/train2017/*.jpg   + data/annotations/instances_train2017.json   （训练）
    data/val2017/*.jpg     + data/annotations/instances_val2017.json     （验证）
    data/subsets/*_ids.json  可选：图片 id 清单；配置为 null/缺失时 = 使用该 json 中全部图片

类别映射约定（train/eval/infer 全局一致）：
    categories 按 id 升序，label 索引 = 排序后位置 + 1（1..80，0=背景）；meta['names'] 长度 81。

性能要点（GPU 训练）：
    - build_coco_index 只保留 file_name 与 GT 框（舍弃 segmentation 等大字段），
      避免 DataLoader 多进程把上百 MB 的原始 json 结构反复 pickle；
    - DataLoader 按设备自动启用 pin_memory / persistent_workers。
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
    """解析标注 json，返回常用索引表（评估 / 可视化 / 类别名使用）。"""
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
        "cats": cats,
        "images": images,
        "anns_by_image": anns_by_image,
        "label_of_cat": {c["id"]: i + 1 for i, c in enumerate(cats)},
        "cat_id_to_name": {c["id"]: c["name"] for c in cats},
        "names": ["__background__"] + [c["name"] for c in cats],
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


def build_coco_index(
    anno_json: str,
    image_ids=None,
    require_objects: bool = False,
    verbose: bool = True,
) -> dict:
    """构建轻量索引 {"ids", "file_names", "boxes"}，供 COCODataset 使用。

    image_ids=None → 使用 json 中全部图片；require_objects=True → 只保留有标注的图片（训练集）。
    """
    if not os.path.isfile(anno_json):
        raise FileNotFoundError(f"标注文件不存在: {anno_json}（全量数据请先下载并解压到 data/）")

    meta = load_coco_meta(anno_json)
    if image_ids is None:
        ids = sorted(meta["images"].keys())
    else:
        missing = [i for i in image_ids if int(i) not in meta["images"]]
        if missing:
            raise ValueError(f"json 中找不到这些图片 id（前 5 个）：{missing[:5]}")
        ids = [int(i) for i in image_ids]

    if require_objects:
        ids = [i for i in ids if meta["anns_by_image"].get(i)]

    file_names, boxes, n_box = [], [], 0
    for i in ids:
        file_names.append(meta["images"][i]["file_name"])
        b = anns_to_boxes(meta["anns_by_image"].get(i, []), meta["label_of_cat"])
        boxes.append(b)
        n_box += len(b)

    if verbose:
        print(f"[coco] 索引完成: {len(ids)} 张图 / {n_box} 个框 "
              f"(来源 {os.path.basename(anno_json)})")
    return {"ids": ids, "file_names": file_names, "boxes": boxes}


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
        index: dict,
        image_dir: str,
        is_training: bool = True,
        image_size: int = 300,
        default_boxes: np.ndarray | None = None,
        default_boxes_tlbr: np.ndarray | None = None,
        color_jitter: bool = True,
    ):
        if is_training and (default_boxes is None or default_boxes_tlbr is None):
            raise ValueError("训练数据集必须传入 default_boxes / default_boxes_tlbr")
        self.ids = index["ids"]
        self.file_names = index["file_names"]
        self.boxes = index["boxes"]
        self.image_dir = image_dir
        self.is_training = is_training
        self.image_size = image_size
        self.default_boxes = default_boxes
        self.default_boxes_tlbr = default_boxes_tlbr
        self.color_jitter = color_jitter

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        image = _read_rgb(os.path.join(self.image_dir, self.file_names[idx]))
        boxes = self.boxes[idx]

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
    index: dict | None = None,
    verbose: bool = True,
):
    """按 cfg 构建 train/val 的 DataLoader（cfg 需为 resolve_config_paths() 后的版本）。

    返回 (loader, dataset)。训练 split 自动过滤无标注图片；DataLoader 在 CUDA 下启用
    pin_memory / persistent_workers，num_workers 由 cfg['data']['num_workers'] 控制。"""
    assert split in ("train", "val"), split
    dcfg = cfg["data"][split]
    is_training = split == "train"

    if index is None:
        ids_json = dcfg.get("ids_json")
        ids, src = None, "json 全部图片"
        if ids_json and os.path.isfile(ids_json):
            ids = _read_ids(ids_json)
            src = f"ids 清单 {os.path.basename(ids_json)}"
        elif ids_json:
            src = f"json 全部图片（ids 清单缺失，已回退: {ids_json}）"
        index = build_coco_index(
            dcfg["anno_json"], ids, require_objects=is_training, verbose=verbose
        )
    else:
        src = "外部传入的 index"

    ds = COCODataset(
        index=index,
        image_dir=dcfg["image_dir"],
        is_training=is_training,
        image_size=cfg["model"]["input_size"][0],
        default_boxes=default_boxes,
        default_boxes_tlbr=default_boxes_tlbr,
        color_jitter=bool(cfg["data"].get("color_jitter", True)),
    )

    batch_size = int(cfg["train"]["batch_size"]) if is_training else int(cfg["eval"].get("batch_size", 1))
    num_workers = int(cfg["data"].get("num_workers", 0))
    pin_memory = bool(cfg["data"].get("pin_memory", True)) and torch.cuda.is_available()

    kwargs = dict(
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=int(cfg["data"].get("prefetch_factor", 4)))
    loader = DataLoader(ds, **kwargs)

    if verbose:
        print(f"[coco] {split}: {len(ds)} 张 / batch {batch_size} / {len(loader)} steps / "
              f"workers {num_workers} / pin_memory {pin_memory} / 来源 {src}")
    return loader, ds
