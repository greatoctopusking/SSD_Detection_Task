"""配置模块：读取/校验 configs/*.yaml，路径解析与目录准备。

配置结构（GPU/CPU 通用）见 configs/ssd300_coco.yaml：
    device: auto|cuda|cpu
    data: {root, train:{image_dir,anno_json,ids_json}, val:{...}, num_workers, pin_memory}
    model: {...}
    loss: {type: multibox|focal, neg_pos_ratio, alpha, gamma}
    train: {batch_size, epochs, lr_*, amp, grad_clip, eval_every_n_epochs, best_metric, ...}
    eval: {batch_size, min_score, nms_threshold, max_boxes, ckpt_path}
    paths: {ckpt_root, output_dir, plot, log_name}
"""
from __future__ import annotations

import copy
import os

import yaml

# ssd/config.py 的上级目录 = 项目根（repo root）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "ssd300_coco.yaml")

# 需要解析为绝对路径的字段（section 可为两级，如 "data.train"）
_PATH_KEYS = (
    ("data", "root"),
    ("data.train", "image_dir"),
    ("data.train", "anno_json"),
    ("data.train", "ids_json"),
    ("data.val", "image_dir"),
    ("data.val", "anno_json"),
    ("data.val", "ids_json"),
    ("eval", "ckpt_path"),
    ("paths", "ckpt_root"),
    ("paths", "output_dir"),
)


def project_root() -> str:
    return PROJECT_ROOT


def load_config(path: str | None = None) -> dict:
    """读取 yaml 配置。path 缺省用 configs/ssd300_coco.yaml。"""
    path = resolve_path(path) if path else DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(p):
    """相对项目根的路径（如 ./data/xxx）解析为绝对路径；绝对路径原样返回。"""
    if not isinstance(p, str):
        return p
    p = os.path.expanduser(os.path.expandvars(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(PROJECT_ROOT, p))


def _get_in(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_in(cfg: dict, dotted: str, value):
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve_config_paths(cfg: dict) -> dict:
    """返回深拷贝，把所有路径字段解析为绝对路径（ids_json 允许为 null）。"""
    c = copy.deepcopy(cfg)
    for section, key in _PATH_KEYS:
        val = _get_in(c, f"{section}.{key}")
        if isinstance(val, str) and val.strip():
            _set_in(c, f"{section}.{key}", resolve_path(val))
    return c


def ensure_dirs(cfg: dict) -> None:
    """创建 ckpt_root / output_dir / subsets 目录（若已配置）。"""
    for key in ("ckpt_root", "output_dir"):
        d = _get_in(cfg, f"paths.{key}")
        if isinstance(d, str) and d:
            os.makedirs(resolve_path(d), exist_ok=True)
    ids_json = _get_in(cfg, "data.train.ids_json")
    if isinstance(ids_json, str) and ids_json:
        os.makedirs(os.path.dirname(resolve_path(ids_json)), exist_ok=True)


def validate_config(cfg: dict, check_files: bool = True) -> list:
    """基础校验：缺关键段落抛 ValueError；文件缺失等只收集为警告列表返回。"""
    warnings = []
    for key in ("model", "train", "data"):
        if key not in cfg:
            raise ValueError(f"配置缺少必需的顶级段: {key}")

    for split in ("train", "val"):
        d = _get_in(cfg, f"data.{split}") or {}
        if check_files:
            anno, img_dir, ids = d.get("anno_json"), d.get("image_dir"), d.get("ids_json")
            if not anno or not os.path.isfile(anno):
                warnings.append(f"data.{split}.anno_json 不存在: {anno}")
            if not img_dir or not os.path.isdir(img_dir):
                warnings.append(f"data.{split}.image_dir 不存在: {img_dir}")
            if ids and not os.path.isfile(ids):
                warnings.append(f"data.{split}.ids_json 不存在（将回退为使用 json 全部图片）: {ids}")

    if (_get_in(cfg, "loss.type") or "multibox").lower() not in ("multibox", "focal"):
        warnings.append(f"loss.type={_get_in(cfg, 'loss.type')} 未知，将按 multibox 处理")
    if (_get_in(cfg, "train.best_metric") or "map").lower() not in ("map", "loss"):
        warnings.append(f"train.best_metric={_get_in(cfg, 'train.best_metric')} 未知，将按 map 处理")
    return warnings
