"""配置模块：读取/校验 configs/*.yaml，并提供路径解析与目录准备。"""
from __future__ import annotations

import os
import copy
import shutil

import yaml

# ssd/config.py 的上级目录 = 项目根（repo root）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "ssd300_coco.yaml")


def project_root() -> str:
    return PROJECT_ROOT


def load_config(path: str | None = None) -> dict:
    """读取 yaml 配置（浅层 dict）。path 缺省用默认 configs/ssd300_coco.yaml。"""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(p) -> str:
    """把相对项目根的路径（如 ./data/xxx）解析为绝对路径；绝对路径原样返回。"""
    if not isinstance(p, str):
        return p
    p = os.path.expanduser(os.path.expandvars(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(PROJECT_ROOT, p))


def resolve_config_paths(cfg: dict) -> dict:
    """返回一份深拷贝，把常见路径字段解析为绝对路径（不改动原 dict）。

    处理范围：data.{root,image_dir,anno_json,subset.*} / eval.ckpt_path / paths.*
    """
    c = copy.deepcopy(cfg)

    data = c.get("data", {})
    for key in ("root", "image_dir", "anno_json"):
        if isinstance(data.get(key), str):
            data[key] = resolve_path(data[key])
    subset = data.get("subset", {})
    for key in ("train_ids_json", "val_ids_json"):
        if isinstance(subset.get(key), str):
            subset[key] = resolve_path(subset[key])
    data["subset"] = subset
    c["data"] = data

    if isinstance(c.get("eval", {}).get("ckpt_path"), str):
        c["eval"]["ckpt_path"] = resolve_path(c["eval"]["ckpt_path"])
    for key, val in (c.get("paths") or {}).items():
        if isinstance(val, str):
            c["paths"][key] = resolve_path(val)
    return c


def ensure_dirs(cfg: dict) -> None:
    """确保输出/数据子目录存在。"""
    paths = cfg.get("paths") or {}
    for key in ("ckpt_dir", "log_dir", "output_dir"):
        d = paths.get(key)
        if d:
            os.makedirs(resolve_path(d), exist_ok=True)
    subset_dir = os.path.dirname((cfg.get("data", {}).get("subset", {}).get("train_ids_json") or ""))
    if subset_dir:
        os.makedirs(resolve_path(subset_dir), exist_ok=True)


def remove_dir_tree(d: str) -> None:
    """删除目录树（供 --clean 等场景）。"""
    d = resolve_path(d)
    if os.path.isdir(d):
        shutil.rmtree(d)
