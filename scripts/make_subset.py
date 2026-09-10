"""图片 id 清单导出 / 子集切分（不依赖模型，可独立运行）。

两种模式：
    1) 全量（GPU 训练推荐）：--all
       把 data.<split>.anno_json 中所有图片导出为 id 清单（train 只保留有标注的图片）。
    2) 抽样（CPU 冒烟）：--train N --val M
       从各 split 的 json 中随机抽 N/M 张，互不重叠、固定种子。

用法：
    conda activate ssd
    cd D:\\GithubRepositories\\SSD_Detection_Task

    # 全量 COCO2017（先下载好 data/train2017、data/val2017、annotations）
    python scripts/make_subset.py --all

    # 小数据冒烟（用当前本地数据）
    python scripts/make_subset.py --train 500 --val 300 --vis 6

产出：写入 yaml 中 data.<split>.ids_json 指向的文件；--vis N 额外输出 GT 抽查图。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssd.config import ensure_dirs, load_config, resolve_config_paths, resolve_path  # noqa: E402
from ssd.data.coco import load_coco_meta  # noqa: E402
from ssd.utils.viz import draw_gt_sample  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="导出图片 id 清单 / 切分子集")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="使用该 split 的全部图片（全量训练）")
    ap.add_argument("--train", type=int, default=500, help="抽样模式下训练集张数")
    ap.add_argument("--val", type=int, default=300, help="抽样模式下验证集张数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vis", type=int, default=0, help="画前 N 张训练图的 GT 框到 outputs/data_check/")
    return ap.parse_args()


def export_split(split: str, args, cfg: dict) -> None:
    dcfg = cfg["data"][split]
    anno_json = dcfg["anno_json"]
    image_dir = dcfg["image_dir"]
    ids_json = dcfg.get("ids_json") or resolve_path(f"./data/subsets/{split}_ids.json")

    if not os.path.isfile(anno_json):
        raise SystemExit(f"[make_subset] 找不到标注文件: {anno_json}\n"
                         f"  请先下载解压 COCO2017（train2017 / val2017 图片 + annotations），"
                         f"或先用 --split {split} 之外的另一个 split。")

    meta = load_coco_meta(anno_json)
    # 训练集必须有 gt；验证集允许包含无标注图（COCO val 中有少量）
    pool = sorted(i for i, _ in meta["images"].items()
                  if (meta["anns_by_image"].get(i) if split == "train" else True))
    print(f"[make_subset] {split}: 候选图 {len(pool)} 张（json 共 {len(meta['images'])} 张）")

    if args.all:
        ids = pool
        mode = "全量"
    else:
        n = args.train if split == "train" else args.val
        if n > len(pool):
            raise SystemExit(f"[make_subset] 抽样数 {n} 超过候选池 {len(pool)}")
        rng = np.random.default_rng(args.seed)
        ids = sorted(int(i) for i in np.asarray(pool, dtype=int)[rng.permutation(len(pool))[:n]])
        mode = f"抽样(seed={args.seed})"

    missing = [i for i in ids
               if not os.path.isfile(os.path.join(image_dir, meta["images"][i]["file_name"]))]
    if missing:
        raise SystemExit(f"[make_subset] 图片文件缺失（前 5 个 id: {missing[:5]}），"
                         f"请检查 data.{split}.image_dir = {image_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(ids_json)), exist_ok=True)
    with open(ids_json, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    cats = set()
    n_box = 0
    for i in ids:
        for a in meta["anns_by_image"].get(i, []):
            cats.add(a["category_id"])
            n_box += 1
    print(f"[make_subset] {split}: {mode} {len(ids)} 张 -> {ids_json}")
    print(f"[make_subset] {split}: {n_box} 个框 / 覆盖 {len(cats)} 类")

    if split == "train" and args.vis > 0:
        vis_out = resolve_path(os.path.join(cfg["paths"]["output_dir"], "data_check"))
        os.makedirs(vis_out, exist_ok=True)
        for i in ids[: args.vis]:
            p = draw_gt_sample(anno_json, image_dir, i, os.path.join(vis_out, f"gt_{i:012d}.jpg"))
            print(f"[make_subset] GT 抽查: {p}")


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    ensure_dirs(cfg)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        export_split(s, args, cfg)
    print("[make_subset] done")


if __name__ == "__main__":
    main()
