"""子集切分（可独立运行，不依赖模型/不依赖 torch 训练栈）：从 val2017 全量 5000 张中
随机抽出互不重叠的训练/验证子集，输出图片 id 清单，并可选把 GT 画出来做数据抽查。

用法：
    python scripts/make_subset.py                      # 默认 train=500, val=300
    python scripts/make_subset.py --train 800 --val 400 --vis 8
产出：
    data/subsets/train_ids.json / val_ids.json         # 图片 id 列表
    outputs/data_check/gt_*.jpg                        # --vis N 的 GT 抽查图
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# 允许从仓库任意位置运行：把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssd.config import load_config, project_root, resolve_config_paths, resolve_path  # noqa: E402
from ssd.data.coco import load_coco_meta  # noqa: E402
from ssd.utils.viz import draw_gt_sample  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="从 COCO val2017 切出 train/val 子集")
    ap.add_argument("--anno", default=None, help="instances json（默认取 configs yaml 的 data.anno_json）")
    ap.add_argument("--image-dir", default=None, help="图片目录（默认取 config）")
    ap.add_argument("--out", default=None, help="输出 json 目录（默认 data/subsets）")
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vis", type=int, default=0, help="画前 N 张训练图的 GT 框到 outputs/data_check/")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config())
    data_cfg = cfg["data"]

    anno_json = args.anno or data_cfg["anno_json"]
    image_dir = args.image_dir or data_cfg["image_dir"]
    default_train_json = os.path.abspath(resolve_path(data_cfg["subset"]["train_ids_json"]))
    out_dir = args.out or os.path.dirname(default_train_json)

    meta = load_coco_meta(anno_json)
    # 候选池：json 中存在 + 至少有 1 个非 crowd 标注（训练/验证都要求有 gt）
    pool = sorted(i for i, im in meta["images"].items() if meta["anns_by_image"].get(i))
    print(f"[make_subset] 候选图片（有标注）: {len(pool)} / json 总 {len(meta['images'])}")

    need = args.train + args.val
    if need > len(pool):
        raise SystemExit(f"train+val ({need}) 超过候选池 {len(pool)}，请调小 --train/--val")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    shuffled = np.asarray(pool, dtype=int)[order]

    train_ids = sorted(int(i) for i in shuffled[: args.train])
    val_ids = sorted(int(i) for i in shuffled[args.train : args.train + args.val])

    # 校验所选图片文件确实存在
    missing = [i for i in train_ids + val_ids
               if not os.path.isfile(os.path.join(image_dir, meta["images"][i]["file_name"]))]
    if missing:
        raise SystemExit(f"图片文件缺失（前 5 个 id: {missing[:5]}），请检查 --image-dir={image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "train_ids.json")
    val_out = os.path.join(out_dir, "val_ids.json")
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_ids, f)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_ids, f)

    # 统计
    def coverage(ids):
        cats = set()
        for i in ids:
            for a in meta["anns_by_image"].get(i, []):
                cats.add(a["category_id"])
        return cats

    tr_cats, va_cats = coverage(train_ids), coverage(val_ids)
    print(f"[make_subset] train_ids.json = {len(train_ids)} 张 -> {train_out}")
    print(f"[make_subset] val_ids.json   = {len(val_ids)} 张 -> {val_out}")
    print(f"[make_subset] 类别覆盖: train {len(tr_cats)} 类 / val {len(va_cats)} 类"
          f"（80 类中，重复类别 {len(tr_cats & va_cats)} 类）")
    print(f"[make_subset] seed={args.seed}，train 与 val 互不重叠 OK")

    # GT 抽查
    if args.vis > 0:
        vis_out = resolve_path(os.path.join(cfg["paths"]["output_dir"], "data_check"))
        os.makedirs(vis_out, exist_ok=True)
        for i in train_ids[: args.vis]:
            out = os.path.join(vis_out, f"gt_{i:012d}.jpg")
            p = draw_gt_sample(anno_json, image_dir, i, out)
            print(f"[make_subset] GT 抽查: {p}")
    print("[make_subset] done")


if __name__ == "__main__":
    main()
