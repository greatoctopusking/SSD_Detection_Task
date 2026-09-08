"""子集切分脚本（第 03b 步实现）：从 val2017 的 5000 张中切出训练/验证子集。

产出（写入 data/subsets/）：
    train_ids.json / val_ids.json   —— 图片 id 清单（与 configs yaml 对应）
功能：固定随机种子、可配置大小(默认 train=500/val=500，CPU 友好)、
     按类别分布大致均衡抽样、保证 train 与 val 无重叠。

用法（待实现后）：
    python scripts/make_subset.py --train 500 --val 500
"""
if __name__ == "__main__":
    raise NotImplementedError("make_subset.py 将在第 03b 步（数据切子集）实现")
