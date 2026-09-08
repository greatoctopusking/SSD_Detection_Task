"""训练入口（第 06 步实现）。

用法（待实现后）：
    conda activate ssd
    python train.py --config configs/ssd300_coco.yaml
流程规划：读配置 -> 构建数据集(子集) -> SSD300(+预训练骨干) -> SSDLoss
         -> warmup+cosine LR + Momentum -> 循环 epoch/step 打印 loss -> 定期存 ckpt
"""
if __name__ == "__main__":
    raise NotImplementedError("train.py 将在第 06 步（CPU 小规模训练）实现")
