"""SSD300 组装 —— Extra 层 + MultiBox 检测头（手册 block8~11 / MultiBox / SSD300Vgg16）。★ 待与用户一起实现 ★

【接口契约 —— train.py / eval.py / benchmark_cpu.py 依赖它，请严格实现】：
    class SSD300(nn.Module):
        def __init__(self, num_classes: int = 81, pretrained_backbone: bool = True)
        def forward(self, x) -> (loc, cls)
            x  : (B,3,300,300)
            loc: (B,8732,4)   —— 4 = [cy偏移, cx偏移, h对数, w对数]（相对锚点的编码量，方差在 loss/解码处处理）
            cls: (B,8732,81)  —— 每类原始 logits（训练算损失用；eval 由外部 sigmoid）

  结构要点：
    - Extra block8~11：每块 = 1x1 降维(输出一半) + 3x3 提取；输出尺寸 10x10/5x5/3x3/1x1
    - 检测头 MultiBox：对 6 层(38,19,10,5,3,1)特征各用 3x3 卷积，
      loc 通道 = 4*每层锚点数，cls 通道 = num_classes*每层锚点数
    - FlattenConcat：把每层输出按 (B, H*W*每点锚数, 4 或 81) 展平并沿 8732 维拼接
      （注意手册 loc 的 4 通道顺序为 [cy,cx,h,w] 的编码偏移，与 anchor.py 的 [cy,cx,h,w] 对齐）
"""

