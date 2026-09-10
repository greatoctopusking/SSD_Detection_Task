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
import torch
import torch.nn as nn
from ssd.model import backbone

class ExtraBlock(nn.Module):
    def __init__(self, inc, midc, outc, pd, sd):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, padding=pd, stride=sd)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        return self.relu2(self.conv2(x))

class Head(nn.Module):
    def __init__(self, inc, k, n_classes=81):
        super().__init__()
        self.loc_conv = nn.Conv2d(in_channels=inc, out_channels=4 * k, padding=1, kernel_size=3)
        self.cls_conv = nn.Conv2d(in_channels=inc, out_channels=k * n_classes, padding=1, kernel_size=3)

    def forward(self, x):
        loc = self.loc_conv(x)
        cls = self.cls_conv(x)
        return loc, cls

class SSD300(nn.Module):
    def __init__(self, pretrained_backbone=True, num_classes=81):
        super().__init__()
        self.n_classes = num_classes
        self.backbone = backbone.build_backbone(pretrained=pretrained_backbone)
        self.extrab8 = ExtraBlock(inc=1024, midc=256, outc=512, pd=1, sd=2)
        self.extrab9 = ExtraBlock(inc=512, midc=128, outc=256, pd=1, sd=2)
        self.extrab10 = ExtraBlock(inc=256, midc=128, outc=256, pd=0, sd=1)
        self.extrab11 = ExtraBlock(inc=256, midc=128, outc=256, pd=0, sd=1)

        self.head4 = Head(inc=512, k=4, n_classes=num_classes)
        self.headfeat = Head(inc=1024, k=6, n_classes=num_classes)
        self.head8 = Head(inc=512, k=6, n_classes=num_classes)
        self.head9 = Head(inc=256, k=6, n_classes=num_classes)
        self.head10 = Head(inc=256, k=4, n_classes=num_classes)
        self.head11 = Head(inc=256, k=4, n_classes=num_classes)

    def flatten(self, x, tail):
        assert len(x.size()) == 4
        bs = x.size(0)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(bs, -1, tail)

    def forward(self, x):
        b4, feat = self.backbone(x)
        b8 = self.extrab8(feat)
        b9 = self.extrab9(b8)
        b10 = self.extrab10(b9)
        b11 = self.extrab11(b10)

        blocks = (b4, feat, b8, b9, b10, b11)
        heads = (self.head4, self.headfeat, self.head8, self.head9, self.head10, self.head11)
        loc_seq = []
        cls_seq = []
        for block, head in zip(blocks, heads):
            l, c = head(block)
            l = self.flatten(l, 4)
            c = self.flatten(c, self.n_classes)
            loc_seq.append(l)
            cls_seq.append(c)
        loc = torch.cat(loc_seq, dim=1)
        cls = torch.cat(cls_seq, dim=1)

        return loc, cls