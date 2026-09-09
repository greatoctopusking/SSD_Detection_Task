"""骨干网络 —— VGG16 前 13 层卷积(block1~5) + fc6/fc7 卷积化。★ 待与用户一起实现 ★

【接口契约 —— SSD300 依赖它，请严格实现】：
    def build_backbone(pretrained: bool = True) -> nn.Module
        forward(x): x 为 (B,3,300,300) float32
        → 返回 (block4, feat) 元组
          block4: (B,512,38,38)   —— 第 4 层最大池化前输出
          feat  : (B,1024,19,19)  —— block5 + pool(m5=3x3 stride1 pad SAME) + block6(空洞 dilation=6,padding=6)
                                       + block7(1x1) 之后的结果（手册结构：19x19）

  结构要点：block1~5 即 VGG16 前 13 个卷积层（m5 的 2x2 stride2 池化改为 3x3 stride1 SAME，
  保持 19x19 不缩）；fc6(4096) -> Conv2d(512,1024,3,dilation=6,padding=6)，fc7(4096)->Conv2d(1024,1024,1)。
  pretrained=True 时从 torchvision vgg16(IMAGENET1K_V1) 迁移权重（features 部分直接拷贝，
  fc6/fc7 权重 reshape 成卷积核并转置），并返回已迁移 backbone。
"""

import torch
import torch.nn as nn

import torchvision.models as mds

class DoubleConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.cv1(x))
        x = self.relu2(self.cv2(x))
        return x

class TripleConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.cv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.cv1(x))
        x = self.relu2(self.cv2(x))
        x = self.relu3(self.cv3(x))
        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Sequential(
            DoubleConv(inc=3, outc=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(inc=64, outc=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            TripleConv(inc=128, outc=256),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            TripleConv(inc=256, outc=512)
        )

        self.part2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            TripleConv(inc=512, outc=512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        block4 = self.part1(x)
        x = self.part2(block4)
        return block4, x

def load_pretrained_weights(net):
    vgg = mds.vgg16(weights=mds.VGG16_Weights.IMAGENET1K_V1, progress=True)
    src_convs = [m for m in vgg.features if isinstance(m, nn.Conv2d)]
    dst_convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    print(f"预训练权重卷积层数：{len(src_convs)}")
    print(f"目标网络卷积层数：{len(dst_convs)}")
    assert len(src_convs) >= 13
    for src, dst in zip(src_convs, dst_convs):
        dst.weight.data.copy_(src.weight.data)
        if dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)

    src_lins = [m for m in vgg.classifier if isinstance(m, nn.Linear)][:2]  # 只留 fc6, fc7
    conv6, conv7 = dst_convs[-2:]

    # fc6(4096,25088) → 前1024行 → 还原(512,7,7) → 3×3采样(0::3) → (1024,512,3,3)
    w6 = src_lins[0].weight.detach()[:1024].view(1024, 512, 7, 7)[:, :, 0::3, 0::3].contiguous()
    conv6.weight.data.copy_(w6)
    conv6.bias.data.copy_(src_lins[0].bias.detach()[:1024])

    # fc7(4096,4096) → 前1024×1024 → (1024,1024,1,1)
    w7 = src_lins[1].weight.detach()[:1024, :1024].view(1024, 1024, 1, 1)
    conv7.weight.data.copy_(w7)
    conv7.bias.data.copy_(src_lins[1].bias.detach()[:1024])

    print("已加载预训练参数")
    

def build_backbone(pretrained=True) -> nn.Module:
    net = Backbone()
    if(pretrained):
        load_pretrained_weights(net)
    return net

if __name__ == "__main__":
    model = build_backbone()