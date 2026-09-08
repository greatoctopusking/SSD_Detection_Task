"""骨干网络 —— VGG16 前 13 层卷积(block1~5) + fc6/fc7 卷积化。

- block6: fc6 -> 3x3 空洞卷积（dilation=6, padding=6, 512->1024）
- block7: fc7 -> 1x1 卷积 (1024->1024)
- 返回 block4(38x38) 与 block7 之后特征供检测层使用
- 支持加载 torchvision ImageNet 预训练 VGG16 权重（pretrained_backbone）
TODO(step 04): 实现 build_backbone + 预训练权重加载。
"""
