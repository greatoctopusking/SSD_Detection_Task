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

