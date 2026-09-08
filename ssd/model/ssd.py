"""SSD300 组装 —— Extra 层 + MultiBox 检测头（对应手册 block8~11 / MultiBox / SSD300Vgg16）。

- Extra: block8~11（1x1 降维 + 3x3 提取，1024->512 / 512->256 / 256->256 / 256->256）
- MultiBox: 对 6 层各做 3x3 卷积 -> loc(8732x4) / cls(8732x81)，FlattenConcat 拼接
- forward: 返回 (pred_loc, pred_label)；eval 模式对 cls 做 sigmoid
TODO(step 04): 实现 SSD300(nn.Module)；forward 形状单测 (B,8732,4)/(B,8732,81)。
"""
