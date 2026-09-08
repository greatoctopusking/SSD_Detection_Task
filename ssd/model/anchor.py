"""锚点（PriorBox）生成 —— 对应手册 GeneratDefaultBoxes。

6 个特征层 (38/19/10/5/3/1)，每点 4/6/6/6/4/4 个框，共 8732 个。
表示形式 [cy, cx, h, w]（百分比），另附 tlbr 版本供 IoU 计算。
TODO(step 04): 实现 AnchorGenerator / default_boxes 预计算（numpy 即可，与手册一致）。
"""
