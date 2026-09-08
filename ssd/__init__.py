"""ssd —— PyTorch 版 SSD300 目标检测代码包（对应实验手册《MindSpore搭建SSD网络实现目标检测》）。

子模块：
    config      : 读取/校验 configs/*.yaml
    model       : anchor(8732) / backbone(VGG16) / SSD300(Extra+MultiBox)
    data        : COCODataset / transforms(数据增强) / encode(匹配编码)
    losses      : SmoothL1 定位损失 + focal 置信度损失 + hard negative mining
    utils       : lr_schedule / nms / coco_eval(mAP) / viz(画框)
"""
