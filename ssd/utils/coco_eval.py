"""COCO 评估 —— 对应手册 COCOMetrics（基于 pycocotools COCOeval 计算 mAP/AP/AR）。

流程：NMS 过滤 -> 生成 predictions.json -> COCOeval(iouType='bbox') -> E.summarize()
返回：mAP = E.stats[0]（IoU=0.50:0.95 的 AP）
TODO(step 07): 实现 COCOMetrics.update / get_metrics。
"""
