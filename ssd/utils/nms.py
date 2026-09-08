"""非极大值抑制 —— 对应手册 apply_nms（推理阶段用，训练不用）。

按置信度降序贪心选取，删除与已选框 IoU > 阈值 的候选，直到候选耗尽或达到 max_boxes。
TODO(step 07): 实现 nms(all_boxes, all_scores, thres, max_boxes)。
"""
