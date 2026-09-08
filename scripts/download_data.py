"""数据获取辅助脚本。

本机当前数据已就绪（用户手动放置）：
    data/images/            COCO2017 val2017 图片 5000 张
    data/annotations/       instances_val2017.json 等官方标注

若日后需重建/补充数据，可用官方直链下载（约 741MB + 241MB）：
    https://images.cocodataset.org/zips/val2017.zip
    https://images.cocodataset.org/annotations/annotations_trainval2017.zip
（本脚本仅作下载入口保留，非必需；如需实现自动下载再补充。）
"""
