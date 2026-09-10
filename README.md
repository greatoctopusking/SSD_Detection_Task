# SSD 目标检测 · PyTorch 本地 CPU 版

> 对应实验手册：《MindSpore搭建SSD网络实现目标检测任务》（华为 ModelArts / MindSpore 2.2）。
> 本项目**不使用 MindSpore**：改用 **PyTorch**（纯 CPU）在本机从零搭建 **SSD300 + VGG16**，使用 COCO2017（精简）数据完成「数据准备 → 模型构建 → 训练 → mAP 评估 → 推理可视化」全流程。

---

## 1. 项目背景与决策

- 原实验在 ModelArts 云端 Notebook + MindSpore 2.2 上训练（60 epochs × 5000 张，batch 5）。
- 本机**无 NVIDIA GPU**，只能 CPU 训练 → 必须大幅缩小训练规模，但仍保证流程完整、指标趋势正确。
- 代码结构与实验手册一一对应（见 §6 对照表），便于边写边对照理解。

| 决策点 | 选择 | 原因 |
|---|---|---|
| 框架 | PyTorch（CPU 版） | 生态成熟、API 与手册代码最接近 |
| 模型 | SSD300，结构手写 | 教学目的，逐层复现 |
| 骨干 | VGG16（block1~5 + fc6/fc7 转卷积） | 与手册一致 |
| 预训练 | 骨干加载 ImageNet 预训练权重（含 fc6/fc7 → conv6/conv7 迁移） | 收敛快、mAP 明显更高 |
| 数据 | 官方 COCO2017 val2017（5000 张图）+ 官方标注，脚本切子集 | 手册 mini 集是 ModelArts 内置 MindRecord 二进制，PyTorch 不可用；train2017 全量 18GB 不下载 |

> ⚠️ 说明：为在本地 CPU 上跑通，训练样本取自 val2017 图片的子集（演示性质，非标准训练协议）。
> 需要严谨结果时可自行下载完整 train2017（18GB）后切换数据源，代码无需改动。

---

## 2. 环境

本机专用 conda 环境：`ssd`（Python 3.12）。

| 组件 | 版本 |
|---|---|
| Python | 3.12.14 |
| torch / torchvision | 2.14.0+cpu / 0.29.0+cpu |
| opencv-python | 5.0.0 |
| numpy | 2.5.2 |
| pycocotools | 2.0.11 |
| pyyaml / matplotlib / tqdm | 已装 |

激活与（如需重建）安装命令：

```powershell
conda activate ssd
# 如从零重建环境：
conda create -n ssd python=3.12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pycocotools pyyaml matplotlib tqdm
```

---

## 3. 目录结构

```
SSD_Detection_Task/
├── README.md                  # 本文件
├── requirements.txt           # 依赖清单
├── .gitignore                 # 忽略 data/checkpoints/runs/outputs
├── configs/
│   └── ssd300_coco.yaml       # ★ 集中配置：路径、81 类、anchor、epoch/batch/lr
├── ssd/                       # ★ 主代码包（PyTorch）
│   ├── __init__.py
│   ├── config.py              # 读取/校验 yaml 配置 + 路径解析
│   ├── model/                 # ★ 模型三兄弟（已实现）
│   │   ├── anchor.py          # PriorBox 生成 8732 个（手册 GeneratDefaultBoxes）
│   │   ├── backbone.py        # VGG16 前13层 + fc6/fc7→conv6/conv7 + 预训练权重迁移
│   │   └── SSD300.py          # Extra 层(block8~11) + MultiBox 检测头 + SSD300 组装
│   ├── data/
│   │   ├── coco.py            # COCO Dataset（读 json 标注 → img/box/label）
│   │   ├── transforms.py      # 随机裁剪采样/翻转/颜色抖动/归一化（手册代码迁移）
│   │   └── encode.py          # gt→anchor 匹配编码 8732×4（手册 ssd_bboxes_encode）
│   ├── losses.py              # SSDLoss：SmoothL1 定位 + focal 置信度 + 归一化
│   └── utils/
│       ├── lr_schedule.py     # warmup + cosine 学习率
│       ├── nms.py             # 推理 NMS（手册 apply_nms）
│       ├── postprocess.py     # 解码预测框 + 逐类阈值/NMS 过滤（手册 SsdInferWithDecoder）
│       ├── coco_eval.py       # pycocotools 计算 mAP（手册 COCOMetrics）
│       └── viz.py             # GT 抽查 / 检测结果画框
├── scripts/
│   ├── download_data.py       # 数据获取说明（本机数据已就位）
│   ├── make_subset.py         # 切分 CPU 友好的训练/验证子集
│   └── benchmark_cpu.py       # 测本机单步耗时 → 决定 epoch/子集规模
├── train.py                   # 训练入口
├── eval.py                    # mAP 评估入口
├── infer.py                   # 单图推理 + 画框
├── data/                      # (git 忽略) 图像与标注 + subsets/
├── checkpoints/               # 权重输出 ssd-*.pth
├── runs/                      # 训练日志
└── outputs/                   # 评估 json / 可视化图片
```

> 说明：`ssd/model/` 三个文件（anchor / backbone / SSD300）已实现并通过验收；
> 其余模块亦已实现。当前仅剩训练/评估/推理的实际运行待执行（§7）。

**运行约定（重要）**：库文件不要用 `python 路径/文件.py` 直接运行（会因 `sys.path` 与文件名遮蔽产生各种 `ModuleNotFoundError`），
统一从**仓库根目录**用包路径或 `-m` 调用：

```powershell
cd D:\GithubRepositories\SSD_Detection_Task
D:\anaconda3\envs\ssd\python.exe -m ssd.model.SSD300     # 模型形状冒烟测试
D:\anaconda3\envs\ssd\python.exe train.py                # 训练
D:\anaconda3\envs\ssd\python.exe scripts\benchmark_cpu.py
```

---

## 4. 数据说明与状态

**数据来源（官方直链）**

- 图片：`https://images.cocodataset.org/zips/val2017.zip`（5000 张，约 741 MB）
- 标注：`https://images.cocodataset.org/annotations/annotations_trainval2017.zip`（约 241 MB，内含 `instances_val2017.json` 等）

**当前状态**

- [x] `val2017` 图片已解压到 `data/images/`（5000 张，约 777 MB，纯 .jpg）
- [x] 官方标注已就位：`data/annotations/instances_val2017.json`（5000 图 / 36781 标注 / 80 类，类 id 1~90 非连续）
- [x] 子集已切分：`data/subsets/train_ids.json`（500 张，覆盖全部 80 类）+ `val_ids.json`（300 张，78 类），train/val 互不重叠
- [x] GT 抽查图已生成：`outputs/data_check/gt_*.jpg`（含 2~17 个真实框），数据通路（读图→增强→编码→tensor）单测通过

---

## 5. 使用方法

```powershell
conda activate ssd
cd D:\GithubRepositories\SSD_Detection_Task

# 1) 数据准备（下载/解压 + 切子集；数据已就位，可重跑切分）
python scripts/make_subset.py --train 500 --val 300 --vis 6

# 2) 模型冒烟测试（形状：8732 锚点 / (B,8732,4) / (B,8732,81)）
python -m ssd.model.SSD300

# 3) CPU 测速，决定子集与 epoch 规模
python scripts/benchmark_cpu.py --batch 4 --steps 3

# 4) 训练
python train.py --config configs/ssd300_coco.yaml

# 5) 评估 mAP
python eval.py --ckpt checkpoints/ssd-05.pth

# 6) 单图推理画框
python infer.py --ckpt checkpoints/ssd-05.pth --image data/images/000000000139.jpg
```

---

## 6. 与实验手册代码对照表

| 手册（MindSpore） | 本项目（PyTorch） | 文件 |
|---|---|---|
| `GeneratDefaultBoxes`（8732 锚点） | `generate_default_boxes` | `ssd/model/anchor.py` |
| `Vgg16`（block1~5）+ block6/7 | `Backbone` / `build_backbone` | `ssd/model/backbone.py` |
| block8~11 + `MultiBox` + `SSD300Vgg16` | `SSD300` / `Head` / `ExtraBlock` | `ssd/model/SSD300.py` |
| `preprocess_fn` / `_data_aug` / 随机采样 | `preprocess_train` / `random_sample_crop` | `ssd/data/transforms.py` |
| `ssd_bboxes_encode`（匹配编码） | `match_and_encode` | `ssd/data/encode.py` |
| `create_ssd_dataset`（MindDataset） | `COCODataset` + `build_dataloader` | `ssd/data/coco.py` |
| `class_loss`(focal) + SmoothL1 定位损失 | `SSDLoss` | `ssd/losses.py` |
| `apply_nms` | `nms` | `ssd/utils/nms.py` |
| `SsdInferWithDecoder` | `decode_boxes` / `filter_predictions` | `ssd/utils/postprocess.py` |
| `COCOMetrics` / `COCOeval`（mAP） | `COCOMetrics` | `ssd/utils/coco_eval.py` |
| `get_lr`（warmup+cosine） | `build_lr_schedule` | `ssd/utils/lr_schedule.py` |

---

## 7. 进度

- [x] 01 项目结构设计
- [x] 02 环境搭建（conda `ssd` + 全部依赖验证通过）
- [x] 02b 目录骨架：全部目录 + 占位文件 + `configs/ssd300_coco.yaml` 模板
- [x] 03a 数据：`val2017` 图片（5000 张，777 MB）
- [x] 03b-1 数据：官方标注 `instances_val2017.json`
- [x] 03b-2 数据：切分 train500/val300 + GT 抽查图 + 数据通路单测
- [x] 03c 代码：除 `model/` 外的全部模块（config/data/losses/utils + train/eval/infer + scripts）
- [x] 04a 模型：`model/anchor.py`（8732 锚点，验收断言 + 数据通路联调通过）
- [x] 04b 模型：`model/backbone.py`（VGG16 + conv6/conv7 权重迁移，四个等号验收通过）
- [x] 04c 模型：`model/SSD300.py`（Extra + MultiBox，输出 (B,8732,4)/(B,8732,81)；backbone 20.48M / 总计 34.30M）
- [ ] 05 跑通 1 个训练 step（loss 为有限值）+ benchmark 测速定规模
- [ ] 06 CPU 小规模训练（观察 loss 下降）
- [ ] 07 mAP 评估（pycocotools）
- [ ] 08 推理画框 + 使用说明收尾

---

## 8. 免责声明

本项目为学习用途的本地 CPU 复现实验：训练子集规模远小于原实验，最终指标仅用于验证流程正确性（loss 下降、mAP 随训练提升、能输出检测框），不代表模型精度水平。
