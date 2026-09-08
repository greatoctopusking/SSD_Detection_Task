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
| 模型 | SSD300，结构全部手写 | 教学目的，逐层复现 |
| 骨干 | VGG16（block1~5 + fc6/fc7 转卷积） | 与手册一致 |
| 预训练 | 骨干加载 ImageNet 预训练权重 | 你选定的方案：收敛快、mAP 明显更高 |
| 数据 | 官方 COCO2017 val2017（5000 张图）+ 官方标注，脚本切分 | 手册 mini 集是 ModelArts 内置 MindRecord 二进制，PyTorch 不可用；train2017 全量 18GB 不下载 |

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
│   ├── config.py              # 读取/校验 yaml 配置
│   ├── model/
│   │   ├── anchor.py          # PriorBox 生成 8732 个（手册 GeneratDefaultBoxes）
│   │   ├── backbone.py        # VGG16 前13层 + fc6/fc7→卷积（block6 空洞 dilation=6）+ 预训练加载
│   │   └── ssd.py             # Extra 层(block8~11) + MultiBox 检测头 + SSD300 组装
│   ├── data/
│   │   ├── coco.py            # COCO Dataset（读 json 标注 → img/box/label）
│   │   ├── transforms.py      # 随机裁剪采样/翻转/颜色抖动（手册代码迁移）
│   │   └── encode.py          # gt→anchor 匹配编码 8732×4（手册 ssd_bboxes_encode）
│   ├── losses.py              # SmoothL1 定位 + focal 置信度 + hard negative mining
│   └── utils/
│       ├── lr_schedule.py     # warmup + cosine 学习率
│       ├── nms.py             # 推理 NMS（手册 apply_nms）
│       ├── coco_eval.py       # pycocotools 计算 mAP（手册 COCOMetrics）
│       └── viz.py             # 检测结果画框
├── scripts/
│   ├── download_data.py       # 下载 COCO 数据 → data/
│   ├── make_subset.py         # CPU 适配：从全量标注切出小训练/验证子集
│   └── benchmark_cpu.py       # 先测本机单步耗时 → 决定 epoch/子集规模
├── train.py                   # 训练入口
├── eval.py                    # mAP 评估入口
├── infer.py                   # 单图推理 + 画框
├── data/                      # (git 忽略) 图像与标注
├── checkpoints/               # 权重输出 ssd-*.pth
├── runs/                      # 训练日志
└── outputs/                   # 评估 json / 可视化图片
```

> 注：目录骨架已建立：`requirements.txt`、`.gitignore`、`configs/ssd300_coco.yaml` 就绪；
> `ssd/**`、`scripts/*`、`train.py / eval.py / infer.py` 目前为**占位文件**（docstring 注明对应手册章节与待实现步骤，入口脚本会提示 NotImplementedError），功能代码将在后续步骤中填充。

---

## 4. 数据说明与状态

**数据来源（官方直链）**

- 图片：`https://images.cocodataset.org/zips/val2017.zip`（5000 张，约 741 MB）
- 标注：`https://images.cocodataset.org/annotations/annotations_trainval2017.zip`（约 241 MB，内含 `instances_val2017.json` 等）

**当前状态**

- [x] `val2017` 图片已下载并解压到 `data/images/`（5000 张，约 777 MB，纯 .jpg）
- [x] 官方标注已就位：`data/annotations/instances_val2017.json`（5000 图 / 36781 标注 / 80 类，类 id 1~90 非连续）
- [ ] `make_subset.py` 切分：从 5000 张中划出独立训练/验证子集（数量可配置，CPU 友好默认值待基准测试后确定）

---

## 5. 使用方法

（待对应脚本实现后更新为可直接执行的命令）

```powershell
conda activate ssd

# 1) 数据准备（下载/解压 + 切子集）
python scripts/make_subset.py

# 2) 训练
python train.py --config configs/ssd300_coco.yaml

# 3) 评估 mAP
python eval.py --config configs/ssd300_coco.yaml --ckpt checkpoints/ssd-*.pth

# 4) 单图推理画框
python infer.py --config configs/ssd300_coco.yaml --ckpt checkpoints/ssd-*.pth --image xxx.jpg
```

---

## 6. 与实验手册代码对照表

| 手册（MindSpore） | 本项目（PyTorch） | 文件 |
|---|---|---|
| `GeneratDefaultBoxes`（8732 锚点） | `AnchorGenerator` | `ssd/model/anchor.py` |
| `Vgg16`（block1~5）+ block6/7 | `build_backbone` | `ssd/model/backbone.py` |
| block8~11 + `MultiBox` + `SSD300Vgg16` | `SSD300` | `ssd/model/ssd.py` |
| `preprocess_fn` / `_data_aug` / 随机采样 | `Transforms` | `ssd/data/transforms.py` |
| `ssd_bboxes_encode`（匹配编码） | `encode_boxes` | `ssd/data/encode.py` |
| `create_ssd_dataset`（MindDataset） | `COCODataset` + `DataLoader` | `ssd/data/coco.py` |
| `class_loss`(focal) + SmoothL1 定位损失 | `SSDLoss` | `ssd/losses.py` |
| `apply_nms` | `nms` | `ssd/utils/nms.py` |
| `COCOMetrics` / `COCOeval`（mAP） | `COCOMetrics` | `ssd/utils/coco_eval.py` |
| `get_lr`（warmup+cosine） | `build_lr_schedule` | `ssd/utils/lr_schedule.py` |
| `SsdInferWithDecoder` | 推理解码 | `infer.py` 内 |

---

## 7. 进度

- [x] 01 项目结构设计（已确认）
- [x] 02 环境搭建（conda `ssd` + 全部依赖验证通过）
- [x] 02b 目录骨架：全部目录 + 占位文件 + `configs/ssd300_coco.yaml` 模板（compileall 通过）
- [x] 03a 数据：`val2017` 图片已下载并解压（5000 张，777 MB）
- [x] 03b-1 数据：官方标注已就位（`instances_val2017.json`）
- [ ] 03b-2 数据：`make_subset.py` 切分训练/验证子集 + 可视化抽查
- [ ] 04 模型实现（anchor / backbone / ssd）+ forward 形状单测（8732）
- [ ] 05 数据通路 + 损失函数（跑通 1 个训练 step）
- [ ] 06 CPU 小规模训练（观察 loss 下降）
- [ ] 07 mAP 评估（pycocotools）
- [ ] 08 推理画框 + README 使用说明收尾

---

## 8. 免责声明

本项目为学习用途的本地 CPU 复现实验：训练子集规模远小于原实验，最终指标仅用于验证流程正确性（loss 下降、mAP 随训练提升、能输出检测框），不代表模型精度水平。
