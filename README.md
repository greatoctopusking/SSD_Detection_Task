# SSD 目标检测 · PyTorch（CUDA / CPU 通用）

> 对应实验手册：《MindSpore搭建SSD网络实现目标检测任务》（华为 ModelArts / MindSpore 2.2）。
> 本项目**不使用 MindSpore**：改用 **PyTorch** 从零搭建 **SSD300 + VGG16**，数据用 COCO2017，
> 支持 **CUDA 训练（AMP 混合精度）与 CPU 调试**，覆盖「数据准备 → 模型构建 → 训练 → mAP 评估 → 推理可视化」全流程。

---

## 1. 特性一览

| 能力 | 说明 |
|---|---|
| 设备自适应 | `device: auto`：有 CUDA 自动上 GPU，否则 CPU；可用 `--device cuda/cpu` 覆盖 |
| 混合精度 | CUDA 下自动启用 AMP（`torch.amp`），可用 `--no-amp` 关闭 |
| 训练稳定性 | 梯度裁剪 + 非有限 loss 自动跳过 + warmup & cosine 学习率 |
| 断点续训 | `python train.py --resume` 自动续最近一次 run，或 `--resume <ckpt>` |
| 检查点管理 | 每次训练一个 `checkpoints/<日期>_<时间>/` 目录，内含 `best_checkpoint.pth` 与 `current_checkpoint.pth` |
| 可视化与日志 | 每 epoch 生成 `curves.png`（loss / 学习率 / mAP 三联图）、`history.json`、`train_log.txt` |
| 损失函数 | 默认**论文原版 MultiBox**（softmax + hard negative mining）；可切回手册版 focal（`loss.type: focal`） |
| 概率方式一致 | 评估/推理的概率方式按 `loss.type` **自动选择**：multibox→softmax，focal→sigmoid（可用 `--sigmoid` 强制） |
| 数据管线 | 轻量 COCO 索引（丢弃 segmentation）、`pin_memory` / `persistent_workers` / 多 worker |

---

## 2. 环境

### GPU 服务器（推荐）
```bash
conda create -n ssd python=3.12 -y && conda activate ssd

# 先看驱动支持：nvidia-smi 右上角 "CUDA Version: 12.x"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按驱动选 cu121/cu124/cu126
pip install -r requirements.txt

# 验收（必须打印 True 和显卡名）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 本机 CPU（调试用）
```powershell
conda create -n ssd python=3.12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

本机当前：Python 3.12.14 / torch 2.14.0+cpu / torchvision 0.29.0+cpu / opencv 5.0.0 / numpy 2.5.2 / pycocotools 2.0.11。

---

## 3. 数据准备

**全量 COCO2017（GPU 训练推荐）**
```
data/
├── train2017/                                   # 118k 张训练图（18GB）
├── val2017/                                     # 5000 张验证图（1GB）
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── subsets/                                     # 由脚本生成
    ├── train_ids.json
    └── val_ids.json
```
官方直链：
- 图片 `http://images.cocodataset.org/zips/train2017.zip`、`http://images.cocodataset.org/zips/val2017.zip`
- 标注 `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

生成图片 id 清单（`null`/缺失时将自动使用 json 中全部图片，显式生成便于控制规模）：
```powershell
python scripts/make_subset.py --all                 # 全量：train 用有标注图，val 用全部
python scripts/make_subset.py --train 500 --val 300 # 小数据抽样（CPU 冒烟）
```

> 小数据冒烟：把 `configs/ssd300_coco.yaml` 里 `data.train/val` 临时指向 `./data/images` 与
> `instances_val2017.json`（yaml 中已给出注释示例）。

---

## 4. 运行

```powershell
cd D:\GithubRepositories\SSD_Detection_Task
conda activate ssd

# 0) 测速：定 batch 与 epoch 规模（自动用 GPU）
python scripts/benchmark.py --batch 16 --steps 5

# 1) 训练（首次）
python train.py

# 2) 断点续训（自动续最近一次 run）
python train.py --resume

# 3) 评估 mAP（默认自动取最近一次 run 的 best_checkpoint）
python eval.py --ckpt checkpoints/20260101_120000/best_checkpoint.pth

# 4) 单图推理画框
python infer.py --ckpt checkpoints/20260101_120000/best_checkpoint.pth --image data/val2017/000000000139.jpg
```

**训练产物**（`checkpoints/<日期>_<时间>/`）：
```
├── best_checkpoint.pth     # best_metric 最优（默认 val mAP）
├── current_checkpoint.pth  # 最新（含 optimizer/AMP/epoch/历史，用于 --resume）
├── curves.png              # loss / lr / mAP 曲线（每 epoch 刷新）
├── history.json            # 曲线数据
├── train_log.txt           # 完整日志
├── config.yaml             # 本次训练配置快照
└── predictions_epochN.json # 训练中途评估的预测结果
```

**运行约定**：库文件不要用 `python 路径/文件.py` 直接跑（会因 `sys.path` 与文件名遮蔽报 `ModuleNotFoundError`），
一律从仓库根目录调用，或 `python -m ssd.model.SSD300` 这样的模块方式。

---

## 5. 配置说明（`configs/ssd300_coco.yaml`）

| 段 | 关键项 | 说明 |
|---|---|---|
| `device` | `auto/cuda/cpu` | 设备选择 |
| `data.train/val` | `image_dir` `anno_json` `ids_json` | 两个 split 独立配置；`ids_json: null` = 用全部图片 |
| `data` | `num_workers` `pin_memory` `prefetch_factor` `color_jitter` | GPU 建议 workers 8 |
| `loss.type` | `multibox` / `focal` | 论文版（softmax+HNM）/ 手册版（sigmoid+focal） |
| `train` | `batch_size` `epochs` `lr_init` `lr_max` `warmup_epochs` `momentum` `weight_decay` | `lr_max` 随 batch 线性放大（batch16→0.04，batch32→0.08） |
| `train` | `amp` `grad_clip` `deterministic` | 混合精度 / 梯度裁剪 / 可复现模式 |
| `train` | `save_every_n_epochs` `eval_every_n_epochs` `best_metric` | 保存频率 / 训练中评估频率 / best 判据（map 或 loss） |
| `eval` | `min_score` `nms_threshold` `max_boxes` `ckpt_path` | softmax 概率阈值（0.1 起；focal 训练可 0.2~0.45） |
| `paths` | `ckpt_root` `output_dir` `plot` `log_name` | run 目录根 / 输出目录 / 是否绘曲线 / 日志名 |

---

## 6. 与实验手册代码对照表

| 手册（MindSpore） | 本项目（PyTorch） | 文件 |
|---|---|---|
| `GeneratDefaultBoxes`（8732 锚点） | `generate_default_boxes` | `ssd/model/anchor.py` |
| `Vgg16`（block1~5）+ block6/7 | `Backbone` / `build_backbone` | `ssd/model/backbone.py` |
| block8~11 + `MultiBox` + `SSD300Vgg16` | `SSD300` / `Head` / `ExtraBlock` | `ssd/model/SSD300.py` |
| `preprocess_fn` / `_data_aug` / 随机采样 | `preprocess_train` / `random_sample_crop` | `ssd/data/transforms.py` |
| `ssd_bboxes_encode`（匹配编码） | `match_and_encode` | `ssd/data/encode.py` |
| `create_ssd_dataset`（MindDataset） | `COCODataset` / `build_coco_index` / `build_dataloader` | `ssd/data/coco.py` |
| `class_loss`(focal) + SmoothL1 | `MultiBoxLoss`（默认）/ `FocalLoss` | `ssd/losses.py` |
| `apply_nms` | `nms` | `ssd/utils/nms.py` |
| `SsdInferWithDecoder` | `decode_boxes` / `filter_predictions` | `ssd/utils/postprocess.py` |
| `COCOMetrics` / `COCOeval`（mAP） | `COCOMetrics` / `evaluate_map` | `ssd/utils/coco_eval.py`、`ssd/utils/evaluator.py` |
| `get_lr`（warmup+cosine） | `build_lr_schedule` | `ssd/utils/lr_schedule.py` |
| （新增）| 断点续训 / 曲线 / 日志 | `ssd/utils/{ckpt,plot,logger,device}.py` |

---

## 7. 目录结构

```
SSD_Detection_Task/
├── configs/ssd300_coco.yaml     # 唯一配置（GPU/CPU 通用）
├── ssd/
│   ├── config.py                # 配置读取/校验/路径解析
│   ├── model/                   # anchor.py / backbone.py / SSD300.py
│   ├── data/                    # coco.py / transforms.py / encode.py
│   ├── losses.py                # MultiBoxLoss / FocalLoss / build_loss
│   └── utils/                   # nms / postprocess / coco_eval / evaluator /
│                                # lr_schedule / ckpt / plot / logger / device / viz
├── scripts/
│   ├── make_subset.py           # id 清单导出（--all 或抽样）
│   ├── benchmark.py             # 训练 step 测速（CUDA/CPU）
│   └── download_data.py         # 数据下载说明
├── train.py / eval.py / infer.py
├── data/                        # (git 忽略) 数据集 + subsets
├── checkpoints/                 # (git 忽略) 每次训练的 run 目录
├── outputs/                     # (git 忽略) 评估 json / 可视化图
└── runs/                        # (保留) 早期日志目录
```

---

## 8. 训练规模参考

| 场景 | 配置 | 说明 |
|---|---|---|
| 本机 CPU 冒烟 | train 500 张 / batch 4 / 6 epoch | 约 20~30 分钟（约 0.54 s/图像） |
| 单卡 GPU 小规模 | train 5000 张 / batch 16 / 30 epoch | 先用 `scripts/benchmark.py` 实测 |
| 单卡 GPU 全量 | train 118k 张 / batch 16 / 60 epoch | 对齐手册 epoch 数，用 `make_subset.py --all` |

> 说明：训练样本取自 COCO2017 官方图片；早期本地调试版曾使用 val2017 子集（演示性质），
> 全量训练请使用 train2017 训练、val2017 评估，这符合标准协议。

---

## 9. 免责声明

本项目为学习用途：模型结构按手册逐层复现并做了工程增强（预训练骨干、标准 MultiBox 损失、AMP、断点续训等），
指标仅用于验证流程正确性与训练趋势，不代表论文级精度。
