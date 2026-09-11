"""训练入口 —— CUDA / CPU 通用。

特性：
    - 设备自适应（yaml `device: auto|cuda|cpu`，可用 --device 覆盖）；
    - AMP 混合精度（CUDA 自动启用，可用 --no-amp 关闭）+ 梯度裁剪 + 非有限 loss 保护；
    - 每次训练创建独立 run 目录：checkpoints/<YYYYmmdd_HHMMSS>/
        best_checkpoint.pth    指标最优（best_metric: map|loss）的权重
        current_checkpoint.pth 最新权重（含优化器 / AMP / epoch / 历史，供断点续训）
        curves.png             loss / 学习率 / mAP 曲线
        train_log.txt          运行日志（txt）
        history.json           loss/lr/mAP 历史
        config.yaml            本次训练配置快照
    - 断点续训：python train.py --resume            （自动续最近一次 run）
                python train.py --resume <ckpt路径>  （续指定 checkpoint）

用法：
    cd D:\\GithubRepositories\\SSD_Detection_Task
    conda activate ssd
    python train.py                                    # 用 configs/ssd300_coco.yaml
    python train.py --epochs 30 --batch-size 16 --device cuda
    python train.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from ssd.config import (
    DEFAULT_CONFIG_PATH,
    ensure_dirs,
    load_config,
    resolve_config_paths,
    resolve_path,
    validate_config,
)
from ssd.data.coco import build_dataloader
from ssd.losses import build_loss
from ssd.model.anchor import generate_default_boxes
from ssd.model.SSD300 import SSD300
from ssd.utils.ckpt import load_checkpoint, save_checkpoint
from ssd.utils.device import configure_backends, describe_device, resolve_device
from ssd.utils.evaluator import evaluate_map
from ssd.utils.logger import RunLogger
from ssd.utils.lr_schedule import build_lr_schedule
from ssd.utils.plot import plot_training_curves


def empty_history() -> dict:
    return {
        "epoch": [], "loss": [], "lr": [], "map": [], "elapsed": [],
        "step_loss": [], "step_lr": [], "step_epoch": [],
    }


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="yaml 路径（默认 configs/ssd300_coco.yaml）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 train.epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 train.batch_size")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu（默认取 yaml 的 device）")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 train.seed")
    ap.add_argument("--no-amp", action="store_true", help="关闭 AMP 混合精度")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="断点续训：--resume 自动续最近一次 run，或 --resume <ckpt路径>")
    ap.add_argument("--run-dir", default=None, help="显式指定 run 目录（默认按时间戳新建）")
    return ap.parse_args()


def latest_run_dir(ckpt_root: str) -> str | None:
    if not os.path.isdir(ckpt_root):
        return None
    runs = [d for d in os.listdir(ckpt_root)
            if os.path.isdir(os.path.join(ckpt_root, d)) and len(d) == 15 and d[8] == "_"]
    if not runs:
        return None
    return os.path.join(ckpt_root, sorted(runs)[-1])


def main():
    args = parse_args()
    cfg = resolve_config_paths(load_config(args.config))
    ensure_dirs(cfg)
    for w in validate_config(cfg):
        print(f"[train][警告] {w}")

    m, t = cfg["model"], cfg["train"]
    if args.epochs:
        t["epochs"] = int(args.epochs)
    if args.batch_size:
        t["batch_size"] = int(args.batch_size)
    seed = int(args.seed if args.seed is not None else t.get("seed", 1))
    set_seed(seed, deterministic=bool(t.get("deterministic", False)))

    # ---------- 设备 ----------
    device = resolve_device(args.device or cfg.get("device", "auto"))
    configure_backends(device)
    amp_enabled = bool(t.get("amp", True)) and device.type == "cuda" and not args.no_amp
    grad_clip = float(t.get("grad_clip", 10.0))
    log_every = int(t.get("log_every_n_steps", 20))
    save_every = int(t.get("save_every_n_epochs", 1))
    keep_every = int(t.get("keep_every_n_epochs", 0))   # 中间快照：每 N epoch 另存一份（0=关闭）
    eval_every = int(t.get("eval_every_n_epochs", 0))
    best_metric = str(t.get("best_metric", "map")).lower()

    # ---------- run 目录（时间戳 / 续训复用） ----------
    ckpt_root = resolve_path(cfg["paths"]["ckpt_root"])
    resume_ckpt = None
    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
    elif args.resume:
        if args.resume == "auto":
            run_dir = latest_run_dir(ckpt_root)
            if run_dir is None:
                raise SystemExit(f"[train] checkpoints/ 下没有可续训的 run 目录: {ckpt_root}")
            resume_ckpt = os.path.join(run_dir, "current_checkpoint.pth")
        else:
            resume_ckpt = resolve_path(args.resume)
            run_dir = os.path.dirname(resume_ckpt)
    else:
        run_dir = os.path.join(ckpt_root, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    log = RunLogger(os.path.join(run_dir, cfg["paths"].get("log_name", "train_log.txt")))
    cur_ckpt_path = os.path.join(run_dir, "current_checkpoint.pth")
    best_ckpt_path = os.path.join(run_dir, "best_checkpoint.pth")
    history_path = os.path.join(run_dir, "history.json")
    curves_path = os.path.join(run_dir, "curves.png")

    # ---------- 锚点 / 数据 ----------
    default_boxes, default_boxes_tlbr = generate_default_boxes(
        feature_map_sizes=m["feature_map_sizes"],
        boxes_per_point=m["boxes_per_point"],
        fk_divisors=m["fk_divisors"],
        img_size=m["input_size"][0],
        scales_min=m["scales_min"],
        scales_max=m["scales_max"],
    )
    train_loader, train_ds = build_dataloader(cfg, "train", default_boxes, default_boxes_tlbr)
    steps_per_epoch = len(train_loader)
    val_loader = None
    if eval_every > 0:
        val_loader, val_ds = build_dataloader(cfg, "val")

    # ---------- 模型 / 损失 / 优化 ----------
    net = SSD300(num_classes=m["num_classes"], pretrained_backbone=bool(m["pretrained_backbone"])).to(device)
    criterion = build_loss(m["num_classes"], cfg.get("loss"))
    # 概率方式与损失配套：multibox(softmax) / focal(sigmoid)
    loss_is_focal = str((cfg.get("loss") or {}).get("type", "multibox")).lower() == "focal"
    lr_arr = build_lr_schedule(
        total_epochs=t["epochs"],
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=t.get("warmup_epochs", 2),
        lr_init=t["lr_init"],
        lr_max=t["lr_max"],
        lr_end_ratio=t.get("lr_end_ratio", 0.05),
    )
    opt = torch.optim.SGD(
        [p for p in net.parameters() if p.requires_grad],
        lr=float(lr_arr[0]), momentum=float(t["momentum"]), weight_decay=float(t["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = empty_history()
    start_epoch, global_step = 1, 0
    # best_map / best_loss 必须分开跟踪：mAP 与 loss 尺度完全不同，混用会让 best 选错
    best_map, best_loss, best_value = None, None, None

    # ---------- 断点续训 ----------
    if resume_ckpt:
        ck = load_checkpoint(resume_ckpt, map_location=device)
        net.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        if scaler.is_enabled() and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        global_step = int(ck.get("global_step", 0))
        best_value = ck.get("best_value")
        best_map = ck.get("best_map")
        best_loss = ck.get("best_loss")
        history = ck.get("history") or empty_history()
        log.log(f"续训：从 {resume_ckpt} 恢复 → 已完成 {start_epoch - 1} epoch / "
                f"global_step={global_step} / best_map={best_map} / best_loss={best_loss}")

    # ---------- 启动信息 ----------
    log.section("SSD300 训练开始")
    log.log(f"device      : {describe_device(device)}")
    log.log(f"run 目录    : {run_dir}")
    log.log(f"数据        : 训练 {len(train_ds)} 张 ({steps_per_epoch} steps/epoch)"
            + (f" / 验证 {len(val_ds)} 张" if val_loader is not None else " / 训练中不评估"))
    log.log(f"超参        : epochs={t['epochs']} batch={t['batch_size']} lr_init={t['lr_init']} "
            f"lr_max={t['lr_max']} warmup={t.get('warmup_epochs')} momentum={t['momentum']} "
            f"wd={t['weight_decay']} seed={seed}")
    log.log(f"损失        : {type(criterion).__name__}   AMP: {amp_enabled}   grad_clip: {grad_clip}")
    log.log(f"评估        : eval_every_n_epochs={eval_every} best_metric={best_metric}")
    log.log(f"保存        : current 每 {save_every} epoch / 中间快照每 {keep_every} epoch（0=关闭）")
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    net.train()
    skipped = 0
    train_start = time.time()

    for epoch in range(start_epoch, int(t["epochs"]) + 1):
        ep_t0 = time.time()
        ep_loss, ep_steps, ep_lr = 0.0, 0, float(lr_arr[0])

        for step, (img, gt_loc, gt_label, num_match) in enumerate(train_loader, start=1):
            cur_lr = float(lr_arr[global_step]) if global_step < len(lr_arr) else float(lr_arr[-1])
            for g in opt.param_groups:
                g["lr"] = cur_lr
            ep_lr = cur_lr

            img = img.to(device, non_blocking=True)
            gt_loc = gt_loc.to(device, non_blocking=True)
            gt_label = gt_label.to(device, non_blocking=True)
            num_match = num_match.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred_loc, pred_cls = net(img)
                loss = criterion(pred_loc, pred_cls, gt_loc, gt_label, num_match)

            if not torch.isfinite(loss):
                skipped += 1
                log.log(f"[警告] epoch {epoch} step {step} loss 非有限({loss.item()})，已跳过该步"
                        f"（累计跳过 {skipped}）")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_value = float(loss.item())
            ep_loss += loss_value
            ep_steps += 1
            global_step += 1

            if step % log_every == 0 or step == steps_per_epoch:
                history["step_loss"].append(loss_value)
                history["step_lr"].append(cur_lr)
                history["step_epoch"].append(epoch)
                mem = (f" mem {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                log.log(f"epoch {epoch}/{t['epochs']} step {step}/{steps_per_epoch} "
                        f"loss {loss_value:.4f} lr {cur_lr:.2e}{mem}")

        avg_loss = ep_loss / max(ep_steps, 1)
        ep_time = time.time() - ep_t0
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["lr"].append(ep_lr)
        history["elapsed"].append(ep_time)

        # ---------- 训练中评估 ----------
        map_value = None
        if val_loader is not None and (epoch % eval_every == 0 or epoch == int(t["epochs"])):
            log.log(f"epoch {epoch}: 开始验证集评估（{len(val_ds)} 张）...")
            ev_t0 = time.time()
            map_value = evaluate_map(
                net, val_loader, default_boxes, cfg["data"]["val"]["anno_json"], device,
                min_score=float(cfg["eval"]["min_score"]),
                nms_threshold=float(cfg["eval"]["nms_threshold"]),
                max_boxes=int(cfg["eval"]["max_boxes"]),
                save_path=os.path.join(run_dir, f"predictions_epoch{epoch}.json"),
                use_sigmoid=loss_is_focal,
                log_fn=log.log,
            )
            log.log(f"epoch {epoch}: val mAP = {map_value:.4f}（耗时 {time.time() - ev_t0:.0f}s）")
        history["map"].append(map_value)

        # ---------- 最优判定（best_map / best_loss 分开跟踪，禁止跨尺度比较） ----------
        if map_value is not None and (best_map is None or map_value > best_map):
            best_map = map_value
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss

        if best_metric == "map":
            # 只有"评估轮次"才参与 best 竞争；未评估的轮次不用 loss 混进来
            is_best = map_value is not None and map_value >= best_map
            candidate = best_map
        else:
            is_best = avg_loss <= best_loss
            candidate = best_loss
        if is_best:
            best_value = candidate

        # ---------- 保存 ----------
        if is_best:
            save_checkpoint(best_ckpt_path, net, epoch=epoch, global_step=global_step,
                            best_value=best_value, best_map=best_map, best_loss=best_loss,
                            history=history, config=cfg)
            log.log(f"epoch {epoch}: 已更新 best_checkpoint（{best_metric}={best_value:.4f}）")
        if save_every > 0 and (epoch % save_every == 0 or epoch == int(t["epochs"])):
            save_checkpoint(cur_ckpt_path, net, optimizer=opt, scaler=scaler, epoch=epoch,
                            global_step=global_step, best_value=best_value, best_map=best_map,
                            best_loss=best_loss, history=history, config=cfg)
        if keep_every > 0 and epoch % keep_every == 0:
            snapshot = os.path.join(run_dir, f"epoch{epoch:03d}_checkpoint.pth")
            save_checkpoint(snapshot, net, epoch=epoch, global_step=global_step,
                            best_value=best_value, best_map=best_map, best_loss=best_loss,
                            history=history, config=cfg)
            log.log(f"epoch {epoch}: 已保存中间快照 {os.path.basename(snapshot)}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        if bool(cfg["paths"].get("plot", True)):
            plot_training_curves(history, curves_path, title=f"SSD300 {os.path.basename(run_dir)}")

        log.log(f"epoch {epoch}/{t['epochs']} 完成 | avg loss {avg_loss:.4f} | "
                f"{ep_time:.0f}s ({ep_time / max(ep_steps, 1):.2f} s/step) | "
                f"best_map={'-' if best_map is None else round(best_map, 4)} | "
                f"best_loss={best_loss:.4f}"
                + (f" | val mAP {map_value:.4f}" if map_value is not None else ""))

    # 兜底：best_metric=map 但整轮都没评估（或从未改进）时，至少留一个 best 文件
    if not os.path.isfile(best_ckpt_path) and os.path.isfile(cur_ckpt_path):
        save_checkpoint(best_ckpt_path, net, epoch=int(t["epochs"]), global_step=global_step,
                        best_value=best_value, best_map=best_map, best_loss=best_loss,
                        history=history, config=cfg)
        log.log("[警告] 本次训练未产生 best（可能未开启评估），已用最终权重补写 best_checkpoint")

    total = time.time() - train_start
    log.log(f"训练结束：总耗时 {total / 60:.1f} 分钟，跳过非有限步 {skipped} 次")
    log.log(f"best_map={best_map} | best_loss={best_loss}")
    log.log(f"best_checkpoint: {best_ckpt_path}")
    log.log(f"current_checkpoint: {cur_ckpt_path}")
    log.log(f"曲线: {curves_path} | 日志: {log.log_path}")
    log.close()


if __name__ == "__main__":
    main()
