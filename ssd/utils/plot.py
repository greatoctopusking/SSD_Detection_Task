"""训练曲线绘制：loss / 学习率 / mAP（matplotlib Agg 后端，不弹窗、可无显示器运行）。"""
from __future__ import annotations

import os

# matplotlib 默认把字体缓存写到用户主目录；在只读 HOME / 沙箱 / HPC 环境下会报
# "Could not save font_manager cache" 并中断。这里优先指向项目内可写目录。
_MPL_DIR = os.environ.get("MPLCONFIGDIR") or os.path.join(os.getcwd(), ".mplconfig")
try:
    os.makedirs(_MPL_DIR, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _MPL_DIR
except OSError:
    pass

import matplotlib

matplotlib.use("Agg")  # 服务器无 GUI 必须
import matplotlib.pyplot as plt  # noqa: E402


def _smooth(values, window: int = 20):
    """简单滑动平均（用于 step 级噪声曲线）。"""
    if not values:
        return []
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot_training_curves(history: dict, out_path: str, title: str = "SSD300 training") -> str:
    """把 history 画成 loss / lr / mAP 三（或两）联图，保存到 out_path（png）。"""
    epochs = history.get("epoch", []) or []
    ep_loss = history.get("loss", []) or []
    ep_lr = history.get("lr", []) or []
    ep_map = history.get("map", []) or []
    step_loss = history.get("step_loss", []) or []
    step_lr = history.get("step_lr", []) or []

    has_map = any(v is not None for v in ep_map)
    n_ax = 3 if has_map else 2
    fig, axes = plt.subplots(n_ax, 1, figsize=(8, 3.2 * n_ax), sharex=False)

    # ---- loss ----
    ax = axes[0]
    if step_loss:
        ax.plot(range(1, len(step_loss) + 1), _smooth(step_loss, 20),
                color="tab:blue", alpha=0.5, linewidth=1,
                label=f"step loss (smooth-20)")
    if epochs and ep_loss:
        ax.plot(epochs, ep_loss, color="tab:red", marker="o", markersize=4,
                label="epoch avg loss")
    ax.set_yscale("log")
    ax.set_xlabel("step / epoch"), ax.set_ylabel("loss (log scale)")
    ax.set_title(f"{title} — loss")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)

    # ---- lr ----
    ax = axes[1]
    if step_lr:
        ax.plot(range(1, len(step_lr) + 1), step_lr, color="tab:green", linewidth=1,
                label="lr (per step)")
    elif epochs and ep_lr:
        ax.plot(epochs, ep_lr, color="tab:green", marker="o", markersize=4, label="lr (per epoch)")
    ax.set_yscale("log")
    ax.set_xlabel("step / epoch"), ax.set_ylabel("learning rate (log)")
    ax.set_title("learning rate schedule")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)

    # ---- mAP ----
    if has_map:
        ax = axes[2]
        xs = [e for e, m in zip(epochs, ep_map) if m is not None]
        ys = [m for m in ep_map if m is not None]
        ax.plot(xs, ys, color="tab:purple", marker="s", markersize=4, label="val mAP@[.5:.95]")
        ax.set_xlabel("epoch"), ax.set_ylabel("mAP")
        ax.set_title("validation mAP")
        ax.grid(alpha=0.3), ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
