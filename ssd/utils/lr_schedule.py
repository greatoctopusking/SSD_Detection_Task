"""学习率调度 —— 对应手册 get_lr：warmup(线性) + cosine 退火。

    lr_init(0.001) --warmup--> lr_max(0.05) --cosine--> lr_end(=lr_init*ratio)
TODO(step 06): 实现 build_lr_schedule(total_steps, warmup_steps, ...) -> np.ndarray / tensor。
"""
