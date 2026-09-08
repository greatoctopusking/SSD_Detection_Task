"""CPU 测速脚本（第 03b/04 步之间运行）：量测本机 1 个训练 step 的真实耗时。

目的：纯 CPU 训练下，用实测的 s/step 反推合理规模
      —— 子集张数 × epochs = 总 step 数，控制在可接受的数小时内。
输出：forward+backward 平均耗时、推算 500/1000/2000 张 × N epochs 的总时长。
用法（待模型骨架可用后）：
    python scripts/benchmark_cpu.py
"""
if __name__ == "__main__":
    raise NotImplementedError("benchmark_cpu.py 将在第 04 步（模型单测）后实现")
