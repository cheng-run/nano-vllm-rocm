r"""本栈的「单算子派发开销」实测——闸门2 的中心事实。

为什么存在(2026-10-02):
  probe_attn_breakdown 显示:bs=2、Lk=130 的一次 attention,数学量微不足道([16,130] 矩阵),
  却要 0.57ms。算下来本栈每个小算子的派发成本高得反常。闸门2 的两条杠杆(图捕获 /
  融合 kernel)都只是在用不同方式减少「算子派发次数」,所以必须先把单价量出来——
  单价已知,整步的账就能心算,选哪条杠杆也就有依据了。

  方法:同一串小算子,分别(a)逐个发起、(b)录进 CUDA Graph 后重放,对比每步耗时。
  这直接量化「图能买到多少」,而不靠推测。

用法: .venv\Scripts\python.exe tools/probe_op_overhead.py
"""
import time

import torch

D = 1024
N_OPS = 300      # 约等于 decode 一步的算子数量级(28 层 × ~10)


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms


def main():
    print(f"device {torch.cuda.get_device_name(0)}")
    x = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)

    # (a) 单算子:小矩阵乘
    def one_matmul():
        return x @ w

    ms = timeit(one_matmul)
    print(f"单次 {D}×{D} bf16 matmul          : {ms*1000:7.1f} µs")

    # (b) 一串小算子(模拟 decode 一层的算子流)
    def op_chain():
        y = x
        for _ in range(10):
            y = y * 1.0001
            y = y + 0.0
        return y

    ms_chain = timeit(op_chain)
    print(f"20 个小逐元素算子(形状 {D}×{D}) : {ms_chain*1000:7.1f} µs "
          f"→ {ms_chain*1000/20:5.1f} µs/算子")

    # (c) 把同样的算子流录进图,再重放
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op_chain()   # warmup
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        gemm_out = x @ w          # 放一个真算子进去,模拟真实负载
        ybuf = op_chain()
    ms_graph = timeit(lambda: g.replay())
    print(f"同算子流(含 1 个 matmul) 图重放   : {ms_graph*1000:7.1f} µs")
    print(f"\n对照:上面那串逐个发起 = {ms_chain*1000:.1f} + {ms*1000:.1f} ≈ "
          f"{ms_chain*1000+ms*1000:.1f} µs;图重放 = {ms_graph*1000:.1f} µs "
          f"→ 约 {(ms_chain*1000+ms*1000)/max(ms_graph*1000,1e-9):.1f}× 缩减")

    # (d) 整步推算
    per_op = ms_chain * 1000 / 20
    print(f"\n按每算子 {per_op:.1f}µs 推算:decode 一步约 28 层 × 10 算子/层 + 采样 ≈ 280 算子")
    print(f"  ≈ {280*per_op/1000:.1f} ms/步 的纯派发开销(实测单步墙钟 48.8ms)")


if __name__ == "__main__":
    main()
