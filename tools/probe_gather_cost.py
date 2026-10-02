r"""padded gather 的成本实测:检验「图先行」方案的覆盖范围是否现实。

为什么存在(2026-10-02,闸门2 拷问第 2 轮):
  用户先选了「图先行」(Q2=A)。图捕获要求 attention 无 host 同步、形状静态,而当前手写
  实现按序列 Python 循环 + int(cache_seqlens[b])。图安全的纯 torch 版只能把 KV 收集成
  静态张量 [bs, maxLk, KVH, D](padding + mask)。但这一步的显存与带宽成本随 bs×maxLk 增长,
  而 KV 池本就吃掉 90% 显存、带宽又是 decode 的最终瓶颈——所以必须先量出 gather 自身
  每层每步的代价,再决定图能覆盖到多大 bs。这把「图先行」从直觉变成一个可算的取舍。

用法: .venv\Scripts\python.exe tools\probe_gather_cost.py
"""
import time

import torch

BLOCK_SIZE = 256
KVH = 8      # Qwen3-0.6B 的 KV 头数
D = 128


def measure(bs, Lk, nblocks_pool=486, iters=20, dtype=torch.bfloat16):
    """返回 (每层耗时 ms, 每层搬运 MB, 静态缓冲 MB)。"""
    pool = torch.zeros(nblocks_pool, BLOCK_SIZE, KVH, D, dtype=dtype, device="cuda")
    nb = (Lk + BLOCK_SIZE - 1) // BLOCK_SIZE
    ids = torch.randint(0, nblocks_pool, (bs, nb), device="cuda")
    buf_mb = bs * Lk * KVH * D * 2 * 2 / 1e6      # K+V 静态缓冲
    moved_mb = buf_mb                              # 每层写入量 = 缓冲大小

    def once():
        k = pool[ids].reshape(bs, -1, KVH, D)[:, :Lk]
        v = pool[ids].reshape(bs, -1, KVH, D)[:, :Lk]
        return k, v

    for _ in range(3):
        once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    del pool
    torch.cuda.empty_cache()
    return ms, moved_mb, buf_mb


def main():
    print(f"device {torch.cuda.get_device_name(0)} | K/V 单元素 2B(bf16)")
    print(f"{'bs':>5} {'paddedLk':>9} {'每层ms':>8} {'×28层/步ms':>11} {'静态缓冲MB':>11} {'每步流量MB':>11}")
    for bs, Lk in [(8, 4096), (32, 4096), (128, 1024), (128, 4096), (256, 1280)]:
        ms, moved, buf = measure(bs, Lk)
        print(f"{bs:>5} {Lk:>9} {ms:>8.2f} {ms*28:>11.1f} {buf:>11.1f} {moved*28:>11.1f}")
    print("\n对照:decode 单步墙钟现状 48.8ms;GPU 数学地板 ≈2ms/步;带宽 640GB/s(规格)")


if __name__ == "__main__":
    main()
