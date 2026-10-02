r"""当前移植实现的 attention 内部分解:gather 搬运 vs 数学计算。

为什么存在(2026-10-02,回答「padded gather 成本是什么、上游是多少」):
  上游 nano-vllm 用 flash_attn_with_kvcache,按 block_table 在 kernel 内就地读 KV,
  不物化连续缓冲——所以它的「gather 成本」是 0。本 fork 的路线 B 实现有两次搬运
  (分页池 → 连续张量),要么逐条循环(launch 多)、要么补齐(带宽爆)。
  本探针把当前实现的 gather 与 matmul 分开计时,量化 triton paged kernel 能消掉多少。

用法: .venv\Scripts\python.exe tools\probe_attn_breakdown.py
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "attention_rocm",
    Path(__file__).resolve().parent.parent / "nanovllm" / "layers" / "attention_rocm.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["attention_rocm"] = _mod
_spec.loader.exec_module(_mod)

BLOCK_SIZE, KVH, D, H = 256, 8, 128, 16
NB_POOL = 486


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    dtype = torch.bfloat16
    pool_k = torch.zeros(NB_POOL, BLOCK_SIZE, KVH, D, dtype=dtype, device="cuda")
    pool_v = torch.zeros_like(pool_k)
    print(f"device {torch.cuda.get_device_name(0)} | bs×Lk 组合下 每层每步 的耗时(ms)")
    print(f"{'bs':>4} {'Lk':>6} {'gather':>8} {'repeatKV':>9} {'matmul/softmax':>15} {'合计':>7} {'×28层':>8}")
    for bs, Lk in [(2, 130), (32, 200), (32, 1024), (8, 4096)]:
        nb = (Lk + BLOCK_SIZE - 1) // BLOCK_SIZE
        ids = torch.randint(0, NB_POOL, (bs, nb), device="cuda")
        q = torch.randn(bs, 1, H, D, dtype=dtype, device="cuda")

        def gather():
            return [_mod._gather_seq_kv(pool_k, pool_v, ids[b], Lk) for b in range(bs)]

        kvs = gather()
        ki, vi = kvs[0]

        def rep():
            return (ki.repeat_interleave(H // KVH, dim=1).transpose(0, 1),
                    vi.repeat_interleave(H // KVH, dim=1).transpose(0, 1))

        ke, ve = rep()

        def mm():
            return _mod._attn_single(q[0], ki, vi, D ** -0.5)

        t_g = bench(gather)
        t_r = bench(rep)
        t_m = bench(mm) * bs      # _attn_single 单条,×bs 估算整批
        tot = t_g + t_r + t_m
        print(f"{bs:>4} {Lk:>6} {t_g:>8.2f} {t_r:>9.2f} {t_m:>15.2f} {tot:>7.2f} {tot*28:>8.1f}")

    print("\n参照:上游 flash-attn 的 paged kernel 就地读池,gather+repeatKV 两栏均归零;")
    print("      decode 单步墙钟现状 48.8ms(bs=1) / GPU 数学地板 ≈2ms/步")


if __name__ == "__main__":
    main()
