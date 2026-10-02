#!/usr/bin/env python
r"""等价性探针:attention_rocm 替代实现 vs 朴素逐元素参考。

为什么要它(教学区 NOTES.md 教训 ×3:涉及时序/边界的结论必须先跑实验):
  移植前先证明替代实现与「flash_attn 应有的语义」一致。参考实现写成显式双重循环,
  每个 (query 行 j, key 列 m) 单独判 mask,肉眼可核;真值不复用实现的 gather 逻辑——
  池子用 scatter 建、真值直接吃连续 KV,这样 gather 的 bug 也逃不过对比。

用法: D:\项目\python\nano-vllm-rocm\.venv\Scripts\python.exe tools\probe_attention_equivalence.py
r"""
import importlib.util
import sys
from pathlib import Path

import torch

# 用单文件加载而不是 from nanovllm... import ——后者的包 __init__ 连锁导入会先撞上
# flash_attn(此探针本来就是给 attention.py 移植把关的,移植前它还不存在)。
# 这是教学区 learning-records/0002 的同款技巧。
_spec = importlib.util.spec_from_file_location(
    "attention_rocm",
    Path(__file__).resolve().parent.parent / "nanovllm" / "layers" / "attention_rocm.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["attention_rocm"] = _mod
_spec.loader.exec_module(_mod)
rocm_varlen_attn = _mod.rocm_varlen_attn
rocm_with_kvcache = _mod.rocm_with_kvcache


def naive_attn(qi, ki, vi, scale):
    """朴素参考:显式双重循环 + 右下 causal。fp32 计算,做真值。"""
    Lq, H, D = qi.shape
    Lk, KVH, _ = ki.shape
    G = H // KVH
    out = torch.zeros(Lq, H, D, dtype=torch.float32, device=qi.device)
    for j in range(Lq):
        limit = j + (Lk - Lq)              # 右下对齐:行 j 看列 0..limit
        for h in range(H):
            kvh = h // G
            scores = (qi[j, h].float() @ ki[: limit + 1, kvh].float().T) * scale
            p = scores.softmax(dim=-1)
            out[j, h] = p @ vi[: limit + 1, kvh].float()
    return out


def scatter_to_pool(k_seq, v_seq, block_row, num_blocks, block_size):
    """把连续 KV 按 block_table 散射进池子(独立于实现的 gather,做对照构造)。"""
    KVH, D = k_seq.shape[1], k_seq.shape[2]
    pool_k = torch.zeros(num_blocks, block_size, KVH, D, dtype=k_seq.dtype, device=k_seq.device)
    pool_v = torch.zeros_like(pool_k)
    for m in range(k_seq.shape[0]):
        slot = int(block_row[m // block_size]) * block_size + (m % block_size)
        pool_k[slot // block_size, slot % block_size] = k_seq[m]
        pool_v[slot // block_size, slot % block_size] = v_seq[m]
    return pool_k, pool_v


def check(name, out, ref, tol):
    err = (out.float() - ref).abs().max().item()
    ok = err < tol
    print(f"{'✓' if ok else '✗'} {name:48} max|err| = {err:.3e} (tol {tol})")
    return ok


def main():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = 128 ** -0.5
    all_ok = True

    for dtype, tol in [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)]:
        print(f"\n=== dtype={dtype}(tol {tol}) device={device} ===")

        # --- 用例1:prefill 无前缀缓存(Lk==Lq),两条序列 ---
        H, KVH, D = 4, 2, 32   # 小头数即可,G=2 与 Qwen3 同构
        lens = [5, 7]
        total = sum(lens)
        q = torch.randn(total, H, D, dtype=dtype, device=device)
        k = torch.randn(total, KVH, D, dtype=dtype, device=device)
        v = torch.randn(total, KVH, D, dtype=dtype, device=device)
        cu_q = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
        cu_k = cu_q.clone()
        out = rocm_varlen_attn(q, k, v, 5, cu_q, 7, cu_k, scale, block_table=None)
        ref = torch.cat([naive_attn(q[:5], k[:5], v[:5], scale), naive_attn(q[5:], k[5:], v[5:], scale)])
        all_ok &= check("prefill 无缓存 两序列", out, ref, tol)

        # --- 用例2:prefill 带前缀缓存(右下 causal 的主场),含混合批 ---
        # seqA: Lk=20, Lq=6(缓存 14);seqB: Lk=33, Lq=33(全量,但同批里被挂上 block_table)
        block_size, num_blocks = 8, 12
        LkA, LqA, LkB = 20, 6, 33
        kA = torch.randn(LkA, KVH, D, dtype=dtype, device=device)
        vA = torch.randn(LkA, KVH, D, dtype=dtype, device=device)
        kB = torch.randn(LkB, KVH, D, dtype=dtype, device=device)
        vB = torch.randn(LkB, KVH, D, dtype=dtype, device=device)
        blocksA = torch.tensor([3, 5, 9], dtype=torch.int32, device=device)   # ceil(20/8)=3 块
        blocksB = torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32, device=device)  # ceil(33/8)=5 块
        pool_k, pool_v = scatter_to_pool(kA, vA, blocksA, num_blocks, block_size)
        pool_k2, pool_v2 = scatter_to_pool(kB, vB, blocksB, num_blocks, block_size)
        pool_k = pool_k + pool_k2  # 两个 scatter 落在不相交的块上,可叠加
        pool_v = pool_v + pool_v2
        bt = torch.zeros(2, 5, dtype=torch.int32, device=device) - 1
        bt[0, :3] = blocksA
        bt[1, :5] = blocksB
        q2 = torch.randn(LqA + LkB, H, D, dtype=dtype, device=device)
        cu_q2 = torch.tensor([0, LqA, LqA + LkB], dtype=torch.int32, device=device)
        cu_k2 = torch.tensor([0, LkA, LkA + LkB], dtype=torch.int32, device=device)
        out2 = rocm_varlen_attn(q2, pool_k, pool_v, LqA, cu_q2, LkB, cu_k2, scale, block_table=bt)
        ref2 = torch.cat([
            naive_attn(q2[:LqA], kA, vA, scale),       # 真值吃连续 KV(不走实现的 gather)
            naive_attn(q2[LqA:], kB, vB, scale),
        ])
        all_ok &= check("prefill 带缓存+混合批(右下 causal)", out2, ref2, tol)

        # --- 用例3:decode,长度跨块/含 -1 填充/含共享块 ---
        lens_d = [1, 8, 25]   # 1 行;恰好一块;跨 3 块
        maxb = 4
        bt_d = torch.zeros(3, maxb, dtype=torch.int32, device=device) - 1
        bt_d[0, 0] = 7
        bt_d[1, 0] = 2
        bt_d[2] = torch.tensor([1, 3, 5, 8], dtype=torch.int32)  # 25 token 用 4 块中 3 块
        seqs = [torch.randn(L, KVH, D, dtype=dtype, device=device) for L in lens_d]
        pool_dk = torch.zeros(num_blocks, block_size, KVH, D, dtype=dtype, device=device)
        pool_dv = torch.zeros_like(pool_dk)
        for b, L in enumerate(lens_d):
            pk, pv = scatter_to_pool(seqs[b], seqs[b] * 0.5, bt_d[b], num_blocks, block_size)
            pool_dk += pk
            pool_dv += pv
        q3 = torch.randn(3, 1, H, D, dtype=dtype, device=device)
        out3 = rocm_with_kvcache(q3, pool_dk, pool_dv,
                                 torch.tensor(lens_d, dtype=torch.int32, device=device),
                                 bt_d, scale)
        # naive 返回 [1,H,D],stack 后 [3,1,H,D] 即与 rocm_with_kvcache 同形
        ref3 = torch.stack([
            naive_attn(q3[b], seqs[b], seqs[b] * 0.5, scale) for b in range(3)
        ])
        all_ok &= check("decode 跨块+共享池+−1 填充", out3, ref3, tol)

        # --- 用例4:边界——Lq==Lk==1 单 token ---
        q4 = torch.randn(1, H, D, dtype=dtype, device=device)
        k4 = torch.randn(1, KVH, D, dtype=dtype, device=device)
        v4 = torch.randn(1, KVH, D, dtype=dtype, device=device)
        out4 = rocm_varlen_attn(q4, k4, v4, 1,
                                torch.tensor([0, 1], dtype=torch.int32, device=device),
                                1, torch.tensor([0, 1], dtype=torch.int32, device=device),
                                scale, block_table=None)
        all_ok &= check("边界 Lq==Lk==1", out4, naive_attn(q4, k4, v4, scale), tol)

    print("\nVERDICT:", "全部用例等价 ✓" if all_ok else "存在不等价 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
