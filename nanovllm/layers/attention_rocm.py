"""flash_attn 的 ROCm/Windows 替代实现(路线 B:gather + 手写分块 attention)。

为什么存在(2026-09-30 调研,见 ../nano-vllm/research/2026-09-30-rocm-feasibility.md):
  flash-attn 在 Windows ROCm 上没有任何可安装产物——PyPI 只有 sdist、ROCm fork 的
  release wheel 全是 CUDA/linux、AMD whl-next 索引无此包。vLLM 在 RDNA4 上同样不用
  flash-attn 包而走 Triton attention。本文件等价复现两个调用点的语义:
    rocm_varlen_attn  ≙ flash_attn_varlen_func(prefill,含 block_table 分页读取)
    rocm_with_kvcache ≙ flash_attn_with_kvcache(decode)

为什么不用 torch SDPA(实测,tools/probe_sdpa_stability.py 留档):
  torch 2.13.0+rocm10.0.0(Windows)上的 scaled_dot_product_attention 调用交替爆
  hipErrorInvalidValue,且错误是异步的——会溢出到下一处 CUDA 算子,连"成功"的状态都
  不可信。所以本文件用纯 torch 算子(matmul/softmax/masked_fill)手写 attention,
  按 query 行分块限制分数矩阵内存。

语义要点(逐条由 tools/probe_attention_equivalence.py 对朴素双重循环验证):
  1. causal 右下对齐:query 行 j 允许看 key 列 0..j+(Lk-Lq)。这是 chunked prefill
     带前缀缓存的正确语义——块内新 token 必须看到全部缓存前缀和块内自己之前的 token。
  2. GQA:query 头 h 用 kv 头 h // (H/KVH)(HF/vLLM 约定),repeat_interleave 展开。
  3. decode 的 block_table 末尾有 -1 填充,clamp 到 0 防越界,靠 Lk 长度丢弃填充行。

已知性能取舍(gate 1 只求正确,优化是后续增量):
  - 序列级 Python 循环(prefill 一批几条、decode 一批几百条),换内存安全;
  - repeat_interleave 物化 2× KV、分数矩阵按行分块无 flash 融合。
"""
import torch

from nanovllm.utils.context import get_context


def _attn_single(
    qi: torch.Tensor,
    ki: torch.Tensor,
    vi: torch.Tensor,
    scale: float,
    causal: bool = True,
    row_chunk: int = 256,
) -> torch.Tensor:
    """单序列 attention:qi [Lq,H,D], ki/vi [Lk,KVH,D] → [Lq,H,D]。要求 Lk >= Lq。"""
    Lq, H, D = qi.shape
    Lk, KVH, _ = ki.shape
    G = H // KVH
    # q 头 h ↔ kv 头 h//G(HF/vLLM 的 repeat_kv 约定)
    ke = ki.repeat_interleave(G, dim=1).transpose(0, 1)   # [H, Lk, D]
    ve = vi.repeat_interleave(G, dim=1).transpose(0, 1)   # [H, Lk, D]
    out = torch.empty(Lq, H, D, dtype=qi.dtype, device=qi.device)
    device = qi.device
    if causal:
        # 允许矩阵 [Lq, Lk]:行 j(块内)可看列 0..j+(Lk-Lq)(右下对齐)
        jj = torch.arange(Lq, device=device) + (Lk - Lq)
        mm = torch.arange(Lk, device=device)
        allow = mm[None, :] <= jj[:, None]                  # [Lq, Lk] bool
    for j0 in range(0, Lq, row_chunk):
        j1 = min(j0 + row_chunk, Lq)
        qc = qi[j0:j1].transpose(0, 1).float()              # [H, c, D]
        scores = torch.matmul(qc, ke.float().transpose(1, 2)) * scale  # [H, c, Lk]
        if causal:
            scores = scores.masked_fill(~allow[j0:j1][None], float("-inf"))
        p = scores.softmax(dim=-1).to(vi.dtype)             # softmax 用 fp32,乘回低精度
        out[j0:j1] = torch.matmul(p, ve).transpose(0, 1)    # [H, c, D] → [c, H, D]
    return out


def _gather_seq_kv(pool_k: torch.Tensor, pool_v: torch.Tensor, block_row: torch.Tensor, Lk: int):
    """按一条序列的 block_table 从分页池收出连续 KV。

    pool: [num_blocks, block_size, KVH, D];block_row: [max_blocks] int32(尾部 -1 填充)。
    返回 ki, vi: [Lk, KVH, D]。
    """
    block_size = pool_k.shape[1]
    nblocks = (Lk + block_size - 1) // block_size
    # -1 填充 clamp 到 0 防 gather 越界;这些行不在 [0,Lk) 内,自然被丢弃。
    # 先按【块】索引 [nblocks, block_size, KVH, D] 再展平——不能 reshape 后按行号取,
    # 块 3 的内容在展平行号 24..31,直接取行 3 是别的数据(等价性探针抓过这个 bug)。
    ids = block_row[:nblocks].long().clamp(min=0)
    ki = pool_k[ids].reshape(-1, *pool_k.shape[2:])[:Lk]
    vi = pool_v[ids].reshape(-1, *pool_v.shape[2:])[:Lk]
    return ki, vi


def rocm_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """≙ flash_attn_varlen_func。q: [sumLq, H, D] → [sumLq, H, D]。

    block_table 为 None 时 k/v 是连续张量 [sumLk, KVH, D](此时 Lk==Lq);
    非 None 时 k/v 是分页池 [num_blocks, block_size, KVH, D],按行 block_table[i] 收集。
    """
    n = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)
    # 长度取自上下文的 CPU 侧列表:曾用 cu_seqlens_q.tolist(),那是一次 D2H 同步,
    # 本机约 3ms/次(闸门2 a 段)。列表由 prepare_prefill 在张量化前留好,内容相同。
    ctx = get_context()
    cu_q = ctx.cu_seqlens_q_host if ctx.cu_seqlens_q_host is not None else cu_seqlens_q.tolist()
    cu_k = ctx.cu_seqlens_k_host if ctx.cu_seqlens_k_host is not None else cu_seqlens_k.tolist()
    for i in range(n):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        qi = q[qs:qe]
        if block_table is None:
            ki, vi = k[ks:ke], v[ks:ke]
        else:
            ki, vi = _gather_seq_kv(k, v, block_table[i], ke - ks)
        out[qs:qe] = _attn_single(qi, ki, vi, softmax_scale, causal)
    return out


def rocm_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """≙ flash_attn_with_kvcache。q: [B, 1, H, D] → [B, 1, H, D]。"""
    B = q.shape[0]
    ctx = get_context()
    # 长度取自上下文的 CPU 侧列表(张量化前的原始值),不回读 GPU 标量——
    # 这行曾是逐层同步的热点:本机 D2H 标量同步 3ms × 28 层,吃掉约 1/3 吞吐
    lens = ctx.cache_seqlens_host
    if lens is None:      # 没有上下文时(独立调用/探针)退回张量读取,保持接口可用
        lens = cache_seqlens.tolist()
    outs = []
    for b in range(B):
        Lk = lens[b]
        ki, vi = _gather_seq_kv(k_cache, v_cache, block_table[b], Lk)
        outs.append(_attn_single(q[b], ki, vi, softmax_scale, causal))   # q[b]: [1, H, D]
    # 每条 [1, H, D],沿 batch 维 stack → [B, 1, H, D](与 flash_attn_with_kvcache 同形)
    return torch.stack(outs, dim=0)
