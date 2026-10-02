"""kernel 后端:triton paged attention(仅 decode)。

为什么存在(2026-10-02,闸门2 b 段):
  参考后端靠 gather 把分页 KV 搬成连续张量再算,搬运量随 bs×Lk 增长——实测
  bs=32/Lk=4096 时 46ms/步、静态缓冲 537MB,而权重读取固定 1.14GB,于是 batch 一大
  搬运就压过权重(这正是公式卡 §5「吞吐的最终墙是 KV」)。上游 flash-attn 与 vLLM 的做法
  是在 kernel 内按 block_table 就地读,搬运成本为 0——本文件复现那条路。

为什么天然图安全(闸门2 c 段的前置):
  seqlen 与 block_id 都在 kernel 内从设备张量读取,host 侧不需要 `.item()`/`.tolist()`,
  也没有依赖 GPU 值的 Python 分支。整个 decode 步因此可以被 CUDA Graph 捕获。

特化(先求正确,不求通用):
  D=128、BLOCK_SIZE=256、GQA 比、bf16、causal 全部以 constexpr 固定到本模型
  (Qwen3-0.6B / gfx1201)。不做通用化——泛化是零收益的工程量。

范围:只接管 decode。prefill(Lq>1)仍走参考后端。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,                  # [B, H, D]  每个序列一个 query token
    k_cache_ptr,            # [num_blocks, BLOCK_SIZE, KVH, D]
    v_cache_ptr,            # 同上
    out_ptr,                # [B, H, D]
    cache_seqlens_ptr,      # [B] int32  每条序列的 KV 长度
    block_table_ptr,        # [B, max_blocks] int32,尾部 -1 填充
    bt_stride,              # block_table 行跨距
    scale,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // (H // KVH)          # GQA:q 头 h 用 kv 头 h//G

    seqlen = tl.load(cache_seqlens_ptr + seq_idx)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + seq_idx * H * D + head_idx * D + offs_d).to(tl.float32)

    # online softmax 状态(flash decoding 的标准累加式)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(seqlen, BLOCK_SIZE)   # 设备端算,无 host 同步
    for b in range(0, num_blocks):
        block_id = tl.load(block_table_ptr + seq_idx * bt_stride + b)
        # 池内定位该块该 KV 头的首地址(K/V 交错布局:每块内 KVH 个头的 D 维相邻)
        base = block_id * BLOCK_SIZE * KVH * D + kv_head * D
        k = tl.load(k_cache_ptr + base + offs_t[:, None] * (KVH * D) + offs_d[None, :]).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        # 最后一块可能不满:按真实长度屏蔽,块内的 -1 填充行自然也落在这里
        valid = (b * BLOCK_SIZE + offs_t) < seqlen
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(v_cache_ptr + base + offs_t[:, None] * (KVH * D) + offs_d[None, :]).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(out_ptr + seq_idx * H * D + head_idx * D + offs_d, acc.to(out_ptr.dtype.element_ty))


def paged_attention_decode(
    q: torch.Tensor,            # [B, 1, H, D](与 flash_attn_with_kvcache 同形)
    k_cache: torch.Tensor,      # [num_blocks, BLOCK_SIZE, KVH, D]
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,   # [B] int32(设备张量;长度在 kernel 内读)
    block_table: torch.Tensor,     # [B, max_blocks] int32
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """≙ flash_attn_with_kvcache 的 decode 分支。"""
    B, one, H, D = q.shape
    assert one == 1, "kernel 后端只接管 decode(每序列 1 个 token);prefill 走参考后端"
    KVH = k_cache.shape[2]
    BLOCK_SIZE = k_cache.shape[1]
    q_flat = q.reshape(B, H, D)
    # kernel 内用 fp32 累加,故输出先按 fp32 分配再转回,避免中间精度损失
    out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
    grid = (B, H)
    _paged_attention_decode_kernel[grid](
        q_flat, k_cache, v_cache, out,
        cache_seqlens, block_table, block_table.stride(0), softmax_scale,
        H=H, KVH=KVH, D=D, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out.to(q.dtype).unsqueeze(1)
