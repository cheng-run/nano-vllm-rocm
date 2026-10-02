#!/usr/bin/env python
r"""SDPA 稳定性探针:torch 2.13.0+rocm10.0.0(Windows)上 scaled_dot_product_attention 交替报
hipErrorInvalidValue 的现象定性。

为什么存在(2026-09-30 实测发现):连续/非连续、带/不带 scale、同/新张量都无法解释失败模式——
同输入连发会出现 ok/FAIL 交替。移植 attention 前必须知道:成功时结果对不对、哪个后端稳定。

用法: .venv\Scripts\python.exe tools\probe_sdpa_stability.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def manual_attn(q, k, v, scale):
    """手写显式 attention(参考真值 + 备用路径):q/k/v [1, H, L, D]。"""
    scores = torch.einsum("bhld,bhmd->bhlm", q.float(), k.float()) * scale
    L = q.shape[2]
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1).to(q.dtype) @ v


def try_call(fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        return "ok", out
    except Exception as e:
        return f"FAIL({type(e).__name__})", None


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H, L, D = 4, 64, 32
    scale = D ** -0.5
    q = torch.randn(1, H, L, D, device=dev)
    k = torch.randn(1, H, L, D, device=dev)
    v = torch.randn(1, H, L, D, device=dev)
    ref = manual_attn(q, k, v, scale)

    def sdpa_default():
        return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

    # --- 1) 默认后端连发 8 次:状态模式 + 成功时的正确性 ---
    print("=== 默认后端 8 连发(每行=单次调用) ===")
    for i in range(8):
        status, out = try_call(sdpa_default)
        note = ""
        if out is not None:
            err = (out.float() - ref).abs().max().item()
            note = f" max|err|={err:.3e}" + (" ✓对" if err < 5e-2 else " ✗错!")
        print(f"  #{i+1}: {status}{note}")

    # --- 2) 强制各后端连发 4 次 ---
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for be_name, be in [("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                        ("MATH", SDPBackend.MATH)]:
        print(f"=== 强制 {be_name} 4 连发 ===")

        def call(be=be):
            with sdpa_kernel([be]):
                return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        for i in range(4):
            status, out = try_call(call)
            note = ""
            if out is not None:
                err = (out.float() - ref).abs().max().item()
                note = f" max|err|={err:.3e}" + (" ✓对" if err < 5e-2 else " ✗错!")
            print(f"  #{i+1}: {status}{note}")

    # --- 3) 手写路径连发 8 次(对照:纯 torch 算子是否稳定) ===
    print("=== 手写 einsum 路径 8 连发 ===")
    for i in range(8):
        status, out = try_call(lambda: manual_attn(q, k, v, scale))
        note = ""
        if out is not None:
            err = (out - ref).abs().max().item()
            note = f" max|err|={err:.3e}"
        print(f"  #{i+1}: {status}{note}")

    print("\n参考真值 = 手写显式 attention(fp32 朴素实现)")


if __name__ == "__main__":
    main()
