r"""decode 步耗时画像:attention(手写实现)占整步的比例。

为什么存在(2026-10-02,闸门2 拷问前的事实采集):
  「写 triton paged attention kernel」是不是正确的杠杆,取决于 decode 步的时间
  到底花在哪。本探针给 rocm_with_kvcache 套计时壳(sync 前后夹),与 generate
  总墙钟对比,得到 attention 份额的**上界**(壳内 sync 会排空流水线,略微高估);
  同时给出均摊每调用耗时,作为 kernel 优化的基线。

用法: .venv\Scripts\python.exe tools\probe_decode_profile.py
"""
import os
import time

import torch
from transformers import AutoTokenizer

import nanovllm.layers.attention as att_mod
from nanovllm import LLM, SamplingParams

# 模型路径:优先读 NANOVLLM_MODEL 环境变量,默认值为本机约定(作者名一级目录);
# 换机器只需设这一个变量,不必改代码
PATH = os.environ.get("NANOVLLM_MODEL", r"D:\llama.cpp\models\Qwen\Qwen3-0.6B")


def main():
    orig = att_mod.flash_attn_with_kvcache
    stats = {"t": 0.0, "n": 0}

    def wrapped(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        torch.cuda.synchronize()
        stats["t"] += time.perf_counter() - t0
        stats["n"] += 1
        return out

    att_mod.flash_attn_with_kvcache = wrapped

    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "介绍一下你自己"}],
        tokenize=False, add_generation_prompt=True,
    )
    n_tokens = 128
    t0 = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=n_tokens), use_tqdm=False)
    wall = time.perf_counter() - t0
    steps = stats["n"] // 28
    print(f"墙钟 {wall:.2f}s | attention 累计 {stats['t']:.2f}s | 份额 {stats['t']/wall*100:.1f}% (上界)")
    print(f"调用 {stats['n']} 次 = 28 层 × ~{steps} 步 | 均摊 {stats['t']/max(stats['n'],1)*1000:.2f} ms/调用")
    print(f"decode 吞吐 ≈ {n_tokens/wall:.1f} tok/s (batch=1)")


if __name__ == "__main__":
    main()
