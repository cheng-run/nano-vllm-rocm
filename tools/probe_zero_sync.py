r"""热路径零同步回归检查:decode 步内 D2H 标量回读次数必须为 0。

为什么存在(2026-10-02,闸门2 a 段的验收与防回归手段):
  本机 GPU→CPU 标量同步约 3ms/次(正常 ~10µs),而参考后端曾在每层每序列读一次
  `int(cache_seqlens[b])`,一步付 28 次 → 吃掉约 1/3 吞吐(对照实测 28.1→36.9 tok/s)。
  这个检查断言的是**运行时事实**(发生了多少次标量回读),不是某行代码怎么写,
  所以实现换了也不会误报——它只认「热路径有没有再长出同步」这一件事。

  口径:`aten::_local_scalar_dense` 是 `.item()` / `int(tensor)` / `float(tensor)`
  的实现;`Tensor.tolist()` 走另一条路径(每步采样必须回读一次 token id,那是必要的,
  不计入)。prefill 步允许一次性回读(它每步只发生一次,不在逐层热路径上)。

用法: .venv\Scripts\python.exe tools\probe_zero_sync.py
"""
import os
import sys

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

# 模型路径必须由 NANOVLLM_MODEL 提供——刻意不设默认值:回落作者本机路径会让换机器的人静默读错目录
PATH = os.environ.get("NANOVLLM_MODEL")
if not PATH:
    raise SystemExit("未设置 NANOVLLM_MODEL:请指向 Qwen3-0.6B 权重目录(含 config.json/tokenizer.json/model.safetensors)")
SCALAR_OP = "aten::_local_scalar_dense"


def count_scalar_readbacks(llm, prompt, n_tokens=16):
    """跑一段 decode,返回 (标量回读次数, decode 步数)。"""
    from torch.profiler import ProfilerActivity, profile

    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=4))
    while not llm.is_finished():      # 热跑,排除 warmup/首次编译
        llm.step()

    steps = 0
    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=n_tokens))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        while not llm.is_finished():
            llm.step()
            steps += 1
    hits = sum(e.count for e in prof.key_averages() if e.key == SCALAR_OP)
    return hits, steps


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "介绍一下你自己"}],
        tokenize=False, add_generation_prompt=True,
    )
    hits, steps = count_scalar_readbacks(llm, prompt)
    print(f"decode {steps} 步,{SCALAR_OP} 共 {hits} 次")
    if hits == 0:
        print("VERDICT: 热路径零标量同步 ✓")
        return 0
    print(f"VERDICT: 热路径仍有 {hits} 次标量回读(≈{hits/steps:.1f} 次/步)✗")
    return 1


if __name__ == "__main__":
    sys.exit(main())
