r"""差距归因:全量 bench 的 528 tok/s 与 decode 图评估的 1600 tok/s 差在哪。

为什么存在(2026-10-02,补掉交付里唯一的软肋):
  README 记着"全量 528.67 vs 图评估 1600.2,差距尚未归因"。两个候选:
    (a) prefill 未 kernel 化——本轮只把 decode 换成 kernel 后端
    (b) 全量场景上下文长得多——decode 成本随上下文线性增长
  本探针用**单变量对照**直接分离 (a):固定同一提示分布,只切换 `NANOVLLM_ATTN_BACKEND`,
  对比两后端的端到端耗时。因为唯一的差异是 decode 走哪个后端,而 prefill 两份都走
  参考后端,**若两后端耗时接近,则 prefill 占主导**;若 kernel 后端明显更快,
  则 decode 在总耗时中的份额不可忽略。

  刻意不跑 decode 重的场景——那是图评估已经量过的(1600 tok/s),重复测只会再烧一遍时间。

用法: .venv\Scripts\python.exe tools\probe_phase_split.py
"""
import os
import subprocess
import sys
import time

PATH = r"D:\llama.cpp\models\Qwen\Qwen3-0.6B"

# 上游同配置的输入分布,输出压到 8 让耗时主要落在 prefill 侧;
# 再用一组长输出做对照,看 decode 份额如何随输出长度变化。
CHILD = r'''
import time, sys
from random import randint, seed
from nanovllm import LLM, SamplingParams
seed(0)
n, in_lo, in_hi, out_len = %d, %d, %d, %d
llm = LLM(r"%s", enforce_eager=True, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(randint(in_lo, in_hi))] for _ in range(n)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len) for _ in range(n)]
llm.generate(["warm"], SamplingParams())
import torch; torch.cuda.synchronize(); t0 = time.perf_counter()
llm.generate(prompts, sps, use_tqdm=False)
torch.cuda.synchronize(); wall = time.perf_counter() - t0
in_tok = sum(len(p) for p in prompts); out_tok = n * out_len
print(f"SPLIT {in_tok} {out_tok} {wall:.2f}")
'''


def run(backend, n, in_lo, in_hi, out_len):
    code = CHILD % (n, in_lo, in_hi, out_len, PATH)
    env = {**os.environ, "NANOVLLM_ATTN_BACKEND": backend}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("SPLIT")]
    if not line:
        tail = (r.stderr.strip().splitlines() or ["(无输出)"])[-1]
        return None, tail[:110]
    _, in_tok, out_tok, wall = line[-1].split()
    return (int(in_tok), int(out_tok), float(wall)), None


def main():
    print(f"device: {PATH.split(chr(92))[-1]} | eager | 单变量:只切 backend")
    scenarios = [
        ("prefill 主导(输出 8)", 128, 100, 1024, 8),
        ("上游同配置(输出 512)", 128, 100, 1024, 512),
    ]
    for label, n, lo, hi, out_len in scenarios:
        print(f"\n=== {label} ===")
        res = {}
        for backend in ("reference", "kernel"):
            got, err = run(backend, n, lo, hi, out_len)
            if got is None:
                print(f"  {backend:10} 失败: {err}")
                continue
            in_tok, out_tok, wall = got
            res[backend] = (in_tok, out_tok, wall)
            tot = in_tok + out_tok
            print(f"  {backend:10} 输入 {in_tok:>6} + 输出 {out_tok:>6} = {tot:>6} tok | "
                  f"{wall:>6.1f}s | {tot/wall:>7.1f} tok/s")
        if len(res) == 2:
            wr = res["reference"][2]
            wk = res["kernel"][2]
            print(f"  → kernel 比参考快 {wr/wk:.2f}×  (说明 decode 在总耗时里占 "
                  f"{(1-wk/wr)*100:.1f}%)")


if __name__ == "__main__":
    main()
