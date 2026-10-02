r"""prefill / decode 两相拆分:全量 bench 的差距归属(闸门2「差距去哪了」的证据)。

为什么存在(2026-10-02):
  全量 bench(kernel 后端 + 图)= 528.67 tok/s,而纯 decode 在图评估里 1600 tok/s。
  本轮只把 decode 换成了 kernel 后端,prefill 仍走参考后端——所以差距应当落在 prefill。
  本探针用两种极端配置把两相分开测,把「差距去哪了」从推断变成测量:
    - prefill 重:输入长、输出 1(几乎全是 prefill)
    - decode 重:输入短、输出长(几乎全是 decode)
  报数纪律:全部为无探针基线(不做任何插桩)。

用法: .venv\Scripts\python.exe tools\probe_phase_split.py
"""
import os
import subprocess
import sys
import time

PATH = r"D:\llama.cpp\models\Qwen\Qwen3-0.6B"

CHILD = r'''
import time, os, torch
from random import randint, seed
from nanovllm import LLM, SamplingParams
seed(0)
n, in_lo, in_hi, out_len = %d, %d, %d, %d
llm = LLM(r"%s", enforce_eager=%s, max_model_len=4096)
prompts = [[randint(0, 10000) for _ in range(randint(in_lo, in_hi))] for _ in range(n)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len) for _ in range(n)]
llm.generate(["warm"], SamplingParams())
torch.cuda.synchronize(); t0 = time.perf_counter()
llm.generate(prompts, sps, use_tqdm=False)
torch.cuda.synchronize(); wall = time.perf_counter() - t0
in_tok = sum(len(p) for p in prompts); out_tok = n * out_len
print(f"SPLIT {in_tok} {out_tok} {wall:.2f} {in_tok/wall:.1f} {out_tok/wall:.1f}")
'''


def run(label, n, in_lo, in_hi, out_len, eager):
    code = CHILD % (n, in_lo, in_hi, out_len, PATH, "True" if eager else "False")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ})
    line = [l for l in r.stdout.splitlines() if l.startswith("SPLIT")]
    if not line:
        print(f"{label}: 失败 {(r.stderr.strip().splitlines() or [''])[-1][:100]}")
        return
    _, in_tok, out_tok, wall, in_tps, out_tps = line[-1].split()
    print(f"{label:26} 输入 {int(in_tok):>7} tok | 输出 {int(out_tok):>6} tok | "
          f"{float(wall):>6.1f}s | prefill {float(in_tps):>7.1f} tok/s | decode {float(out_tps):>7.1f} tok/s")


def main():
    backend = os.environ.get("NANOVLLM_ATTN_BACKEND", "kernel")
    print(f"backend={backend} | 无探针基线")
    for eager in (True, False):
        tag = "eager" if eager else "graph"
        print(f"--- {tag} ---")
        run(f"prefill 重({tag})", 256, 100, 1024, 1, eager)
        run(f"decode 重({tag})", 256, 8, 16, 256, eager)
        run(f"上游同配置({tag})", 256, 100, 1024, 512, eager)


if __name__ == "__main__":
    main()
