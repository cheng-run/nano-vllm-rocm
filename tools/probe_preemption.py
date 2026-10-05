r"""大 batch 吞吐的墙:KV 池超配 -> preempt -> 重算(闸门2 差距归因的第二步)。

为什么存在(2026-10-02):
  两相拆分发现 prefill 只占 0.8%,差距在 decode 侧。观察到一个整齐的相关性:
    bs=32  需 KV ~2.6K  池 124K(不超配) -> 1600 tok/s
    bs=128 需 KV ~137K  池 124K(1.1x)    -> 1107 tok/s
    bs=256 需 KV ~288K  池 124K(2.3x)    ->  529 tok/s
  候选机制:池装不下 -> preempt(扔掉重算) -> 重算即额外 prefill 开销。
  本探针直接数 preempt 次数,把相关性变成因果。

  池容量取自 model_runner.allocate_kv_cache 的实测值(num_kvcache_blocks x block_size)。

用法: .venv\Scripts\python.exe tools\probe_preemption.py
"""
import os
import subprocess
import sys

# 模型路径必须由 NANOVLLM_MODEL 提供——刻意不设默认值:回落作者本机路径会让换机器的人静默读错目录
PATH = os.environ.get("NANOVLLM_MODEL")
if not PATH:
    raise SystemExit("未设置 NANOVLLM_MODEL:请指向 Qwen3-0.6B 权重目录(含 config.json/tokenizer.json/model.safetensors)")

CHILD = r'''
import time, os
from random import randint, seed
from nanovllm import LLM, SamplingParams
import nanovllm.engine.scheduler as sch

seed(0)
n, in_lo, in_hi, out_len = %d, %d, %d, %d
llm = LLM(r"%s", enforce_eager=True, max_model_len=4096)

# 数 preempt:每次抢占都是一次"扔掉重算",重算的 token 会重新走 prefill
counts = {"preempt": 0, "hash_blocks": 0}
orig_preempt = sch.Scheduler.preempt
def counting_preempt(self, seq):
    counts["preempt"] += 1
    return orig_preempt(self, seq)
sch.Scheduler.preempt = counting_preempt

pool = llm.model_runner.config.num_kvcache_blocks * llm.model_runner.block_size

prompts = [[randint(0, 10000) for _ in range(randint(in_lo, in_hi))] for _ in range(n)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len) for _ in range(n)]
llm.generate(["warm"], SamplingParams())
import torch; torch.cuda.synchronize(); t0 = time.perf_counter()
llm.generate(prompts, sps, use_tqdm=False)
torch.cuda.synchronize(); wall = time.perf_counter() - t0

in_tok = sum(len(p) for p in prompts); out_tok = n * out_len
need = in_tok + out_tok     # 上界:所有序列同时活着所需的 KV token
print(f"RESULT {n} {in_tok} {out_tok} {wall:.2f} {pool} {counts['preempt']} {need}")
'''


def run(backend, n, in_lo, in_hi, out_len):
    code = CHILD % (n, in_lo, in_hi, out_len, PATH)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ, "NANOVLLM_ATTN_BACKEND": backend})
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return line.split()[1:]
    return None


def main():
    backend = os.environ.get("NANOVLLM_ATTN_BACKEND", "kernel")
    print(f"backend={backend} | eager | 输出固定 256,只改序列数")
    print(f"{'序列':>5} {'输入tok':>9} {'需KV tok':>10} {'池容量':>9} {'超配':>6} "
          f"{'preempt':>8} {'耗时s':>7} {'吞吐':>8}")
    for n in (32, 128, 256):
        got = run(backend, n, 100, 1024, 256)
        if not got:
            print(f"{n:>5} 失败")
            continue
        n_, in_tok, out_tok, wall, pool, prem, need = (int(got[0]), int(got[1]), int(got[2]),
                                                      float(got[3]), int(got[4]), int(got[5]), int(got[6]))
        tot = in_tok + out_tok
        print(f"{n_:>5} {in_tok:>9} {need:>10} {pool:>9} {need/pool:>5.2f}x "
              f"{prem:>8} {wall:>7.1f} {tot/wall:>8.1f}")


if __name__ == "__main__":
    main()
