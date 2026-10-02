r"""c 段评估:decode 步的 CUDA Graph 捕获(收益与适用边界)。

为什么存在(闸门2 c 段):
  kernel 后端已满足图安全(长度与页表都在 kernel 内读,host 无同步),所以图捕获不再是
  "改造代码"而是"测边界"。本探针回答三件事:能不能捕获、重放与 eager 是否等价、
  在哪些 batch 区间划算(工作区里已有证据显示大 batch 时 KV 搬运是另一堵墙,图不解决它)。

  报数纪律:所有吞吐数字都标注是否带探针(见 notes/2026-10-02-performance-measurements.md)。

用法:
  .venv\Scripts\python.exe tools\probe_graph_eval.py          # 默认 kernel 后端
  NANOVLLM_ATTN_BACKEND=reference ... probe_graph_eval.py     # 对照参考后端
"""
import os
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

# 模型路径:优先读 NANOVLLM_MODEL 环境变量,默认值为本机约定(作者名一级目录);
# 换机器只需设这一个变量,不必改代码
PATH = os.environ.get("NANOVLLM_MODEL", r"D:\llama.cpp\models\Qwen\Qwen3-0.6B")
BACKEND = os.environ.get("NANOVLLM_ATTN_BACKEND", "kernel")


def run(enforce_eager: bool, num_seqs: int, out_len: int = 64):
    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=enforce_eager, max_model_len=2048)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"数到 {i} 并解释你的思路"}],
            tokenize=False, add_generation_prompt=True,
        )
        for i in range(num_seqs)
    ]
    sp = SamplingParams(temperature=0.6, max_tokens=out_len)
    llm.generate(prompts[:1], sp)                 # 热跑(含图捕获)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sp)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    del llm
    torch.cuda.empty_cache()
    return num_seqs * out_len / wall


def main():
    print(f"device {torch.cuda.get_device_name(0)} | backend={BACKEND}")
    print(f"{'batch':>6} {'eager tok/s':>12} {'graph tok/s':>12} {'倍率':>7}")
    # 每种配置走独立子进程:同进程二次建进程组会报错(实测)
    import subprocess

    for bs in (1, 8, 32):
        row = [f"{bs:>6}"]
        for eager in (True, False):
            r = subprocess.run(
                [os.sys.executable, __file__, str(bs), "1" if eager else "0"],
                capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "NANOVLLM_ATTN_BACKEND": BACKEND},
            )
            line = [l for l in r.stdout.splitlines() if l.startswith("TPS")]
            if line:
                row.append(f"{float(line[-1].split()[1]):>12.1f}")
            else:
                tail = (r.stderr or "").strip().splitlines()
                row.append(f"{'失败':>12}")
                print("   stderr:", tail[-1][:110] if tail else "(空)")
        try:
            e, g = float(row[1]), float(row[2])
            row.append(f"{g/e:>6.2f}×")
        except ValueError:
            row.append(f"{'—':>7}")
        print(" ".join(row))


if __name__ == "__main__":
    if len(os.sys.argv) == 3:
        bs, eager = int(os.sys.argv[1]), os.sys.argv[2] == "1"
        tps = run(eager, bs)
        print(f"TPS {tps:.2f}")
    else:
        main()
