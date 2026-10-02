r"""decode 步的内核级真账:torch.profiler 分解。

为什么存在(2026-10-02,自我纠错):
  前面两个探针用了 cuda events / 手动 sync 夹壳,结果自相矛盾——
  不带探针时 example.py 约 30ms/步(33 tok/s),带探针变 48.8ms/步。
  即**探针自己加了 18ms 以上**,「attention 占 43%」是被污染的数。
  torch.profiler 用 CUPTI/HIP 侧计时,开销低得多,能给出每个 kernel 的真实 GPU 时间
  与 CPU 侧派发时间,从而把「GPU 真在算什么」和「CPU 供不上货」分开。

用法: .venv\Scripts\python.exe tools\probe_profiler.py
"""
import sys

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

PATH = r"D:\llama.cpp\models\Qwen\Qwen3-0.6B"


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "介绍一下你自己"}],
        tokenize=False, add_generation_prompt=True,
    )
    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=40))
    # 先跑热,避免把 warmup/首次编译算进去
    while not llm.is_finished():
        llm.step()

    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=20))
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        while not llm.is_finished():
            llm.step()

    print("=== 按 CUDA 时间排序(前 15) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("=== 按 CPU 时间排序(前 10) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # 注意:torch.profiler 自身有开销,绝对值被放大(实测单步墙钟 ~28-35ms,profiler 下翻倍),
    # 只用来看**相对份额**与调用次数——.item() 的 532 次调用就是这样定位到的。
    evs = list(prof.key_averages())
    total_cpu = sum(getattr(e, "cpu_time_total", 0.0) for e in evs)
    total_cuda = sum(getattr(e, "device_time_total", 0.0) for e in evs)
    print(f"\n合计: CPU {total_cpu/1000:.1f} ms | 设备 {total_cuda/1000:.1f} ms(含 profiler 开销,看份额不看绝对值)")


if __name__ == "__main__":
    sys.exit(main())
