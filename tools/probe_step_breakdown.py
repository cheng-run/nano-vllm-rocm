r"""decode 步耗时的细分解:CPU(Python 调度/准备) vs 模型调用(内核+launch+同步等待)。

为什么存在(2026-10-02):probe_decode_profile 发现手写 attention 只占 decode 步 ~30%(上界),
另外 ~70% 在哪决定了闸门2 的真杠杆是「图捕获」还是「triton kernel」还是「Python 开销」。
方法:对 scheduler.schedule / postprocess 计纯 CPU 墙钟(不 sync);
model_runner.call 的墙钟单列(内含 sampler 的 .tolist() 同步,即整步 GPU 等待的落点);
attention 的 GPU 流时间用 cuda events 单独量,把「launch/Python 开销」与「真 GPU 计算」分开。

用法: .venv\Scripts\python.exe tools\probe_step_breakdown.py
"""
import os
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine

# 模型路径必须由 NANOVLLM_MODEL 提供——刻意不设默认值:回落作者本机路径会让换机器的人静默读错目录
PATH = os.environ.get("NANOVLLM_MODEL")
if not PATH:
    raise SystemExit("未设置 NANOVLLM_MODEL:请指向 Qwen3-0.6B 权重目录(含 config.json/tokenizer.json/model.safetensors)")


def main():
    # attention GPU 流时间:event 夹住原始调用(不含 sync 等待)
    import nanovllm.layers.attention as att_mod
    orig = att_mod.flash_attn_with_kvcache
    ev_stats = {"t": 0.0, "n": 0}

    pending = []   # (start, end) 对,结束时统一收账——
    # 中途收账会撞 "Both events must be completed":下一次调用入队时上一次 kernel
    # 还在跑(CPU 领先 GPU,恰是 launch-bound 的直接证据,见输出)

    def wrapped(*a, **kw):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig(*a, **kw)
        e.record()
        pending.append((s, e))
        ev_stats["n"] += 1
        return out

    att_mod.flash_attn_with_kvcache = wrapped

    # 三段 CPU 墙钟
    cpu = {"schedule": 0.0, "call": 0.0, "post": 0.0, "steps": 0}

    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=True)
    engine = llm   # LLM 直接继承 LLMEngine(llm.py 只有 pass),没有 .engine 属性

    orig_schedule = engine.scheduler.schedule
    orig_call = engine.model_runner.call
    orig_post = engine.scheduler.postprocess

    def schedule():
        t0 = time.perf_counter()
        r = orig_schedule()
        cpu["schedule"] += time.perf_counter() - t0
        return r

    def call(name, *a, **kw):
        t0 = time.perf_counter()
        r = orig_call(name, *a, **kw)
        cpu["call"] += time.perf_counter() - t0
        return r

    def post(*a, **kw):
        t0 = time.perf_counter()
        r = orig_post(*a, **kw)
        cpu["post"] += time.perf_counter() - t0
        return r

    engine.scheduler.schedule = schedule
    engine.model_runner.call = call
    engine.scheduler.postprocess = post

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "介绍一下你自己"}],
        tokenize=False, add_generation_prompt=True,
    )
    n_tokens = 128
    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=n_tokens))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not engine.is_finished():
        engine.step()
        cpu["steps"] += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    for s, e in pending:      # 全部完成后统一收账
        ev_stats["t"] += s.elapsed_time(e)
    S = cpu["steps"]
    print(f"步数 {S} | 墙钟 {wall*1000/S:.2f} ms/步 | decode ≈ {n_tokens/wall:.1f} tok/s")
    print(f"schedule(CPU)      {cpu['schedule']*1000/S:6.2f} ms/步  ({cpu['schedule']/wall*100:4.1f}%)")
    print(f"model call(含同步) {cpu['call']*1000/S:6.2f} ms/步  ({cpu['call']/wall*100:4.1f}%)")
    print(f"postprocess(CPU)   {cpu['post']*1000/S:6.2f} ms/步  ({cpu['post']/wall*100:4.1f}%)")
    att_ms = ev_stats["t"] / ev_stats["n"] if ev_stats["n"] else 0
    wall_ms = wall * 1000 / S
    att_per_step = att_ms * (ev_stats["n"] / S)
    print(f"attention GPU 流时间 {att_ms:.3f} ms/调用 × {ev_stats['n']/S:.1f} 次/步 "
          f"= {att_per_step:.2f} ms/步 (占墙钟 {att_per_step/wall_ms*100:.1f}%——异步口径,不含 host 等待)")
    print(f"步内非 attention 部分(其余 kernel + launch 间隙 + CPU)≈ {wall_ms - att_per_step:.2f} ms/步")


if __name__ == "__main__":
    main()
