r"""验证「热路径 `.item()` 是主瓶颈」——闸门2 的决定性实验。

为什么存在(2026-10-02,profiler 指向):
  torch.profiler 显示 aten::_local_scalar_dense(= .item())占 CPU 时间 64%,
  每次 3.026ms,532 次 ≈ 20 步 × 28 层——正对应 attention_rocm.rocm_with_kvcache 里
  的 `Lk = int(cache_seqlens[b])`:每层每序列一次 GPU→CPU 同步。本栈上的 D2H 标量同步
  异常昂贵(正常 ~10µs,本机 3ms)。本探针把它换成「调用方已知的 Python 长度列表」,
  其余实现完全不变,测吞吐变化——同一份代码、只去掉同步,所以差异归属明确。

用法: .venv\Scripts\python.exe tools\probe_no_item.py
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import nanovllm.engine.model_runner as mr
import nanovllm.layers.attention as att_mod
from nanovllm import LLM, SamplingParams

PATH = r"D:\llama.cpp\models\Qwen\Qwen3-0.6B"

_spec = importlib.util.spec_from_file_location(
    "attention_rocm",
    Path(__file__).resolve().parent.parent / "nanovllm" / "layers" / "attention_rocm.py",
)
_ar = importlib.util.module_from_spec(_spec)
sys.modules["attention_rocm"] = _ar
_spec.loader.exec_module(_ar)

_lens: list[int] = []   # 由 prepare_decode 钩子填充(CPU 侧本来就有的 Python 列表)


def run(avoid_item: bool, n_tokens=64):
    _lens.clear()
    tok = AutoTokenizer.from_pretrained(PATH)
    llm = LLM(PATH, enforce_eager=True)

    orig_prep = mr.ModelRunner.prepare_decode

    def prep(self, seqs):
        _lens[:] = [len(s) for s in seqs]     # 长度在 CPU 侧已存在,无需回读
        return orig_prep(self, seqs)

    mr.ModelRunner.prepare_decode = prep

    if avoid_item:
        def patched(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal=True):
            # 与 rocm_with_kvcache 逐行相同,只把 int(cache_seqlens[b]) 换成 _lens[b]
            outs = []
            for b in range(q.shape[0]):
                ki, vi = _ar._gather_seq_kv(k_cache, v_cache, block_table[b], _lens[b])
                outs.append(_ar._attn_single(q[b], ki, vi, softmax_scale, causal))
            return torch.stack(outs, dim=0)

        att_mod.flash_attn_with_kvcache = patched

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "介绍一下你自己"}],
        tokenize=False, add_generation_prompt=True,
    )
    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=8))
    while not llm.is_finished():      # 热跑
        llm.step()

    llm.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=n_tokens))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not llm.is_finished():
        llm.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    mr.ModelRunner.prepare_decode = orig_prep
    return n_tokens / wall, wall, n_tokens


def main():
    print(f"device {torch.cuda.get_device_name(0)} | Qwen3-0.6B | batch=1 | eager")
    # 每种模式走独立子进程:LLM 初始化会建默认进程组,同进程建两次会报
    # "trying to initialize the default process group twice!"(实测)
    import subprocess

    for label, flag in [("现状(每层 .item() 同步)", "0"), ("去掉同步(长度从 CPU 侧传入)", "1")]:
        r = subprocess.run(
            [sys.executable, __file__, flag],
            capture_output=True, text=True, encoding="utf-8",
        )
        line = [l for l in r.stdout.splitlines() if "tok/s" in l]
        print(f"{label:34} {line[-1].strip() if line else '失败: ' + r.stderr[-200:]}")
    print("\n注意:唯一差异 = 是否有 D2H 标量同步")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        tps, wall, n = run(sys.argv[1] == "1")
        print(f"结果 {tps:.1f} tok/s ({n} token / {wall:.2f}s)")
    else:
        main()

