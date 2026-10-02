"""nano-vllm bench —— 与上游同配置,便于直接并列比较。

上游 README 的基准配置(RTX 4070 Laptop 8GB):256 序列、输入 100–1024、输出 100–1024,
Nano-vLLM 1434.13 tok/s / vLLM 1361.84 tok/s。本文件的默认参数即该配置。

历史(闸门1,2026-09-30):移植初期参考后端无融合,全量要 1 小时以上,故曾缩量到
32 × 128 作为迭代用。闸门2 完成后(2026-10-02)恢复上游配置——
参考后端 75 tok/s → kernel 后端 811 tok/s → 开图 1600 tok/s,全量已可接受。

环境变量 NANOVLLM_ATTN_BACKEND 选后端(reference / kernel);
enforce_eager 通过 --eager 开关控制(默认开图,与上游一致)。
"""
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    # 上游同配置
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    eager = os.environ.get("NANOVLLM_EAGER", "0") == "1"

    # 模型路径:优先读 NANOVLLM_MODEL 环境变量,默认值为本机约定(作者名一级目录)
    path = os.environ.get("NANOVLLM_MODEL", r"D:\llama.cpp\models\Qwen\Qwen3-0.6B")
    llm = LLM(path, enforce_eager=eager, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    backend = os.environ.get("NANOVLLM_ATTN_BACKEND", "reference")
    print(f"backend={backend} eager={eager}")
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()

