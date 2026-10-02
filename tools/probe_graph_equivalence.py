r"""图等价性:同一提示下 eager 与 graph 路径的输出必须一致(闸门2 c 段验收)。

为什么存在:
  图捕获把整步录成固定图,若输入槽填充或长度张量处理有偏差,输出会静默漂移。
  本探针用**确定性采样**(temperature 极低 + 固定种子)让两条路径可比——
  采样本身是随机的,所以必须先把随机性压到最小,否则比对无意义。
  注意:temperature 必须 > 0(本引擎断言不允许纯 greedy),故取一个极小值。

用法: .venv\Scripts\python.exe tools\probe_graph_equivalence.py
"""
import os
import subprocess
import sys

PATH = r"D:\llama.cpp\models\Qwen\Qwen3-0.6B"
BACKEND = os.environ.get("NANOVLLM_ATTN_BACKEND", "kernel")

CHILD = r'''
import sys, torch, os
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(r"%s")
llm = LLM(r"%s", enforce_eager=%s, max_model_len=1024)
p = tok.apply_chat_template([{"role":"user","content":"列出 10 以内的素数"}],
                            tokenize=False, add_generation_prompt=True)
torch.manual_seed(0)
out = llm.generate([p], SamplingParams(temperature=0.01, max_tokens=24), use_tqdm=False)
print("OUT", repr(out[0]["token_ids"]))
''' % (PATH, PATH, "%s")


def run(eager: bool):
    code = CHILD % ("True" if eager else "False")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ, "NANOVLLM_ATTN_BACKEND": BACKEND})
    for line in r.stdout.splitlines():
        if line.startswith("OUT"):
            return line[4:].strip()
    return "失败: " + (r.stderr.strip().splitlines() or ["(无输出)"])[-1][:120]


def main():
    print(f"backend={BACKEND} | temperature=0.01(把采样随机性压到最小,使两条路径可比)")
    a, b = run(True), run(False)
    print(f"eager: {a}")
    print(f"graph: {b}")
    same = a == b and not a.startswith("失败")
    print("VERDICT:", "eager 与 graph 输出一致 ✓" if same else "输出不一致 ✗")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
