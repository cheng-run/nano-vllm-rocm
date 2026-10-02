import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 模型路径:优先读 NANOVLLM_MODEL 环境变量,默认值为本机约定(作者名一级目录);
    # 换机器只需设这一个变量,不必改代码。HF 缓存的 config/tokenizer 需与本目录放在一起。
    path = os.environ.get("NANOVLLM_MODEL", r"D:\llama.cpp\models\Qwen\Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
