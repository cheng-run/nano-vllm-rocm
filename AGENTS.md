# nano-vllm-rocm — 项目说明(给 agent 读的)

本仓库是 `GeeeekExplorer/nano-vllm` 的 **ROCm/Windows 适配 fork**,目标平台:
AMD RX 9070 (RDNA4 / gfx1201) + Windows 11 + ROCm 10 + torch 2.13.0+rocm10.0.0。

**为什么另建仓库**:nano-vllm 是别人的项目,不改别人的仓库——本仓库是从上游拷贝出来的
独立工作区,所有改动在这里进行。

## 当前状态

- **闸门 1 已通过(2026-09-30)**:`example.py` / `bench.py` 在 RX 9070 上端到端跑通。
- 移植面:`nanovllm/layers/attention_rocm.py`(参考后端)+ `attention_triton.py`
  (kernel 后端)+ `attention.py` 的 seam 切换 + `model_runner.py` 的 nccl→gloo 回退。
  其余 1300 行未改。
- **闸门 2 已完成(2026-10-02)**:decode 路径三步优化,同配置全量 A/B 对照 75.2 → **547.0 tok/s**
  (2026-10-03 复测修正;原记录 810.9 未能复现,见 `README.md` 的复测修正节)。
  图在全量配置下仅 1.01×(短输出 bs=32 时 1600.2 tok/s,两组口径不可比)。
  差距归因见 `README.md`(四个假设全部被证伪,主因仍未知)。

## 工程纪律

- **涉及时序/边界/数字的结论必须先跑实验**,禁止推演后直接下结论。
  验证脚本一律留在 `tools/`,不删(证据价值)。
- **份额类测量必须附带无探针基线**:本项目的头号教训——cuda events/手动同步探针
  自身会给每步加 18ms,导致"attention 占 43%"这类结论完全失真。详见
  `notes/2026-10-02-performance-measurements.md`。判"平台不可用"之前先怀疑探针。
- **改主代码前先写等价性探针**:参考实现写成显式朴素版本,独立构造输入,
  不与被测实现共享逻辑——移植期的 3 个 bug 都是这样当场抓住的。
- **跨脚本比较数字前先对齐口径**:`bench.py` 只算输出 token,多个探针算输入+输出,
  同一实验换口径差 16%。
- 代码注释写**为什么**(中文),引用实测现场或反例,不复述代码在做什么;
  标识符保持英文。
- 领域词汇以 `CONTEXT.md` 为准(后端/参考后端/kernel 后端/seam/图安全/热路径/无探针基线)。

## 环境

- venv:`.venv`(uv 管理,`uv sync` 可复现)
- **跑任何脚本前先设 `PYTHONUTF8=1`**:中文 Windows 下不设则 `import nanovllm` 即崩
  (`UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`,torch inductor 读模板未指定编码)。
  这是区域设置问题,与本 fork 无关,但踩中时现象像"代码坏了"。
- 权重:Qwen3-0.6B 的 safetensors + config/tokenizer;路径由环境变量 `NANOVLLM_MODEL` 提供,
  **没有默认值**——不设即报错(刻意为之:防止换机器时静默读错目录)
- **参考后端不可被 CUDA Graph 捕获**(捕获期它会回退 `cache_seqlens.tolist()`,而捕获期禁止
  CPU↔CUDA 拷贝)。测参考后端必须 `NANOVLLM_EAGER=1`;要开图只能用 kernel 后端。
- **不要用 `torch.nn.functional.scaled_dot_product_attention`**:实测在本栈上交替抛
  `hipErrorInvalidValue` 且异步污染流,证据见 `tools/probe_sdpa_stability.py`。
  需要 attention 就用手写算子或 triton kernel。
- 进程组端口默认 2333(沿用上游值),同机并发两个引擎实例会撞端口而失败
  (实测报 `DistNetworkError`)。需要并行对照实验时给其中一个设 `NANOVLLM_MASTER_PORT`。
