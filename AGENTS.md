# nano-vllm-rocm — 项目说明(给 agent 读的)

本仓库是 `GeeeekExplorer/nano-vllm` 的 **ROCm/Windows 适配 fork**,目标平台:
AMD RX 9070 (RDNA4 / gfx1201) + Windows 11 + ROCm 10 + torch 2.13.0+rocm10.0.0。

**为什么不改上游仓库**:用户规矩——不改别人的仓库。本目录是从上游拷贝出来的独立工作区,
上游副本保留在 `D:\项目\python\nano-vllm`(教学用,只读,含 8 课讲义与速查卡)。

## 当前状态

- **闸门 1 已通过(2026-09-30)**:`example.py` / `bench.py` 在 RX 9070 上端到端跑通,
  缩量 bench = 3731 tok / 58.77 s / 63.49 tok/s。
- 移植面:`nanovllm/layers/attention_rocm.py`(新增,替代 flash-attn)+ `attention.py`
  换 import + `model_runner.py` 一行 nccl→gloo 回退。其余 1300 行未改。
- **闸门 2 进行中**:性能增量。候选与实测依据见 `README.md` 的待办与基准表。

## 工程纪律(沿用教学区教训)

- **涉及时序/边界/数字的结论必须先跑实验**,禁止推演后直接下结论。
  验证脚本一律留在 `tools/`,不删(证据价值)。
- **份额类测量必须附带无探针基线**:本项目的头号教训——cuda events/手动同步探针
  自身会给每步加 18ms,导致"attention 占 43%"这类结论完全失真。详见
  `notes/2026-10-02-performance-measurements.md`。判"平台不可用"之前先怀疑探针。
- **改主代码前先写等价性探针**:参考实现写成显式朴素版本,独立构造输入,
  不与被测实现共享逻辑——移植期的 3 个 bug 都是这样当场抓住的。
- 代码注释写**为什么**(中文),引用实测现场或反例,不复述代码在做什么;
  标识符保持英文。
- 下载与联网安装一律由用户本人执行;agent 只提供直链、大小、校验方法。
- 领域词汇以 `CONTEXT.md` 为准(后端/参考后端/图安全/热路径/闸门2/降级路径)。

## 环境

- venv:`D:\项目\python\nano-vllm-rocm\.venv`(uv 管理,`uv sync` 可复现)
- 权重:`D:\llama.cpp\models\Qwen\Qwen3-0.6B\`(用户下载,已校验 sha256)
- **不要用 `torch.nn.functional.scaled_dot_product_attention`**:实测在本栈上交替抛
  `hipErrorInvalidValue` 且异步污染流,证据见 `tools/probe_sdpa_stability.py`。
  需要 attention 就用手写算子或 triton kernel。

## Agent skills

### Issue tracker

Issues live as local markdown under `.scratch/<feature>/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
