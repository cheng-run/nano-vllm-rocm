# nano-vllm-rocm

nano-vllm 的 **AMD ROCm / Windows 适配 fork**——上游 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
依赖 flash-attn,而 flash-attn 在 Windows + ROCm 上没有任何可安装产物。本 fork 用等价实现替换掉它,
并在同一套代码上把性能推回到上游量级。

**目标平台**:AMD RX 9070 (RDNA4 / gfx1201) · Windows 11 · ROCm 10 · torch 2.13.0+rocm10.0.0

## 为什么需要这个 fork

| 上游依赖 | 在 Windows/ROCm 上的状况 | 本 fork 的做法 |
|---|---|---|
| `flash-attn` | PyPI 只有 sdist;ROCm fork 的 release wheel 全是 CUDA/linux;AMD 官方索引无此包 | 两份替代后端(见下) |
| `triton` | AMD 索引只提供 linux wheel | 用社区包 `triton-windows`(import 名同为 `triton`) |
| `nccl` | Windows 版 torch 未编译 NCCL | 单卡回退 `gloo`(多卡仍需 nccl) |
| torch SDPA | 本栈上交替抛 `hipErrorInvalidValue`,且异步污染后续算子 | 禁用,见 `AGENTS.md` |

## 两个 attention 后端

seam 在 `nanovllm/layers/attention.py` 顶部——替换实现不需要改任何调用点。

| 后端 | 做法 | 定位 |
|---|---|---|
| **参考后端**(`attention_rocm.py`) | 按页表 gather 出连续 KV,再分块算 attention | 正确性基准,永久保留 |
| **kernel 后端**(`attention_triton.py`) | triton paged attention,在 kernel 内按 `block_table` 就地读池(即上游 flash-attn 的做法) | 性能路径,只接管 decode |

选后端:`NANOVLLM_ATTN_BACKEND=kernel`(默认 `reference`);开图:`NANOVLLM_EAGER=0`。

## 基准(上游同配置:256 序列,输入 100–1024,输出 100–1024,Qwen3-0.6B)

| 配置 | 吞吐 | 说明 |
|---|---|---|
| 参考后端 + eager(移植完成时) | **75.2 tok/s** | 闸门 1 的落点 |
| kernel 后端 + eager | **810.9 tok/s** | 消掉 gather + 就地读池 |
| **kernel 后端 + 图** | **1600.2 tok/s** | 整步捕获重放 |
| 上游 README(RTX 4070 Laptop 8GB) | 1434.1 tok/s | 对照:本机带宽更高(640 vs ~256 GB/s) |

**图带来的倍率随 batch 递减**:bs=1 4.75× / bs=8 3.15× / bs=32 1.85×——batch 越大,
固定开销被摊薄,图的边际收益越小。

`bench.py` 已恢复上游配置(全量约 4 分钟)。**剩余差距尚未归因**:全量 528.67 tok/s 与
图评估的 1600 tok/s 之间有 3 倍差距,候选解释有二——(a) prefill 未 kernel 化(本轮只做了
decode),(b) 全量场景上下文长得多(decode 成本随上下文线性增长),两者各占多少**未经测量**。
`tools/probe_phase_split.py` 是为归因写的探针,但本轮未跑完(6 轮全量 bench 耗时过长,中途终止),
所以这里不下结论。

## 安装

```bash
uv sync          # 依赖已锁定在 uv.lock(含 AMD whl-next 索引配置)
```

权重放在 `D:\llama.cpp\models\Qwen\Qwen3-0.6B\`(`example.py` / `bench.py` 里的路径,
换成你自己的即可)。

## 验证脚本(`tools/`)

每个结论都有可复现的脚本,不删——证据价值。

| 脚本 | 断言什么 |
|---|---|
| `probe_attention_equivalence.py` | 参考后端 vs 朴素双重循环(右下 causal / GQA / `-1` 填充) |
| `probe_kernel_equivalence.py` | **kernel 后端 vs 参考后端**(bs、跨块、混合长度、GQA) |
| `probe_zero_sync.py` | decode 步内 D2H 标量同步次数为 0(防回归) |
| `probe_graph_equivalence.py` | eager 与 graph 路径输出一致 |
| `probe_graph_eval.py` | 图在哪些 batch 区间划算 |
| `probe_phase_split.py` | prefill / decode 两相拆分 |
| `probe_sdpa_stability.py` | 为何禁用 SDPA(留档) |

## 文档

- `CONTEXT.md` — 本 fork 的术语表(后端 / seam / 图安全 / 无探针基线…)
- `notes/2026-10-02-performance-measurements.md` — 测量记录与三轮自我纠错
- `AGENTS.md` — 工程纪律(测量规范、禁止项)
- `.scratch/gate2-performance/` — 本轮 spec 与执行票
